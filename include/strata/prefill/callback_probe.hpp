// Opt-in callback diagnostics. Host-only; reads at most 64KiB at a time and never writes device memory.
#pragma once
#include "strata/prefill/audit.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::prefill::detail {
inline bool callback_probe_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MAIN_CALLBACK_PROBE"); return v && std::strcmp(v, "1") == 0; }();
    return on;
}
inline thread_local const char** callback_probe_path = nullptr;
inline void callback_probe_draft_path(const char* path) { if (callback_probe_path) *callback_probe_path = path; }
struct CallbackProbeRange {
    const char* name; int64_t layer, ordinal, row;
    const void* data; uint64_t bytes;
    uint64_t recurrence_elements = 0; // GDN row suffix is conv; zero for non-GDN ranges.
};
class CallbackProbe {
    struct Directory {
        std::mutex mutex;
        bool initialized = false;
        std::string root, error;
        uint64_t sequence = 0;
        bool reserve(std::filesystem::path& path, uint64_t& seq, std::string& err) {
            std::lock_guard<std::mutex> lock(mutex);
            std::error_code ec;
            if (!initialized) {
                initialized = true;
                const char* v = std::getenv("STRATA_MAIN_CALLBACK_PROBE_DIR");
                root = v ? v : "";
                if (root.empty()) error = "callback probe: unique STRATA_MAIN_CALLBACK_PROBE_DIR required";
                else if (!std::filesystem::create_directory(root, ec))
                    error = "callback probe: cannot exclusively create directory " + root + ": " + (ec ? ec.message() : "already exists");
            }
            if (!error.empty()) { err = error; return false; }
            seq = sequence++;
            path = std::filesystem::path(root) / ("chunk-" + std::to_string(seq));
            if (!std::filesystem::create_directory(path, ec)) {
                err = "callback probe: cannot exclusively create chunk directory: " + (ec ? ec.message() : "already exists");
                return false;
            }
            return true;
        }
    };
    std::filesystem::path path_;
    std::ofstream meta_;
    std::vector<CallbackProbeRange> ranges_;
    uint64_t sequence_ = 0;
    const char* actual_ = "draft_kv_not_called";
    const char** previous_ = nullptr;
    bool active_ = false;
public:
    ~CallbackProbe() { if (active_) callback_probe_path = previous_; }
    bool start(int device, int64_t lb, int64_t le, int64_t p0, int64_t n, int64_t capacity,
               bool pipeline, std::vector<CallbackProbeRange> ranges, std::string& err,
               const void* scratch = nullptr, uint64_t scratch_bytes = 0,
               const void* residual = nullptr, uint64_t residual_bytes = 0) {
        static Directory directory;
        if (!directory.reserve(path_, sequence_, err)) return false;
        ranges_ = std::move(ranges);
        meta_.open(path_ / "meta.txt", std::ios::out);
        if (!meta_) { err = "callback probe: cannot write metadata"; return false; }
        const uint32_t endian = 1;
        meta_ << "format=strata-main-callback-v1 sequence=" << sequence_ << " device=" << device
              << " owner=" << std::hash<std::thread::id>{}(std::this_thread::get_id())
              << " layers=" << lb << ':' << le << " basepos=" << p0 << " n=" << n << " end=" << p0 + n
              << " capacity=" << capacity << " pipeline=" << pipeline << " dtype=fp32 byte_order="
              << (*reinterpret_cast<const uint8_t*>(&endian) ? "little" : "big")
              << " coverage=listed_ranges requested=gdn45+gdn46+lastR qsa=excluded sync=device-wide algorithm=strata-fnv1a64 diagnostic_timing=excluded\n";
        meta_ << "scratch_device_pointer=0x" << std::hex << (uintptr_t) scratch << std::dec << " scratch_bytes=" << scratch_bytes
              << " activeR_device_pointer=0x" << std::hex << (uintptr_t) residual << std::dec << " activeR_bytes=" << residual_bytes << '\n';
        auto overlap = [](const void* a, uint64_t na, const void* b, uint64_t nb) {
            const uintptr_t x = (uintptr_t) a, y = (uintptr_t) b;
            return a && b && na && nb && (x <= y ? y - x < na : x - y < nb);
        };
        for (const auto& range : ranges_) {
            if (!range.data || range.bytes == 0 || range.bytes % sizeof(float) != 0 || range.recurrence_elements > range.bytes / 4) {
                err = "callback probe: invalid active FP32 range"; return false;
            }
            meta_ << "range=" << range.name << " layer=" << range.layer << " ordinal=" << range.ordinal
                  << " allocation_row=" << range.row << " bytes=" << range.bytes << " elements=" << range.bytes / 4
                  << " recurrence_elements=" << range.recurrence_elements
                  << " conv_elements=" << (range.recurrence_elements ? range.bytes / 4 - range.recurrence_elements : 0)
                  << " device_pointer=0x" << std::hex << (uintptr_t) range.data << std::dec
                  << " scratch_overlap=" << overlap(range.data, range.bytes, scratch, scratch_bytes)
                  << " activeR_overlap=" << overlap(range.data, range.bytes, residual, residual_bytes) << '\n';
        }
        previous_ = callback_probe_path;
        callback_probe_path = &actual_;
        active_ = true;
        meta_.flush();
        return true;
    }
    // phase is a fixed internal name. before_repeat checks stable rereads before invoking the callback.
    template<class Read>
    bool sample(const char* phase, Read&& read, std::string& err) {
        std::array<uint8_t, audit_scratch_bytes> bytes, before;
        const bool compare = std::strcmp(phase, "before") != 0;
        for (const auto& range : ranges_) {
            const std::string name = range.name;
            std::ofstream out(path_ / (std::string(phase) + "-" + name + ".fp32"), std::ios::binary);
            std::ifstream baseline;
            if (compare) baseline.open(path_ / ("before-" + name + ".fp32"), std::ios::binary);
            if (!out || (compare && !baseline)) { err = "callback probe: cannot open raw files"; return false; }
            uint64_t hash = audit_seed, finite = 0, nonfinite = 0, bit_diff = 0;
            int64_t first = -1;
            uint32_t first_before = 0, first_after = 0;
            double sum = 0, max_abs_diff = 0;
            float lo = std::numeric_limits<float>::infinity(), hi = -lo;
            for (uint64_t at = 0; at < range.bytes;) {
                const size_t count = (size_t) std::min<uint64_t>(bytes.size(), range.bytes - at);
                if (!read(range.data, at, bytes.data(), count)) { if (err.empty()) err = "callback probe: range read failed"; return false; }
                out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) count);
                if (compare) {
                    baseline.read(reinterpret_cast<char*>(before.data()), (std::streamsize) count);
                    if (!baseline) { err = "callback probe: short before raw file"; return false; }
                }
                hash = audit_fnv(bytes.data(), count, hash);
                for (size_t i = 0; i < count; i += 4) {
                    float x; uint32_t xb;
                    std::memcpy(&x, bytes.data() + i, 4); std::memcpy(&xb, bytes.data() + i, 4);
                    if (std::isfinite(x)) { ++finite; sum += x; lo = std::min(lo, x); hi = std::max(hi, x); }
                    else ++nonfinite;
                    if (compare) {
                        float y; uint32_t yb;
                        std::memcpy(&y, before.data() + i, 4); std::memcpy(&yb, before.data() + i, 4);
                        if (xb != yb) {
                            ++bit_diff;
                            if (first < 0) { first = (int64_t) ((at + i) / 4); first_before = yb; first_after = xb; }
                        }
                        if (std::isfinite(x) && std::isfinite(y)) max_abs_diff = std::max(max_abs_diff, std::abs((double) x - y));
                    }
                }
                at += count;
            }
            out.close();
            if (!out) { err = "callback probe: raw write failed"; return false; }
            char hashes[96];
            std::snprintf(hashes, sizeof hashes, " hash=%016llx first_before=%08x first_after=%08x",
                          (unsigned long long) hash, first_before, first_after);
            meta_.precision(17);
            meta_ << "phase=" << phase << " range=" << name << hashes << " finite=" << finite << " nonfinite=" << nonfinite
                  << " sum=" << sum << " min=" << lo << " max=" << hi << " comparison=" << (compare ? "before" : "none")
                  << " bit_diff=" << bit_diff << " first_element=" << first << " max_abs_finite_diff=" << max_abs_diff << '\n';
        }
        meta_.flush();
        if (!meta_) { err = "callback probe: metadata write failed"; return false; }
        return true;
    }
    bool finish(bool callback_ok, std::string& err) {
        meta_ << "complete=1 callback_ok=" << callback_ok << " actual_draft_kv=" << actual_
              << " fallback=consult_DRAFT_PREFILL_or_MTP_audit\n";
        meta_.close();
        if (!meta_) { err = "callback probe: metadata close failed"; return false; }
        std::fprintf(stderr, "strata callback probe: sequence=%llu path=%s actual_draft_kv=%s callback_ok=%d complete=1\n",
                     (unsigned long long) sequence_, path_.string().c_str(), actual_, callback_ok);
        return true;
    }
};
} // namespace strata::prefill::detail
