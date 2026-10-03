// Opt-in first-warmup GDN45 boundary diagnostic. Host-only: streams owner-device arrays to the host
// in bounded 64KiB reads and never writes device memory. Default off: no CUDA calls, no file output.
// Separates H1 (bad initial GDN state) / H2 (layer-44 -> 45 producer inputs) / H3 (the recurrence
// itself differing on identical inputs); see docs/AGC8F_CALLBACK_PROBE.md and the iteration-006 report.
#pragma once
#include "strata/prefill/audit.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>

namespace strata::prefill::detail {

inline bool boundary_probe_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_GDN45_BOUNDARY_PROBE"); return v && std::strcmp(v, "1") == 0; }();
    return on;
}

// The hooks need layer 44's end (the producer residual) and layer 45 (GDN), first chunk (p0 == 0) of the
// first prefill - the serving warmup. One claim per process: no later request or repeated chunk recaptures.
inline bool boundary_probe_want(int64_t lb, int64_t le, int64_t p0) {
    static std::atomic<bool> noted{false}, claimed{false};
    if (!boundary_probe_enabled() || p0 != 0) return false;
    if (!(lb <= 44 && 45 < le)) {
        bool quiet = false;
        if (noted.compare_exchange_strong(quiet, true))
            std::fprintf(stderr, "strata boundary probe: stage layers %lld:%lld do not own 44/45; nothing captured\n",
                         (long long) lb, (long long) le);
        return false;
    }
    return !claimed.exchange(true);
}

class BoundaryProbe {
public:
    enum class Raw : int { None = 0, Recurrence = 1, All = 2 };
    static Raw raw_policy() {
        static const Raw v = [] {
            const char* e = std::getenv("STRATA_GDN45_BOUNDARY_RAW");
            return e && !std::strcmp(e, "all") ? Raw::All : e && !std::strcmp(e, "none") ? Raw::None : Raw::Recurrence;
        }();
        return v;
    }
    static const char* raw_name(Raw r) { return r == Raw::All ? "all" : r == Raw::None ? "none" : "recurrence"; }

    bool start(int device, int64_t lb, int64_t le, int64_t p0, int64_t T, int64_t capacity, int64_t n_embd,
               int64_t dim, int64_t conv_channels, int64_t hv, int64_t zv, uint64_t gdn_row_floats,
               uint64_t gdn_recurrence_floats, int64_t gdn_ord0, int64_t gdn_alloc, int64_t ordinal45,
               std::string& err) {
        const char* v = std::getenv("STRATA_GDN45_BOUNDARY_PROBE_DIR");
        const std::string root = v ? v : "";
        std::error_code ec;
        if (root.empty()) { err = "boundary probe: unique STRATA_GDN45_BOUNDARY_PROBE_DIR required"; return false; }
        if (!std::filesystem::create_directory(root, ec)) {
            err = "boundary probe: cannot exclusively create " + root + ": " + (ec ? ec.message() : "already exists");
            return false;
        }
        path_ = std::filesystem::path(root);
        meta_.open(path_ / "meta.txt", std::ios::out);
        if (!meta_) { err = "boundary probe: cannot write metadata"; return false; }
        if (T <= 0 || capacity < T || ordinal45 < gdn_ord0 || ordinal45 - gdn_ord0 >= gdn_alloc ||
            gdn_recurrence_floats == 0 || gdn_row_floats <= gdn_recurrence_floats) {
            err = "boundary probe: layer-45 carve or chunk geometry invalid"; return false;
        }
        T_ = T; n_embd_ = n_embd; dim_ = dim; conv_ = conv_channels; hv_ = hv; zv_ = zv;
        gdn_row_ = gdn_row_floats; gdn_rec_ = gdn_recurrence_floats;
        const uint32_t endian = 1;
        meta_ << "format=strata-gdn45-boundary-v1 device=" << device
              << " owner=" << std::hash<std::thread::id>{}(std::this_thread::get_id()) << " layers=" << lb << ':' << le
              << " basepos=" << p0 << " n=" << T << " capacity=" << capacity << " n_embd=" << n_embd << " dim=" << dim
              << " conv_channels=" << conv_channels << " hv=" << hv << " zv=" << zv
              << " gdn_row_floats=" << gdn_row_floats << " recurrence_floats=" << gdn_recurrence_floats
              << " conv_hist_floats=" << (gdn_row_floats - gdn_recurrence_floats) << " gdn_ord0=" << gdn_ord0
              << " gdn_alloc=" << gdn_alloc << " ordinal45=" << ordinal45 << " row45=" << (ordinal45 - gdn_ord0)
              << " raw_policy=" << raw_name(raw_policy())
              << " byte_order=" << (*reinterpret_cast<const uint8_t*>(&endian) ? "little" : "big")
              << " coverage=layer44_end_residual+layer45_gdn_boundaries qsa=excluded"
              << " sync=owner_stream_checked device_writes=none algorithm=strata-fnv1a64 read_limit=65536"
              << " diagnostic_timing=excluded\n";
        meta_ << "markers=" << kCount << " required:";
        for (const Spec& s : kSpecs) meta_ << ' ' << s.name;
        meta_ << '\n';
        meta_.flush();
        if (!meta_) { err = "boundary probe: metadata write failed"; return false; }
        return true;
    }

    // Streams the marker's whole active array through `read` (<= 64KiB per call): whole-array FNV-1a64,
    // finite/nonfinite and exact-nonzero counts, min/max/sum, and raw bytes per the dump policy. The
    // marker table fixes each range's dtype and geometry, so a wrong pointer, width or dtype fails loudly.
    template <class Read>
    bool capture(const char* marker, const void* data, uint64_t bytes, Read&& read, std::string& err) {
        const Spec* spec = find(marker);
        const size_t index = spec ? (size_t) (spec - kSpecs) : kCount;
        if (!spec) { err = std::string("boundary probe: unknown marker ") + marker; return false; }
        if (seen_[index]) { err = std::string("boundary probe: duplicate marker ") + marker; return false; }
        const size_t elem = std::strcmp(spec->dtype, "fp32") == 0 ? 4 : 2;
        const uint64_t expected = expected_elements(spec->rows) * elem;
        if (data == nullptr || bytes == 0) {
            err = "boundary probe: marker " + std::string(marker) + " has no active " + spec->dtype + " range";
            return false;
        }
        if (bytes != expected) {
            err = "boundary probe: marker " + std::string(marker) + " bytes " + std::to_string(bytes) + " != geometry " +
                  std::to_string(expected) + " (" + spec->dtype + ")";
            return false;
        }
        const bool raw = raw_policy() == Raw::All || (raw_policy() == Raw::Recurrence && spec->raw);
        std::ofstream out;
        if (raw) {
            out.open(path_ / (std::string(marker) + '.' + spec->dtype), std::ios::binary);
            if (!out) { err = "boundary probe: cannot open raw file"; return false; }
        }
        const auto started = std::chrono::steady_clock::now();
        std::array<uint8_t, audit_scratch_bytes> buf;
        uint64_t hash = audit_seed, finite = 0, nonfinite = 0, nonzero = 0;
        int64_t first_nonzero = -1;
        double sum = 0;
        float lo = std::numeric_limits<float>::infinity(), hi = -lo;
        for (uint64_t at = 0; at < bytes;) {
            const size_t n = (size_t) std::min<uint64_t>(buf.size(), bytes - at);
            if (!read(data, at, buf.data(), n)) { if (err.empty()) err = "boundary probe: range read failed"; return false; }
            if (raw) out.write(reinterpret_cast<const char*>(buf.data()), (std::streamsize) n);
            hash = audit_fnv(buf.data(), n, hash);
            for (size_t i = 0; i < n; i += elem) {
                float x;
                if (elem == 4) std::memcpy(&x, buf.data() + i, 4);
                else { uint16_t h; std::memcpy(&h, buf.data() + i, 2); x = spec->dtype[0] == 'b' ? bf16(h) : f16(h); }
                if (std::isfinite(x)) {
                    ++finite;
                    sum += x;
                    lo = std::min(lo, x);
                    hi = std::max(hi, x);
                    if (x != 0.0f) {  // signed zero counts as zero: the reset carve must be exactly zero
                        ++nonzero;
                        if (first_nonzero < 0) first_nonzero = (int64_t) ((at + i) / elem);
                    }
                } else ++nonfinite;
            }
            at += n;
        }
        if (raw) {
            out.close();
            if (!out) { err = "boundary probe: raw write failed"; return false; }
        }
        seen_[index] = true;
        ++captured_;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        perturbation_ms_ += ms;
        char hex[24];
        std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) hash);
        meta_.precision(17);
        meta_ << "marker=" << marker << " dtype=" << spec->dtype << " bytes=" << bytes << " elements=" << bytes / elem
              << " row_elements=" << expected_elements(spec->rows);
        if (spec->rows == Rows::GdnRow)
            meta_ << " recurrence_elements=" << gdn_rec_ << " conv_elements=" << (gdn_row_ - gdn_rec_);
        meta_ << " hash=" << hex << " finite=" << finite << " nonfinite=" << nonfinite << " nonzero=" << nonzero
              << " first_nonzero_element=" << first_nonzero << " min=" << lo << " max=" << hi << " sum=" << sum
              << " raw=" << (raw ? 1 : 0) << " elapsed_ms=" << ms << '\n';
        meta_.flush();
        if (!meta_) { err = "boundary probe: metadata write failed"; return false; }
        return true;
    }

    // Loud unless every required marker of the fixed checklist was captured exactly once.
    bool finish(std::string& err) {
        std::string missing;
        for (size_t i = 0; i < kCount; ++i)
            if (!seen_[i]) { missing += ' '; missing += kSpecs[i].name; }
        if (!missing.empty() || captured_ != kCount) {
            err = "boundary probe: INCOMPLETE capture, missing:" + missing;
            std::fprintf(stderr, "strata boundary probe: INCOMPLETE capture, missing:%s\n", missing.c_str());
            return false;
        }
        meta_ << "complete=1 markers=" << captured_ << " perturbation_ms=" << perturbation_ms_
              << " note=diagnostic_run_excluded_from_all_timings\n";
        meta_.close();
        if (!meta_) { err = "boundary probe: metadata close failed"; return false; }
        std::fprintf(stderr, "strata boundary probe: complete=1 markers=%zu raw_policy=%s path=%s perturbation_ms=%.1f\n",
                     (size_t) captured_, raw_name(raw_policy()), path_.string().c_str(), perturbation_ms_);
        return true;
    }

private:
    enum class Rows : uint8_t { GdnRow, Residual, Mixed, Conv, Zv, Ab, GateBeta };
    struct Spec { const char* name; const char* dtype; Rows rows; bool raw; };
    static constexpr Spec kSpecs[] = {
        {"l44_residual_end",             "fp32", Rows::Residual, false},
        {"l45_mixed_f32_after_mix",      "fp32", Rows::Mixed,    false},
        {"l45_mixed_bf16_after_mix",     "bf16", Rows::Mixed,    false},
        {"l45_mixed_f16_after_mix",      "f16",  Rows::Mixed,    false},
        {"l45_state_initial_before_gdn", "fp32", Rows::GdnRow,   true },
        {"l45_qkv_after_proj",           "fp32", Rows::Conv,     false},
        {"l45_z_after_proj",             "fp32", Rows::Zv,       false},
        {"l45_ab_after_proj",            "fp32", Rows::Ab,       false},
        {"l45_gate_after_gates",         "fp32", Rows::GateBeta, true },
        {"l45_beta_after_gates",         "fp32", Rows::GateBeta, true },
        {"l45_h_after_conv",             "fp32", Rows::Conv,     true },
        {"l45_state_after_recurrence",   "fp32", Rows::GdnRow,   true },
        {"l45_y_f32_after_recurrence",   "fp32", Rows::Zv,       false},
        {"l45_y_f16_after_recurrence",   "f16",  Rows::Zv,       false},
    };
    static constexpr size_t kCount = sizeof kSpecs / sizeof kSpecs[0];
    static const Spec* find(const char* marker) {
        for (const Spec& s : kSpecs) if (!std::strcmp(s.name, marker)) return &s;
        return nullptr;
    }
    uint64_t expected_elements(Rows r) const {
        switch (r) {
            case Rows::GdnRow: return gdn_row_;
            case Rows::Residual: return (uint64_t) T_ * dim_;
            case Rows::Mixed: return (uint64_t) T_ * n_embd_;
            case Rows::Conv: return (uint64_t) T_ * conv_;
            case Rows::Zv: return (uint64_t) T_ * zv_;
            case Rows::Ab: return (uint64_t) T_ * 2 * hv_;
            case Rows::GateBeta: return (uint64_t) T_ * hv_;
        }
        return 0;
    }
    static float bf16(uint16_t h) { const uint32_t u = (uint32_t) h << 16; float f; std::memcpy(&f, &u, 4); return f; }
    static float f16(uint16_t h) {
        const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
        if (exp == 31) { const uint32_t u = sign | 0x7f800000u; float f; std::memcpy(&f, &u, 4); return f; }
        if (exp == 0) return std::ldexp((float) man, -24) * (sign ? -1.0f : 1.0f);
        const uint32_t u = sign | ((exp - 15 + 127) << 23) | (man << 13);
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    std::filesystem::path path_;
    std::ofstream meta_;
    bool seen_[kCount] = {};
    uint64_t captured_ = 0;
    int64_t T_ = 0, n_embd_ = 0, dim_ = 0, conv_ = 0, hv_ = 0, zv_ = 0;
    uint64_t gdn_row_ = 0, gdn_rec_ = 0;
    double perturbation_ms_ = 0;
};
}  // namespace strata::prefill::detail
