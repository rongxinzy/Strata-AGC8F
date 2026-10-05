#include "strata/program/mtp_prefill_audit.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>
#include <cstring>

using namespace strata::program::mtp_audit;

std::vector<uint8_t> bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    assert(f.good());
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void half_stats() {
    assert(half_value(0x3c00) == 1.0 && half_value(0xbc00) == -1.0);
    assert(half_value(0x0001) == std::ldexp(1.0, -24));
    assert(half_value(0x0400) == std::ldexp(1.0, -14));
    assert(half_value(0x7bff) == 65504.0);
    assert(std::signbit(half_value(0x8000)) && !std::signbit(half_value(0)));
    Stats s;
    for (uint16_t x : {0x3c00, 0xbc00, 0x8000, 0x0001, 0x7c00, 0xfc00, 0x7e01}) s.add(x);
    assert(s.elements == 7 && s.finite == 4 && s.inf == 2 && s.nan == 1 && s.negative_zero == 1);
    assert(s.minimum == -1 && s.maximum == 1 && s.max_abs == 1);
    std::ostringstream out; json_stats(out, s);
    assert(out.str().find("\"finite\":4") != std::string::npos);
}

void mapped_partial_rows(const std::filesystem::path& root) {
    const Geometry g{2, 3, 4, 4, 4, 16};
    const Request r{7, 5, 7, 7, true}; // nonzero cell0 and first_needed clip, partial pages
    const std::vector<int32_t> mapping{2, 0, 3, 1};
    std::vector<uint16_t> pools[2];
    for (auto& p : pools) p.assign(4 * 2 * 4 * 3, 0x7e01); // stale/padding sentinel
    for (int cell = 7; cell < 12; ++cell) for (int head = 0; head < 2; ++head) for (int dim = 0; dim < 3; ++dim) {
        const int at = ((mapping[(size_t) cell / 4] * 2 + head) * 4 + cell % 4) * 3 + dim;
        for (int pool = 0; pool < 2; ++pool) pools[pool][at] = (uint16_t) (0x3c00 + pool * 128 + cell * 6 + head * 3 + dim);
    }
    int copies = 0;
    auto read = [&](bool key, uint64_t at, uint16_t* out, size_t n, std::string&) {
        assert(n * 2 <= copy_limit);
        // Every copied row belongs to the effective interval, not early cells.
        const int physical = (int) (at / 3 / 4 / 2);
        const int logical = (int) (std::find(mapping.begin(), mapping.end(), physical) - mapping.begin());
        const int start = logical * 4 + (int) ((at / 3) % 4);
        assert(start >= 7 && start + (int) n / 3 <= 12 && n % 3 == 0);
        assert(at + n <= pools[key ? 0 : 1].size());
        std::memcpy(out, pools[key ? 0 : 1].data() + at, n * 2); ++copies;
        return true;
    };
    std::string err;
    assert(write(root, 0, g, r, 1, {0, 3}, read, err));
    assert(copies == 8);
    for (int pool = 0; pool < 2; ++pool) {
        const auto raw = bytes(root / "prefill-0" / (pool ? "v.fp16le" : "k.fp16le"));
        assert(raw.size() == 5 * 2 * 3 * 2);
        size_t i = 0;
        for (int cell = 7; cell < 12; ++cell) for (int head = 0; head < 2; ++head) for (int dim = 0; dim < 3; ++dim) {
            const uint16_t expected = (uint16_t) (0x3c00 + pool * 128 + cell * 6 + head * 3 + dim);
            assert(raw[i++] == (uint8_t) expected); assert(raw[i++] == (uint8_t) (expected >> 8));
        }
    }
    const auto raw_meta = bytes(root / "prefill-0" / "metadata.json");
    const std::string meta(raw_meta.begin(), raw_meta.end());
    assert(meta.find("\"active_begin\":7") != std::string::npos && meta.find("\"active_end\":12") != std::string::npos);
    assert(meta.find("\"nan\":0") != std::string::npos && meta.find("\"finite\":30") != std::string::npos);
    assert(!write(root, 0, g, r, 1, {0, 3}, read, err)); // cannot overwrite a prior receipt
    assert(err.find("overwrite") != std::string::npos);
    auto never = [](bool, uint64_t, uint16_t*, size_t, std::string&) { assert(false); return false; };
    assert(write(root, 1, g, Request{7, 5, 7, 20, false}, 3, {}, never, err));
    assert(bytes(root / "prefill-1" / "k.fp16le").empty());
    assert(!write(root, 2, g, r, 1, {0, 0}, never, err)); // alias
    assert(!write(root, 2, g, r, 1, {0, -1}, never, err));
    assert(!write(root, 2, g, r, 1, {0, 4}, never, err));
    assert(!write(root, 2, g, r, 0, {0, 3}, never, err));
    assert(!write(root, 2, g, Request{7, -1, 7, 0, false}, 0, {}, never, err));
    assert(!write(root, 2, g, Request{7, 15, 2, 0, false}, 0, {}, never, err));
    assert(!write(root, 2, Geometry{2, 3, 4, 4, INT64_MAX, 16}, r, 1, {0, 3}, never, err));
    int failed_copy = 0;
    auto fail = [&](bool key, uint64_t at, uint16_t* out, size_t n, std::string& e) {
        if (++failed_copy == 2) { e = "injected copy failure"; return false; }
        return read(key, at, out, n, e);
    };
    assert(!write(root, 3, g, r, 1, {0, 3}, fail, err));
    assert(err == "injected copy failure" && !std::filesystem::exists(root / "prefill-3" / "metadata.json"));
    std::cout << "permuted physical pages, nonzero cell0, active clip, partial tail, empty interval, failure/overwrite rejection\n";
}

void bounded_full_pages(const std::filesystem::path& root) {
    Geometry g{2, 256, 256, 2, 2, 512};
    Request r{7, 63, 385, 65, false};
    size_t largest = 0;
    auto read = [&](bool key, uint64_t at, uint16_t* out, size_t n, std::string&) {
        largest = std::max(largest, n * 2); assert(n * 2 <= copy_limit);
        for (size_t i = 0; i < n; ++i) out[i] = (uint16_t) (0x3c00 + (at + i + (key ? 0 : 11)) % 1024);
        return true;
    };
    std::string err;
    assert(write(root, 4, g, r, 0, {1, 0}, read, err));
    assert(largest == 32768);
    assert(bytes(root / "prefill-4" / "k.fp16le").size() == (448 - 65) * 2 * 256 * 2);
    Range range;
    assert(active_range(Request{7, 15, 0, 0, false}, 16, range, err) && range.begin == 15 && range.end == 15);
    assert(active_range(Request{7, 1, 63, -100, false}, 100, range, err) && range.begin == 1 && range.end == 64);
    assert(active_range(Request{7, 1, 64, 0, false}, 100, range, err) && range.end == 65);
    assert(active_range(Request{7, 1, 65, 0, false}, 100, range, err) && range.end == 66);
    std::cout << "bounded 64KiB copies, head-major to canonical raw layout, range lengths 0/63/64/65\n";
}

int main(int argc, char** argv) {
    const bool preserve = argc == 2;
    const auto root = preserve ? std::filesystem::path(argv[1]) : std::filesystem::temp_directory_path() /
        ("strata-mtp-audit-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    assert(!std::filesystem::exists(root));
    half_stats(); mapped_partial_rows(root); bounded_full_pages(root);
    if (!preserve) std::filesystem::remove_all(root);
    std::cout << "binary16 statistics and audit output passed; CUDA/MTP math not exercised\n";
}
