// Host-only regression for the GDN45 boundary probe helper. Validates the diagnostic, not CUDA math:
// the read callback here is host memory, standing in for the engine's checked owner-stream D2H reads.
#include "strata/prefill/boundary_probe.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace strata::prefill::detail;
static std::string read_text(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
static bool has(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

// The engine's fixed chunk geometry (prefill.cpp: N, D, C, HV, ZV and the GDN carve) at a small T.
static constexpr int64_t kT = 3, kN = 2560, kD = 10240, kC = 10240, kHV = 48, kZV = 6144;
static constexpr uint64_t kRow = 817152, kRec = 786432;  // recurrence + conv-history floats

int main(int argc, char** argv) {
    assert(argc == 3);
    const std::string mode = argv[1];
    const std::filesystem::path root = std::string(argv[2]) + "-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::string err;
    setenv("STRATA_GDN45_BOUNDARY_PROBE", mode == "disabled" ? "0" : "1", 1);
    if (mode != "missing") setenv("STRATA_GDN45_BOUNDARY_PROBE_DIR", root.c_str(), 1);
    else unsetenv("STRATA_GDN45_BOUNDARY_PROBE_DIR");
    if (mode != "rawnone" && mode != "rawall") unsetenv("STRATA_GDN45_BOUNDARY_RAW");
    else setenv("STRATA_GDN45_BOUNDARY_RAW", mode == "rawall" ? "all" : "none", 1);

    if (mode == "disabled") {
        assert(!boundary_probe_enabled());
        assert(!boundary_probe_want(42, 48, 0));
        assert(!std::filesystem::exists(root));
        std::cout << "disabled flag performs no capture and no output\n";
        return 0;
    }
    if (mode == "claim") {
        assert(boundary_probe_want(0, 6, 0) == false);   // not the 44/45 owner; says so once on stderr
        assert(boundary_probe_want(42, 48, 5) == false);  // not the first chunk
        assert(boundary_probe_want(42, 48, 0));           // the one claim of the process
        assert(boundary_probe_want(42, 48, 0) == false);  // no second request or repeated chunk
        assert(!std::filesystem::exists(root));
        std::cout << "claim is once per process and only for the 44/45 owner's first chunk\n";
        return 0;
    }
    assert(boundary_probe_enabled());

    std::vector<float> row(kRow, 0.0f), residual(kT * kD), mixed(kT * kN), conv(kT * kC), zv(kT * kZV),
        ab(kT * 2 * kHV), gb(kT * kHV);
    std::vector<uint16_t> mixed_bf(kT * kN), mixed_f16(kT * kN), y16(kT * kZV);
    row[5000] = 2.5f;  // the initial state is exactly zero except one planted element
    uint32_t negative_zero = 0x80000000u;
    for (size_t i = 0; i < residual.size(); ++i) residual[i] = (float) ((int64_t) i % 7) - 3.0f;
    std::memcpy(&residual[9], &negative_zero, 4);  // signed zero is a zero, not a nonzero
    for (size_t i = 0; i < mixed_bf.size(); ++i) mixed_bf[i] = (uint16_t) (i % 3 == 0 ? 0x3f80 : i % 3 == 1 ? 0x8000 : 0x7fc0);
    for (size_t i = 0; i < mixed_f16.size(); ++i) mixed_f16[i] = (uint16_t) (i % 3 == 0 ? 0x3c00 : i % 3 == 1 ? 0x0000 : 0x7e00);
    for (size_t i = 0; i < y16.size(); ++i) y16[i] = 0x4000;  // f16 2.0

    BoundaryProbe probe;
    if (mode == "existing") {
        std::filesystem::create_directory(root);
        std::ofstream(root / "keep.txt") << "unmodified";
    }
    const bool started = probe.start(7, 42, 48, 0, kT, 2048, kN, kD, kC, kHV, kZV, kRow, kRec, 32, 4, 34, err);
    if (mode == "missing" || mode == "existing") {
        assert(!started && !err.empty());
        assert(mode == "missing" ? !std::filesystem::exists(root) : read_text(root / "keep.txt") == "unmodified");
        std::cout << mode << " rejected: " << err << '\n';
        return 0;
    }
    assert(started && err.empty());
    size_t largest = 0;
    auto copy = [&](const void* p, uint64_t at, uint8_t* host, size_t count) {
        assert(count <= audit_scratch_bytes && count % 2 == 0);
        largest = std::max(largest, count);
        std::memcpy(host, (const uint8_t*) p + at, count);
        return true;
    };
    const std::pair<const char*, uint64_t> ranges[] = {
        {"l44_residual_end", (uint64_t) (residual.size() * 4)},
        {"l45_mixed_f32_after_mix", (uint64_t) (mixed.size() * 4)},
        {"l45_mixed_bf16_after_mix", (uint64_t) (mixed_bf.size() * 2)},
        {"l45_mixed_f16_after_mix", (uint64_t) (mixed_f16.size() * 2)},
        {"l45_state_initial_before_gdn", kRow * 4},
        {"l45_qkv_after_proj", (uint64_t) (conv.size() * 4)},
        {"l45_z_after_proj", (uint64_t) (zv.size() * 4)},
        {"l45_ab_after_proj", (uint64_t) (ab.size() * 4)},
        {"l45_gate_after_gates", (uint64_t) (gb.size() * 4)},
        {"l45_beta_after_gates", (uint64_t) (gb.size() * 4)},
        {"l45_h_after_conv", (uint64_t) (conv.size() * 4)},
        {"l45_state_after_recurrence", kRow * 4},
        {"l45_y_f32_after_recurrence", (uint64_t) (zv.size() * 4)},
        {"l45_y_f16_after_recurrence", (uint64_t) (y16.size() * 2)},
    };
    const void* data[] = {residual.data(),  mixed.data(),  mixed_bf.data(), mixed_f16.data(), row.data(),
                          conv.data(),      zv.data(),     ab.data(),       gb.data(),        gb.data(),
                          conv.data(),      row.data(),    zv.data(),       y16.data()};

    if (mode == "badmarker") {
        assert(!probe.capture("l45_unknown_marker", residual.data(), ranges[0].second, copy, err) && has(err, "unknown marker"));
        assert(!probe.capture("l44_residual_end", residual.data(), ranges[1].second, copy, err) &&
               has(err, "bytes 30720 != geometry 122880"));  // wrong width is loud, not silent
        assert(!probe.capture("l45_h_after_conv", nullptr, ranges[10].second, copy, err) && has(err, "no active fp32 range"));
        std::cout << "unknown marker and range/dtype failures are loud: " << err << '\n';
        return 0;
    }
    for (size_t i = 0; i < std::size(ranges); ++i) {
        if (mode == "incomplete" && i == 4) break;  // stop before the checklist is complete
        if (mode == "readfail" && i == 2) {
            assert(!probe.capture(ranges[i].first, data[i], ranges[i].second,
                                  [](const void*, uint64_t, uint8_t*, size_t) { return false; }, err) &&
                   has(err, "read failed"));
            assert(!has(read_text(root / "meta.txt"), "complete=1"));
            std::cout << "read failure leaves incomplete evidence\n";
            return 0;
        }
        assert(probe.capture(ranges[i].first, data[i], ranges[i].second, copy, err));
        if (mode == "duplicate" && i == 0) {
            assert(!probe.capture(ranges[i].first, data[i], ranges[i].second, copy, err) && has(err, "duplicate marker"));
            std::cout << "duplicate marker rejected: " << err << '\n';
            return 0;
        }
    }
    assert(largest == audit_scratch_bytes);  // whole arrays stream in bounded 64KiB reads
    if (mode == "incomplete") {
        assert(!probe.finish(err) && has(err, "INCOMPLETE") && has(err, "l45_state_initial_before_gdn"));
        assert(!has(read_text(root / "meta.txt"), "complete=1"));
        std::cout << "missing required markers fail loudly: " << err << '\n';
        return 0;
    }
    assert(probe.finish(err));
    const std::string meta = read_text(root / "meta.txt");
    assert(has(meta, "complete=1 markers=14"));
    assert(has(meta, "format=strata-gdn45-boundary-v1 device=7 owner=") && has(meta, " layers=42:48 basepos=0 n=3 capacity=2048"));
    assert(has(meta, "gdn_row_floats=817152 recurrence_floats=786432 conv_hist_floats=30720 gdn_ord0=32 gdn_alloc=4 ordinal45=34 row45=2"));
    assert(has(meta, "sync=owner_stream_checked device_writes=none"));
    assert(has(meta, "marker=l45_state_initial_before_gdn dtype=fp32 bytes=3268608 elements=817152"));
    assert(has(meta, "nonzero=1 first_nonzero_element=5000"));                    // H1's exact-zero check
    assert(has(meta, "marker=l44_residual_end dtype=fp32 bytes=122880 elements=30720"));
    assert(has(meta, "marker=l45_mixed_bf16_after_mix dtype=bf16 bytes=15360 elements=7680"));
    assert(has(meta, "marker=l45_y_f16_after_recurrence dtype=f16 bytes=36864 elements=18432"));
    {   // whole-array digests cover every byte, so cross-boot equality needs no sampled rows
        uint64_t h = audit_seed;
        for (float v : residual) h = audit_fnv((const uint8_t*) &v, 4, h);
        char hex[24];
        std::snprintf(hex, sizeof hex, "hash=%016llx", (unsigned long long) h);
        const auto line = meta.substr(meta.find("marker=l44_residual_end"));
        assert(has(line.substr(0, line.find('\n')), hex));
    }
    {   // bf16/f16 decode: thirds of 1.0 / signed zero / NaN, and f16 2.0 for the recurrence output
        const auto line = meta.substr(meta.find("marker=l45_mixed_bf16_after_mix"));
        assert(has(line.substr(0, line.find('\n')), "finite=5120 nonfinite=2560 nonzero=2560"));
        const auto l16 = meta.substr(meta.find("marker=l45_y_f16_after_recurrence"));
        assert(has(l16.substr(0, l16.find('\n')), "finite=18432 nonfinite=0 nonzero=18432 first_nonzero_element=0 min=2 max=2"));
    }
    const std::pair<const char*, uint64_t> raw_set[] = {
        {"l45_state_initial_before_gdn.fp32", kRow * 4},
        {"l45_gate_after_gates.fp32", (uint64_t) (gb.size() * 4)},
        {"l45_beta_after_gates.fp32", (uint64_t) (gb.size() * 4)},
        {"l45_h_after_conv.fp32", (uint64_t) (conv.size() * 4)},
        {"l45_state_after_recurrence.fp32", kRow * 4},
    };
    for (const auto& [f, bytes] : raw_set) {
        if (mode == "rawnone") assert(!std::filesystem::exists(root / f));
        else assert(std::filesystem::file_size(root / f) == bytes);
    }
    assert(std::filesystem::exists(root / "l45_state_initial_before_gdn.fp32") == (mode != "rawnone"));
    assert(std::filesystem::exists(root / "l44_residual_end.fp32") == (mode == "rawall"));
    if (mode == "rawall") assert(std::filesystem::file_size(root / "l44_residual_end.fp32") == (uint64_t) kT * kD * 4);
    std::cout << mode << ": complete checklist, exact-zero state, whole-array digests, raw policy "
              << BoundaryProbe::raw_name(BoundaryProbe::raw_policy()) << '\n';
    return 0;
}
