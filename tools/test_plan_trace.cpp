// A producer-shaped plan fixture: resident and PCIe groups share one entry array.
#include "plan_trace.h"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
using namespace strata::core::plantrace;
int main(int argc, char** argv) {
    assert(argc == 3);
    const std::string mode = argv[1];
    const std::filesystem::path root = std::filesystem::path(argv[2]) /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    if (mode == "disabled") {
        unsetenv("STRATA_VERIFY_PLAN_TRACE");
        assert(Recorder::acquire() == nullptr && !std::filesystem::exists(root));
        return 0;
    }
    std::filesystem::create_directories(root.parent_path());
    setenv("STRATA_VERIFY_PLAN_TRACE", root.c_str(), 1);
    setenv("STRATA_VERIFY_PLAN_TRACE_RAW", "4", 1);
    Recorder* rec = Recorder::acquire();
    assert(rec);
    SampleLabels lb;
    lb.device = 7; lb.stage_first = 42; lb.stage_last = 48; lb.layer = 45;
    lb.T = 4; lb.groups = 1; lb.cap = 80; lb.quant_gu = 21; lb.quant_down = 20;
    int32_t counts[4]; // only the producer's three populated slots are initialized
    counts[0] = 3; counts[1] = 9; counts[2] = 2;
    int32_t resident[] = {0, 1, 3, 6};
    int32_t pcie[] = {6, 7, 9};
    int32_t dst[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int32_t tok[9] = {0, 0, 1, 0, 1, 2, 3, 2, 3};
    if (mode == "invalid_total") counts[1] = 8;
    else if (mode == "invalid_offset") pcie[0] = 0;
    else if (mode == "invalid_range") counts[0] = 81;
    else if (mode == "happy" || mode == "reserved") {
        if (mode == "reserved") counts[3] = -123456789;
    } else assert(false);
    rec->record(lb, counts, resident, pcie, dst, tok);
    if (mode == "reserved") {
        counts[3] = 123456789;
        rec->record(lb, counts, resident, pcie, dst, tok);
    }
    rec->flush();
    std::filesystem::path run;
    for (const auto& p : std::filesystem::directory_iterator(root)) run = p.path();
    std::ifstream f(run / "plan_hist.tsv");
    const std::string text((std::istreambuf_iterator<char>(f)), {});
    if (mode.find("invalid_") == 0) {
        assert(text.find("errors=1") != std::string::npos);
        assert(text.find("\nagg ") == std::string::npos);
    } else {
        assert(text.find("errors=0") != std::string::npos);
        int aggregate_rows = 0;
        std::istringstream lines(text);
        std::string line;
        while (std::getline(lines, line)) {
            if (line.rfind("agg ", 0) != 0) continue;
            std::istringstream fields(line);
            std::string tag;
            int d, first, last, layer, T, G, grp, qgu, qdown, empty;
            char share;
            long long samples, groups, entries, total, reserved, bins[10];
            fields >> tag >> d >> first >> last >> layer >> T >> G >> grp >> qgu >> qdown >> empty
                   >> share >> samples >> groups >> entries >> total >> reserved;
            for (auto& b : bins) fields >> b;
            assert(fields && reserved == 0 && d == 7 && layer == 45 && qgu == 21);
            const long long n = mode == "reserved" ? 2 : 1;
            assert(samples == n);
            if (share == 'r') {
                assert(groups == 3*n && entries == 6*n && total == 9*n);
                assert(bins[1] == n && bins[2] == n && bins[3] == n);
            } else {
                assert(share == 'p' && groups == 2*n && entries == 3*n);
                assert(bins[1] == n && bins[2] == n && bins[3] == 0);
            }
            ++aggregate_rows;
        }
        assert(aggregate_rows == 2);
    }
}
