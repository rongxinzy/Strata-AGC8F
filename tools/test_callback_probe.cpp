#include "strata/prefill/callback_probe.hpp"
#include <cassert>
#include <iostream>
#include <iterator>
#include <chrono>
using namespace strata::prefill::detail;
static std::string read_text(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
int main(int argc, char** argv) {
    assert(argc == 3);
    const std::string mode = argv[1];
    const std::filesystem::path root = std::string(argv[2]) + "-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::string err;
    setenv("STRATA_MAIN_CALLBACK_PROBE", mode == "disabled" ? "0" : "1", 1);
    if (mode != "missing") setenv("STRATA_MAIN_CALLBACK_PROBE_DIR", root.c_str(), 1);
    else unsetenv("STRATA_MAIN_CALLBACK_PROBE_DIR");
    if (mode == "disabled") {
        assert(!callback_probe_enabled());
        assert(!std::filesystem::exists(root));
        std::cout << "disabled flag leaves output absent\n"; return 0;
    }
    assert(callback_probe_enabled());
    std::vector<float> gdn(17501, 0.0f); // spans the bounded 64KiB read boundary
    std::vector<float> residual{1, -2, 3};
    const auto original = gdn;
    const std::vector<CallbackProbeRange> ranges{{"gdn45",45,34,4,gdn.data(),gdn.size()*4,17000},
        {"lastR",-1,-1,2,residual.data(),residual.size()*4}};
    if (mode == "existing") {
        std::filesystem::create_directory(root);
        std::ofstream(root / "keep.txt") << "unmodified";
    }
    CallbackProbe probe;
    const bool started = probe.start(7,42,48,1024,63,2048,true,ranges,err,gdn.data()+17000,501*4,residual.data(),residual.size()*4);
    if (mode == "existing" || mode == "missing") {
        assert(!started && !err.empty());
        if (mode == "existing") assert(read_text(root / "keep.txt") == "unmodified");
        else assert(!std::filesystem::exists(root));
        std::cout << mode << " rejected: " << err << '\n'; return 0;
    }
    assert(started);
    size_t largest = 0;
    auto copy = [&](const void* p, uint64_t at, uint8_t* h, size_t n) {
        assert(n <= audit_scratch_bytes && at % 4 == 0 && n % 4 == 0);
        largest = std::max(largest,n);
        std::memcpy(h, static_cast<const uint8_t*>(p)+at,n); return true;
    };
    if (mode == "failure") {
        assert(!probe.sample("before", [](const void*,uint64_t,uint8_t*,size_t){ return false; },err));
        assert(read_text(root / "chunk-0/meta.txt").find("complete=1") == std::string::npos);
        std::cout << "read failure leaves incomplete evidence\n"; return 0;
    }
    assert(probe.sample("before",copy,err));
    assert(probe.sample("before_repeat",copy,err));
    assert(std::memcmp(gdn.data(),original.data(),gdn.size()*4)==0);
    uint32_t negative_zero=0x80000000u;
    std::memcpy(&gdn[0],&negative_zero,4);
    gdn[16384]=2.0f;
    gdn[17400]=1.0f; // conv suffix, proving the same row dump covers both portions
    residual[2]=std::numeric_limits<float>::infinity();
    callback_probe_draft_path("batched_q8");
    assert(probe.sample("after",copy,err));
    assert(probe.finish(true,err));
    assert(largest==65536);
    const auto chunk=root / "chunk-0";
    assert(read_text(chunk / "before-gdn45.fp32")==read_text(chunk / "before_repeat-gdn45.fp32"));
    const auto meta=read_text(chunk / "meta.txt");
    assert(meta.find("basepos=1024 n=63 end=1087 capacity=2048 pipeline=1")!=std::string::npos);
    assert(meta.find("allocation_row=4 bytes=70004 elements=17501")!=std::string::npos);
    assert(meta.find("recurrence_elements=17000 conv_elements=501")!=std::string::npos);
    assert(meta.find("first_before=00000000 first_after=80000000")!=std::string::npos);
    assert(meta.find("bit_diff=3 first_element=0 max_abs_finite_diff=2")!=std::string::npos);
    assert(meta.find("finite=2 nonfinite=1")!=std::string::npos);
    assert(meta.find("scratch_overlap=1 activeR_overlap=0")!=std::string::npos);
    assert(meta.find("scratch_overlap=0 activeR_overlap=1")!=std::string::npos);
    assert(meta.find("scratch_bytes=2004")!=std::string::npos);
    assert(meta.find("activeR_bytes=12")!=std::string::npos);
    assert(meta.find("actual_draft_kv=batched_q8")!=std::string::npos);
    assert(meta.find("complete=1")!=std::string::npos);
    std::cout << "bounded read, exact raw preservation, stable reread, signed-zero/nonfinite diff, and path metadata pass\n";
}
