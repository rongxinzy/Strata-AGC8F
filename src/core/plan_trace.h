// src/core/plan_trace.h - STRATA_VERIFY_PLAN_TRACE: a host-only entries-per-group histogram of the verify
// window's expert plan.  Default OFF: with the variable unset, Recorder::acquire() is a null check and
// nothing is read or written.  The expert pool calls Verifier::publish_plan after it has written the mapped
// plan (counts | start | dst | tok | ptr | ptr2 | start2) and before it raises flag A, so those arrays are
// quiescent host memory there - the one safe CPU read point.  This file never touches CUDA, device
// pointers or the plan itself: it is read-only, a malformed plan is logged as a tracing error and never
// "fixed", and no flag, order or plan byte changes.  Tracing runs are diagnostics, excluded from formal
// timings; a counts profile alone proves nothing about speed, latency or correctness.
#ifndef STRATA_CORE_PLAN_TRACE_H
#define STRATA_CORE_PLAN_TRACE_H

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#define PT_MKDIR(p) _mkdir(p)
#define PT_PID() _getpid()
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define PT_MKDIR(p) mkdir(p, 0777)
#define PT_PID() getpid()
#endif

namespace strata::core::plantrace {

constexpr int kBinMax = 8;         // entries-per-group bins 1..8 (the NC tile widths worth separating)
constexpr int kOver = kBinMax + 1; // bin 9 collects ">8"; bin 0 counts anomalous zero-entry groups

struct SampleLabels {
    int device = 0;         // the stage's GPU ordinal; no addresses, device or host, are ever written out
    int stage_first = 0;    // the stage's layer range [stage_first, stage_last) of a layer split
    int stage_last = 0;
    int layer = 0;          // global layer index
    int T = 0;              // the window's token count
    int groups = 1;         // the window's token-group count (a split window: 2)
    int grp = 0;            // which token group this plan covers
    int cap = 0;            // sink_.cap = max_t * k: every count and prefix sum is bounds-checked against it
    int quant_gu = 0;       // the layer's native gate/up and down types (0: the non-native S2 layout)
    int quant_down = 0;
    bool empty_plan = false; // the verifier wrote the empty plan itself (the pool did not publish)
};

struct Key {
    int device, stage_first, stage_last, layer, T, groups, grp, quant_gu, quant_down;
    char share;            // 'r' = resident/VRAM groups, 'p' = the PCIe share
    bool empty_plan;
    bool operator<(const Key& o) const {
        return std::tie(device, stage_first, stage_last, layer, T, groups, grp, quant_gu, quant_down, share,
                        empty_plan) <
               std::tie(o.device, o.stage_first, o.stage_last, o.layer, o.T, o.groups, o.grp, o.quant_gu,
                        o.quant_down, o.share, o.empty_plan);
    }
};

struct Agg {
    long long samples = 0, groups = 0, entries = 0, rows_gpu = 0;
    long long bin[kOver + 1] = {0};   // bin[1..8], bin[kOver] = ">8", bin[0] = zero-entry groups
};

class Recorder {
public:
    // nullptr unless STRATA_VERIFY_PLAN_TRACE is set: when off, the engine's whole cost is this check.
    // The recorder is a function-local static, so its destructor flushes at process exit (stdio only - no
    // CUDA, so teardown order cannot bite).
    static Recorder* acquire();
    // counts[0]/counts[2] are the resident/PCIe group counts, start/start2 their prefix sums, dst/tok the
    // destination-row / token mapping for the optional bounded raw lines.  Read-only.
    void record(const SampleLabels& lb, const int32_t* counts, const int32_t* start, const int32_t* start2,
                const int32_t* dst, const int32_t* tok);
    void note_excluded(const char* why);   // e.g. STRATA_VERIFY_DEVICE_PLAN=1: the pool plan is not what ran
    void flush();                          // rewrites the aggregate snapshot; once per window is cheap
    ~Recorder();

private:
    explicit Recorder(const char* base);
    void error_locked(const SampleLabels& lb, const char* why, int gn, int rows, int gn2);
    bool disabled_ = false, io_noted_ = false, raw_truncated_ = false;
    long long samples_ = 0, errors_ = 0, drops_ = 0, raw_lines_ = 0, raw_cap_ = 0;
    std::string dir_;
    std::mutex mu_;   // publish_plan runs on the pool's threads, flush() on the window's own thread
    std::map<Key, Agg> agg_;
    std::map<std::string, long long> excl_;
    std::FILE* raw_ = nullptr;
    std::FILE* err_ = nullptr;
};

inline Recorder* Recorder::acquire() {
    static Recorder* const r = []() -> Recorder* {
        const char* base = std::getenv("STRATA_VERIFY_PLAN_TRACE");
        if (base == nullptr || base[0] == '\0') return nullptr;   // default off, decided once
        static Recorder rec(base);
        return &rec;
    }();
    return r;
}

inline Recorder::Recorder(const char* base) {
    const char* raw = std::getenv("STRATA_VERIFY_PLAN_TRACE_RAW");
    long long n = raw != nullptr ? std::atoll(raw) : 0;
    raw_cap_ = n < 0 ? 0 : (n > 10000000 ? 10000000 : n);
    if (PT_MKDIR(base) != 0 && errno != EEXIST) { disabled_ = true; }
    for (int seq = 0; !disabled_ && seq < 10000; ++seq) {   // a fresh directory per run, never overwritten
        std::string cand = std::string(base) + "/run-" + std::to_string((long long) PT_PID()) + "-" +
                           std::to_string(seq);
        if (PT_MKDIR(cand.c_str()) == 0) { dir_ = cand; break; }
        if (errno != EEXIST) break;
    }
    if (dir_.empty()) disabled_ = true;
    if (!disabled_) {
        raw_ = std::fopen((dir_ + "/plan_raw.tsv").c_str(), "w");   // optional: may stay null
        err_ = std::fopen((dir_ + "/plan_errors.log").c_str(), "w");
        if (err_ == nullptr) disabled_ = true;   // errors must be preservable, or tracing is worthless
    }
    if (disabled_ && !io_noted_) {
        io_noted_ = true;
        std::fprintf(stderr, "strata plan trace: cannot start under %s; tracing disabled, the run is unaffected\n",
                     base);
    }
}

inline void Recorder::error_locked(const SampleLabels& lb, const char* why, int gn, int rows, int gn2) {
    ++errors_;   // the sample is dropped whole; the plan itself is never touched
    if (err_ != nullptr)
        std::fprintf(err_, "err sample=%lld dev=%d stage=%d:%d layer=%d T=%d G=%d grp=%d cap=%d "
                           "counts=%d,%d,%d reason=%s\n",
                     samples_, lb.device, lb.stage_first, lb.stage_last, lb.layer, lb.T, lb.groups, lb.grp,
                     lb.cap, gn, rows, gn2, why);
}

inline void Recorder::record(const SampleLabels& lb, const int32_t* counts, const int32_t* start,
                             const int32_t* start2, const int32_t* dst, const int32_t* tok) {
    std::lock_guard<std::mutex> hold(mu_);
    if (disabled_) { ++drops_; return; }
    if (counts == nullptr) { ++samples_; error_locked(lb, "null counts", 0, 0, 0); return; }
    const long long sid = ++samples_;
    const int gn = *(const volatile int32_t*) (counts + 0);
    const int rows = *(const volatile int32_t*) (counts + 1);
    const int gn2 = *(const volatile int32_t*) (counts + 2);
    const int32_t* st[2] = {start, start2};
    const int ns[2] = {gn, gn2};
    for (int s = 0; s < 2; ++s) {   // validate both shares first: a malformed plan is never half-histogrammed
        if (ns[s] < 0 || ns[s] > lb.cap) { error_locked(lb, "group count out of range", gn, rows, gn2); return; }
        if (rows < 0 || rows > lb.cap) { error_locked(lb, "row count out of range", gn, rows, gn2); return; }
        if (ns[s] == 0) continue;
        if (st[s] == nullptr) { error_locked(lb, "null start array", gn, rows, gn2); return; }
        long long prev = 0;
        for (int g = 0; g <= ns[s]; ++g) {
            const long long cur = *(const volatile int32_t*) (st[s] + g);
            if (cur < prev || cur > lb.cap) {
                error_locked(lb, "start is not a monotone prefix sum within cap", gn, rows, gn2);
                return;
            }
            prev = cur;
        }
    }
    // expert_source.cpp publishes exactly counts[0..2]. Slot 3 is reserved and uninitialized;
    // never read it. counts[1] covers BOTH GPU shares, whose prefixes share one entry array.
    const int resident_end = gn ? start[gn] : 0;
    const int gpu_end = gn2 ? start2[gn2] : resident_end;
    if ((gn && start[0] != 0) || (gn2 && start2[0] != resident_end) || gpu_end != rows) {
        error_locked(lb, "resident/PCIe prefix does not cover total GPU rows", gn, rows, gn2);
        return;
    }
    for (int s = 0; s < 2; ++s) {
        Key k{lb.device,  lb.stage_first, lb.stage_last, lb.layer,    lb.T,   lb.groups, lb.grp,
              lb.quant_gu, lb.quant_down, s == 0 ? 'r' : 'p', lb.empty_plan};
        Agg& a = agg_[k];
        ++a.samples;
        if (s == 0) {
            a.rows_gpu += rows;
        }
        if (ns[s] == 0) continue;
        const volatile int32_t* sv = st[s];
        for (int g = 0; g < ns[s]; ++g) {
            const long long e = (long long) sv[g + 1] - (long long) sv[g];
            ++a.groups;
            a.entries += e;
            ++a.bin[e >= 1 && e <= kBinMax ? (int) e : (e == 0 ? 0 : kOver)];
            if (raw_ != nullptr && !raw_truncated_ && raw_lines_ < raw_cap_) {   // bounded, numeric, no addresses
                for (long long j = 0; j < e && raw_lines_ < raw_cap_; ++j) {
                    const int idx = (int) (sv[g] + j);
                    if (idx < 0 || idx >= lb.cap) break;
                    std::fprintf(raw_, "raw %lld %d %d %d %d %d %d %d %c %d %lld %d %d\n", sid, lb.device,
                                 lb.stage_first, lb.stage_last, lb.layer, lb.T, lb.groups, lb.grp, k.share, g,
                                 e, dst != nullptr ? dst[idx] : -1, tok != nullptr ? tok[idx] : -1);
                    ++raw_lines_;
                }
            }
        }
    }
    if (raw_ != nullptr && !raw_truncated_ && raw_cap_ > 0 && raw_lines_ >= raw_cap_) {
        std::fprintf(raw_, "# raw lines truncated at the STRATA_VERIFY_PLAN_TRACE_RAW bound\n");
        std::fflush(raw_);
        raw_truncated_ = true;
    }
}

inline void Recorder::note_excluded(const char* why) {
    std::lock_guard<std::mutex> hold(mu_);
    ++excl_[why];
}

inline void Recorder::flush() {
    std::lock_guard<std::mutex> hold(mu_);
    if (disabled_) return;
    const std::string path = dir_ + "/plan_hist.tsv";
    std::FILE* f = std::fopen(path.c_str(), "w");   // a snapshot: rewritten each flush, not appended
    if (f == nullptr) {
        disabled_ = true;
        if (!io_noted_) {
            io_noted_ = true;
            std::fprintf(stderr, "strata plan trace: cannot write %s; tracing disabled\n", path.c_str());
        }
        return;
    }
    std::fprintf(f, "# strata plan trace v1 (tracing only; excluded from formal timings; no device addresses)\n");
    std::fprintf(f, "# agg device stage_first stage_last layer T G grp qgu qdown empty share samples groups "
                    "entries gpu_entries_total reserved_unread b0 b1 b2 b3 b4 b5 b6 b7 b8 bover\n");
    for (const auto& kv : agg_) {
        const Key& k = kv.first;
        const Agg& a = kv.second;
        std::fprintf(f, "agg %d %d %d %d %d %d %d %d %d %d %c %lld %lld %lld %lld %lld", k.device,
                     k.stage_first, k.stage_last, k.layer, k.T, k.groups, k.grp, k.quant_gu, k.quant_down,
                     (int) k.empty_plan, k.share, a.samples, a.groups, a.entries, a.rows_gpu, 0LL);
        for (int b = 0; b <= kOver; ++b) std::fprintf(f, " %lld", a.bin[b]);
        std::fprintf(f, "\n");
    }
    for (const auto& we : excl_) std::fprintf(f, "excl %lld %s\n", we.second, we.first.c_str());
    std::fprintf(f, "# counters samples=%lld errors=%lld drops=%lld raw_lines=%lld raw_cap=%lld\n", samples_,
                 errors_, drops_, raw_lines_, raw_cap_);
    std::fclose(f);
    if (raw_ != nullptr) std::fflush(raw_);
    if (err_ != nullptr) std::fflush(err_);
}

inline Recorder::~Recorder() {
    flush();   // the last window's aggregates land even if the engine never flushes again (stdio only)
    if (raw_ != nullptr) std::fclose(raw_);
    if (err_ != nullptr) std::fclose(err_);
}

}  // namespace strata::core::plantrace
#endif  // STRATA_CORE_PLAN_TRACE_H
