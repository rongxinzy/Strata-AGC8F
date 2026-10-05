// Request-scoped, host-only scheduling for the opt-in layer-split prompt pipeline.
// The GPU owner calls consumed() ONLY after its input's H2D has completed. That
// releases a producer's pinned slot; finish() is the separate full-chain barrier.
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::prefill::detail {

struct PipelineJob {
    const int64_t* tokens = nullptr;
    int64_t n = 0, pos0 = 0;
    const float* rows = nullptr;
    size_t producer = 0;
    int slot = 0;
};

class Pipeline {
public:
    using Handler = std::function<bool(size_t, const PipelineJob&, std::string&)>;
    Pipeline(size_t stages, std::function<bool()> stop, bool trace = false)
        : queues_(stages), closed_(stages, false), busy_(stages), next_slot_(stages, 0),
          stop_(std::move(stop)), trace_(trace), started_(Clock::now()) {
        if (stages < 2) throw std::invalid_argument("prefill pipeline needs two stages");
        for (auto& slots : busy_) slots = {false, false};
    }
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    ~Pipeline() {
        { std::lock_guard<std::mutex> lk(mu_); failed_ = true; }
        cv_.notify_all();
        join();
    }

    // Stage zero stays on the calling thread. Exactly one worker owns every
    // downstream stage's session, scratch, stream and callbacks for this request.
    void start(Handler handler) {
        try {
            for (size_t stage = 1; stage < queues_.size(); ++stage) {
                workers_.emplace_back([this, stage, handler] {
                    try {
                        PipelineJob job;
                        while (take(stage, job)) {
                            std::string err;
                            event("start", stage, job.pos0, job.n);
                            if (!handler(stage, job, err)) {
                                fail(err.empty() ? "prefill pipeline stage failed" : err);
                                return;
                            }
                            event("end", stage, job.pos0, job.n);
                        }
                        if (stage + 1 < queues_.size()) close(stage + 1);
                    } catch (const std::exception& e) {
                        fail(std::string("prefill pipeline worker: ") + e.what());
                    } catch (...) { fail("prefill pipeline worker: unknown exception"); }
                });
            }
        } catch (...) {
            fail("prefill pipeline: cannot start stage workers");
            join();
            throw;
        }
    }

    bool stopped() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (failed_) return true;
        }
        // Never invoke a caller's callback while holding the scheduler mutex.
        if (stop_ && stop_()) { fail("cancelled"); return true; }
        return false;
    }

    // Reserve one of the two producer slots before D2H writes it. Keeping this
    // across calls (not a run-local counter) is essential for downstream stages.
    bool acquire(size_t stage, int& slot) {
        std::unique_lock<std::mutex> lk(mu_);
        slot = next_slot_[stage];
        while (busy_[stage][slot]) {
            if (!pause(lk)) return false;
        }
        if (!check(lk)) return false;
        busy_[stage][slot] = true;
        next_slot_[stage] ^= 1;
        return true;
    }

    bool send(size_t stage, const PipelineJob& job) {
        std::unique_lock<std::mutex> lk(mu_);
        while (queues_[stage].size() >= 2) {
            if (!pause(lk)) return false;
        }
        if (!check(lk)) return false;
        if (closed_[stage]) { lk.unlock(); fail("prefill pipeline: send after close"); return false; }
        queues_[stage].push_back(job);
        cv_.notify_all();
        return true;
    }

    void consumed(const PipelineJob& job) {
        std::lock_guard<std::mutex> lk(mu_);
        busy_[job.producer][job.slot] = false;
        cv_.notify_all();
    }

    void fail(const std::string& err) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!failed_) { failed_ = true; error_ = err; }
        cv_.notify_all();
    }
    std::string error() const {
        std::lock_guard<std::mutex> lk(mu_);
        return error_;
    }

    // Closing propagates after each single worker drains its FIFO. No state may
    // be read for decode/checkpoint/refill until all workers have joined.
    bool finish(std::string& err) {
        close(1);
        join();
        err = error();
        return err.empty();
    }

    void event(const char* what, size_t stage, int64_t pos, int64_t n, int device = -1) {
        if (!trace_) return;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started_).count();
        char affinity[256] = "unavailable";
        int actual_cpu = -1;
#if defined(__linux__)
        actual_cpu = sched_getcpu();
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask) == 0) {
            size_t at = 0;
            affinity[0] = 0;
            for (int cpu = 0; cpu < CPU_SETSIZE && at + 16 < sizeof(affinity); ++cpu) {
                if (!CPU_ISSET(cpu, &mask)) continue;
                const int wrote = std::snprintf(affinity + at, sizeof(affinity) - at, "%s%d", at ? "," : "", cpu);
                if (wrote > 0) at += (size_t) wrote;
            }
        }
#endif
        std::fprintf(stderr, "strata prefill pipeline trace: stage=%zu device=%d pos=%lld n=%lld event=%s us=%lld cpu=%d affinity=%s\n",
                     stage, device, (long long) pos, (long long) n, what, (long long) us, actual_cpu, affinity);
    }

private:
    using Clock = std::chrono::steady_clock;
    bool check(std::unique_lock<std::mutex>& lk) {
        if (failed_) return false;
        lk.unlock();
        const bool stop = stopped();
        lk.lock();
        return !stop && !failed_;
    }
    bool pause(std::unique_lock<std::mutex>& lk) {
        if (!check(lk)) return false;
        cv_.wait_for(lk, std::chrono::milliseconds(10));
        return check(lk);
    }
    bool take(size_t stage, PipelineJob& job) {
        std::unique_lock<std::mutex> lk(mu_);
        while (queues_[stage].empty() && !closed_[stage]) {
            if (!pause(lk)) return false;
        }
        if (!check(lk) || queues_[stage].empty()) return false;
        job = queues_[stage].front();
        queues_[stage].pop_front();
        cv_.notify_all();
        return true;
    }
    void close(size_t stage) {
        std::lock_guard<std::mutex> lk(mu_);
        closed_[stage] = true;
        cv_.notify_all();
    }
    void join() {
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::deque<PipelineJob>> queues_;
    std::vector<bool> closed_;
    std::vector<std::array<bool, 2>> busy_;
    std::vector<int> next_slot_;
    std::vector<std::thread> workers_;
    std::function<bool()> stop_;
    bool failed_ = false, trace_ = false;
    std::string error_;
    Clock::time_point started_;
};

}  // namespace strata::prefill::detail
