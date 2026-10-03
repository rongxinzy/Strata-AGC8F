// Host-only regression of queue ownership/acks/barriers, not a CUDA correctness test.
#include "strata/prefill/pipeline.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using strata::prefill::detail::Pipeline;
using strata::prefill::detail::PipelineJob;
using namespace std::chrono_literals;

void ordered_deep_pipeline() {
    constexpr size_t stages = 8;
    constexpr int chunks = 40;
    Pipeline pipeline(stages, [] { return false; });
    std::vector<std::array<float, 2>> buffers(stages);
    std::array<int64_t, chunks> tokens{};
    std::array<int, stages> seen{}, active{};
    std::atomic<int> concurrent{0}, peak{0};
    std::mutex seen_mu;
    pipeline.start([&](size_t stage, const PipelineJob& job, std::string& err) {
        const int now = concurrent.fetch_add(1) + 1;
        int old = peak.load();
        while (now > old && !peak.compare_exchange_weak(old, now)) {}
        {
            std::lock_guard<std::mutex> lock(seen_mu);
            assert(++active[stage] == 1);
            assert(job.pos0 == seen[stage]++);
        }
        // Model H2D: keep a copy before releasing the producer's host buffer.
        const float input = *job.rows;
        assert(input == (float) (job.pos0 + stage - 1));
        std::this_thread::sleep_for(100us);
        assert(*job.rows == input); // producer cannot overwrite before ack
        pipeline.consumed(job);
        std::this_thread::sleep_for(2ms); // independent stage compute / callback
        if (stage + 1 < stages) {
            int slot = 0;
            if (!pipeline.acquire(stage, slot)) { err = pipeline.error(); return false; }
            buffers[stage][slot] = input + 1;
            const PipelineJob next{job.tokens, job.n, job.pos0, &buffers[stage][slot], stage, slot};
            if (!pipeline.send(stage + 1, next)) { err = pipeline.error(); return false; }
        }
        {
            std::lock_guard<std::mutex> lock(seen_mu);
            --active[stage];
        }
        concurrent.fetch_sub(1);
        return true;
    });
    for (int i = 0; i < chunks; ++i) {
        tokens[i] = i;
        int slot = 0;
        assert(pipeline.acquire(0, slot));
        buffers[0][slot] = (float) i;
        assert(pipeline.send(1, PipelineJob{&tokens[i], 1, i, &buffers[0][slot], 0, slot}));
    }
    std::string err;
    assert(pipeline.finish(err));
    assert(err.empty());
    assert(pipeline.finish(err)); // root's normal barrier + wrapper's cleanup barrier
    for (size_t stage = 1; stage < stages; ++stage) {
        assert(seen[stage] == chunks);
        assert(active[stage] == 0);
    }
    assert(peak.load() >= 3); // excludes the old root + sequential tail schedule
    std::cout << "ordered 8-stage pipeline: " << chunks << " chunks; peak workers=" << peak.load() << "\n";
}

void ack_is_not_completion() {
    Pipeline pipeline(2, [] { return false; });
    float buffers[2]{};
    int64_t token = 0;
    std::promise<void> consumed, release;
    auto release_future = release.get_future().share();
    std::atomic<int> completed{0};
    pipeline.start([&](size_t, const PipelineJob& job, std::string&) {
        pipeline.consumed(job);
        if (job.pos0 == 0) { consumed.set_value(); release_future.wait(); }
        completed.fetch_add(1);
        return true;
    });
    int slot = 0;
    assert(pipeline.acquire(0, slot));
    assert(pipeline.send(1, PipelineJob{&token, 1, 0, &buffers[slot], 0, slot}));
    consumed.get_future().wait();
    assert(completed.load() == 0);
    // Both slots are available even while stage 1 is still processing chunk 0.
    for (int i = 1; i <= 2; ++i) {
        assert(pipeline.acquire(0, slot));
        assert(pipeline.send(1, PipelineJob{&token, 1, i, &buffers[slot], 0, slot}));
    }
    auto final = std::async(std::launch::async, [&] { std::string err; return pipeline.finish(err); });
    assert(final.wait_for(10ms) == std::future_status::timeout);
    release.set_value();
    assert(final.get());
    assert(completed.load() == 3);
    std::cout << "input ack releases a slot; finish waits for all callbacks\n";
}

void failure_releases_blocked_producer(bool throwing) {
    Pipeline pipeline(3, [] { return false; });
    float buffer[2]{};
    int64_t token = 0;
    std::promise<void> release;
    auto ready = release.get_future().share();
    pipeline.start([&](size_t, const PipelineJob&, std::string& err) {
        ready.wait(); // deliberately fail without acknowledging the input
        if (throwing) throw std::runtime_error("injected exception");
        err = "injected failure";
        return false;
    });
    int slot = 0;
    for (int i = 0; i < 2; ++i) {
        assert(pipeline.acquire(0, slot));
        assert(pipeline.send(1, PipelineJob{&token, 1, i, &buffer[slot], 0, slot}));
    }
    auto blocked = std::async(std::launch::async, [&] { int next = 0; return pipeline.acquire(0, next); });
    assert(blocked.wait_for(10ms) == std::future_status::timeout);
    release.set_value();
    assert(!blocked.get());
    std::string err;
    assert(!pipeline.finish(err));
    assert(err.find(throwing ? "injected exception" : "injected failure") != std::string::npos);
    std::cout << "worker " << (throwing ? "exception" : "failure") << " releases blocked producer and joins\n";
}

void cancellation_releases_idle_and_busy_workers() {
    std::atomic<bool> stop{false};
    Pipeline pipeline(8, [&] { return stop.load(); });
    float buffer[2]{};
    int64_t token = 0;
    pipeline.start([&](size_t, const PipelineJob&, std::string& err) {
        while (!pipeline.stopped()) std::this_thread::sleep_for(1ms);
        err = "cancelled";
        return false;
    });
    int slot = 0;
    for (int i = 0; i < 2; ++i) {
        assert(pipeline.acquire(0, slot));
        assert(pipeline.send(1, PipelineJob{&token, 1, i, &buffer[slot], 0, slot}));
    }
    auto blocked = std::async(std::launch::async, [&] { int next = 0; return pipeline.acquire(0, next); });
    assert(blocked.wait_for(10ms) == std::future_status::timeout);
    stop.store(true);
    assert(!blocked.get());
    std::string err;
    assert(!pipeline.finish(err));
    assert(err == "cancelled");
    std::cout << "cancellation releases slot/FIFO waits across 8 stages\n";
}

void empty_pipeline_and_scope_cleanup() {
    {
        Pipeline pipeline(8, [] { return false; });
        pipeline.start([](size_t, const PipelineJob&, std::string&) { assert(false); return false; });
        std::string err;
        assert(pipeline.finish(err));
    }
    {
        Pipeline pipeline(8, [] { return false; });
        pipeline.start([](size_t, const PipelineJob&, std::string&) { assert(false); return false; });
        // No explicit finish: scope cleanup wakes every idle FIFO and joins.
    }
    std::cout << "empty FIFO closure and unfinished-scope cleanup join all workers\n";
}

int main() {
    // Fresh request-scoped workers/queues repeatedly, without retained handoff state.
    for (int i = 0; i < 3; ++i) ordered_deep_pipeline();
    ack_is_not_completion();
    failure_releases_blocked_producer(false);
    failure_releases_blocked_producer(true);
    cancellation_releases_idle_and_busy_workers();
    empty_pipeline_and_scope_cleanup();
}
