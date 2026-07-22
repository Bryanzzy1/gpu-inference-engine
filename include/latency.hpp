#ifndef LATENCY_HPP
#define LATENCY_HPP

#include <chrono>
#include <cstddef>
#include <functional>
#include <vector>

// Tail latency summary in nanoseconds, computed from the full sorted sample.
struct Stats {
    std::size_t count = 0;
    double p50 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double max = 0.0;
    double iqr = 0.0;
    double mean = 0.0;

    // Sorts samples_ns in place and fills a Stats.
    static Stats from(std::vector<double>& samples_ns);

    // One-line report to stdout.
    void print(const char* label) const;
};

// Timing run parameters.
struct HarnessConfig {
    std::size_t warmup = 1000;
    std::size_t iters = 200000;
    double target_rate_hz = 0.0; // 0 disables replay pacing
};

// Times any callable and reports the tail. Reused unchanged by every backend.
class LatencyHarness {
public:
    explicit LatencyHarness(HarnessConfig cfg) : cfg_(cfg) {}

    // Runs work cfg_.iters times keeping every sample, returns the tail stats.
    Stats run(const std::function<void()>& work);

private:
    HarnessConfig cfg_;
    using Clock = std::chrono::steady_clock;
};

#endif
