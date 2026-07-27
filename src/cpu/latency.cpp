#include "latency.hpp"

#include <algorithm>
#include <cstdio>
#include <thread>

namespace {

// Value at a percentile in a sorted vector, using nearest-rank.
double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    std::size_t rank = static_cast<std::size_t>(p * sorted.size());
    if (rank >= sorted.size()) rank = sorted.size() - 1;
    return sorted[rank];
}

}

Stats Stats::from(std::vector<double>& samples_ns) {
    Stats s;
    s.count = samples_ns.size();
    if (samples_ns.empty()) return s;

    std::sort(samples_ns.begin(), samples_ns.end());

    double sum = 0.0;
    for (double v : samples_ns) sum += v;
    s.mean = sum / static_cast<double>(samples_ns.size());

    s.p50 = percentile(samples_ns, 0.50);
    s.p99 = percentile(samples_ns, 0.99);
    s.p999 = percentile(samples_ns, 0.999);
    s.max = samples_ns.back();
    s.iqr = percentile(samples_ns, 0.75) - percentile(samples_ns, 0.25);
    return s;
}

void Stats::print(const char* label) const {
    std::printf("%-16s n=%zu  p50=%.1f  p99=%.1f  p999=%.1f  max=%.1f  "
                "iqr=%.1f  mean=%.1f  (ns)\n",
                label, count, p50, p99, p999, max, iqr, mean);
}

Stats LatencyHarness::run(const std::function<void()>& work) {
    // Warm-up runs are discarded so cold caches and clock ramp do not skew the tail.
    for (std::size_t i = 0; i < cfg_.warmup; ++i) work();

    std::vector<double> samples;
    samples.reserve(cfg_.iters);

    const bool paced = cfg_.target_rate_hz > 0.0;
    const auto start = Clock::now();
    // Nanoseconds between scheduled events when replay pacing is on.
    const double period_ns = paced ? 1e9 / cfg_.target_rate_hz : 0.0;

    for (std::size_t i = 0; i < cfg_.iters; ++i) {
        // In replay mode, wait until this event's scheduled send time.
        if (paced) {
            const auto due = start + std::chrono::nanoseconds(
                static_cast<int64_t>(period_ns * static_cast<double>(i)));
            std::this_thread::sleep_until(due);
        }

        const auto t0 = Clock::now();
        work();
        const auto t1 = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::nano>(t1 - t0).count());
    }

    return Stats::from(samples);
}
