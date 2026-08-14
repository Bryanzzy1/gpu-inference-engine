#include "latency.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
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

    // Population standard deviation, a second view of spread next to IQR.
    double var = 0.0;
    for (double v : samples_ns) {
        const double d = v - s.mean;
        var += d * d;
    }
    s.stddev = std::sqrt(var / static_cast<double>(samples_ns.size()));

    s.min = samples_ns.front();
    s.p25 = percentile(samples_ns, 0.25);
    s.p50 = percentile(samples_ns, 0.50);
    s.p75 = percentile(samples_ns, 0.75);
    s.p99 = percentile(samples_ns, 0.99);
    s.p999 = percentile(samples_ns, 0.999);
    s.p9999 = percentile(samples_ns, 0.9999);
    s.max = samples_ns.back();
    s.iqr = s.p75 - s.p25;
    return s;
}

void Stats::print(const char* label) const {
    std::printf("%-16s n=%zu  p50=%.1f  p99=%.1f  p999=%.1f  p9999=%.1f  max=%.1f  "
                "iqr=%.1f  mean=%.1f  sd=%.1f  (ns)\n",
                label, count, p50, p99, p999, p9999, max, iqr, mean, stddev);
}

std::vector<HistBucket> histogram(std::vector<double> samples_ns, int bins) {
    std::vector<HistBucket> out;
    if (samples_ns.empty() || bins < 1) return out;
    std::sort(samples_ns.begin(), samples_ns.end());

    // Log-spaced edges between min and max. Guard a zero/degenerate min so log is safe.
    double lo = samples_ns.front();
    const double hi = samples_ns.back();
    if (lo <= 0.0) lo = 1.0;              // sub-ns floor; steady_clock is >= ~100ns anyway
    if (hi <= lo) {                       // all samples equal: one bucket
        out.push_back({samples_ns.front(), samples_ns.back(), samples_ns.size()});
        return out;
    }
    const double log_lo = std::log10(lo);
    const double step = (std::log10(hi) - log_lo) / bins;

    out.resize(bins);
    for (int b = 0; b < bins; ++b) {
        out[b].lo_ns = std::pow(10.0, log_lo + step * b);
        out[b].hi_ns = std::pow(10.0, log_lo + step * (b + 1));
    }
    for (double v : samples_ns) {
        int b = static_cast<int>((std::log10(std::max(v, lo)) - log_lo) / step);
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;      // include the max in the last bucket
        out[b].count++;
    }
    return out;
}

void write_histogram_csv(const std::string& path, const char* label,
                         const std::vector<HistBucket>& hist) {
    const bool exists = std::ifstream(path).good();
    std::ofstream f(path, std::ios::app);
    if (!f) return;
    if (!exists) f << "label,bucket_lo_ns,bucket_hi_ns,count\n";
    for (const HistBucket& b : hist) {
        f << label << ',' << b.lo_ns << ',' << b.hi_ns << ',' << b.count << '\n';
    }
}

Stats LatencyHarness::run_capture(const std::function<void()>& work,
                                  std::vector<double>& samples) {
    // Warm-up runs are discarded so cold caches and clock ramp do not skew the tail.
    for (std::size_t i = 0; i < cfg_.warmup; ++i) work();

    samples.clear();
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

    return Stats::from(samples);  // sorts `samples` in place
}

Stats LatencyHarness::run(const std::function<void()>& work) {
    std::vector<double> samples;
    return run_capture(work, samples);
}
