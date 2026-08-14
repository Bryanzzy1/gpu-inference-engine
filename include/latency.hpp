#ifndef LATENCY_HPP
#define LATENCY_HPP

#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// Tail latency summary in nanoseconds, computed from the full sorted sample.
// p9999 is included because at high event rates p999 fires often: at 100k events/s
// the 1-in-1000 spike happens ~100x/second, so the 1-in-10000 point is the one a
// tight SLA actually cares about.
struct Stats {
    std::size_t count = 0;
    double min = 0.0;
    double p50 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double p9999 = 0.0;
    double max = 0.0;
    double p25 = 0.0;
    double p75 = 0.0;
    double iqr = 0.0;   // p75 - p25, robust jitter
    double mean = 0.0;
    double stddev = 0.0;

    // Sorts samples_ns in place and fills a Stats.
    static Stats from(std::vector<double>& samples_ns);

    // One-line report to stdout.
    void print(const char* label) const;
};

// One histogram bucket: [lo, hi) nanoseconds and how many samples fell in it.
struct HistBucket {
    double lo_ns = 0.0;
    double hi_ns = 0.0;
    std::size_t count = 0;
};

// Log-spaced latency histogram over the sample set. Log-spaced because latency spans
// orders of magnitude (sub-us median, hundreds-of-us tail) and linear buckets would
// put everything in one bin. `bins` buckets between the min and max sample.
std::vector<HistBucket> histogram(std::vector<double> samples_ns, int bins = 20);

// Write a histogram to CSV: bucket_lo_ns,bucket_hi_ns,count. Header first if new.
void write_histogram_csv(const std::string& path, const char* label,
                         const std::vector<HistBucket>& hist);

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

    // Same run, but also returns every raw per-call latency (ns) via `samples_out`,
    // so callers can build a histogram or dump the full distribution. `samples_out`
    // is left sorted (Stats::from sorts in place).
    Stats run_capture(const std::function<void()>& work,
                      std::vector<double>& samples_out);

private:
    HarnessConfig cfg_;
    using Clock = std::chrono::steady_clock;
};

#endif
