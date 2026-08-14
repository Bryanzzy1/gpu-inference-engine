// Self-check for the extended Stats and the histogram helper. No GPU, no timing:
// feeds known samples and asserts the percentiles, ordering, and bucket counts.
#include <cstdio>
#include <vector>

#include "latency.hpp"

int main() {
    bool ok = true;
    auto check = [&](bool cond, const char* what) {
        if (!cond) { std::printf("FAIL: %s\n", what); ok = false; }
    };

    // 1..1000 ns. Percentiles are nearest-rank on the sorted sample.
    std::vector<double> xs;
    for (int i = 1; i <= 1000; ++i) xs.push_back(static_cast<double>(i));
    Stats s = Stats::from(xs);

    check(s.count == 1000, "count");
    check(s.min == 1.0, "min");
    check(s.max == 1000.0, "max");
    check(s.p50 > 490 && s.p50 < 520, "p50 near middle");
    check(s.p99 > 980 && s.p99 <= 1000, "p99 near top");
    check(s.p999 >= 990, "p999 in top decile of top percentile");
    // Ordering must hold across the whole ladder.
    check(s.min <= s.p25 && s.p25 <= s.p50 && s.p50 <= s.p75 && s.p75 <= s.p99
          && s.p99 <= s.p999 && s.p999 <= s.p9999 && s.p9999 <= s.max, "monotone ladder");
    check(s.iqr == s.p75 - s.p25, "iqr = p75 - p25");
    check(s.mean > 490 && s.mean < 510, "mean near 500");
    check(s.stddev > 0.0, "stddev positive");

    // Histogram: buckets cover the range, counts sum to the sample size, edges rise.
    std::vector<HistBucket> h = histogram(xs, 10);
    check(!h.empty(), "histogram non-empty");
    std::size_t total = 0;
    bool rising = true;
    for (std::size_t i = 0; i < h.size(); ++i) {
        total += h[i].count;
        if (h[i].hi_ns <= h[i].lo_ns) rising = false;
        if (i && h[i].lo_ns < h[i - 1].lo_ns) rising = false;
    }
    check(total == 1000, "histogram counts sum to N");
    check(rising, "histogram edges strictly increasing");

    // Degenerate input: all equal -> one bucket, no crash.
    std::vector<double> flat(50, 7.0);
    std::vector<HistBucket> hf = histogram(flat, 8);
    check(!hf.empty(), "flat histogram non-empty");

    // Empty input -> empty stats, empty histogram, no crash.
    std::vector<double> empty;
    Stats es = Stats::from(empty);
    check(es.count == 0, "empty stats count 0");
    check(histogram(empty, 8).empty(), "empty histogram");

    std::printf(ok ? "OK: latency stats + histogram\n" : "FAILED\n");
    return ok ? 0 : 1;
}
