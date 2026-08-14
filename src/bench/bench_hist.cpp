// Latency-distribution benchmark for the CPU forward pass. Times Model::forward over
// real feature rows, prints the extended tail stats (min/p50/p99/p999/p9999/max, IQR,
// stddev), and writes a log-spaced latency histogram to CSV. The histogram shows the
// SHAPE of the distribution, not just percentiles, which is what makes a tail argument
// convincing: you can see the bimodality and where the mass sits, not one number.
//
// CPU only, builds with CMake (no GPU needed). The GPU backends can reuse the same
// LatencyHarness::run_capture + histogram() to dump their own distributions.
//
// Usage: bench_hist <trades.csv> <model-stem> <out.csv> [iters] [bins]
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "features.hpp"
#include "latency.hpp"
#include "model.hpp"
#include "parser.hpp"
#include "trade.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <trades.csv> <model-stem> <out.csv> [iters] [bins]\n";
        return 1;
    }
    const std::string trades_path = argv[1];
    const std::string model_stem = argv[2];
    const std::string out_path = argv[3];
    const std::size_t iters = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 200000;
    const int bins = (argc > 5) ? std::atoi(argv[5]) : 20;

    Model cpu;
    try {
        cpu = Model::load(model_stem + ".meta");
    } catch (const std::exception& e) {
        std::cerr << "model load error: " << e.what() << "\n";
        return 1;
    }

    std::vector<Trade> trades;
    try {
        trades = parse_agg_trades(trades_path);
    } catch (const std::exception& e) {
        std::cerr << "parse error: " << e.what() << "\n";
        return 1;
    }

    // Rebuild the model input rows: [ret_1, volatility, imbalance, intensity].
    FeatureEngine engine(50, 1000000);
    std::vector<std::vector<float>> rows;
    Features f;
    double prev_mid = 0.0;
    bool have_prev = false;
    for (const Trade& t : trades) {
        if (!engine.update(t, f)) continue;
        if (have_prev) {
            const double ret_1 = std::log(f.mid / prev_mid);
            rows.push_back({static_cast<float>(ret_1), static_cast<float>(f.volatility),
                            static_cast<float>(f.imbalance), static_cast<float>(f.intensity)});
        }
        prev_mid = f.mid;
        have_prev = true;
    }
    if (rows.empty()) { std::cerr << "error: no feature rows\n"; return 1; }

    HarnessConfig cfg;
    cfg.iters = iters;
    LatencyHarness h(cfg);

    std::size_t idx = 0;
    double sink = 0.0;
    std::vector<double> samples;
    Stats s = h.run_capture([&]() {
        sink += cpu.forward(rows[idx]);
        if (++idx >= rows.size()) idx = 0;
    }, samples);

    s.print("cpu.forward");
    std::vector<HistBucket> hist = histogram(samples, bins);
    write_histogram_csv(out_path, "cpu", hist);

    std::cout << "wrote " << hist.size() << " buckets to " << out_path << "\n";
    std::cout << "sink " << sink << "\n";
    return 0;
}
