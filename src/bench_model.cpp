// Benchmarks the CPU model forward pass and reports tail latency.
// Usage: bench_model <trades.csv> <model-stem> [warmup] [iters] [rate_hz]
#include <cmath>
#include <cstdint>
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
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <trades.csv> <model-stem> [warmup] [iters] [rate_hz]\n";
        return 1;
    }
    const std::string trades_path = argv[1];
    const std::string model_stem = argv[2];

    HarnessConfig cfg;
    if (argc > 3) cfg.warmup = std::strtoull(argv[3], nullptr, 10);
    if (argc > 4) cfg.iters = std::strtoull(argv[4], nullptr, 10);
    if (argc > 5) cfg.target_rate_hz = std::atof(argv[5]);

    Model model;
    try {
        model = Model::load(model_stem + ".meta");
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

    // Build the model input rows: [ret_1, volatility, imbalance, intensity].
    // ret_1 is the log return between consecutive feature mids.
    FeatureEngine engine(50, 1000000);
    std::vector<std::vector<float>> rows;
    Features f;
    double prev_mid = 0.0;
    bool have_prev = false;
    for (const Trade& t : trades) {
        if (!engine.update(t, f)) continue;
        if (have_prev) {
            const double ret_1 = std::log(f.mid / prev_mid);
            rows.push_back({static_cast<float>(ret_1),
                            static_cast<float>(f.volatility),
                            static_cast<float>(f.imbalance),
                            static_cast<float>(f.intensity)});
        }
        prev_mid = f.mid;
        have_prev = true;
    }

    if (rows.empty()) {
        std::cerr << "error: no feature rows\n";
        return 1;
    }

    std::cout << "trades " << trades.size() << "  feature rows " << rows.size()
              << "  input_dim " << model.input_dim() << "\n";
    std::cout << "warmup " << cfg.warmup << "  iters " << cfg.iters;
    if (cfg.target_rate_hz > 0.0)
        std::cout << "  replay " << cfg.target_rate_hz << " Hz";
    std::cout << "\n";

    // Cycle through the real rows, accumulate the logit so the call is not elided.
    std::size_t idx = 0;
    double sink = 0.0;
    auto work = [&]() {
        sink += model.forward(rows[idx]);
        ++idx;
        if (idx >= rows.size()) idx = 0;
    };

    LatencyHarness harness(cfg);
    Stats s = harness.run(work);
    s.print("model.forward");

    // Print the sink so the compiler cannot drop the timed work.
    std::cout << "sink " << sink << "\n";
    return 0;
}
