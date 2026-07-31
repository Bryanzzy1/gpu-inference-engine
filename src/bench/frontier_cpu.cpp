// Sweeps the CPU backend over the 2D frontier (batch size x arrival rate) and writes
// a CSV of per-inference tail latency per cell. This is the frontier scaffold: it runs
// with no GPU so the sweep, batching, pacing, and CSV are verified on the CPU path here.
// The four-backend GPU version (frontier_all) reuses frontier.hpp and the same axes on
// a machine with nvcc.
// Usage: frontier_cpu <trades.csv> <model-stem> <out.csv> [iters]
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "features.hpp"
#include "frontier.hpp"
#include "latency.hpp"
#include "model.hpp"
#include "parser.hpp"
#include "trade.hpp"

// Builds the [ret_1, volatility, imbalance, intensity] rows the model consumes.
static std::vector<std::vector<float>> build_rows(const std::vector<Trade>& trades) {
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
    return rows;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <trades.csv> <model-stem> <out.csv> [iters]\n";
        return 1;
    }
    const std::string trades_path = argv[1];
    const std::string model_stem = argv[2];
    const std::string out_path = argv[3];
    const std::size_t iters = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 100000;

    Model cpu;
    try {
        cpu = Model::load(model_stem + ".meta");
    } catch (const std::exception& e) {
        std::cerr << "init error: " << e.what() << "\n";
        return 1;
    }

    std::vector<Trade> trades;
    try {
        trades = parse_agg_trades(trades_path);
    } catch (const std::exception& e) {
        std::cerr << "parse error: " << e.what() << "\n";
        return 1;
    }

    const std::vector<std::vector<float>> rows = build_rows(trades);
    if (rows.empty()) { std::cerr << "error: no feature rows\n"; return 1; }
    std::cout << "feature rows " << rows.size() << "\n";

    const std::vector<std::size_t> batches = default_batch_axis();
    const std::vector<double> rates = default_rate_axis();

    std::vector<FrontierCell> cells;
    std::size_t idx = 0;
    double sink = 0.0;
    const int in_dim = cpu.input_dim();

    std::vector<float> batch_in;
    std::vector<float> batch_out;

    for (std::size_t batch : batches) {
        for (double rate : rates) {
            HarnessConfig cfg;
            cfg.warmup = 1000;
            cfg.iters = iters;
            cfg.target_rate_hz = rate;
            LatencyHarness harness(cfg);

            batch_in.resize(batch * static_cast<std::size_t>(in_dim));

            // One timed unit = one batched inference over `batch` rows, packed
            // row-major, matching the GPU batched contract in batch.hpp.
            auto work = [&]() {
                for (std::size_t b = 0; b < batch; ++b) {
                    const std::vector<float>& r = rows[idx];
                    std::copy(r.begin(), r.end(),
                              batch_in.begin() + b * static_cast<std::size_t>(in_dim));
                    if (++idx >= rows.size()) idx = 0;
                }
                cpu.forward_batch(batch_in, batch, batch_out);
                for (float v : batch_out) sink += v;
            };
            FrontierCell cell;
            cell.backend = "cpu";
            cell.batch = batch;
            cell.rate_hz = rate;
            cell.stats = harness.run(work);
            cells.push_back(cell);

            std::printf("cpu  batch=%-4zu rate=%-9.0f p50=%8.1f p99=%8.1f p999=%9.1f ns\n",
                        batch, rate, cell.stats.p50, cell.stats.p99, cell.stats.p999);
        }
    }

    write_frontier_csv(out_path, cells);
    std::cout << "wrote " << cells.size() << " cells to " << out_path << "\n";
    std::cout << "sink " << sink << "\n";
    return 0;
}
