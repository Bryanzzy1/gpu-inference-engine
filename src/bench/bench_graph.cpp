// Benchmarks the CUDA Graphs backend against the naive GPU path and the CPU on the
// same tail-latency harness, after checking all three agree. The graph replays a
// captured copy-launch-copy sequence, so its per-call launch overhead is lower than the
// naive path's three separate submissions.
// Usage: bench_graph <trades.csv> <model-stem> [warmup] [iters]
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "features.hpp"
#include "graph_model.hpp"
#include "gpu_model.hpp"
#include "latency.hpp"
#include "model.hpp"
#include "parser.hpp"
#include "trade.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <trades.csv> <model-stem> [warmup] [iters]\n";
        return 1;
    }
    const std::string trades_path = argv[1];
    const std::string model_stem = argv[2];

    HarnessConfig cfg;
    if (argc > 3) cfg.warmup = std::strtoull(argv[3], nullptr, 10);
    if (argc > 4) cfg.iters = std::strtoull(argv[4], nullptr, 10);

    Model cpu;
    GpuModel gpu;
    GraphModel graph;
    try {
        cpu = Model::load(model_stem + ".meta");
        gpu.load(cpu);
        graph.load(cpu);
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

    // Build model input rows [ret_1, volatility, imbalance, intensity], as in bench_gpu.
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

    // Correctness gate: both GPU paths must match the CPU before timing means anything.
    const std::size_t check_n = std::min<std::size_t>(rows.size(), 50000);
    double max_gpu = 0.0;
    double max_graph = 0.0;
    for (std::size_t i = 0; i < check_n; ++i) {
        const float c = cpu.forward(rows[i]);
        max_gpu = std::max(max_gpu, static_cast<double>(std::fabs(c - gpu.forward(rows[i]))));
        max_graph = std::max(max_graph, static_cast<double>(std::fabs(c - graph.forward(rows[i]))));
    }
    std::cout << "match over " << check_n << " rows: max|cpu-gpu| = " << max_gpu
              << "  max|cpu-graph| = " << max_graph << "\n";
    if (max_gpu > 1e-4 || max_graph > 1e-4) {
        std::cerr << "FAIL: a GPU path disagrees with the CPU beyond 1e-4\n";
        return 1;
    }

    std::cout << "trades " << trades.size() << "  feature rows " << rows.size() << "\n";
    std::cout << "warmup " << cfg.warmup << "  iters " << cfg.iters << "\n";

    LatencyHarness harness(cfg);
    std::size_t idx = 0;
    double sink = 0.0;
    auto cycle = [&](auto&& fn) {
        idx = 0;
        return harness.run([&]() {
            sink += fn(rows[idx]);
            if (++idx >= rows.size()) idx = 0;
        });
    };

    Stats cpu_s = cycle([&](const std::vector<float>& r) { return cpu.forward(r); });
    Stats gpu_s = cycle([&](const std::vector<float>& r) { return gpu.forward(r); });
    Stats gph_s = cycle([&](const std::vector<float>& r) { return graph.forward(r); });

    cpu_s.print("cpu.forward");
    gpu_s.print("gpu.forward(naive)");
    gph_s.print("gpu.forward(graph)");
    if (gpu_s.p99 > 0.0) {
        std::cout << "p99 graph/naive = " << (gph_s.p99 / gpu_s.p99) << "\n";
    }
    if (gpu_s.p999 > 0.0) {
        std::cout << "p999 graph/naive = " << (gph_s.p999 / gpu_s.p999) << "\n";
    }
    std::cout << "sink " << sink << "\n";
    return 0;
}
