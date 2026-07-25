// Benchmarks the naive GPU forward pass against the CPU one on the same tail-latency
// harness, after checking the two agree. This is the first CPU-vs-GPU p999 comparison.
// Usage: bench_gpu <trades.csv> <model-stem> [warmup] [iters]
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "features.hpp"
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
    try {
        cpu = Model::load(model_stem + ".meta");
    } catch (const std::exception& e) {
        std::cerr << "model load error: " << e.what() << "\n";
        return 1;
    }

    GpuModel gpu;
    try {
        gpu.load(cpu);
    } catch (const std::exception& e) {
        std::cerr << "gpu init error: " << e.what() << "\n";
        return 1;
    }

    std::vector<Trade> trades;
    try {
        trades = parse_agg_trades(trades_path);
    } catch (const std::exception& e) {
        std::cerr << "parse error: " << e.what() << "\n";
        return 1;
    }

    // Build model input rows [ret_1, volatility, imbalance, intensity], same as bench_model.
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

    // Correctness gate: the GPU forward must match the CPU forward before timing means
    // anything. Check a large slice and report the worst disagreement.
    const std::size_t check_n = std::min<std::size_t>(rows.size(), 50000);
    double max_abs = 0.0;
    for (std::size_t i = 0; i < check_n; ++i) {
        const float c = cpu.forward(rows[i]);
        const float g = gpu.forward(rows[i]);
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(c - g)));
    }
    std::cout << "match check over " << check_n << " rows: max|cpu - gpu| = "
              << max_abs << "\n";
    if (max_abs > 1e-4) {
        std::cerr << "FAIL: gpu forward disagrees with cpu beyond 1e-4\n";
        return 1;
    }

    std::cout << "trades " << trades.size() << "  feature rows " << rows.size() << "\n";
    std::cout << "warmup " << cfg.warmup << "  iters " << cfg.iters << "\n";

    // Time the CPU path.
    std::size_t idx = 0;
    double sink = 0.0;
    LatencyHarness harness(cfg);
    Stats cpu_s = harness.run([&]() {
        sink += cpu.forward(rows[idx]);
        if (++idx >= rows.size()) idx = 0;
    });

    // Time the naive GPU path on the same harness.
    idx = 0;
    Stats gpu_s = harness.run([&]() {
        sink += gpu.forward(rows[idx]);
        if (++idx >= rows.size()) idx = 0;
    });

    cpu_s.print("cpu.forward");
    gpu_s.print("gpu.forward(naive)");
    if (cpu_s.p999 > 0.0) {
        std::cout << "p999 ratio gpu/cpu = " << (gpu_s.p999 / cpu_s.p999) << "\n";
    }
    std::cout << "sink " << sink << "\n";
    return 0;
}
