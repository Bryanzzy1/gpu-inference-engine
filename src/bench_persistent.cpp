// Benchmarks the persistent megakernel against the naive GPU path and the CPU on the
// same tail-latency harness, after checking all three agree. The persistent path pays
// no per-event launch cost, so its p999 should sit below the naive GPU's.
// Usage: bench_persistent <trades.csv> <model-stem> [warmup] [iters]
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "features.hpp"
#include "gpu_model.hpp"
#include "latency.hpp"
#include "model.hpp"
#include "parser.hpp"
#include "persistent_model.hpp"
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
    PersistentModel pers;
    try {
        cpu = Model::load(model_stem + ".meta");
        gpu.load(cpu);
        pers.load(cpu);
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

    // Naive GPU must match the CPU before timing means anything (persistent is checked
    // below, inside its own start/stop window).
    const std::size_t check_n = std::min<std::size_t>(rows.size(), 50000);
    double max_gpu = 0.0;
    for (std::size_t i = 0; i < check_n; ++i) {
        const float c = cpu.forward(rows[i]);
        max_gpu = std::max(max_gpu, static_cast<double>(std::fabs(c - gpu.forward(rows[i]))));
    }
    std::cout << "match over " << check_n << " rows: max|cpu-gpu| = " << max_gpu << "\n";
    if (max_gpu > 1e-4) {
        std::cerr << "FAIL: naive GPU disagrees with the CPU beyond 1e-4\n";
        return 1;
    }

    std::cout << "trades " << trades.size() << "  feature rows " << rows.size() << "\n";
    std::cout << "warmup " << cfg.warmup << "  iters " << cfg.iters << "\n";

    LatencyHarness harness(cfg);
    std::size_t idx = 0;
    double sink = 0.0;
    auto cycle = [&](LatencyHarness& h, auto&& fn) {
        idx = 0;
        return h.run([&]() {
            sink += fn(rows[idx]);
            if (++idx >= rows.size()) idx = 0;
        });
    };

    // CPU and naive GPU: no persistent kernel running, so no TDR pressure.
    Stats cpu_s = cycle(harness, [&](const std::vector<float>& r) { return cpu.forward(r); });
    Stats gpu_s = cycle(harness, [&](const std::vector<float>& r) { return gpu.forward(r); });

    // Persistent: the kernel busy-spins, so its whole live window must stay under the
    // ~2s WDDM TDR watchdog. Cap iters and keep start()->stop() tight around the timing.
    HarnessConfig pcfg = cfg;
    pcfg.iters = std::min<std::size_t>(cfg.iters, 100000);
    LatencyHarness pharness(pcfg);

    pers.start();
    double max_pers = 0.0;
    const std::size_t pcheck = std::min<std::size_t>(rows.size(), 2000);
    for (std::size_t i = 0; i < pcheck; ++i) {
        max_pers = std::max(max_pers, static_cast<double>(std::fabs(cpu.forward(rows[i]) - pers.forward(rows[i]))));
    }
    Stats per_s = cycle(pharness, [&](const std::vector<float>& r) { return pers.forward(r); });
    pers.stop();

    std::cout << "match over " << pcheck << " rows: max|cpu-persistent| = " << max_pers << "\n";
    if (max_pers > 1e-4) {
        std::cerr << "FAIL: persistent GPU disagrees with the CPU beyond 1e-4\n";
        return 1;
    }

    cpu_s.print("cpu.forward");
    gpu_s.print("gpu.forward(naive)");
    per_s.print("gpu.forward(persistent)");
    if (gpu_s.p999 > 0.0) {
        std::cout << "p999 persistent/naive = " << (per_s.p999 / gpu_s.p999) << "\n";
    }
    std::cout << "sink " << sink << "\n";
    return 0;
}
