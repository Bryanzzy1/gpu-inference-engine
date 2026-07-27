// One process, all four backends, one run: the defensible latency table. Times the CPU
// baseline, naive CUDA, CUDA Graphs, and persistent megakernel on the same harness after
// checking each agrees with the CPU. The steady_clock timer brackets the full end-to-end
// forward() call, including H2D/D2H copies, which is the tick-to-prediction latency that
// matters. Batch size one throughout.
// Usage: bench_all <trades.csv> <model-stem> [warmup] [iters]
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
#include "persistent_model.hpp"
#include "trade.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <trades.csv> <model-stem> [warmup] [iters]\n";
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
    PersistentModel pers;
    try {
        cpu = Model::load(model_stem + ".meta");
        gpu.load(cpu);
        graph.load(cpu);
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

    // Correctness gates: no number is meaningful unless the backend computes the right logit.
    const std::size_t check_n = std::min<std::size_t>(rows.size(), 50000);
    double m_gpu = 0.0, m_graph = 0.0;
    for (std::size_t i = 0; i < check_n; ++i) {
        const float c = cpu.forward(rows[i]);
        m_gpu = std::max(m_gpu, static_cast<double>(std::fabs(c - gpu.forward(rows[i]))));
        m_graph = std::max(m_graph, static_cast<double>(std::fabs(c - graph.forward(rows[i]))));
    }
    std::cout << "match over " << check_n << " rows: max|cpu-naive| = " << m_gpu
              << "  max|cpu-graph| = " << m_graph << "\n";
    if (m_gpu > 1e-4 || m_graph > 1e-4) { std::cerr << "FAIL: a GPU path disagrees\n"; return 1; }

    std::cout << "trades " << trades.size() << "  feature rows " << rows.size() << "\n";

    LatencyHarness harness(cfg);
    std::size_t idx = 0;
    double sink = 0.0;
    auto cycle = [&](LatencyHarness& h, auto&& fn) {
        idx = 0;
        return h.run([&]() { sink += fn(rows[idx]); if (++idx >= rows.size()) idx = 0; });
    };

    // Non-spinning backends run at the full iteration count.
    Stats cpu_s = cycle(harness, [&](const std::vector<float>& r) { return cpu.forward(r); });
    Stats gpu_s = cycle(harness, [&](const std::vector<float>& r) { return gpu.forward(r); });
    Stats gph_s = cycle(harness, [&](const std::vector<float>& r) { return graph.forward(r); });

    // Persistent busy-spins, so its live window must stay under the 2s WDDM TDR watchdog.
    HarnessConfig pcfg = cfg;
    pcfg.iters = std::min<std::size_t>(cfg.iters, 100000);
    LatencyHarness pharness(pcfg);
    pers.start();
    double m_pers = 0.0;
    const std::size_t pcheck = std::min<std::size_t>(rows.size(), 2000);
    for (std::size_t i = 0; i < pcheck; ++i)
        m_pers = std::max(m_pers, static_cast<double>(std::fabs(cpu.forward(rows[i]) - pers.forward(rows[i]))));
    Stats per_s = cycle(pharness, [&](const std::vector<float>& r) { return pers.forward(r); });
    pers.stop();
    std::cout << "match over " << pcheck << " rows: max|cpu-persistent| = " << m_pers << "\n";
    if (m_pers > 1e-4) { std::cerr << "FAIL: persistent disagrees\n"; return 1; }

    std::cout << "warmup " << cfg.warmup << " (discarded)\n";
    std::cout << "\nbackend            iters      p50_us   p99_us   p999_us  transfer\n";
    auto row_us = [](const char* name, const Stats& s, const char* xfer) {
        std::printf("%-18s %-9zu  %8.2f %8.2f %9.2f  %s\n",
                    name, s.count, s.p50 / 1000.0, s.p99 / 1000.0, s.p999 / 1000.0, xfer);
    };
    row_us("cpu", cpu_s, "n/a (host only)");
    row_us("cuda-naive", gpu_s, "included (H2D+D2H)");
    row_us("cuda-graphs", gph_s, "included (H2D+D2H)");
    row_us("persistent", per_s, "included (zero-copy)");
    std::cout << "sink " << sink << "\n";
    return 0;
}
