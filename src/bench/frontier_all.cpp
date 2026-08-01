// The 2D frontier over all four backends. For each (batch size x arrival rate) cell
// it times CPU, naive GPU, CUDA Graphs, and persistent, after gating each on matching
// the CPU batched reference, and writes one CSV row per (backend, batch, rate). Feed
// the CSV to python/plot_frontier.py for the winner heatmap.
//
// Build with nvcc, not CMake (nvcc needs the MSVC host compiler on Windows):
//   nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
//     src/bench/frontier_all.cpp src/bench/frontier.cpp \
//     src/gpu/gpu_model.cu src/gpu/graph_model.cu src/gpu/gpu_weights.cu \
//     src/gpu/persistent_model.cu src/io/parser.cpp src/io/features.cpp \
//     src/cpu/model.cpp src/cpu/latency.cpp -o build/frontier_all.exe
//
// The batch axis is only meaningful here, see batch.hpp; frontier_cpu is the CPU
// control. Persistent is measured unpaced only (see the row-budget note below).
//
// Usage: frontier_all <trades.csv> <model-stem> <out.csv> [iters]
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "batch.hpp"
#include "features.hpp"
#include "frontier.hpp"
#include "graph_model.hpp"
#include "gpu_model.hpp"
#include "latency.hpp"
#include "model.hpp"
#include "parser.hpp"
#include "persistent_model.hpp"
#include "trade.hpp"

namespace {

std::vector<std::vector<float>> build_rows(const std::vector<Trade>& trades) {
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

// Packs `batch` rows starting at *idx (wrapping) into out, row-major.
void pack(const std::vector<std::vector<float>>& rows, std::size_t& idx, int in_dim,
          std::size_t batch, std::vector<float>& out) {
    out.resize(batch * static_cast<std::size_t>(in_dim));
    for (std::size_t b = 0; b < batch; ++b) {
        const std::vector<float>& r = rows[idx];
        std::copy(r.begin(), r.end(), out.begin() + b * static_cast<std::size_t>(in_dim));
        if (++idx >= rows.size()) idx = 0;
    }
}

// Max abs difference between a backend's batched output and the CPU reference.
double batch_mismatch(const std::vector<float>& got, const std::vector<float>& ref) {
    double m = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i)
        m = std::max(m, static_cast<double>(std::fabs(got[i] - ref[i])));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <trades.csv> <model-stem> <out.csv> [iters]\n";
        return 1;
    }
    const std::string trades_path = argv[1];
    const std::string model_stem = argv[2];
    const std::string out_path = argv[3];
    const std::size_t iters = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 50000;

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
    const std::vector<std::vector<float>> rows = build_rows(trades);
    if (rows.empty()) { std::cerr << "error: no feature rows\n"; return 1; }

    const int in_dim = cpu.input_dim();
    const std::vector<double> rates = default_rate_axis();
    std::vector<FrontierCell> cells;

    std::vector<float> in, ref, got;
    std::size_t idx = 0;
    double sink = 0.0;

    // Persistent runs unpaced only (rate 0). On Windows WDDM a resident busy-spinning
    // kernel is reset by the 2s TDR watchdog, so a host-paced run (iters/rate seconds
    // resident) always trips it, and even unpaced the live window grows with the rows
    // processed. Cap the persistent window to a row budget so it stays under the
    // watchdog; the CSV records the actual sample count per cell.
    constexpr std::size_t kPersRowBudget = 300000;

    for (std::size_t batch : kFrontierBatches) {
        // Correctness gate per batch size, before timing anything at this size.
        pack(rows, idx, in_dim, batch, in);
        cpu.forward_batch(in, batch, ref);
        gpu.forward_batch(in, batch, got);
        const double m_gpu = batch_mismatch(got, ref);
        graph.forward_batch(in, batch, got);
        const double m_graph = batch_mismatch(got, ref);
        pers.start();
        pers.forward_batch(in, batch, got);
        pers.stop();
        const double m_pers = batch_mismatch(got, ref);
        if (std::max({m_gpu, m_graph, m_pers}) > 1e-4) {
            std::cerr << "FAIL: a backend disagrees at batch " << batch
                      << " (naive " << m_gpu << " graph " << m_graph
                      << " persistent " << m_pers << ")\n";
            return 1;
        }

        for (double rate : rates) {
            HarnessConfig cfg;
            cfg.warmup = 1000;
            cfg.iters = iters;
            cfg.target_rate_hz = rate;

            auto run_backend = [&](const char* name, const HarnessConfig& c, auto&& fn) {
                LatencyHarness h(c);
                Stats s = h.run([&]() {
                    pack(rows, idx, in_dim, batch, in);
                    fn(in, batch, got);
                    for (float v : got) sink += v;
                });
                cells.push_back({name, batch, rate, s});
            };

            run_backend("cpu", cfg, [&](const std::vector<float>& i, std::size_t n, std::vector<float>& o) { cpu.forward_batch(i, n, o); });
            run_backend("cuda-naive", cfg, [&](const std::vector<float>& i, std::size_t n, std::vector<float>& o) { gpu.forward_batch(i, n, o); });
            run_backend("cuda-graphs", cfg, [&](const std::vector<float>& i, std::size_t n, std::vector<float>& o) { graph.forward_batch(i, n, o); });

            if (rate == 0.0) {
                HarnessConfig pcfg = cfg;
                pcfg.iters = std::max<std::size_t>(1000, std::min(iters, kPersRowBudget / batch));
                pcfg.warmup = std::min<std::size_t>(cfg.warmup, pcfg.iters / 10);
                pers.start();
                run_backend("persistent", pcfg, [&](const std::vector<float>& i, std::size_t n, std::vector<float>& o) { pers.forward_batch(i, n, o); });
                pers.stop();
            }

            std::printf("batch=%-4zu rate=%-9.0f done\n", batch, rate);
        }
    }

    write_frontier_csv(out_path, cells);
    std::cout << "wrote " << cells.size() << " cells to " << out_path << "\n";
    std::cout << "sink " << sink << "\n";
    return 0;
}
