// The jitter autopsy: where does GPU inference latency come from, and how much of the
// tail is each stage? Runs the naive backend forward_timed N times, collecting per-call
// H2D / launch / compute / D2H durations from CUDA events, then reports a per-stage tail
// (p50/p99/p999) and writes a CSV for python/plot_autopsy.py.
//
// The point: the frontier showed the GPU loses at batch 1. This says WHY, stage by
// stage, instead of as one number. Run it twice, once with the GPU clocks locked
// (nvidia-smi --lock-gpu-clocks=<min>,<max>) and once unlocked, to separate clock-ramp
// jitter from real per-stage variance.
//
// Build with nvcc (needs the MSVC host env on Windows):
//   nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
//     src/bench/autopsy.cpp src/gpu/gpu_model.cu src/cpu/stage_timing.cpp \
//     src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp \
//     -o build/autopsy.exe
//
// Usage: autopsy <trades.csv> <model-stem> <out.csv> [label] [iters]
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "features.hpp"
#include "gpu_model.hpp"
#include "model.hpp"
#include "parser.hpp"
#include "stage_timing.hpp"
#include "trade.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <trades.csv> <model-stem> <out.csv> [label] [iters]\n";
        return 1;
    }
    const std::string trades_path = argv[1];
    const std::string model_stem = argv[2];
    const std::string out_path = argv[3];
    const std::string label = (argc > 4) ? argv[4] : "naive";
    const std::size_t iters = (argc > 5) ? std::strtoull(argv[5], nullptr, 10) : 100000;

    Model cpu;
    GpuModel gpu;
    try {
        cpu = Model::load(model_stem + ".meta");
        gpu.load(cpu);
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

    // Warm-up, discarded, so cold caches and clock ramp do not enter the sample.
    StageSample scratch;
    double sink = 0.0;
    for (std::size_t i = 0; i < 1000; ++i) sink += gpu.forward_timed(rows[i % rows.size()], scratch);

    std::vector<StageSample> samples;
    samples.reserve(iters);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < iters; ++i) {
        StageSample s;
        sink += gpu.forward_timed(rows[idx], s);
        samples.push_back(s);
        if (++idx >= rows.size()) idx = 0;
    }

    StageBreakdown b = StageBreakdown::from(samples);
    b.print(label.c_str());
    write_autopsy_csv(out_path, label.c_str(), b);
    std::cout << "wrote " << out_path << "  sink " << sink << "\n";
    return 0;
}
