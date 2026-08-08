// The closed-loop SLA controller, driven by REAL backend latency instead of a simulation.
// Each control tick it measures the actual p99 of the current (backend, batch) over a
// window of inferences on the GPU, feeds that to Controller::step, and applies the batch +
// backend the controller returns. It logs a trace so we can see the loop climb to the
// SLA-limited batch and hold it under real jitter. This closes the capstone loop that
// test_controller only simulated.
//
// Build with nvcc (needs the MSVC host env on Windows):
//   nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
//     src/bench/controller_live.cpp src/gpu/gpu_model.cu src/gpu/graph_model.cu \
//     src/gpu/gpu_weights.cu src/gpu/persistent_model.cu src/cpu/controller.cpp \
//     src/cpu/router.cpp src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp \
//     src/cpu/latency.cpp -o build/controller_live.exe
//
// Usage: controller_live <trades.csv> <model-stem> <frontier.csv> <out.csv> [sla_us] [window]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "controller.hpp"
#include "features.hpp"
#include "graph_model.hpp"
#include "gpu_model.hpp"
#include "model.hpp"
#include "parser.hpp"
#include "persistent_model.hpp"
#include "router.hpp"
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

void pack(const std::vector<std::vector<float>>& rows, std::size_t& idx, int in_dim,
          std::size_t batch, std::vector<float>& out) {
    out.resize(batch * static_cast<std::size_t>(in_dim));
    for (std::size_t b = 0; b < batch; ++b) {
        const std::vector<float>& r = rows[idx];
        std::copy(r.begin(), r.end(), out.begin() + b * static_cast<std::size_t>(in_dim));
        if (++idx >= rows.size()) idx = 0;
    }
}

double p99_ns(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    // Nearest-rank p99, matching the harness convention.
    std::size_t k = static_cast<std::size_t>(std::ceil(0.99 * samples.size()));
    if (k > 0) --k;
    return samples[k];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <trades.csv> <model-stem> <frontier.csv> <out.csv> [sla_us] [window]\n";
        return 1;
    }
    const std::string trades_path = argv[1];
    const std::string model_stem = argv[2];
    const std::string frontier_csv = argv[3];
    const std::string out_path = argv[4];
    const double sla_ns = (argc > 5 ? std::strtod(argv[5], nullptr) : 150.0) * 1000.0;
    const std::size_t window = (argc > 6) ? std::strtoull(argv[6], nullptr, 10) : 3000;

    Model cpu;
    GpuModel gpu;
    GraphModel graph;
    PersistentModel pers;
    Router router = Router::load(frontier_csv); // throws if unreadable
    router.set_metric(Router::Metric::P99); // hold a p99 SLA, so route on p99 not p999
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

    // Measure the real p99 of one (backend, batch) over `window` inferences, unpaced.
    // Persistent is capped to a row budget and wrapped in start()/stop() so its resident
    // window stays under the WDDM TDR watchdog (see frontier_all).
    std::vector<float> in, out;
    std::size_t idx = 0;
    double sink = 0.0;
    auto measure = [&](const std::string& backend, std::size_t batch) -> double {
        std::size_t iters = window;
        std::vector<double> lat;
        lat.reserve(iters);
        const bool is_pers = (backend == "persistent");
        if (is_pers) {
            iters = std::max<std::size_t>(200, std::min(iters, std::size_t(200000) / batch));
            pers.start();
        }
        for (std::size_t i = 0; i < iters; ++i) {
            pack(rows, idx, in_dim, batch, in);
            const auto t0 = std::chrono::steady_clock::now();
            if (backend == "cpu") cpu.forward_batch(in, batch, out);
            else if (backend == "cuda-graphs") graph.forward_batch(in, batch, out);
            else if (is_pers) pers.forward_batch(in, batch, out);
            else gpu.forward_batch(in, batch, out); // cuda-naive and default
            const auto t1 = std::chrono::steady_clock::now();
            lat.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            for (float v : out) sink += v;
        }
        if (is_pers) pers.stop();
        return p99_ns(lat);
    };

    // A bursty arrival-rate schedule, quiet -> spike -> quiet. Rate drives the router's
    // backend choice; the controller adapts batch to hold the p99 SLA under the load.
    const double rate_schedule[] = {1e4, 1e4, 5e4, 1e5, 5e5, 1e6, 1e6, 5e5, 1e5, 5e4, 1e4,
                                    1e4, 5e4, 1e5, 5e5, 1e6, 1e6, 5e5, 1e5, 1e4};
    const int ticks = static_cast<int>(std::size(rate_schedule));

    ControllerConfig cfg;
    cfg.sla_p99_ns = sla_ns;
    Controller ctrl(std::move(router), cfg);

    std::ofstream csv(out_path);
    csv << "tick,rate_hz,batch,backend,observed_p99_ns,sla_ns,under_sla\n";
    std::printf("SLA p99 = %.0f us, window = %zu inferences/tick\n", sla_ns / 1000.0, window);
    std::printf("%-5s %-10s %-6s %-12s %-12s %-6s\n",
                "tick", "rate", "batch", "backend", "p99_us", "ok");

    std::size_t batch = cfg.min_batch;
    std::string backend = "cpu";
    int under = 0, settled_under = 0, settled_ticks = 0;
    for (int t = 0; t < ticks; ++t) {
        const double rate = rate_schedule[t];
        const double p99 = measure(backend, batch); // real latency of the current config
        const bool ok = p99 <= sla_ns;
        if (ok) ++under;
        if (t >= ticks / 3) { settled_ticks++; if (ok) settled_under++; } // ignore the climb

        csv << t << ',' << rate << ',' << batch << ',' << backend << ','
            << p99 << ',' << sla_ns << ',' << (ok ? 1 : 0) << '\n';
        std::printf("%-5d %-10.0f %-6zu %-12s %-12.1f %-6s\n",
                    t, rate, batch, backend.c_str(), p99 / 1000.0, ok ? "yes" : "NO");

        const Decision d = ctrl.step(p99, rate); // close the loop: decide the next config
        batch = d.batch;
        backend = d.backend.empty() ? backend : d.backend;
    }

    std::printf("under SLA: %d/%d ticks overall, %d/%d after settling\n",
                under, ticks, settled_under, settled_ticks);
    std::cout << "wrote " << out_path << "  sink " << sink << "\n";
    return 0;
}
