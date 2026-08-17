// Offered-load sweep: what does end-to-end latency (queue wait + service) do as real load
// rises, per backend and batch? Uses measured per-batch service times from the frontier CSV
// and the discrete-event queue model (queue_sim.hpp). This answers the question the
// synchronous benchmarks cannot: under Poisson arrivals a batching server has a throughput
// ceiling of batch/service, and batching is what raises that ceiling. At small batch a
// modest load already overloads the queue; only large batch stays stable.
//
// No GPU: the service times are already measured (frontier.csv), this only replays them
// under load. Build it with a plain C++ compiler.
//
// Usage: queue_load <frontier.csv> <out.csv> [n_rows]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "queue_sim.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <frontier.csv> <out.csv> [n_rows]\n";
        return 1;
    }
    const std::string frontier_csv = argv[1];
    const std::string out_path = argv[2];
    const std::size_t n_rows = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 200000;

    ServiceModel model;
    try {
        model = ServiceModel::load(frontier_csv);
    } catch (const std::exception& e) {
        std::cerr << "load error: " << e.what() << "\n";
        return 1;
    }

    // The two backends whose crossover the frontier is about; both are launch-per-call in
    // this model. (Persistent is unpaced-only and not part of the offered-load story here.)
    const std::vector<std::string> backends = {"cpu", "cuda-naive"};
    const std::vector<std::size_t> batches = {1, 8, 32, 128, 256};
    // Offered load in rows/s, spanning below the smallest ceiling to above the largest.
    const std::vector<double> rates = {1e4, 5e4, 1e5, 5e5, 1e6, 2e6, 3e6};

    std::ofstream csv(out_path);
    csv << "backend,batch,offered_rows_s,capacity_rows_s,utilization,"
           "p50_ns,p99_ns,p999_ns,max_ns,stable\n";

    std::printf("%-11s %-6s %-11s %-11s %-6s %-10s %-10s %-6s\n",
                "backend", "batch", "offered", "capacity", "util", "p99_us", "p999_us", "ok");
    for (const std::string& backend : backends) {
        for (std::size_t batch : batches) {
            const double cap = model.throughput_rows_s(backend, batch);
            const double s_ns = model.service_ns(backend, batch);
            for (double rate : rates) {
                const QueueResult r = simulate_queue(s_ns, batch, rate, n_rows, 12345);
                const bool stable = r.utilization < 1.0;
                csv << backend << ',' << batch << ',' << rate << ',' << cap << ','
                    << r.utilization << ',' << r.latency.p50 << ',' << r.latency.p99 << ','
                    << r.latency.p999 << ',' << r.latency.max << ',' << (stable ? 1 : 0) << '\n';
                std::printf("%-11s %-6zu %-11.0f %-11.0f %-6.2f %-10.1f %-10.1f %-6s\n",
                            backend.c_str(), batch, rate, cap, r.utilization,
                            r.latency.p99 / 1000.0, r.latency.p999 / 1000.0,
                            stable ? "yes" : "NO");
            }
        }
    }

    std::cout << "wrote " << out_path << "\n";
    return 0;
}
