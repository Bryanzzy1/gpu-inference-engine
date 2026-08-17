#ifndef QUEUE_SIM_HPP
#define QUEUE_SIM_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "latency.hpp"

// A discrete-event queue model of one backend under real offered load.
//
// The synchronous benchmarks (bench_all, frontier, controller_live) feed one input, wait
// for the result, feed the next. Arrival rate never builds a queue there, so latency is
// independent of load and batching looks like a pure latency tradeoff. It is not. Under
// real Poisson arrivals a server that batches has a throughput ceiling of batch/service,
// and any offered load above that ceiling makes the queue grow without bound. This model
// makes that visible: it replays MEASURED per-batch service times (the frontier p50) under
// a Poisson arrival stream and reports the end-to-end latency (queue wait + service) tail.
//
// The service times are real (measured on the GPU, loaded from the frontier CSV); only the
// arrival process and the queue are simulated. That keeps the finding grounded: the numbers
// come from the hardware, the model just puts them under load the synchronous harness could
// not.

// Per-(backend, batch) service time, loaded from the frontier CSV (rate==0 rows, p50_ns).
struct ServiceModel {
    // s[backend][batch] = service time in ns of one batched call at that batch size.
    std::map<std::string, std::map<std::size_t, double>> s;

    static ServiceModel load(const std::string& frontier_csv) {
        std::ifstream f(frontier_csv);
        if (!f) throw std::runtime_error("queue_sim: cannot open " + frontier_csv);
        ServiceModel m;
        std::string line;
        bool header = true;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (header) { header = false; continue; }
            std::vector<std::string> c;
            std::string field;
            std::istringstream ss(line);
            while (std::getline(ss, field, ',')) c.push_back(field);
            if (c.size() < 5) continue;                 // backend,batch,rate,count,p50,...
            if (std::atof(c[2].c_str()) != 0.0) continue; // unpaced rows are the service time
            m.s[c[0]][std::strtoull(c[1].c_str(), nullptr, 10)] = std::atof(c[4].c_str());
        }
        if (m.s.empty()) throw std::runtime_error("queue_sim: no unpaced rows in " + frontier_csv);
        return m;
    }

    // Service time (ns) of `backend` at `batch`. Throws if that cell is not in the CSV.
    double service_ns(const std::string& backend, std::size_t batch) const {
        auto b = s.find(backend);
        if (b == s.end()) throw std::runtime_error("queue_sim: no backend " + backend);
        auto it = b->second.find(batch);
        if (it == b->second.end())
            throw std::runtime_error("queue_sim: no batch cell for " + backend);
        return it->second;
    }

    // Rows per second the backend sustains at this batch: batch / service. The queue is
    // stable only while the offered rate stays below this ceiling.
    double throughput_rows_s(const std::string& backend, std::size_t batch) const {
        return static_cast<double>(batch) / (service_ns(backend, batch) / 1e9);
    }
};

// Result of one simulated run: the end-to-end latency tail plus stability info.
struct QueueResult {
    Stats latency;              // arrival-to-completion latency, ns
    double offered_rate_s = 0;  // the Poisson rate fed in, rows/s
    double capacity_rows_s = 0; // batch/service ceiling
    double utilization = 0;     // offered / capacity; >= 1 means the queue is unstable
    std::size_t rounds = 0;     // service rounds executed
};

// Simulate opportunistic fixed-batch service under Poisson arrivals.
//
// The server, whenever free, takes up to `batch` rows that have already arrived and serves
// them together in one round costing `service_ns` (a GPU kernel launches a fixed-shape grid,
// so a partly full batch still costs the full service time, which is the realistic model).
// Rows arriving mid-round wait for the next one. Each row's latency is completion - arrival.
//
// `n_rows` sets the run length; under overload the tail keeps climbing with n (the signature
// of an unstable queue), so compare two run lengths to detect instability rather than trust a
// single number. `seed` makes the arrival stream reproducible.
inline QueueResult simulate_queue(double service_ns, std::size_t batch, double rate_rows_s,
                                  std::size_t n_rows, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::exponential_distribution<double> gap(rate_rows_s / 1e9); // mean gap in ns

    // Generate sorted arrival times (ns). Exponential gaps make a Poisson process.
    std::vector<double> arrivals(n_rows);
    double t_arr = 0.0;
    for (std::size_t i = 0; i < n_rows; ++i) {
        t_arr += gap(rng);
        arrivals[i] = t_arr;
    }

    std::vector<double> lat;
    lat.reserve(n_rows);
    std::deque<double> waiting; // arrival times of queued-but-unserved rows, FIFO
    double t = 0.0;             // time the server is next free
    std::size_t i = 0;         // next not-yet-arrived index
    std::size_t rounds = 0;

    while (i < n_rows || !waiting.empty()) {
        while (i < n_rows && arrivals[i] <= t) waiting.push_back(arrivals[i++]);
        if (waiting.empty()) {          // server idle: fast-forward to the next arrival
            t = arrivals[i];
            continue;
        }
        const double complete = t + service_ns;
        for (std::size_t k = 0; k < batch && !waiting.empty(); ++k) {
            lat.push_back(complete - waiting.front());
            waiting.pop_front();
        }
        t = complete;
        ++rounds;
    }

    QueueResult r;
    r.latency = Stats::from(lat);
    r.offered_rate_s = rate_rows_s;
    r.capacity_rows_s = static_cast<double>(batch) / (service_ns / 1e9);
    r.utilization = rate_rows_s / r.capacity_rows_s;
    r.rounds = rounds;
    return r;
}

#endif
