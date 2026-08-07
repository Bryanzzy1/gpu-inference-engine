#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "router.hpp"

// Closed-loop SLA controller: hold a p99 latency target under changing arrival rate by
// choosing the batch size and backend, using the measured frontier as the model of what
// each choice costs. This is the capstone, the feedback half of routing; the static
// Router is the open-loop half it rides on.
//
// The decision each control tick, given the observed arrival rate and the measured p99
// of the last window:
//   - If measured p99 is over the SLA, back off: drop to a smaller batch (lower latency
//     per inference) and let the Router pick the lowest-p999 backend there.
//   - If measured p99 is comfortably under the SLA and load is high, grow the batch to
//     absorb throughput while staying under target.
//   - Hysteresis: only change batch when the gain clears a margin, so it does not
//     oscillate at a boundary.
//
// No GPU here. The controller is pure decision logic over the frontier table plus a
// live p99 reading; a driver or a simulation feeds it observations.

struct ControllerConfig {
    double sla_p99_ns = 50000.0;   // the p99 target to hold, nanoseconds
    double headroom = 0.8;         // grow batch only while p99 < headroom * SLA
    std::size_t min_batch = 1;
    std::size_t max_batch = 256;
};

// One control decision: what to run next and why.
struct Decision {
    std::size_t batch = 1;
    std::string backend;
    bool changed = false; // did batch change from the previous decision
};

class Controller {
public:
    Controller(Router router, ControllerConfig cfg)
        : router_(std::move(router)), cfg_(cfg) {}

    // One control tick. observed_p99_ns is the measured p99 of the last window at the
    // current batch; rate_hz is the observed arrival rate. Returns the next batch +
    // backend to use. Call repeatedly as load and latency evolve.
    Decision step(double observed_p99_ns, double rate_hz);

    std::size_t batch() const { return batch_; }

private:
    Router router_;
    ControllerConfig cfg_;
    std::size_t batch_ = 1;    // current batch size
};

#endif
