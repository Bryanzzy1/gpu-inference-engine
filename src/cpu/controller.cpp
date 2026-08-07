#include "controller.hpp"

#include <iterator>

#include "batch.hpp"

namespace {

// The batch ladder the controller steps along, clamped to the config bounds.
// Returns the index of the largest ladder value <= batch (the current rung).
int rung_of(std::size_t batch) {
    int r = 0;
    for (int i = 0; i < static_cast<int>(std::size(kFrontierBatches)); ++i) {
        if (kFrontierBatches[i] <= batch) r = i;
    }
    return r;
}

}  // namespace

Decision Controller::step(double observed_p99_ns, double rate_hz) {
    const std::size_t prev = batch_;
    int r = rung_of(batch_);
    const int last = static_cast<int>(std::size(kFrontierBatches)) - 1;

    if (observed_p99_ns > cfg_.sla_p99_ns) {
        // Over the SLA: back off one rung to cut per-inference latency.
        if (r > 0) --r;
    } else if (observed_p99_ns < cfg_.headroom * cfg_.sla_p99_ns) {
        // Comfortably under with headroom to spare: grow one rung to absorb load.
        // The headroom margin is the hysteresis: we only grow when clearly safe, so
        // the loop does not flip up and down at the boundary.
        if (r < last) ++r;
    }

    std::size_t next = kFrontierBatches[r];
    if (next < cfg_.min_batch) next = cfg_.min_batch;
    if (next > cfg_.max_batch) next = cfg_.max_batch;
    batch_ = next;

    Decision d;
    d.batch = batch_;
    d.backend = router_.route(batch_, rate_hz);
    d.changed = (batch_ != prev);
    return d;
}
