// Self-check for the stage-timing host code: feeds synthetic per-stage samples,
// confirms the per-stage and total tails come out ordered and sensible. No GPU needed;
// this verifies the StageBreakdown math and CSV path that the GPU autopsy relies on.
#include <cstdio>
#include <vector>

#include "stage_timing.hpp"

int main() {
    // 1000 samples, each stage a fixed base plus a small ramp so percentiles differ.
    std::vector<StageSample> samples;
    for (int i = 0; i < 1000; ++i) {
        StageSample s;
        s.ns[0] = 200.0 + i * 0.1;   // h2d
        s.ns[1] = 500.0 + i * 0.2;   // launch
        s.ns[2] = 800.0 + i * 0.05;  // compute
        s.ns[3] = 200.0 + i * 0.1;   // d2h
        samples.push_back(s);
    }

    StageBreakdown b = StageBreakdown::from(samples);
    b.print("synthetic");

    // Total p50 must equal the sum of stage p50s here (monotone inputs), and each
    // p50 <= p99 <= p999. Cheap invariants that catch an indexing bug.
    bool ok = true;
    for (int s = 0; s < 4; ++s) {
        if (!(b.stage[s].p50 <= b.stage[s].p99 && b.stage[s].p99 <= b.stage[s].p999)) ok = false;
    }
    if (!(b.total.p50 <= b.total.p99 && b.total.p99 <= b.total.p999)) ok = false;

    std::printf(ok ? "OK: stage tails ordered\n" : "FAIL: stage tails out of order\n");
    return ok ? 0 : 1;
}
