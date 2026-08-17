// Self-check for the queue simulator: three properties that must hold, no GPU, no CSV.
//   1. Near-idle load: latency is about one service time (nothing waits in the queue).
//   2. Stable load (offered < capacity): the tail is bounded and does not grow with run
//      length.
//   3. Overload (offered > capacity): the tail grows with run length, the signature of an
//      unstable queue.
// If any fails, the simulator's queue accounting is wrong.
#include <cstdio>

#include "queue_sim.hpp"

int main() {
    const double S = 1000.0;      // 1 us service time per round
    const std::size_t B = 8;      // batch 8 -> capacity 8 / 1us = 8e6 rows/s
    const double cap = static_cast<double>(B) / (S / 1e9); // 8,000,000 rows/s
    bool ok = true;

    // 1. Near-idle: 1000 rows/s, capacity 8e6, so essentially no queueing.
    QueueResult idle = simulate_queue(S, B, 1000.0, 50000, 1);
    std::printf("idle:     p50=%.0fns p99=%.0fns util=%.4f\n",
                idle.latency.p50, idle.latency.p99, idle.utilization);
    if (idle.latency.p99 > 3.0 * S) ok = false; // should sit near one service time

    // 2. Stable: offered = 60% of capacity. Tail must not grow with run length.
    QueueResult stable_short = simulate_queue(S, B, 0.6 * cap, 20000, 2);
    QueueResult stable_long = simulate_queue(S, B, 0.6 * cap, 200000, 2);
    std::printf("stable:   p99 short=%.0fns long=%.0fns util=%.2f\n",
                stable_short.latency.p99, stable_long.latency.p99, stable_long.utilization);
    if (stable_long.latency.p99 > 2.0 * stable_short.latency.p99 + S) ok = false;

    // 3. Overload: offered = 120% of capacity. Tail must grow with run length.
    QueueResult over_short = simulate_queue(S, B, 1.2 * cap, 20000, 3);
    QueueResult over_long = simulate_queue(S, B, 1.2 * cap, 200000, 3);
    std::printf("overload: p99 short=%.0fns long=%.0fns util=%.2f\n",
                over_short.latency.p99, over_long.latency.p99, over_long.utilization);
    if (over_long.latency.p99 <= over_short.latency.p99) ok = false; // must blow up with n

    std::printf(ok ? "OK: queue model behaves\n" : "FAIL: queue model is wrong\n");
    return ok ? 0 : 1;
}
