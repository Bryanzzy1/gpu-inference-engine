// Self-check for the closed-loop SLA controller: writes a small synthetic frontier,
// then simulates latency responding to batch size and asserts the controller (a) backs
// off when p99 exceeds the SLA and (b) grows batch when it has headroom. No GPU needed;
// this verifies the control logic before it drives real backends.
#include <cstdio>
#include <fstream>
#include <string>

#include "controller.hpp"
#include "router.hpp"

int main() {
    // Frontier where p999 rises with batch. The simulated "observed p99" below uses
    // the same shape, so bigger batch = higher latency, the realistic direction.
    const std::string path = "test_ctrl_frontier.csv";
    {
        std::ofstream f(path);
        f << "backend,batch,rate_hz,count,p50_ns,p99_ns,p999_ns,max_ns,iqr_ns,mean_ns\n";
        for (std::size_t b : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
            const double p999 = 1000.0 * static_cast<double>(b);
            f << "cpu," << b << ",0,1000,300," << p999 * 0.6 << "," << p999
              << "," << p999 * 1.2 << ",100,350\n";
        }
    }
    Router r = Router::load(path);
    std::remove(path.c_str());

    ControllerConfig cfg;
    cfg.sla_p99_ns = 20000.0;  // 20 us target
    Controller ctrl(std::move(r), cfg);

    // Model observed p99 at a batch as ~600 ns * batch (matches the frontier shape).
    auto observed_p99 = [](std::size_t batch) { return 600.0 * static_cast<double>(batch); };

    bool ok = true;

    // Start at batch 1 (600 ns, far under SLA): controller should grow.
    Decision d = ctrl.step(observed_p99(ctrl.batch()), 0.0);
    std::printf("under SLA: batch 1 -> %zu (backend %s)\n", d.batch, d.backend.c_str());
    if (d.batch <= 1) ok = false;

    // Drive it up until it would exceed the SLA, then confirm it backs off.
    for (int i = 0; i < 10; ++i) d = ctrl.step(observed_p99(ctrl.batch()), 0.0);
    const std::size_t settled = ctrl.batch();
    std::printf("settled batch under 20us SLA: %zu (p99 ~%.0f ns)\n",
                settled, observed_p99(settled));
    // At the settle point p99 must be under the SLA (the controller holds the target).
    if (observed_p99(settled) > cfg.sla_p99_ns) ok = false;

    // Now slam it with an over-SLA reading and confirm it drops a rung.
    const std::size_t before = ctrl.batch();
    d = ctrl.step(cfg.sla_p99_ns * 2.0, 0.0);
    std::printf("over SLA: batch %zu -> %zu\n", before, d.batch);
    if (before > 1 && d.batch >= before) ok = false;

    std::printf(ok ? "OK: controller holds the SLA\n" : "FAIL: controller mis-behaved\n");
    return ok ? 0 : 1;
}
