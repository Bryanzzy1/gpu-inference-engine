// Self-check for the static router: writes a small synthetic frontier CSV, loads it,
// and asserts the router returns the lowest-p999 backend per cell. No GPU needed; this
// verifies the CSV parse and the nearest-cell lowest-p999 lookup the controller builds on.
#include <cstdio>
#include <fstream>
#include <string>

#include "router.hpp"

int main() {
    // Two cells. At batch 1 the CPU wins; at batch 256 cuda-naive wins. Extra rows
    // present so the router has to pick the minimum, not just the first match.
    const std::string path = "test_frontier.csv";
    {
        std::ofstream f(path);
        f << "backend,batch,rate_hz,count,p50_ns,p99_ns,p999_ns,max_ns,iqr_ns,mean_ns\n";
        f << "cpu,1,0,1000,300,500,700,900,100,350\n";
        f << "cuda-naive,1,0,1000,4000,9000,20000,60000,5000,5000\n";
        f << "persistent,1,0,1000,5500,8000,12000,30000,2500,6000\n";
        f << "cpu,256,0,1000,1350000,1400000,1500000,1600000,50000,1360000\n";
        f << "cuda-naive,256,0,1000,900000,1000000,1017000,1100000,80000,950000\n";
    }

    Router r = Router::load(path);
    std::remove(path.c_str());

    bool ok = true;
    auto check = [&](std::size_t batch, double rate, const std::string& want) {
        const std::string got = r.route(batch, rate);
        std::printf("route(batch=%zu, rate=%.0f) = %-12s want %s\n",
                    batch, rate, got.c_str(), want.c_str());
        if (got != want) ok = false;
    };

    check(1, 0.0, "cpu");          // in-cache CPU wins at batch 1
    check(256, 0.0, "cuda-naive"); // GPU overtakes at large batch
    check(200, 0.0, "cuda-naive"); // nearest batch is 256

    std::printf(ok ? "OK: router picks the lowest-p999 backend\n"
                   : "FAIL: router mis-routed\n");
    return ok ? 0 : 1;
}
