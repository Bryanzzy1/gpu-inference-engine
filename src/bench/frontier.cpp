#include "frontier.hpp"

#include <cstdio>
#include <fstream>

std::vector<std::size_t> default_batch_axis() {
    return {1, 2, 4, 8, 16, 32, 64, 128, 256};
}

std::vector<double> default_rate_axis() {
    // 0 is the unpaced ceiling. The rest span light to heavy sustained load.
    return {0.0, 1000.0, 10000.0, 50000.0, 100000.0, 250000.0};
}

void write_frontier_csv(const std::string& path, const std::vector<FrontierCell>& cells) {
    const bool exists = std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out) return;
    if (!exists) {
        out << "backend,batch,rate_hz,count,p50_ns,p99_ns,p999_ns,max_ns,iqr_ns,mean_ns\n";
    }
    for (const FrontierCell& c : cells) {
        out << c.backend << ',' << c.batch << ',' << c.rate_hz << ','
            << c.stats.count << ',' << c.stats.p50 << ',' << c.stats.p99 << ','
            << c.stats.p999 << ',' << c.stats.max << ',' << c.stats.iqr << ','
            << c.stats.mean << '\n';
    }
}
