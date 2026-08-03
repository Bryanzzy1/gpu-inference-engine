#include "stage_timing.hpp"

#include <cstdio>
#include <fstream>

StageBreakdown StageBreakdown::from(std::vector<StageSample>& samples) {
    StageBreakdown b;
    std::vector<double> col;
    col.reserve(samples.size());

    for (int s = 0; s < 4; ++s) {
        col.clear();
        for (const StageSample& x : samples) col.push_back(x.ns[s]);
        b.stage[s] = Stats::from(col);
    }

    col.clear();
    for (const StageSample& x : samples) {
        col.push_back(x.ns[0] + x.ns[1] + x.ns[2] + x.ns[3]);
    }
    b.total = Stats::from(col);
    return b;
}

void StageBreakdown::print(const char* label) const {
    std::printf("%-14s   p50_us   p99_us  p999_us\n", label);
    for (int s = 0; s < 4; ++s) {
        std::printf("  %-10s %8.2f %8.2f %8.2f\n", stage_name(static_cast<Stage>(s)),
                    stage[s].p50 / 1000.0, stage[s].p99 / 1000.0, stage[s].p999 / 1000.0);
    }
    std::printf("  %-10s %8.2f %8.2f %8.2f\n", "total",
                total.p50 / 1000.0, total.p99 / 1000.0, total.p999 / 1000.0);
}

void write_autopsy_csv(const std::string& path, const char* label,
                       const StageBreakdown& b) {
    const bool exists = std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out) return;
    if (!exists) out << "label,stage,count,p50_ns,p99_ns,p999_ns,max_ns\n";

    auto row = [&](const char* stage, const Stats& s) {
        out << label << ',' << stage << ',' << s.count << ',' << s.p50 << ','
            << s.p99 << ',' << s.p999 << ',' << s.max << '\n';
    };
    for (int s = 0; s < 4; ++s) row(stage_name(static_cast<Stage>(s)), b.stage[s]);
    row("total", b.total);
}
