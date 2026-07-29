#ifndef FRONTIER_HPP
#define FRONTIER_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "latency.hpp"

// One measured point of the 2D frontier: a backend timed at a given batch size and
// arrival rate. stats are per-inference tail latency in nanoseconds.
struct FrontierCell {
    std::string backend;
    std::size_t batch = 1;
    double rate_hz = 0.0; // 0 means as fast as possible, no pacing
    Stats stats;
};

// Default sweep axes. Batch grows geometrically, rate spans idle to bursty.
// rate 0.0 is the unpaced ceiling used as the baseline column.
std::vector<std::size_t> default_batch_axis();
std::vector<double> default_rate_axis();

// Append one row per cell to a CSV, writing the header first if the file is new.
// Columns: backend,batch,rate_hz,count,p50_ns,p99_ns,p999_ns,max_ns,iqr_ns,mean_ns.
void write_frontier_csv(const std::string& path, const std::vector<FrontierCell>& cells);

#endif
