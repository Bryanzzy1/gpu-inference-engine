#ifndef STAGE_TIMING_HPP
#define STAGE_TIMING_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "latency.hpp"

// Per-stage latency decomposition for one GPU inference: where the time actually goes.
// The naive path is H2D copy -> kernel launch -> compute -> D2H copy. Timing the whole
// forward() gives one number; the jitter autopsy splits it so we can see which stage
// owns the tail and which stage's jitter dominates. Stages are timed with CUDA events
// on the device side (see GpuModel::forward_timed), collected here on the host.
enum class Stage { H2D, Launch, Compute, D2H };

inline const char* stage_name(Stage s) {
    switch (s) {
        case Stage::H2D: return "h2d";
        case Stage::Launch: return "launch";
        case Stage::Compute: return "compute";
        case Stage::D2H: return "d2h";
    }
    return "?";
}

// One inference's four stage durations in nanoseconds, in H2D/Launch/Compute/D2H order.
struct StageSample {
    double ns[4] = {0.0, 0.0, 0.0, 0.0};
};

// Full-distribution tail per stage, computed from all collected samples.
struct StageBreakdown {
    Stats stage[4]; // indexed by Stage
    Stats total;    // sum of the four stages per sample

    // Builds a breakdown from raw per-inference samples. Sorts internally.
    static StageBreakdown from(std::vector<StageSample>& samples);

    // Prints a per-stage p50/p99/p999 table plus the total row.
    void print(const char* label) const;
};

// Append one row per stage (plus a total row) to a CSV, header first if new.
// Columns: label,stage,count,p50_ns,p99_ns,p999_ns,max_ns.
void write_autopsy_csv(const std::string& path, const char* label,
                       const StageBreakdown& b);

#endif
