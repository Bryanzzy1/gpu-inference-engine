#ifndef ROUTER_HPP
#define ROUTER_HPP

#include <cstddef>
#include <string>
#include <vector>

// Static frontier routing: given the measured frontier CSV, pick the backend with the
// lowest p999 for the current (batch, arrival rate). This is the lookup-table half of
// the "switch to whatever wins" idea; the closed-loop controller comes later and rides
// on the same table. No GPU needed: the router only reads the CSV the frontier produced.

// One measured point loaded from the frontier CSV.
struct FrontierPoint {
    std::string backend;
    std::size_t batch = 0;
    double rate_hz = 0.0;
    double p999_ns = 0.0;
};

class Router {
public:
    // Load a frontier CSV (backend,batch,rate_hz,count,p50_ns,p99_ns,p999_ns,...).
    // Throws std::runtime_error if the file cannot be read or has no usable rows.
    static Router load(const std::string& csv_path);

    // Backend with the lowest p999 at the grid cell nearest (batch, rate_hz). Nearest
    // by exact batch match then closest rate; falls back to closest batch. Returns the
    // winning backend name, or "" if the table is empty.
    std::string route(std::size_t batch, double rate_hz) const;

    // The winning p999 (ns) for the same cell, for reporting. 0 if empty.
    double best_p999_ns(std::size_t batch, double rate_hz) const;

    std::size_t size() const { return points_.size(); }

private:
    // Winning point at the cell nearest (batch, rate_hz), or nullptr if empty. Shared
    // by route() and best_p999_ns() so the selection rule lives in one place.
    const FrontierPoint* best_point(std::size_t batch, double rate_hz) const;

    std::vector<FrontierPoint> points_;
};

#endif
