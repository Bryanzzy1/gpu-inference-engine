#include "router.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// Split a CSV line into fields. Simple: no quoted-comma handling, the frontier CSV
// has none.
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) out.push_back(field);
    return out;
}

}  // namespace

Router Router::load(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f) throw std::runtime_error("router: cannot open " + csv_path);

    Router r;
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (header) { header = false; continue; }  // skip the column header row
        std::vector<std::string> c = split(line);
        if (c.size() < 7) continue;  // backend,batch,rate,count,p50,p99,p999,...
        FrontierPoint p;
        p.backend = c[0];
        p.batch = std::strtoull(c[1].c_str(), nullptr, 10);
        p.rate_hz = std::atof(c[2].c_str());
        p.p999_ns = std::atof(c[6].c_str());
        r.points_.push_back(p);
    }
    if (r.points_.empty()) throw std::runtime_error("router: no rows in " + csv_path);
    return r;
}

const FrontierPoint* Router::best_point(std::size_t batch, double rate_hz) const {
    // Batch mismatch is weighted far above any rate distance, so an exact-batch cell
    // always beats a nearer-rate cell at the wrong batch. Among the nearest cell(s),
    // the lowest p999 wins.
    const double kBatchWeight = 1e12;
    const FrontierPoint* best = nullptr;
    double best_dist = 0.0;
    for (const FrontierPoint& p : points_) {
        const double batch_pen =
            (p.batch == batch) ? 0.0
                               : kBatchWeight * std::fabs(static_cast<double>(p.batch) -
                                                          static_cast<double>(batch));
        const double dist = batch_pen + std::fabs(p.rate_hz - rate_hz);
        if (!best || dist < best_dist || (dist == best_dist && p.p999_ns < best->p999_ns)) {
            best = &p;
            best_dist = dist;
        }
    }
    return best;
}

std::string Router::route(std::size_t batch, double rate_hz) const {
    const FrontierPoint* p = best_point(batch, rate_hz);
    return p ? p->backend : "";
}

double Router::best_p999_ns(std::size_t batch, double rate_hz) const {
    const FrontierPoint* p = best_point(batch, rate_hz);
    return p ? p->p999_ns : 0.0;
}
