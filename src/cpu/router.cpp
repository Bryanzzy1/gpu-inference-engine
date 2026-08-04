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

std::string Router::route(std::size_t batch, double rate_hz) const {
    std::string best;
    double best_p999 = 0.0;
    double best_dist = 0.0;
    bool have = false;

    for (const FrontierPoint& p : points_) {
        // Distance to the requested cell: exact batch match preferred, then closest
        // rate. Batch mismatch is weighted heavily so a same-batch cell always wins.
        const double batch_pen =
            (p.batch == batch) ? 0.0 : 1e12 * std::fabs(static_cast<double>(p.batch) -
                                                        static_cast<double>(batch));
        const double dist = batch_pen + std::fabs(p.rate_hz - rate_hz);

        // Among the nearest cell(s), keep the lowest p999.
        if (!have || dist < best_dist || (dist == best_dist && p.p999_ns < best_p999)) {
            best = p.backend;
            best_p999 = p.p999_ns;
            best_dist = dist;
            have = true;
        }
    }
    return best;
}

double Router::best_p999_ns(std::size_t batch, double rate_hz) const {
    std::string best;
    double best_p999 = 0.0;
    double best_dist = 0.0;
    bool have = false;
    for (const FrontierPoint& p : points_) {
        const double batch_pen =
            (p.batch == batch) ? 0.0 : 1e12 * std::fabs(static_cast<double>(p.batch) -
                                                        static_cast<double>(batch));
        const double dist = batch_pen + std::fabs(p.rate_hz - rate_hz);
        if (!have || dist < best_dist || (dist == best_dist && p.p999_ns < best_p999)) {
            best_p999 = p.p999_ns;
            best_dist = dist;
            have = true;
        }
    }
    return best_p999;
}
