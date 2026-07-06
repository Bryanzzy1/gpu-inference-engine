#pragma once
#include <cstdint>
#include <vector>
#include "trade.hpp"

// One row of computed features for a single trade.
struct Features {
    int64_t time_us;
    double  mid;         // mid-price proxy = trade price
    double  volatility;  // std dev of last N log returns
    double  imbalance;   // signed-volume imbalance over last N trades, [-1, 1]
    double  intensity;   // trade count in the last time window
};

// Streaming feature computation over a rolling window of trades.
// Feed trades in order with update(); it returns true once a full window exists.
class FeatureEngine {
public:
    FeatureEngine(int window, int64_t intensity_window_us);

    // Push one trade. If a full window is ready, fill `out` and return true.
    bool update(const Trade& t, Features& out);

private:
    int     window_;
    int64_t intensity_window_us_;

    // Log-return ring (for volatility).
    std::vector<double> ret_;
    int    ret_head_  = 0;
    int    ret_count_ = 0;
    double ret_sum_   = 0.0;
    double ret_sumsq_ = 0.0;
    double prev_price_ = 0.0;
    bool   have_prev_  = false;

    // Signed-volume ring (for imbalance).
    std::vector<double> sqty_;
    std::vector<double> qty_;
    int    imb_head_ = 0;
    double sqty_sum_ = 0.0;
    double qty_sum_  = 0.0;

    // Timestamp ring (for intensity).
    std::vector<int64_t> ts_;
    int ts_cap_  = 0;
    int ts_head_ = 0;
    int ts_size_ = 0;
};
