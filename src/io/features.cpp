#include "features.hpp"
#include <cmath>

FeatureEngine::FeatureEngine(int window, int64_t intensity_window_us)
    : window_(window),
      intensity_window_us_(intensity_window_us),
      ret_(window, 0.0),
      sqty_(window, 0.0),
      qty_(window, 0.0) {
    // Size the timestamp ring generously; intensity counts vary with the window.
    ts_cap_ = window * 8;
    if (ts_cap_ < 1024) ts_cap_ = 1024;
    ts_.assign(ts_cap_, 0);
}

bool FeatureEngine::update(const Trade& t, Features& out) {
    // Signed volume for this trade.
    const double sign = t.is_buyer_maker ? -1.0 : 1.0;
    const double sq = sign * t.qty;

    // Update signed-volume ring (overwrite oldest, adjust running sums).
    sqty_sum_ += sq - sqty_[imb_head_];
    qty_sum_  += t.qty - qty_[imb_head_];
    sqty_[imb_head_] = sq;
    qty_[imb_head_]  = t.qty;
    imb_head_ = (imb_head_ + 1) % window_;

    // Update log-return ring (only once we have a previous price).
    if (have_prev_) {
        const double r = std::log(t.price / prev_price_);
        const double old = ret_[ret_head_];
        ret_sum_   += r - old;
        ret_sumsq_ += r * r - old * old;
        ret_[ret_head_] = r;
        ret_head_ = (ret_head_ + 1) % window_;
        if (ret_count_ < window_) ++ret_count_;
    }
    prev_price_ = t.price;
    have_prev_  = true;

    // Update timestamp ring, then drop entries older than the time window.
    ts_[ts_head_] = t.time_us;
    ts_head_ = (ts_head_ + 1) % ts_cap_;
    if (ts_size_ < ts_cap_) ++ts_size_;
    const int64_t cutoff = t.time_us - intensity_window_us_;
    int count = 0;
    for (int k = 0; k < ts_size_; ++k) {
        int idx = ts_head_ - 1 - k;
        if (idx < 0) idx += ts_cap_;
        if (ts_[idx] >= cutoff) ++count; else break;
    }

    // Emit only when a full log-return window exists.
    if (ret_count_ < window_) return false;

    const double mean = ret_sum_ / window_;
    double var = ret_sumsq_ / window_ - mean * mean;
    if (var < 0.0) var = 0.0;  // clamp tiny negatives from float error

    out.time_us    = t.time_us;
    out.mid        = t.price;
    out.volatility = std::sqrt(var);
    out.imbalance  = (qty_sum_ > 0.0) ? (sqty_sum_ / qty_sum_) : 0.0;
    out.intensity  = static_cast<double>(count);
    return true;
}
