#pragma once
#include <cstdint>

// One aggregated trade from Binance spot aggTrades.
struct Trade {
    int64_t agg_id;          // aggTrade id
    double  price;
    double  qty;
    int64_t time_us;         // timestamp, microseconds
    bool    is_buyer_maker;  // true = seller was the aggressor
};
