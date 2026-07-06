// Reads a Binance aggTrades CSV and prints summary stats.
// Usage: parse_trades <path-to-aggTrades.csv>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "parser.hpp"
#include "trade.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <path-to-aggTrades.csv>\n";
        return 1;
    }

    std::size_t bad_lines = 0;
    std::vector<Trade> trades;
    try {
        trades = parse_agg_trades(argv[1], bad_lines);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    if (trades.empty()) {
        std::cerr << "error: no trades parsed\n";
        return 1;
    }

    // Min/max price and buyer-maker counts in one pass.
    int64_t first_ts = trades.front().time_us;
    int64_t last_ts = trades.back().time_us;
    double min_price = std::numeric_limits<double>::max();
    double max_price = std::numeric_limits<double>::lowest();
    std::size_t buyer_maker_true = 0;
    std::size_t buyer_maker_false = 0;

    for (const Trade& t : trades) {
        if (t.price < min_price) min_price = t.price;
        if (t.price > max_price) max_price = t.price;
        if (t.is_buyer_maker) ++buyer_maker_true; else ++buyer_maker_false;
    }

    // Show more digits (default is 6 sig figs).
    std::cout << std::setprecision(8);

    std::cout << "=== parse summary ===\n";
    std::cout << "rows parsed:      " << trades.size() << "\n";
    std::cout << "bad lines:        " << bad_lines << "\n";
    std::cout << "first timestamp:  " << first_ts << "  (us)\n";
    std::cout << "last timestamp:   " << last_ts  << "  (us)\n";
    std::cout << "min price:        " << min_price << "\n";
    std::cout << "max price:        " << max_price << "\n";
    std::cout << "is_buyer_maker True:  " << buyer_maker_true  << "\n";
    std::cout << "is_buyer_maker False: " << buyer_maker_false << "\n";

    // First and last row.
    const Trade& a = trades.front();
    const Trade& b = trades.back();
    std::cout << "\nfirst row: id=" << a.agg_id << " price=" << a.price
              << " qty=" << a.qty << " time_us=" << a.time_us
              << " is_buyer_maker=" << (a.is_buyer_maker ? "True" : "False") << "\n";
    std::cout << "last  row: id=" << b.agg_id << " price=" << b.price
              << " qty=" << b.qty << " time_us=" << b.time_us
              << " is_buyer_maker=" << (b.is_buyer_maker ? "True" : "False") << "\n";

    return 0;
}
