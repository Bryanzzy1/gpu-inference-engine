// Reads an aggTrades CSV, computes features + label, writes a feature CSV.
// Usage: build_features <in.csv> <out.csv> [window] [horizon] [intensity_us]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "features.hpp"
#include "parser.hpp"
#include "trade.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <in.csv> <out.csv> [window] [horizon] [intensity_us]\n";
        return 1;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    const int     window       = (argc > 3) ? std::atoi(argv[3]) : 50;
    const int     horizon      = (argc > 4) ? std::atoi(argv[4]) : 20;
    const int64_t intensity_us = (argc > 5) ? std::atoll(argv[5]) : 1000000;

    std::size_t bad_lines = 0;
    std::vector<Trade> trades;
    try {
        trades = parse_agg_trades(in_path, bad_lines);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    if (static_cast<int>(trades.size()) <= horizon) {
        std::cerr << "error: not enough trades\n";
        return 1;
    }

    FILE* out = std::fopen(out_path.c_str(), "wb");
    if (!out) {
        std::cerr << "error: cannot open output: " << out_path << "\n";
        return 1;
    }
    std::fprintf(out, "time_us,mid,volatility,imbalance,intensity,label\n");

    FeatureEngine engine(window, intensity_us);
    Features f;
    std::size_t rows = 0;
    std::size_t up = 0;

    // Feature at trade i uses trades <= i; label needs price[i + horizon].
    for (std::size_t i = 0; i < trades.size(); ++i) {
        if (!engine.update(trades[i], f)) continue;
        if (i + static_cast<std::size_t>(horizon) >= trades.size()) break;

        const int label = (trades[i + horizon].price > trades[i].price) ? 1 : 0;
        up += static_cast<std::size_t>(label);
        ++rows;

        std::fprintf(out, "%lld,%.8f,%.10f,%.10f,%.1f,%d\n",
                     static_cast<long long>(f.time_us),
                     f.mid, f.volatility, f.imbalance, f.intensity, label);
    }

    std::fclose(out);

    std::cout << "wrote " << rows << " rows to " << out_path << "\n";
    std::cout << "bad lines: " << bad_lines << "\n";
    if (rows > 0) {
        const double pct = 100.0 * static_cast<double>(up) / static_cast<double>(rows);
        std::cout << "label balance: " << up << " up ("
                  << pct << "%), " << (rows - up) << " down/flat\n";
    }
    return 0;
}
