#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <vector>
#include "trade.hpp"

// Parse a Binance aggTrades CSV into a vector<Trade>. Bad lines are skipped and
// counted in bad_lines. Throws std::runtime_error if the file cannot be opened.
std::vector<Trade> parse_agg_trades(const std::string& path, std::size_t& bad_lines);

// Same, without the bad-line count.
std::vector<Trade> parse_agg_trades(const std::string& path);

#endif
