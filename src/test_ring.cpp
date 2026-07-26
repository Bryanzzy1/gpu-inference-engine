// Self-check for SpscRing: one producer thread, one consumer thread, a small ring
// that wraps many times. Proves the memory ordering is right, if it were wrong the
// consumer would see a torn or out-of-order value and the checks below would fire.
// Uses explicit checks, not assert, so it still fails under NDEBUG (Release builds).
#include <cstdint>
#include <iostream>
#include <thread>

#include "ring.hpp"

int main() {
    constexpr std::size_t N = 5'000'000; // far larger than the ring, so it wraps a lot
    SpscRing<std::uint64_t, 1024> ring;

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < N; ++i) {
            while (!ring.push(i)) { /* spin until a slot frees */ }
        }
    });

    std::uint64_t expected = 0; // next value we must see, in FIFO order
    std::uint64_t sum = 0;
    std::size_t got = 0;
    while (got < N) {
        std::uint64_t v;
        if (!ring.pop(v)) continue; // spin until a value arrives
        if (v != expected) {
            std::cerr << "FAIL: out of order, want " << expected << " got " << v << "\n";
            producer.join();
            return 1;
        }
        expected = v + 1;
        sum += v;
        ++got;
    }

    producer.join();

    const std::uint64_t want = N * (N - 1) / 2;
    if (sum != want) {
        std::cerr << "FAIL: sum " << sum << " != " << want << "\n";
        return 1;
    }
    std::cout << "OK: " << got << " items, sum " << sum << " matches " << want << "\n";
    return 0;
}
