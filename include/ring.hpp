#ifndef RING_HPP
#define RING_HPP

#include <atomic>
#include <cstddef>

// Single-producer single-consumer lock-free ring buffer. One thread pushes, one
// thread pops, no mutex. Capacity is a compile-time power of two so wrapping is a
// bitmask instead of a modulo. head_ and tail_ are free-running counters, never
// wrapped; the buffer index is (counter & MASK). Empty when head_ == tail_, full
// when head_ - tail_ == Capacity.
//
// head_ and tail_ sit on separate cache lines so the producer and consumer never
// write the same line (no false sharing on the hot path).
template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(Capacity >= 2, "capacity must be at least 2");
    static_assert((Capacity & (Capacity - 1)) == 0, "capacity must be a power of two");

public:
    // Producer side. Returns false if the ring is full (caller retries or drops).
    bool push(const T& value) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail == Capacity) return false; // full
        buf_[head & MASK] = value;
        head_.store(head + 1, std::memory_order_release); // publish the slot
        return true;
    }

    // Consumer side. Returns false if the ring is empty.
    bool pop(T& out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        if (head == tail) return false; // empty
        out = buf_[tail & MASK];
        tail_.store(tail + 1, std::memory_order_release); // free the slot
        return true;
    }

private:
    static constexpr std::size_t MASK = Capacity - 1;

    T buf_[Capacity];
    alignas(64) std::atomic<std::size_t> head_{0}; // written by producer only
    alignas(64) std::atomic<std::size_t> tail_{0}; // written by consumer only
};

#endif
