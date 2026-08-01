#include "persistent_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>

#include "mlp_forward.cuh"

namespace {

constexpr int CAPACITY = 512;  // ring slots, power of two; > max frontier batch (256)
constexpr int HEAD = 0;        // ctrl[HEAD]: host writes, device reads
constexpr int TAIL = 1;        // ctrl[TAIL]: device writes, host reads
constexpr int STOP = 2;        // ctrl[STOP]: host writes, device reads

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cuda ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

// The megakernel: launched once, loops forever until the host sets ctrl[STOP].
// Thread 0 spins on the ring's head, all threads run the forward pass on each new
// input, thread 0 publishes tail. One resident block, no per-event launch.
__global__ void persistent_kernel(const float* mean, const float* stdv,
                                  const float* weights, const float* biases,
                                  const LayerDesc* layers, int num_layers,
                                  int input_dim, int output_dim,
                                  const float* in_ring, float* out_ring,
                                  volatile unsigned* ctrl) {
    extern __shared__ float smem[];
    const int t = static_cast<int>(threadIdx.x);
    const int mask = CAPACITY - 1;
    unsigned local = 0;          // next slot to consume, mirrored in every thread
    __shared__ int s_stop;

    for (;;) {
        if (t == 0) {
            for (;;) {                        // busy-wait for work or a stop signal
                if (ctrl[STOP]) { s_stop = 1; break; }
                if (ctrl[HEAD] != local) { s_stop = 0; break; } // head moved: work ready
            }
        }
        __syncthreads();                      // broadcast s_stop to the whole block
        if (s_stop) return;

        const int slot = static_cast<int>(local) & mask;
        mlp_forward(mean, stdv, weights, biases, layers, num_layers, input_dim,
                    in_ring + slot * input_dim, out_ring + slot * output_dim, smem);
        __threadfence_system();               // output visible to host before tail moves
        __syncthreads();
        if (t == 0) ctrl[TAIL] = local + 1;   // publish result
        ++local;                              // keep every thread's mirror in step
        __syncthreads();
    }
}

} // namespace

void PersistentModel::load(const Model& cpu) {
    release();
    w_.upload(cpu);

    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };

    // Pinned, mapped ring: host RAM the GPU reads/writes directly over PCIe.
    cuda_check(cudaHostAlloc(&h_in_ring_, fbytes(CAPACITY * w_.input_dim),
                             cudaHostAllocMapped), "hostalloc in");
    cuda_check(cudaHostAlloc(&h_out_ring_, fbytes(CAPACITY * w_.output_dim),
                             cudaHostAllocMapped), "hostalloc out");
    cuda_check(cudaHostAlloc(&h_ctrl_, 3 * sizeof(unsigned),
                             cudaHostAllocMapped), "hostalloc ctrl");
    h_ctrl_[HEAD] = h_ctrl_[TAIL] = h_ctrl_[STOP] = 0;
    next_head_ = 0;

    // Device-side addresses of the same pinned buffers, kept for the kernel launch.
    cuda_check(cudaHostGetDevicePointer(&d_in_ring_, h_in_ring_, 0), "devptr in");
    cuda_check(cudaHostGetDevicePointer(&d_out_ring_, h_out_ring_, 0), "devptr out");
    cuda_check(cudaHostGetDevicePointer(&d_ctrl_, h_ctrl_, 0), "devptr ctrl");
}

void PersistentModel::start() {
    if (launched_) return;
    h_ctrl_[HEAD] = h_ctrl_[TAIL] = h_ctrl_[STOP] = 0; // fresh run
    next_head_ = 0;

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "stream create");
    stream_ = stream;

    const int threads = w_.max_dim;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    persistent_kernel<<<1, threads, shmem, stream>>>(
        w_.mean, w_.stdv, w_.weights, w_.biases,
        w_.layers, w_.num_layers,
        w_.input_dim, w_.output_dim, d_in_ring_, d_out_ring_,
        reinterpret_cast<volatile unsigned*>(d_ctrl_));
    cuda_check(cudaGetLastError(), "kernel launch");
    // Windows WDDM batches launches in a command buffer and does not push them to the
    // GPU until a sync point. Query the stream to flush, so the kernel actually starts
    // spinning before the host enqueues its first input.
    cudaStreamQuery(stream);
    launched_ = true;
}

void PersistentModel::stop() {
    if (!launched_) return;
    h_ctrl_[STOP] = 1;
    std::atomic_thread_fence(std::memory_order_release);
    if (stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
    launched_ = false;
}

float PersistentModel::forward(const std::vector<float>& in) {
    volatile unsigned* ctrl = h_ctrl_;
    const unsigned h = next_head_;
    const int slot = static_cast<int>(h) & (CAPACITY - 1);

    while (h - ctrl[TAIL] == CAPACITY) { /* ring full, wait for the kernel */ }

    float* dst = h_in_ring_ + slot * w_.input_dim;
    for (int i = 0; i < w_.input_dim; ++i) dst[i] = in[i];
    std::atomic_thread_fence(std::memory_order_release); // input written before head moves
    ctrl[HEAD] = h + 1;
    next_head_ = h + 1;

    while (ctrl[TAIL] != h + 1) { /* spin until the kernel finishes this one */ }
    std::atomic_thread_fence(std::memory_order_acquire); // read output after tail seen
    return h_out_ring_[slot * w_.output_dim];
}

void PersistentModel::forward_batch(const std::vector<float>& in, std::size_t n,
                                    std::vector<float>& out) {
    if (in.size() != n * static_cast<std::size_t>(w_.input_dim)) {
        throw std::runtime_error("forward_batch: input size != n * input_dim");
    }
    out.resize(n * static_cast<std::size_t>(w_.output_dim));
    volatile unsigned* ctrl = h_ctrl_;
    const int mask = CAPACITY - 1;

    // Stream the batch through the ring in chunks no larger than the ring. The resident
    // kernel drains one slot per iteration and advances the tail, so publishing a chunk
    // of head positions at once lets it chew through them back to back with no launch.
    std::size_t done = 0;
    while (done < n) {
        const std::size_t chunk = std::min(n - done, static_cast<std::size_t>(CAPACITY));
        const unsigned base = next_head_;

        for (std::size_t j = 0; j < chunk; ++j) {
            const int slot = static_cast<int>(base + j) & mask;
            float* dst = h_in_ring_ + static_cast<std::size_t>(slot) * w_.input_dim;
            const float* src = &in[(done + j) * static_cast<std::size_t>(w_.input_dim)];
            for (int i = 0; i < w_.input_dim; ++i) dst[i] = src[i];
        }
        std::atomic_thread_fence(std::memory_order_release); // inputs written before head moves
        ctrl[HEAD] = base + static_cast<unsigned>(chunk);
        next_head_ = base + static_cast<unsigned>(chunk);

        while (ctrl[TAIL] != base + static_cast<unsigned>(chunk)) { /* wait for the chunk */ }
        std::atomic_thread_fence(std::memory_order_acquire); // read outputs after tail seen

        for (std::size_t j = 0; j < chunk; ++j) {
            const int slot = static_cast<int>(base + j) & mask;
            out[(done + j) * static_cast<std::size_t>(w_.output_dim)] =
                h_out_ring_[static_cast<std::size_t>(slot) * w_.output_dim];
        }
        done += chunk;
    }
}

void PersistentModel::release() noexcept {
    stop(); // signal and join the kernel if still running
    if (h_in_ring_) { cudaFreeHost(h_in_ring_); h_in_ring_ = nullptr; }
    if (h_out_ring_) { cudaFreeHost(h_out_ring_); h_out_ring_ = nullptr; }
    if (h_ctrl_) { cudaFreeHost(h_ctrl_); h_ctrl_ = nullptr; }
    d_in_ring_ = d_out_ring_ = nullptr; // alias the freed pinned buffers
    d_ctrl_ = nullptr;
    w_.free();
    launched_ = false;
}

PersistentModel::~PersistentModel() { release(); }
