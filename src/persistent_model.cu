#include "persistent_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>

#include "mlp_forward.cuh"

namespace {

constexpr int CAPACITY = 256;  // ring slots, power of two
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

    input_dim_ = cpu.input_dim();
    output_dim_ = cpu.output_dim();
    num_layers_ = static_cast<int>(cpu.layers().size());

    // Flatten weights/biases and record per-layer offsets.
    // ponytail: duplicated from GpuModel::load; each backend owns its own device
    // memory, factor into a shared uploader only if a third backend needs it.
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<LayerDesc> descs;
    max_dim_ = cpu.input_dim();
    for (const Layer& L : cpu.layers()) {
        LayerDesc d;
        d.out_dim = L.out_dim;
        d.in_dim = L.in_dim;
        d.w_off = static_cast<int>(weights.size());
        d.b_off = static_cast<int>(biases.size());
        weights.insert(weights.end(), L.weight.begin(), L.weight.end());
        biases.insert(biases.end(), L.bias.begin(), L.bias.end());
        descs.push_back(d);
        max_dim_ = std::max(max_dim_, L.out_dim);
    }

    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };
    LayerDesc* d_layers = nullptr;
    cuda_check(cudaMalloc(&d_mean_, fbytes(cpu.scaler_mean().size())), "malloc mean");
    cuda_check(cudaMalloc(&d_std_, fbytes(cpu.scaler_std().size())), "malloc std");
    cuda_check(cudaMalloc(&d_weights_, fbytes(weights.size())), "malloc weights");
    cuda_check(cudaMalloc(&d_biases_, fbytes(biases.size())), "malloc biases");
    cuda_check(cudaMalloc(&d_layers, descs.size() * sizeof(LayerDesc)), "malloc layers");
    d_layers_ = d_layers;

    cuda_check(cudaMemcpy(d_mean_, cpu.scaler_mean().data(),
                          fbytes(cpu.scaler_mean().size()), cudaMemcpyHostToDevice), "cp mean");
    cuda_check(cudaMemcpy(d_std_, cpu.scaler_std().data(),
                          fbytes(cpu.scaler_std().size()), cudaMemcpyHostToDevice), "cp std");
    cuda_check(cudaMemcpy(d_weights_, weights.data(),
                          fbytes(weights.size()), cudaMemcpyHostToDevice), "cp weights");
    cuda_check(cudaMemcpy(d_biases_, biases.data(),
                          fbytes(biases.size()), cudaMemcpyHostToDevice), "cp biases");
    cuda_check(cudaMemcpy(d_layers, descs.data(),
                          descs.size() * sizeof(LayerDesc), cudaMemcpyHostToDevice), "cp layers");

    // Pinned, mapped ring: host RAM the GPU reads/writes directly over PCIe.
    cuda_check(cudaHostAlloc(&h_in_ring_, fbytes(CAPACITY * input_dim_),
                             cudaHostAllocMapped), "hostalloc in");
    cuda_check(cudaHostAlloc(&h_out_ring_, fbytes(CAPACITY * output_dim_),
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

    const int threads = max_dim_;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    persistent_kernel<<<1, threads, shmem, stream>>>(
        d_mean_, d_std_, d_weights_, d_biases_,
        static_cast<const LayerDesc*>(d_layers_), num_layers_,
        input_dim_, output_dim_, d_in_ring_, d_out_ring_,
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

    float* dst = h_in_ring_ + slot * input_dim_;
    for (int i = 0; i < input_dim_; ++i) dst[i] = in[i];
    std::atomic_thread_fence(std::memory_order_release); // input written before head moves
    ctrl[HEAD] = h + 1;
    next_head_ = h + 1;

    while (ctrl[TAIL] != h + 1) { /* spin until the kernel finishes this one */ }
    std::atomic_thread_fence(std::memory_order_acquire); // read output after tail seen
    return h_out_ring_[slot * output_dim_];
}

void PersistentModel::release() noexcept {
    stop(); // signal and join the kernel if still running
    if (h_in_ring_) { cudaFreeHost(h_in_ring_); h_in_ring_ = nullptr; }
    if (h_out_ring_) { cudaFreeHost(h_out_ring_); h_out_ring_ = nullptr; }
    if (h_ctrl_) { cudaFreeHost(h_ctrl_); h_ctrl_ = nullptr; }
    d_in_ring_ = d_out_ring_ = nullptr; // alias the freed pinned buffers
    d_ctrl_ = nullptr;
    cudaFree(d_mean_);
    cudaFree(d_std_);
    cudaFree(d_weights_);
    cudaFree(d_biases_);
    cudaFree(d_layers_);
    d_mean_ = d_std_ = d_weights_ = d_biases_ = nullptr;
    d_layers_ = nullptr;
    launched_ = false;
}

PersistentModel::~PersistentModel() { release(); }
