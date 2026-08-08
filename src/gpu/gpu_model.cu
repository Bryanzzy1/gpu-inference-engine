#include "gpu_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "mlp_forward.cuh"

namespace {

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cuda ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

// Naive request-response kernel: one launch per event, runs the shared forward pass.
__global__ void forward_kernel(const float* mean, const float* stdv,
                               const float* weights, const float* biases,
                               const LayerDesc* layers, int num_layers,
                               int input_dim, const float* in, float* out) {
    extern __shared__ float smem[];
    mlp_forward(mean, stdv, weights, biases, layers, num_layers, input_dim, in, out, smem);
}

// Batched kernel: one block per row. Block b reads row b of the packed input and
// writes logit b. Each block gets its own shared activations, so the rows are
// independent. One launch covers the whole batch, which is the fixed cost the batch
// amortizes over.
__global__ void forward_kernel_batch(const float* mean, const float* stdv,
                                     const float* weights, const float* biases,
                                     const LayerDesc* layers, int num_layers,
                                     int input_dim, int output_dim,
                                     const float* in, float* out) {
    extern __shared__ float smem[];
    const int row = static_cast<int>(blockIdx.x);
    mlp_forward(mean, stdv, weights, biases, layers, num_layers, input_dim,
                in + row * input_dim, out + row * output_dim, smem);
}

} // namespace

void GpuModel::load(const Model& cpu) {
    release(); // safe to call load twice
    w_.upload(cpu);

    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };
    cuda_check(cudaMalloc(&d_in_, fbytes(w_.input_dim)), "malloc in");
    cuda_check(cudaMalloc(&d_out_, fbytes(w_.output_dim)), "malloc out");
    cap_ = 1; // buffers hold one row until a batch call grows them
}

// Grow the per-call device buffers to hold n rows if they do not already.
void GpuModel::ensure_capacity(std::size_t n) {
    if (n <= cap_) return;
    cudaFree(d_in_);
    cudaFree(d_out_);
    const auto fbytes = [](std::size_t k) { return k * sizeof(float); };
    cuda_check(cudaMalloc(&d_in_, fbytes(n * w_.input_dim)), "malloc in batch");
    cuda_check(cudaMalloc(&d_out_, fbytes(n * w_.output_dim)), "malloc out batch");
    cap_ = n;
}

float GpuModel::forward(const std::vector<float>& in) {
    // Naive request-response: copy up, launch, copy down, every call. Copies come
    // straight from pageable host memory, which is the honest naive path.
    cudaMemcpy(d_in_, in.data(), static_cast<std::size_t>(w_.input_dim) * sizeof(float),
               cudaMemcpyHostToDevice);

    const int threads = w_.max_dim;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    forward_kernel<<<1, threads, shmem>>>(
        w_.mean, w_.stdv, w_.weights, w_.biases,
        w_.layers, w_.num_layers, w_.input_dim, d_in_, d_out_);

    // The D2H copy waits for the kernel, so no explicit sync is needed.
    float out = 0.0f;
    cudaMemcpy(&out, d_out_, sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

void GpuModel::forward_batch(const std::vector<float>& in, std::size_t n,
                             std::vector<float>& out) {
    if (in.size() != n * static_cast<std::size_t>(w_.input_dim)) {
        throw std::runtime_error("forward_batch: input size != n * input_dim");
    }
    out.resize(n * static_cast<std::size_t>(w_.output_dim));
    ensure_capacity(n);

    cudaMemcpy(d_in_, in.data(), n * static_cast<std::size_t>(w_.input_dim) * sizeof(float),
               cudaMemcpyHostToDevice);

    const int threads = w_.max_dim;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    forward_kernel_batch<<<static_cast<unsigned>(n), threads, shmem>>>(
        w_.mean, w_.stdv, w_.weights, w_.biases,
        w_.layers, w_.num_layers, w_.input_dim, w_.output_dim, d_in_, d_out_);

    cudaMemcpy(out.data(), d_out_, n * static_cast<std::size_t>(w_.output_dim) * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void GpuModel::init_timing() {
    if (tstream_) return;
    cudaStream_t s = nullptr;
    cuda_check(cudaStreamCreate(&s), "timing stream");
    tstream_ = s;
    for (int i = 0; i < 4; ++i) {
        cudaEvent_t e = nullptr;
        cuda_check(cudaEventCreate(&e), "timing event");
        ev_[i] = e;
    }
    cuda_check(cudaHostAlloc(&th_in_, static_cast<std::size_t>(w_.input_dim) * sizeof(float),
                             cudaHostAllocDefault), "pin in");
    cuda_check(cudaHostAlloc(&th_out_, static_cast<std::size_t>(w_.output_dim) * sizeof(float),
                             cudaHostAllocDefault), "pin out");
}

float GpuModel::forward_timed(const std::vector<float>& in, StageSample& out) {
    init_timing();
    cudaStream_t s = static_cast<cudaStream_t>(tstream_);
    cudaEvent_t e0 = static_cast<cudaEvent_t>(ev_[0]);
    cudaEvent_t e1 = static_cast<cudaEvent_t>(ev_[1]);
    cudaEvent_t e2 = static_cast<cudaEvent_t>(ev_[2]);
    cudaEvent_t e3 = static_cast<cudaEvent_t>(ev_[3]);

    for (int i = 0; i < w_.input_dim; ++i) th_in_[i] = in[i];

    // Device timeline: bracket H2D, kernel, D2H with events on one stream.
    cudaEventRecord(e0, s);
    cudaMemcpyAsync(d_in_, th_in_, static_cast<std::size_t>(w_.input_dim) * sizeof(float),
                    cudaMemcpyHostToDevice, s);
    cudaEventRecord(e1, s);

    const int threads = w_.max_dim;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    // Host-side launch issue cost: the CPU time to enqueue the kernel.
    const auto t0 = std::chrono::steady_clock::now();
    forward_kernel<<<1, threads, shmem, s>>>(
        w_.mean, w_.stdv, w_.weights, w_.biases,
        w_.layers, w_.num_layers, w_.input_dim, d_in_, d_out_);
    const auto t1 = std::chrono::steady_clock::now();

    cudaEventRecord(e2, s);
    cudaMemcpyAsync(th_out_, d_out_, static_cast<std::size_t>(w_.output_dim) * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaEventRecord(e3, s);
    cudaEventSynchronize(e3); // wait for the whole chain, then read the device times

    float h2d_ms = 0.0f, comp_ms = 0.0f, d2h_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, e0, e1);
    cudaEventElapsedTime(&comp_ms, e1, e2);
    cudaEventElapsedTime(&d2h_ms, e2, e3);

    out.ns[static_cast<int>(Stage::H2D)] = static_cast<double>(h2d_ms) * 1e6;
    out.ns[static_cast<int>(Stage::Launch)] =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    out.ns[static_cast<int>(Stage::Compute)] = static_cast<double>(comp_ms) * 1e6;
    out.ns[static_cast<int>(Stage::D2H)] = static_cast<double>(d2h_ms) * 1e6;
    return th_out_[0];
}

void GpuModel::release() noexcept {
    w_.free();
    cudaFree(d_in_);
    cudaFree(d_out_);
    d_in_ = d_out_ = nullptr;
    for (int i = 0; i < 4; ++i) {
        if (ev_[i]) { cudaEventDestroy(static_cast<cudaEvent_t>(ev_[i])); ev_[i] = nullptr; }
    }
    if (th_in_) { cudaFreeHost(th_in_); th_in_ = nullptr; }
    if (th_out_) { cudaFreeHost(th_out_); th_out_ = nullptr; }
    if (tstream_) { cudaStreamDestroy(static_cast<cudaStream_t>(tstream_)); tstream_ = nullptr; }
}

GpuModel::~GpuModel() { release(); }
