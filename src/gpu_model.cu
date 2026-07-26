#include "gpu_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

} // namespace

void GpuModel::load(const Model& cpu) {
    release(); // safe to call load twice
    w_.upload(cpu);

    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };
    cuda_check(cudaMalloc(&d_in_, fbytes(w_.input_dim)), "malloc in");
    cuda_check(cudaMalloc(&d_out_, fbytes(w_.output_dim)), "malloc out");
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

void GpuModel::release() noexcept {
    w_.free();
    cudaFree(d_in_);
    cudaFree(d_out_);
    d_in_ = d_out_ = nullptr;
}

GpuModel::~GpuModel() { release(); }
