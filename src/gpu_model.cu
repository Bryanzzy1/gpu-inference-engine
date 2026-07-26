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

    input_dim_ = cpu.input_dim();
    output_dim_ = cpu.output_dim();
    num_layers_ = static_cast<int>(cpu.layers().size());

    // Flatten weights and biases, record per-layer offsets, find the widest layer.
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

    // A partial load that throws leaves some pointers set; the destructor's release()
    // frees them, so no explicit rollback is needed here.
    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };
    cuda_check(cudaMalloc(&d_mean_, fbytes(cpu.scaler_mean().size())), "malloc mean");
    cuda_check(cudaMalloc(&d_std_, fbytes(cpu.scaler_std().size())), "malloc std");
    cuda_check(cudaMalloc(&d_weights_, fbytes(weights.size())), "malloc weights");
    cuda_check(cudaMalloc(&d_biases_, fbytes(biases.size())), "malloc biases");
    cuda_check(cudaMalloc(&d_layers_, descs.size() * sizeof(LayerDesc)), "malloc layers");
    cuda_check(cudaMalloc(&d_in_, fbytes(input_dim_)), "malloc in");
    cuda_check(cudaMalloc(&d_out_, fbytes(output_dim_)), "malloc out");

    cuda_check(cudaMemcpy(d_mean_, cpu.scaler_mean().data(),
                          fbytes(cpu.scaler_mean().size()), cudaMemcpyHostToDevice), "cp mean");
    cuda_check(cudaMemcpy(d_std_, cpu.scaler_std().data(),
                          fbytes(cpu.scaler_std().size()), cudaMemcpyHostToDevice), "cp std");
    cuda_check(cudaMemcpy(d_weights_, weights.data(),
                          fbytes(weights.size()), cudaMemcpyHostToDevice), "cp weights");
    cuda_check(cudaMemcpy(d_biases_, biases.data(),
                          fbytes(biases.size()), cudaMemcpyHostToDevice), "cp biases");
    cuda_check(cudaMemcpy(d_layers_, descs.data(),
                          descs.size() * sizeof(LayerDesc), cudaMemcpyHostToDevice), "cp layers");
}

float GpuModel::forward(const std::vector<float>& in) {
    // Naive request-response: copy up, launch, copy down, every call. Copies come
    // straight from pageable host memory, which is the honest naive path.
    cudaMemcpy(d_in_, in.data(), static_cast<std::size_t>(input_dim_) * sizeof(float),
               cudaMemcpyHostToDevice);

    const int threads = max_dim_;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    forward_kernel<<<1, threads, shmem>>>(
        d_mean_, d_std_, d_weights_, d_biases_,
        static_cast<const LayerDesc*>(d_layers_), num_layers_, input_dim_, d_in_, d_out_);

    // The D2H copy waits for the kernel, so no explicit sync is needed.
    float out = 0.0f;
    cudaMemcpy(&out, d_out_, sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

void GpuModel::release() noexcept {
    cudaFree(d_mean_);
    cudaFree(d_std_);
    cudaFree(d_weights_);
    cudaFree(d_biases_);
    cudaFree(d_layers_);
    cudaFree(d_in_);
    cudaFree(d_out_);
    d_mean_ = d_std_ = d_weights_ = d_biases_ = d_in_ = d_out_ = nullptr;
    d_layers_ = nullptr;
}

GpuModel::~GpuModel() { release(); }
