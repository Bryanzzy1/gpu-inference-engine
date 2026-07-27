#include "gpu_weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlp_forward.cuh" // full LayerDesc

namespace {
void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cuda ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}
} // namespace

void GpuWeights::upload(const Model& cpu) {
    free();
    input_dim = cpu.input_dim();
    output_dim = cpu.output_dim();
    num_layers = static_cast<int>(cpu.layers().size());

    // Flatten weights and biases into one array each, record per-layer offsets and the
    // widest layer.
    std::vector<float> w;
    std::vector<float> b;
    std::vector<LayerDesc> descs;
    max_dim = cpu.input_dim();
    for (const Layer& L : cpu.layers()) {
        LayerDesc d;
        d.out_dim = L.out_dim;
        d.in_dim = L.in_dim;
        d.w_off = static_cast<int>(w.size());
        d.b_off = static_cast<int>(b.size());
        w.insert(w.end(), L.weight.begin(), L.weight.end());
        b.insert(b.end(), L.bias.begin(), L.bias.end());
        descs.push_back(d);
        max_dim = std::max(max_dim, L.out_dim);
    }

    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };
    cuda_check(cudaMalloc(&mean, fbytes(cpu.scaler_mean().size())), "malloc mean");
    cuda_check(cudaMalloc(&stdv, fbytes(cpu.scaler_std().size())), "malloc std");
    cuda_check(cudaMalloc(&weights, fbytes(w.size())), "malloc weights");
    cuda_check(cudaMalloc(&biases, fbytes(b.size())), "malloc biases");
    cuda_check(cudaMalloc(&layers, descs.size() * sizeof(LayerDesc)), "malloc layers");

    cuda_check(cudaMemcpy(mean, cpu.scaler_mean().data(),
                          fbytes(cpu.scaler_mean().size()), cudaMemcpyHostToDevice), "cp mean");
    cuda_check(cudaMemcpy(stdv, cpu.scaler_std().data(),
                          fbytes(cpu.scaler_std().size()), cudaMemcpyHostToDevice), "cp std");
    cuda_check(cudaMemcpy(weights, w.data(), fbytes(w.size()), cudaMemcpyHostToDevice), "cp weights");
    cuda_check(cudaMemcpy(biases, b.data(), fbytes(b.size()), cudaMemcpyHostToDevice), "cp biases");
    cuda_check(cudaMemcpy(layers, descs.data(),
                          descs.size() * sizeof(LayerDesc), cudaMemcpyHostToDevice), "cp layers");
}

void GpuWeights::free() noexcept {
    cudaFree(mean);
    cudaFree(stdv);
    cudaFree(weights);
    cudaFree(biases);
    cudaFree(layers);
    mean = stdv = weights = biases = nullptr;
    layers = nullptr;
}
