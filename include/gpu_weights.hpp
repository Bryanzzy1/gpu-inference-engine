#ifndef GPU_WEIGHTS_HPP
#define GPU_WEIGHTS_HPP

#include "model.hpp"

struct LayerDesc; // full definition in mlp_forward.cuh, only a pointer is needed here

// Device-resident MLP weights, uploaded once and shared by every GPU backend (naive,
// persistent, graph). Owns the device buffers. Header stays CUDA-free so plain .cpp
// drivers can include the backends that hold one.
struct GpuWeights {
    float* mean = nullptr;
    float* stdv = nullptr;
    float* weights = nullptr;
    float* biases = nullptr;
    LayerDesc* layers = nullptr;

    int num_layers = 0;
    int input_dim = 0;
    int output_dim = 0;
    int max_dim = 0; // widest layer, sets the block size

    void upload(const Model& cpu); // cudaMalloc + copy, safe to call twice
    void free() noexcept;          // cudaFree everything and null
};

#endif
