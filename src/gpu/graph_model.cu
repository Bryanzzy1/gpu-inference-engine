#include "graph_model.hpp"

#include <cuda_runtime.h>

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

// Same kernel as the naive path: one launch, the shared forward pass.
__global__ void graph_forward_kernel(const float* mean, const float* stdv,
                                     const float* weights, const float* biases,
                                     const LayerDesc* layers, int num_layers,
                                     int input_dim, const float* in, float* out) {
    extern __shared__ float smem[];
    mlp_forward(mean, stdv, weights, biases, layers, num_layers, input_dim, in, out, smem);
}

} // namespace

void GraphModel::load(const Model& cpu) {
    release();
    w_.upload(cpu);

    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };
    cuda_check(cudaMalloc(&d_in_, fbytes(w_.input_dim)), "malloc in");
    cuda_check(cudaMalloc(&d_out_, fbytes(w_.output_dim)), "malloc out");
    cuda_check(cudaHostAlloc(&h_in_, fbytes(w_.input_dim), cudaHostAllocDefault), "hostalloc in");
    cuda_check(cudaHostAlloc(&h_out_, fbytes(w_.output_dim), cudaHostAllocDefault), "hostalloc out");

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "stream create");
    stream_ = stream;

    // Record the copy-launch-copy sequence once. Everything binds to fixed pointers
    // (d_in_, d_out_, h_in_, h_out_, the resident weights), so the replay is valid for
    // every future call as long as those buffers stay put.
    const int threads = w_.max_dim;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    cudaMemcpyAsync(d_in_, h_in_, fbytes(w_.input_dim), cudaMemcpyHostToDevice, stream);
    graph_forward_kernel<<<1, threads, shmem, stream>>>(
        w_.mean, w_.stdv, w_.weights, w_.biases,
        w_.layers, w_.num_layers, w_.input_dim, d_in_, d_out_);
    cudaMemcpyAsync(h_out_, d_out_, fbytes(w_.output_dim), cudaMemcpyDeviceToHost, stream);

    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    graph_ = graph;

    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "graph instantiate");
    exec_ = exec;
}

float GraphModel::forward(const std::vector<float>& in) {
    for (int i = 0; i < w_.input_dim; ++i) h_in_[i] = in[i]; // stage the input
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_), static_cast<cudaStream_t>(stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    return h_out_[0];
}

void GraphModel::release() noexcept {
    if (exec_) { cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_)); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(static_cast<cudaGraph_t>(graph_)); graph_ = nullptr; }
    if (stream_) { cudaStreamDestroy(static_cast<cudaStream_t>(stream_)); stream_ = nullptr; }
    if (h_in_) { cudaFreeHost(h_in_); h_in_ = nullptr; }
    if (h_out_) { cudaFreeHost(h_out_); h_out_ = nullptr; }
    cudaFree(d_in_);
    cudaFree(d_out_);
    d_in_ = d_out_ = nullptr;
    w_.free();
}

GraphModel::~GraphModel() { release(); }
