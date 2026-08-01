#include "graph_model.hpp"

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

// Capture kernel: one block per row (grid of `batch` blocks). Batch 1 is grid of 1,
// so the same kernel serves the single-row graph and every batched graph.
__global__ void graph_forward_kernel_batch(const float* mean, const float* stdv,
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

void GraphModel::load(const Model& cpu) {
    release();
    w_.upload(cpu);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "stream create");
    stream_ = stream;

    capture_graph(1); // single-row graph, the batch-1 fast path
}

// (Re)capture the copy-launch-copy sequence for a given batch size. A captured graph
// binds fixed pointers and a fixed grid, so a new batch size needs new buffers and a
// fresh capture. The frontier sweeps one batch per outer loop, so this cost is paid
// once per batch size and amortized over every timed call at that size.
void GraphModel::capture_graph(std::size_t batch) {
    if (exec_) { cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_)); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(static_cast<cudaGraph_t>(graph_)); graph_ = nullptr; }
    cudaFree(d_in_);
    cudaFree(d_out_);
    d_in_ = d_out_ = nullptr;
    if (h_in_) { cudaFreeHost(h_in_); h_in_ = nullptr; }
    if (h_out_) { cudaFreeHost(h_out_); h_out_ = nullptr; }

    const auto fbytes = [](std::size_t n) { return n * sizeof(float); };
    const std::size_t in_n = batch * static_cast<std::size_t>(w_.input_dim);
    const std::size_t out_n = batch * static_cast<std::size_t>(w_.output_dim);
    cuda_check(cudaMalloc(&d_in_, fbytes(in_n)), "malloc in");
    cuda_check(cudaMalloc(&d_out_, fbytes(out_n)), "malloc out");
    cuda_check(cudaHostAlloc(&h_in_, fbytes(in_n), cudaHostAllocDefault), "hostalloc in");
    cuda_check(cudaHostAlloc(&h_out_, fbytes(out_n), cudaHostAllocDefault), "hostalloc out");

    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    const int threads = w_.max_dim;
    const std::size_t shmem = 2u * static_cast<std::size_t>(threads) * sizeof(float);
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    cudaMemcpyAsync(d_in_, h_in_, fbytes(in_n), cudaMemcpyHostToDevice, stream);
    graph_forward_kernel_batch<<<static_cast<unsigned>(batch), threads, shmem, stream>>>(
        w_.mean, w_.stdv, w_.weights, w_.biases,
        w_.layers, w_.num_layers, w_.input_dim, w_.output_dim, d_in_, d_out_);
    cudaMemcpyAsync(h_out_, d_out_, fbytes(out_n), cudaMemcpyDeviceToHost, stream);

    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    graph_ = graph;

    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "graph instantiate");
    exec_ = exec;
    cap_batch_ = batch;
}

float GraphModel::forward(const std::vector<float>& in) {
    if (cap_batch_ != 1) capture_graph(1); // fall back to the single-row graph
    for (int i = 0; i < w_.input_dim; ++i) h_in_[i] = in[i]; // stage the input
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_), static_cast<cudaStream_t>(stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    return h_out_[0];
}

void GraphModel::forward_batch(const std::vector<float>& in, std::size_t n,
                               std::vector<float>& out) {
    if (in.size() != n * static_cast<std::size_t>(w_.input_dim)) {
        throw std::runtime_error("forward_batch: input size != n * input_dim");
    }
    if (n != cap_batch_) capture_graph(n); // rebind the graph to this batch size
    out.resize(n * static_cast<std::size_t>(w_.output_dim));

    std::copy(in.begin(), in.end(), h_in_); // stage n rows into pinned input
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_), static_cast<cudaStream_t>(stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    std::copy(h_out_, h_out_ + out.size(), out.begin());
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
