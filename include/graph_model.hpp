#ifndef GRAPH_MODEL_HPP
#define GRAPH_MODEL_HPP

#include <vector>

#include "gpu_weights.hpp"
#include "model.hpp"

// CUDA Graphs backend. Same three operations as the naive path (copy input up, launch
// the kernel, copy the logit down), but captured once into a graph and replayed with a
// single cudaGraphLaunch per call. The driver work of scheduling three separate
// operations is paid once at capture time, not on every event, so the per-call CPU
// launch overhead drops. The copies bind to fixed pinned staging buffers, so each call
// just overwrites the staging input, replays, and reads the staging output.
//
// Header stays free of CUDA syntax so plain .cpp drivers can include it.
class GraphModel {
public:
    GraphModel() = default;
    ~GraphModel();
    GraphModel(const GraphModel&) = delete; // owns device + graph resources
    GraphModel& operator=(const GraphModel&) = delete;

    // Upload weights, allocate buffers, capture and instantiate the graph.
    void load(const Model& cpu);

    // Overwrite the staged input, replay the graph, return the logit.
    float forward(const std::vector<float>& in);

    int input_dim() const { return w_.input_dim; }
    int output_dim() const { return w_.output_dim; }

private:
    void release() noexcept;

    GpuWeights w_;             // resident weights, shared upload path
    float* d_in_ = nullptr;    // device input, copied into by the graph
    float* d_out_ = nullptr;   // device output, copied out of by the graph
    float* h_in_ = nullptr;    // pinned staging input (stable source for the H2D copy)
    float* h_out_ = nullptr;   // pinned staging output (stable dest for the D2H copy)

    void* stream_ = nullptr;   // cudaStream_t used for capture and replay
    void* graph_ = nullptr;    // cudaGraph_t, the recorded sequence
    void* exec_ = nullptr;     // cudaGraphExec_t, the instantiated replayable graph
};

#endif
