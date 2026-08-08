#ifndef GPU_MODEL_HPP
#define GPU_MODEL_HPP

#include <vector>

#include "gpu_weights.hpp"
#include "model.hpp"
#include "stage_timing.hpp"

// GPU forward pass for the tiny MLP, naive request-response style: per call it copies
// the input up, launches one forward kernel, copies the logit down. Weights are
// uploaded once by load() and stay resident on the device. Same math as
// Model::forward, so results match within float tolerance.
//
// This is the naive baseline backend: deliberately pays the launch + PCIe cost on
// every event, which is the cost the frontier measures at batch size one.
//
// Header stays free of CUDA syntax so plain .cpp drivers can include it; the kernel
// and all device calls live in gpu_model.cu.
class GpuModel {
public:
    GpuModel() = default;
    ~GpuModel();
    GpuModel(const GpuModel&) = delete; // owns device memory
    GpuModel& operator=(const GpuModel&) = delete;

    // Upload a loaded CPU model's weights and scaler to the device.
    void load(const Model& cpu);

    // Naive request-response forward on one raw feature vector. Returns the logit.
    float forward(const std::vector<float>& in);

    // Batched forward: n rows packed row-major in, n logits out. One H2D copy of
    // n*input_dim floats, one kernel of n blocks (one row per block), one D2H copy of
    // n logits. Grows d_in_/d_out_ to hold the batch. Matches Model::forward_batch.
    void forward_batch(const std::vector<float>& in, std::size_t n,
                       std::vector<float>& out);

    // Single forward that fills `out` with this call's H2D/launch/compute/D2H
    // durations in nanoseconds. cudaEvents on a dedicated stream bracket the H2D copy,
    // the kernel (compute), and the D2H copy; the launch stage is the host time to issue
    // the kernel, measured with steady_clock. This is the jitter-autopsy instrumentation,
    // see stage_timing.hpp.
    float forward_timed(const std::vector<float>& in, StageSample& out);

    int input_dim() const { return w_.input_dim; }
    int output_dim() const { return w_.output_dim; }

private:
    void release() noexcept;
    void ensure_capacity(std::size_t n); // grow d_in_/d_out_ to hold n rows
    void init_timing();          // lazily create the stream/events/pinned staging

    GpuWeights w_;               // resident weights, shared upload path
    float* d_in_ = nullptr;      // per-call input buffer, sized for the max batch
    float* d_out_ = nullptr;     // per-call output buffer, sized for the max batch
    std::size_t cap_ = 0;        // rows the current buffers can hold

    // forward_timed resources (opaque so the header stays CUDA-free), created on first use.
    void* tstream_ = nullptr;    // cudaStream_t the timed forward runs on
    void* ev_[4] = {nullptr, nullptr, nullptr, nullptr}; // cudaEvent_t stage brackets
    float* th_in_ = nullptr;     // pinned staging input for clean event timing
    float* th_out_ = nullptr;    // pinned staging output
};

#endif
