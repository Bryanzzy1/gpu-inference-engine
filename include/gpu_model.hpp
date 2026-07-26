#ifndef GPU_MODEL_HPP
#define GPU_MODEL_HPP

#include <vector>

#include "gpu_weights.hpp"
#include "model.hpp"

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

    int input_dim() const { return w_.input_dim; }
    int output_dim() const { return w_.output_dim; }

private:
    void release() noexcept;

    GpuWeights w_;               // resident weights, shared upload path
    float* d_in_ = nullptr;      // per-call input buffer
    float* d_out_ = nullptr;     // per-call output buffer
};

#endif
