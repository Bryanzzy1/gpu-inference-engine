#ifndef PERSISTENT_MODEL_HPP
#define PERSISTENT_MODEL_HPP

#include <vector>

#include "model.hpp"

// Persistent-megakernel backend. The kernel is launched once and never exits: it
// spins reading a lock-free ring in pinned host memory, runs the forward pass on
// each input the host enqueues, writes the logit back, and advances the ring. No
// per-event kernel launch, which is the launch cost the naive backend pays every
// call and the reason the tail is lower here.
//
// Same request-response contract as GpuModel::forward: one feature vector in, one
// logit out. The difference is entirely in how the work reaches the GPU.
//
// Header stays free of CUDA syntax so plain .cpp drivers can include it.
class PersistentModel {
public:
    PersistentModel() = default;
    ~PersistentModel();
    PersistentModel(const PersistentModel&) = delete; // owns device + pinned memory
    PersistentModel& operator=(const PersistentModel&) = delete;

    // Upload weights and allocate the pinned ring. Does not launch the kernel.
    void load(const Model& cpu);

    // Launch / stop the persistent kernel. On Windows WDDM the kernel busy-spins and
    // the 2s TDR watchdog resets the GPU, so keep the running window short: start(),
    // run the measurement, stop(). Splitting these out is what makes it usable here.
    void start();
    void stop();

    // Enqueue one input, spin until the kernel returns its logit. Requires start().
    float forward(const std::vector<float>& in);

    int input_dim() const { return input_dim_; }
    int output_dim() const { return output_dim_; }

private:
    void release() noexcept;

    int input_dim_ = 0;
    int output_dim_ = 0;
    int num_layers_ = 0;
    int max_dim_ = 0;

    // Resident weights (device memory).
    float* d_mean_ = nullptr;
    float* d_std_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_biases_ = nullptr;
    void* d_layers_ = nullptr; // LayerDesc[num_layers_], opaque to this header

    // Ring in pinned (page-locked) host memory the GPU maps directly. Host pointers
    // for the CPU side; the kernel gets the matching device pointers internally.
    float* h_in_ring_ = nullptr;   // Capacity * input_dim floats
    float* h_out_ring_ = nullptr;  // Capacity * output_dim floats
    unsigned* h_ctrl_ = nullptr;   // [head, tail, stop]

    // Device-side addresses of the same pinned buffers (used by the kernel launch).
    float* d_in_ring_ = nullptr;
    float* d_out_ring_ = nullptr;
    unsigned* d_ctrl_ = nullptr;

    void* stream_ = nullptr;       // cudaStream_t the kernel runs on
    unsigned next_head_ = 0;       // host's running submit counter
    bool launched_ = false;
};

#endif
