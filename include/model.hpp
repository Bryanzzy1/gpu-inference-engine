#ifndef MODEL_HPP
#define MODEL_HPP

#include <cstddef>
#include <string>
#include <vector>

// One fully-connected layer. weight is row-major [out_dim, in_dim]; y = W x + b.
struct Layer {
    int out_dim;
    int in_dim;
    std::vector<float> weight; // out_dim * in_dim, row-major
    std::vector<float> bias;   // out_dim
};

// MLP loaded from a .meta manifest + .bin weights. Runs the same forward pass as
// the Python reference: standardize inputs, then (Linear, ReLU) per hidden layer
// and a final Linear. Output is the raw logit.
class Model {
public:
    // Load from "<stem>.meta" (which names its .bin). Throws std::runtime_error
    // on any format or size mismatch.
    static Model load(const std::string& meta_path);

    // Forward pass on one raw (unscaled) feature vector. Returns the logit.
    // in.size() must equal input_dim().
    float forward(const std::vector<float>& in) const;

    int input_dim() const { return input_dim_; }
    int output_dim() const { return output_dim_; }

private:
    int input_dim_ = 0;
    int output_dim_ = 0;
    std::vector<float> scaler_mean_; // input_dim
    std::vector<float> scaler_std_;  // input_dim
    std::vector<Layer> layers_;
};

#endif
