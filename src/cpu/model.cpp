#include "model.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// Directory part of a path (everything up to the last '/' or '\'), or "" if none.
std::string dir_of(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    return path.substr(0, slash + 1);
}

// Read the whole file into a string.
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}

Model Model::load(const std::string& meta_path) {
    Model m;

    // Parse the flat-text manifest token by token, skipping '#' comment lines.
    std::ifstream mf(meta_path);
    if (!mf) throw std::runtime_error("cannot open manifest: " + meta_path);

    std::string bin_name;
    int num_layers = -1;
    std::vector<std::pair<int, int>> layer_dims; // (out, in) per layer

    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string key;
        ls >> key;

        if (key == "input_dim") {
            ls >> m.input_dim_;
        } else if (key == "output_dim") {
            ls >> m.output_dim_;
        } else if (key == "scaler_mean") {
            float v;
            while (ls >> v) m.scaler_mean_.push_back(v);
        } else if (key == "scaler_std") {
            float v;
            while (ls >> v) m.scaler_std_.push_back(v);
        } else if (key == "num_layers") {
            ls >> num_layers;
        } else if (key == "layer") {
            int out_dim = 0, in_dim = 0;
            ls >> out_dim >> in_dim;
            layer_dims.emplace_back(out_dim, in_dim);
        } else if (key == "bin") {
            ls >> bin_name;
        }
    }

    if (m.input_dim_ <= 0) throw std::runtime_error("manifest: bad input_dim");
    if (num_layers < 0 || static_cast<int>(layer_dims.size()) != num_layers) {
        throw std::runtime_error("manifest: layer count mismatch");
    }
    if (static_cast<int>(m.scaler_mean_.size()) != m.input_dim_ ||
        static_cast<int>(m.scaler_std_.size()) != m.input_dim_) {
        throw std::runtime_error("manifest: scaler size != input_dim");
    }
    if (bin_name.empty()) throw std::runtime_error("manifest: no bin path");

    // Layer shapes must chain: layer[0].in == input_dim, layer[k].in == layer[k-1].out.
    for (std::size_t i = 0; i < layer_dims.size(); ++i) {
        const int expect_in =
            (i == 0) ? m.input_dim_ : layer_dims[i - 1].first;
        if (layer_dims[i].second != expect_in) {
            throw std::runtime_error("manifest: layer input dim does not chain");
        }
    }
    if (layer_dims.back().first != m.output_dim_) {
        throw std::runtime_error("manifest: last layer out != output_dim");
    }

    // Load raw float32 weights, then biases, per layer, in manifest order.
    const std::string bin_path = dir_of(meta_path) + bin_name;
    const std::string blob = read_file(bin_path);
    if (blob.size() % sizeof(float) != 0) {
        throw std::runtime_error("bin: size not a multiple of float32");
    }
    const float* raw = reinterpret_cast<const float*>(blob.data());
    const std::size_t total = blob.size() / sizeof(float);

    std::size_t off = 0;
    for (const auto& d : layer_dims) {
        Layer layer;
        layer.out_dim = d.first;
        layer.in_dim = d.second;

        const std::size_t w_count =
            static_cast<std::size_t>(layer.out_dim) * layer.in_dim;
        const std::size_t b_count = static_cast<std::size_t>(layer.out_dim);

        if (off + w_count + b_count > total) {
            throw std::runtime_error("bin: ran out of data for a layer");
        }
        layer.weight.assign(raw + off, raw + off + w_count);
        off += w_count;
        layer.bias.assign(raw + off, raw + off + b_count);
        off += b_count;

        m.layers_.push_back(std::move(layer));
    }
    if (off != total) throw std::runtime_error("bin: trailing bytes unused");

    return m;
}

float Model::forward(const std::vector<float>& in) const {
    if (static_cast<int>(in.size()) != input_dim_) {
        throw std::runtime_error("forward: input size != input_dim");
    }

    // Standardize with the exported mean/std (same as Python before the net).
    std::vector<float> a(input_dim_);
    for (int i = 0; i < input_dim_; ++i) {
        a[i] = (in[i] - scaler_mean_[i]) / scaler_std_[i];
    }

    // (Linear, ReLU) per hidden layer, then a final Linear (no ReLU).
    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const Layer& layer = layers_[li];
        std::vector<float> out(layer.out_dim);

        // y_o = b_o + sum_i W[o, i] * a[i]. weight is row-major [out, in].
        for (int o = 0; o < layer.out_dim; ++o) {
            float acc = layer.bias[o];
            const float* wrow = &layer.weight[static_cast<std::size_t>(o) * layer.in_dim];
            for (int i = 0; i < layer.in_dim; ++i) {
                acc += wrow[i] * a[i];
            }
            out[o] = acc;
        }

        const bool last = (li + 1 == layers_.size());
        if (!last) {
            for (float& v : out) {
                if (v < 0.0f) v = 0.0f; // ReLU on hidden layers only
            }
        }
        a = std::move(out);
    }

    return a[0]; // single logit
}
