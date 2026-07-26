#ifndef MLP_FORWARD_CUH
#define MLP_FORWARD_CUH

// Shared device-side forward pass for the tiny MLP, used by both the naive
// request-response kernel and the persistent megakernel so their math cannot drift.

// A layer's shape plus its offsets into the flat weight/bias arrays.
struct LayerDesc {
    int out_dim;
    int in_dim;
    int w_off;
    int b_off;
};

// One-block forward pass. blockDim.x must be >= every layer dim. Standardize the
// input, then (Linear, ReLU) per hidden layer and a final Linear. smem is two
// activation buffers of blockDim.x floats each, swapped per layer. Every thread in
// the block must call this (it uses __syncthreads).
__device__ inline void mlp_forward(const float* mean, const float* stdv,
                                   const float* weights, const float* biases,
                                   const LayerDesc* layers, int num_layers,
                                   int input_dim, const float* in, float* out,
                                   float* smem) {
    float* cur = smem;                // activations into this layer
    float* nxt = smem + blockDim.x;   // activations out of this layer
    const int t = static_cast<int>(threadIdx.x);

    if (t < input_dim) cur[t] = (in[t] - mean[t]) / stdv[t];
    __syncthreads();

    for (int li = 0; li < num_layers; ++li) {
        const LayerDesc L = layers[li];
        if (t < L.out_dim) {
            float acc = biases[L.b_off + t];
            const float* wrow = weights + L.w_off + t * L.in_dim;
            for (int i = 0; i < L.in_dim; ++i) acc += wrow[i] * cur[i];
            if (li + 1 < num_layers && acc < 0.0f) acc = 0.0f; // ReLU on hidden only
            nxt[t] = acc;
        }
        __syncthreads(); // nxt fully written before it becomes cur
        float* tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    if (t == 0) out[0] = cur[0];
}

#endif
