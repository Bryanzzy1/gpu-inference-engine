#ifndef BATCH_HPP
#define BATCH_HPP

// The batched inference contract shared by every backend.
//
// A batch is N feature vectors processed by one inference call. Inputs are packed
// row-major: row r occupies in[r*input_dim .. r*input_dim + input_dim). Outputs are
// N logits, out[r] for row r. This is the layout Model::forward_batch produces and
// the layout the GPU batched kernels must match.
//
// Why batching exists here: at batch 1 the GPU pays launch + PCIe per event and the
// in-cache CPU wins. As N grows, one kernel over N rows amortizes that fixed cost, so
// the frontier sweeps N to find where each backend takes over. On the CPU a "batch"
// is just N sequential forward() calls, so the CPU batch axis is a control, not a
// result; the axis is only meaningful on the GPU.
//
// Every backend exposes:
//   int  input_dim() const;
//   int  output_dim() const;
//   void forward_batch(const std::vector<float>& in, std::size_t n,
//                      std::vector<float>& out);   // n rows in, n logits out
// The single-row forward(row) stays as the batch==1 fast path.

#include <cstddef>

// Batch sizes the frontier sweeps by default. Geometric so each step roughly doubles
// the work, which is where amortization crossovers show up.
inline constexpr std::size_t kFrontierBatches[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};

#endif
