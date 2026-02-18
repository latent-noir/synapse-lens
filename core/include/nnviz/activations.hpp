#pragma once
// ─── Activation Functions ───────────────────────────────────────────────────
// Vectorized activation functions for fast inference / analysis.

#include "exports.hpp"
#include <vector>

namespace nnviz::activations {

/// Element-wise ReLU: max(0, x).
NNVIZ_API std::vector<double> relu(const std::vector<double>& x);

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
NNVIZ_API std::vector<double> sigmoid(const std::vector<double>& x);

/// Softmax over the entire vector (numerically stable).
NNVIZ_API std::vector<double> softmax(const std::vector<double>& x);

/// Element-wise GELU approximation.
NNVIZ_API std::vector<double> gelu(const std::vector<double>& x);

/// Element-wise Leaky ReLU with configurable negative slope.
NNVIZ_API std::vector<double> leaky_relu(const std::vector<double>& x,
                                          double negative_slope = 0.01);

}  // namespace nnviz::activations
