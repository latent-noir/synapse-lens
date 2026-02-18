#pragma once
// ─── Loss Functions ─────────────────────────────────────────────────────────
// MSE loss and associated gradient for backpropagation.

#include "exports.hpp"
#include "matrix.hpp"

namespace nnviz::nn {

/// Compute Mean Squared Error loss.
/// @param predictions Matrix of shape (batch_size, output_size).
/// @param targets     Matrix of shape (batch_size, output_size).
/// @return Scalar MSE loss value.
NNVIZ_API double mse_loss(const Matrix& predictions, const Matrix& targets);

/// Compute the gradient of MSE loss w.r.t. predictions.
/// grad = 2 * (predictions - targets) / N
/// @return Matrix of same shape as predictions.
NNVIZ_API Matrix mse_gradient(const Matrix& predictions, const Matrix& targets);

/// Compute Cross-Entropy loss for classification.
/// @param predictions Matrix of shape (batch_size, num_classes) — after softmax.
/// @param targets     Matrix of shape (batch_size, num_classes) — one-hot encoded.
/// @return Scalar cross-entropy loss value.
NNVIZ_API double cross_entropy_loss(const Matrix& predictions, const Matrix& targets);

}  // namespace nnviz::nn
