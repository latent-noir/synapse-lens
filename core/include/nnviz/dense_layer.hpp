#pragma once
// ─── Dense (Fully-Connected) Layer ──────────────────────────────────────────
// A single dense layer with weights, bias, and configurable activation.

#include "exports.hpp"
#include "matrix.hpp"
#include <string>
#include <functional>

namespace nnviz::nn {

/// Supported activation function types
enum class Activation {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
};

class NNVIZ_API DenseLayer {
public:
    /// Construct a dense layer.
    /// @param input_size  Number of input features.
    /// @param output_size Number of output features (neurons).
    /// @param activation  Activation function to apply after the linear transform.
    DenseLayer(std::size_t input_size, std::size_t output_size,
               Activation activation = Activation::ReLU);

    /// Forward pass: output = activation(input * W^T + b)
    /// @param input  Matrix of shape (batch_size, input_size).
    /// @return Matrix of shape (batch_size, output_size).
    Matrix forward(const Matrix& input);

    /// Get the most recent pre-activation values (before activation fn).
    const Matrix& get_pre_activation() const { return z_; }

    /// Get the most recent post-activation output.
    const Matrix& get_output() const { return a_; }

    /// Get the most recent input (cached for backprop).
    const Matrix& get_input() const { return input_cache_; }

    // ── Weight Access ───────────────────────────────────────────────────
    Matrix& weights() { return weights_; }
    const Matrix& weights() const { return weights_; }
    Matrix& bias() { return bias_; }
    const Matrix& bias() const { return bias_; }

    std::size_t input_size() const { return input_size_; }
    std::size_t output_size() const { return output_size_; }
    Activation activation_type() const { return activation_; }

private:
    std::size_t input_size_;
    std::size_t output_size_;
    Activation activation_;

    Matrix weights_;   // Shape: (output_size, input_size)
    Matrix bias_;      // Shape: (1, output_size)

    // Cached values for backpropagation
    Matrix input_cache_;   // Last input
    Matrix z_;             // Pre-activation
    Matrix a_;             // Post-activation

    // ── Activation Functions ────────────────────────────────────────────
    Matrix apply_activation(const Matrix& z) const;
};

/// Convert activation enum to string name
NNVIZ_API std::string activation_to_string(Activation act);

}  // namespace nnviz::nn
