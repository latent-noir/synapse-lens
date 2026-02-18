// ─── Dense Layer Implementation ─────────────────────────────────────────────
#include "nnviz/dense_layer.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace nnviz::nn {

DenseLayer::DenseLayer(std::size_t input_size, std::size_t output_size, Activation activation)
    : input_size_(input_size),
      output_size_(output_size),
      activation_(activation),
      weights_(Matrix::xavier(output_size, input_size)),
      bias_(Matrix::zeros(1, output_size)) {}

Matrix DenseLayer::forward(const Matrix& input) {
    if (input.cols() != input_size_) {
        throw std::invalid_argument(
            "DenseLayer::forward: input has " + std::to_string(input.cols()) +
            " features, expected " + std::to_string(input_size_));
    }

    input_cache_ = input;

    // z = input * W^T + b
    z_ = input * weights_.transpose();
    z_ = z_.add_row_vector(bias_);

    // a = activation(z)
    a_ = apply_activation(z_);

    return a_;
}

Matrix DenseLayer::apply_activation(const Matrix& z) const {
    switch (activation_) {
        case Activation::None:
            return z;

        case Activation::ReLU:
            return z.apply([](double v) { return std::max(0.0, v); });

        case Activation::Sigmoid:
            return z.apply([](double v) { return 1.0 / (1.0 + std::exp(-v)); });

        case Activation::Tanh:
            return z.apply([](double v) { return std::tanh(v); });

        case Activation::Softmax: {
            // Row-wise softmax (numerically stable)
            Matrix result(z.rows(), z.cols());
            for (std::size_t i = 0; i < z.rows(); ++i) {
                // Find row max
                double row_max = z(i, 0);
                for (std::size_t j = 1; j < z.cols(); ++j) {
                    row_max = std::max(row_max, z(i, j));
                }
                // Compute exp(z - max) and sum
                double sum = 0.0;
                for (std::size_t j = 0; j < z.cols(); ++j) {
                    result(i, j) = std::exp(z(i, j) - row_max);
                    sum += result(i, j);
                }
                // Normalize
                for (std::size_t j = 0; j < z.cols(); ++j) {
                    result(i, j) /= sum;
                }
            }
            return result;
        }

        default:
            return z;
    }
}

std::string activation_to_string(Activation act) {
    switch (act) {
        case Activation::None:    return "none";
        case Activation::ReLU:    return "relu";
        case Activation::Sigmoid: return "sigmoid";
        case Activation::Tanh:    return "tanh";
        case Activation::Softmax: return "softmax";
        default: return "unknown";
    }
}

}  // namespace nnviz::nn
