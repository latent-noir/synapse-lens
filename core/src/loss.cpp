// ─── Loss Function Implementations ──────────────────────────────────────────
#include "nnviz/loss.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace nnviz::nn {

double mse_loss(const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("mse_loss: shape mismatch");
    }
    Matrix diff = predictions - targets;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < diff.size(); ++i) {
        sum_sq += diff.data()[i] * diff.data()[i];
    }
    return sum_sq / static_cast<double>(diff.size());
}

Matrix mse_gradient(const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("mse_gradient: shape mismatch");
    }
    double factor = 2.0 / static_cast<double>(predictions.size());
    return (predictions - targets) * factor;
}

double cross_entropy_loss(const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("cross_entropy_loss: shape mismatch");
    }
    constexpr double eps = 1e-12;
    double loss = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        double p = std::clamp(predictions.data()[i], eps, 1.0 - eps);
        loss -= targets.data()[i] * std::log(p);
    }
    return loss / static_cast<double>(predictions.rows());
}

}  // namespace nnviz::nn
