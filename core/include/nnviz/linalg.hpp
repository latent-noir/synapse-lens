#pragma once
// ─── Linear Algebra Operations ──────────────────────────────────────────────
// High-performance matrix operations backed by Eigen.

#include "exports.hpp"
#include <Eigen/Dense>
#include <vector>

namespace nnviz::linalg {

/// Multiply two matrices: C = A * B.
NNVIZ_API Eigen::MatrixXd matmul(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

/// Transpose a matrix.
NNVIZ_API Eigen::MatrixXd transpose(const Eigen::MatrixXd& m);

/// Compute the Frobenius norm of a matrix.
NNVIZ_API double frobenius_norm(const Eigen::MatrixXd& m);

/// Flatten a matrix to a 1-D vector (row-major order).
NNVIZ_API std::vector<double> flatten(const Eigen::MatrixXd& m);

/// Compute the singular values of a matrix (thin SVD).
NNVIZ_API std::vector<double> singular_values(const Eigen::MatrixXd& m);

}  // namespace nnviz::linalg
