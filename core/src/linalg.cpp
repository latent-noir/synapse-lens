// ─── Linear Algebra Implementations ─────────────────────────────────────────
#include "nnviz/linalg.hpp"
#include <Eigen/SVD>
#include <stdexcept>

namespace nnviz::linalg {

Eigen::MatrixXd matmul(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument(
            "matmul: incompatible dimensions (" +
            std::to_string(a.rows()) + "x" + std::to_string(a.cols()) +
            ") * (" + std::to_string(b.rows()) + "x" + std::to_string(b.cols()) + ")");
    }
    return a * b;
}

Eigen::MatrixXd transpose(const Eigen::MatrixXd& m) {
    return m.transpose();
}

double frobenius_norm(const Eigen::MatrixXd& m) {
    return m.norm();
}

std::vector<double> flatten(const Eigen::MatrixXd& m) {
    // Row-major flattening
    Eigen::MatrixXd row_major = m;
    std::vector<double> result(row_major.size());
    Eigen::Map<Eigen::VectorXd>(result.data(), row_major.size()) =
        Eigen::Map<const Eigen::VectorXd>(row_major.data(), row_major.size());
    return result;
}

std::vector<double> singular_values(const Eigen::MatrixXd& m) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m);
    const auto& sv = svd.singularValues();
    return {sv.data(), sv.data() + sv.size()};
}

}  // namespace nnviz::linalg
