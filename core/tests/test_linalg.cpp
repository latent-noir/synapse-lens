// ─── Linear Algebra Unit Tests ───────────────────────────────────────────────
#include <gtest/gtest.h>
#include "nnviz/linalg.hpp"

using namespace nnviz::linalg;

TEST(LinalgTest, MatmulIdentity) {
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
    auto result = matmul(I, A);
    ASSERT_TRUE(result.isApprox(A, 1e-10));
}

TEST(LinalgTest, MatmulDimensionMismatch) {
    Eigen::MatrixXd A(2, 3);
    Eigen::MatrixXd B(4, 2);
    EXPECT_THROW(matmul(A, B), std::invalid_argument);
}

TEST(LinalgTest, Transpose) {
    Eigen::MatrixXd A(2, 3);
    A << 1, 2, 3, 4, 5, 6;
    auto result = transpose(A);
    ASSERT_EQ(result.rows(), 3);
    ASSERT_EQ(result.cols(), 2);
    ASSERT_DOUBLE_EQ(result(0, 0), 1.0);
    ASSERT_DOUBLE_EQ(result(2, 1), 6.0);
}

TEST(LinalgTest, FrobeniusNorm) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3);
    ASSERT_NEAR(frobenius_norm(A), std::sqrt(3.0), 1e-10);
}

TEST(LinalgTest, Flatten) {
    Eigen::MatrixXd A(2, 2);
    A << 1, 2, 3, 4;
    auto flat = flatten(A);
    ASSERT_EQ(flat.size(), 4);
}

TEST(LinalgTest, SingularValues) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3);
    auto sv = singular_values(A);
    ASSERT_EQ(sv.size(), 3);
    for (const auto& s : sv) {
        ASSERT_NEAR(s, 1.0, 1e-10);
    }
}
