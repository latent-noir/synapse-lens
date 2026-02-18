// ─── Neural Network Engine Unit Tests ────────────────────────────────────────
#include <gtest/gtest.h>
#include "nnviz/matrix.hpp"
#include "nnviz/dense_layer.hpp"
#include "nnviz/network.hpp"
#include "nnviz/loss.hpp"
#include <cmath>

using namespace nnviz::nn;

// ═══════════════════════════════════════════════════════════════════════════
//  Matrix Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(MatrixTest, Construction) {
    Matrix m(3, 4, 1.0);
    ASSERT_EQ(m.rows(), 3);
    ASSERT_EQ(m.cols(), 4);
    ASSERT_DOUBLE_EQ(m(0, 0), 1.0);
    ASSERT_DOUBLE_EQ(m(2, 3), 1.0);
}

TEST(MatrixTest, ZerosAndOnes) {
    auto z = Matrix::zeros(2, 3);
    auto o = Matrix::ones(2, 3);
    ASSERT_DOUBLE_EQ(z(0, 0), 0.0);
    ASSERT_DOUBLE_EQ(o(1, 2), 1.0);
}

TEST(MatrixTest, Addition) {
    Matrix a(2, 2, 1.0);
    Matrix b(2, 2, 2.0);
    auto c = a + b;
    ASSERT_DOUBLE_EQ(c(0, 0), 3.0);
}

TEST(MatrixTest, Subtraction) {
    Matrix a(2, 2, 5.0);
    Matrix b(2, 2, 2.0);
    auto c = a - b;
    ASSERT_DOUBLE_EQ(c(0, 0), 3.0);
}

TEST(MatrixTest, MatMul) {
    Matrix a(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix b(3, 2, {7, 8, 9, 10, 11, 12});
    auto c = a * b;
    ASSERT_EQ(c.rows(), 2);
    ASSERT_EQ(c.cols(), 2);
    ASSERT_DOUBLE_EQ(c(0, 0), 58.0);
    ASSERT_DOUBLE_EQ(c(0, 1), 64.0);
    ASSERT_DOUBLE_EQ(c(1, 0), 139.0);
    ASSERT_DOUBLE_EQ(c(1, 1), 154.0);
}

TEST(MatrixTest, MatMulDimensionMismatch) {
    Matrix a(2, 3);
    Matrix b(4, 2);
    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(MatrixTest, ScalarMultiply) {
    Matrix a(2, 2, 3.0);
    auto c = a * 2.0;
    ASSERT_DOUBLE_EQ(c(0, 0), 6.0);
    auto d = 2.0 * a;
    ASSERT_DOUBLE_EQ(d(0, 0), 6.0);
}

TEST(MatrixTest, Transpose) {
    Matrix a(2, 3, {1, 2, 3, 4, 5, 6});
    auto t = a.transpose();
    ASSERT_EQ(t.rows(), 3);
    ASSERT_EQ(t.cols(), 2);
    ASSERT_DOUBLE_EQ(t(0, 0), 1.0);
    ASSERT_DOUBLE_EQ(t(0, 1), 4.0);
    ASSERT_DOUBLE_EQ(t(2, 1), 6.0);
}

TEST(MatrixTest, Hadamard) {
    Matrix a(2, 2, {1, 2, 3, 4});
    Matrix b(2, 2, {5, 6, 7, 8});
    auto c = a.hadamard(b);
    ASSERT_DOUBLE_EQ(c(0, 0), 5.0);
    ASSERT_DOUBLE_EQ(c(1, 1), 32.0);
}

TEST(MatrixTest, SumAndMean) {
    Matrix a(2, 2, {1, 2, 3, 4});
    ASSERT_DOUBLE_EQ(a.sum(), 10.0);
    ASSERT_DOUBLE_EQ(a.mean(), 2.5);
}

TEST(MatrixTest, ArgmaxRow) {
    Matrix a(2, 3, {1, 5, 3, 7, 2, 9});
    ASSERT_EQ(a.argmax_row(0), 1);
    ASSERT_EQ(a.argmax_row(1), 2);
}

TEST(MatrixTest, AddRowVector) {
    Matrix a(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix v(1, 3, {10, 20, 30});
    auto r = a.add_row_vector(v);
    ASSERT_DOUBLE_EQ(r(0, 0), 11.0);
    ASSERT_DOUBLE_EQ(r(1, 2), 36.0);
}

TEST(MatrixTest, XavierInit) {
    auto m = Matrix::xavier(100, 100);
    ASSERT_EQ(m.rows(), 100);
    ASSERT_EQ(m.cols(), 100);
    // Xavier values should be bounded by sqrt(6/(in+out))
    double limit = std::sqrt(6.0 / 200.0);
    ASSERT_LE(m.max_val(), limit + 0.01);  // Small tolerance
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dense Layer Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(DenseLayerTest, ForwardShape) {
    DenseLayer layer(4, 3, Activation::ReLU);
    Matrix input = Matrix::random(2, 4);
    auto output = layer.forward(input);
    ASSERT_EQ(output.rows(), 2);
    ASSERT_EQ(output.cols(), 3);
}

TEST(DenseLayerTest, ReluActivation) {
    DenseLayer layer(2, 2, Activation::ReLU);
    Matrix input(1, 2, {-1.0, 1.0});
    auto output = layer.forward(input);
    // All values should be >= 0 after ReLU
    for (std::size_t j = 0; j < output.cols(); ++j) {
        ASSERT_GE(output(0, j), 0.0);
    }
}

TEST(DenseLayerTest, SigmoidBounds) {
    DenseLayer layer(2, 2, Activation::Sigmoid);
    Matrix input = Matrix::random(5, 2);
    auto output = layer.forward(input);
    for (std::size_t i = 0; i < output.rows(); ++i) {
        for (std::size_t j = 0; j < output.cols(); ++j) {
            ASSERT_GE(output(i, j), 0.0);
            ASSERT_LE(output(i, j), 1.0);
        }
    }
}

TEST(DenseLayerTest, SoftmaxSumsToOne) {
    DenseLayer layer(4, 3, Activation::Softmax);
    Matrix input = Matrix::random(2, 4);
    auto output = layer.forward(input);
    for (std::size_t i = 0; i < output.rows(); ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < output.cols(); ++j) {
            sum += output(i, j);
          }
        ASSERT_NEAR(sum, 1.0, 1e-10);
    }
}

TEST(DenseLayerTest, InputDimensionCheck) {
    DenseLayer layer(4, 3);
    Matrix input(2, 5);  // Wrong input size
    EXPECT_THROW(layer.forward(input), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Network Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(NetworkTest, ForwardPropagation) {
    Network net;
    net.add_layer(4, 8, Activation::ReLU);
    net.add_layer(8, 3, Activation::Softmax);

    Matrix input = Matrix::random(5, 4);
    auto output = net.forward(input);
    ASSERT_EQ(output.rows(), 5);
    ASSERT_EQ(output.cols(), 3);

    // Softmax output: each row sums to 1
    for (std::size_t i = 0; i < output.rows(); ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < output.cols(); ++j) {
            sum += output(i, j);
        }
        ASSERT_NEAR(sum, 1.0, 1e-10);
    }
}

TEST(NetworkTest, Predict) {
    Network net;
    net.add_layer(4, 8, Activation::ReLU);
    net.add_layer(8, 3, Activation::Softmax);

    Matrix input = Matrix::random(5, 4);
    auto preds = net.predict(input);
    ASSERT_EQ(preds.size(), 5);
    for (auto p : preds) {
        ASSERT_LT(p, 3);  // Must be valid class index
    }
}

TEST(NetworkTest, Summary) {
    Network net;
    net.add_layer(784, 128, Activation::ReLU);
    net.add_layer(128, 10, Activation::Softmax);
    std::string summary = net.summary();
    ASSERT_FALSE(summary.empty());
    ASSERT_NE(summary.find("Dense"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Loss Function Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(LossTest, MseLossZero) {
    Matrix a(2, 2, {1, 2, 3, 4});
    ASSERT_NEAR(mse_loss(a, a), 0.0, 1e-12);
}

TEST(LossTest, MseLossBasic) {
    Matrix pred(1, 2, {1.0, 2.0});
    Matrix target(1, 2, {1.5, 2.5});
    // MSE = ((0.5)^2 + (0.5)^2) / 2 = 0.25
    ASSERT_NEAR(mse_loss(pred, target), 0.25, 1e-10);
}

TEST(LossTest, MseGradientShape) {
    Matrix pred(3, 2, {1, 2, 3, 4, 5, 6});
    Matrix target(3, 2, {1.1, 2.1, 3.1, 4.1, 5.1, 6.1});
    auto grad = mse_gradient(pred, target);
    ASSERT_EQ(grad.rows(), 3);
    ASSERT_EQ(grad.cols(), 2);
}

TEST(LossTest, MseDimensionMismatch) {
    Matrix a(2, 2);
    Matrix b(3, 2);
    EXPECT_THROW(mse_loss(a, b), std::invalid_argument);
}

TEST(LossTest, CrossEntropyPositive) {
    // Perfect prediction: loss should be near 0
    Matrix pred(1, 3, {0.0001, 0.9998, 0.0001});
    Matrix target(1, 3, {0, 1, 0});
    double loss = cross_entropy_loss(pred, target);
    ASSERT_LT(loss, 0.01);
}
