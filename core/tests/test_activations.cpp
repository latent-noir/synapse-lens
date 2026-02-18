// ─── Activation Function Unit Tests ──────────────────────────────────────────
#include <gtest/gtest.h>
#include "nnviz/activations.hpp"
#include <cmath>
#include <numeric>

using namespace nnviz::activations;

TEST(ActivationsTest, ReluPositive) {
    auto result = relu({1.0, 2.0, 3.0});
    ASSERT_EQ(result.size(), 3);
    ASSERT_DOUBLE_EQ(result[0], 1.0);
    ASSERT_DOUBLE_EQ(result[2], 3.0);
}

TEST(ActivationsTest, ReluNegative) {
    auto result = relu({-1.0, -2.0, 0.0});
    ASSERT_DOUBLE_EQ(result[0], 0.0);
    ASSERT_DOUBLE_EQ(result[1], 0.0);
    ASSERT_DOUBLE_EQ(result[2], 0.0);
}

TEST(ActivationsTest, SigmoidBounds) {
    auto result = sigmoid({-100.0, 0.0, 100.0});
    ASSERT_NEAR(result[0], 0.0, 1e-6);
    ASSERT_NEAR(result[1], 0.5, 1e-10);
    ASSERT_NEAR(result[2], 1.0, 1e-6);
}

TEST(ActivationsTest, SoftmaxSumsToOne) {
    auto result = softmax({1.0, 2.0, 3.0, 4.0});
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    ASSERT_NEAR(sum, 1.0, 1e-10);
}

TEST(ActivationsTest, SoftmaxEmpty) {
    auto result = softmax({});
    ASSERT_TRUE(result.empty());
}

TEST(ActivationsTest, GeluApproximatesZeroAtZero) {
    auto result = gelu({0.0});
    ASSERT_NEAR(result[0], 0.0, 1e-10);
}

TEST(ActivationsTest, LeakyReluNegativeSlope) {
    auto result = leaky_relu({-10.0, 5.0}, 0.1);
    ASSERT_NEAR(result[0], -1.0, 1e-10);
    ASSERT_DOUBLE_EQ(result[1], 5.0);
}
