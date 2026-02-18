// ─── Statistics Unit Tests ───────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "nnviz/statistics.hpp"
#include <cmath>

using namespace nnviz::statistics;

TEST(StatisticsTest, MeanBasic) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    ASSERT_NEAR(mean(data), 3.0, 1e-10);
}

TEST(StatisticsTest, MeanEmpty) {
    EXPECT_THROW(mean({}), std::invalid_argument);
}

TEST(StatisticsTest, Variance) {
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    ASSERT_NEAR(variance(data), 4.0, 1e-10);
}

TEST(StatisticsTest, Stddev) {
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    ASSERT_NEAR(stddev(data), 2.0, 1e-10);
}

TEST(StatisticsTest, HistogramBinCount) {
    std::vector<double> data = {0.1, 0.5, 0.9, 1.5, 2.0, 2.5, 3.0};
    auto result = histogram(data, 3);
    ASSERT_EQ(result.bin_edges.size(), 4);
    ASSERT_EQ(result.counts.size(), 3);
}

TEST(StatisticsTest, HistogramSumEqualsDataSize) {
    std::vector<double> data = {0.1, 0.5, 0.9, 1.5, 2.0};
    auto result = histogram(data, 10);
    std::size_t total = 0;
    for (auto c : result.counts) total += c;
    ASSERT_EQ(total, data.size());
}

TEST(StatisticsTest, L2Norm) {
    std::vector<double> data = {3.0, 4.0};
    ASSERT_NEAR(l2_norm(data), 5.0, 1e-10);
}
