// ─── Statistics Implementations ──────────────────────────────────────────────
#include "nnviz/statistics.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nnviz::statistics {

double mean(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("mean: empty data vector");
    }
    return std::accumulate(data.begin(), data.end(), 0.0) /
           static_cast<double>(data.size());
}

double variance(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("variance: empty data vector");
    }
    const double m = mean(data);
    double sum_sq = 0.0;
    for (const auto& x : data) {
        const double diff = x - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(data.size());
}

double stddev(const std::vector<double>& data) {
    return std::sqrt(variance(data));
}

HistogramResult histogram(const std::vector<double>& data, std::size_t num_bins) {
    if (data.empty() || num_bins == 0) {
        throw std::invalid_argument("histogram: empty data or zero bins");
    }

    const auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    double min_val = *min_it;
    double max_val = *max_it;

    // Handle constant data: create a single bin around the value
    if (min_val == max_val) {
        max_val = min_val + 1.0;
    }

    const double bin_width = (max_val - min_val) / static_cast<double>(num_bins);

    HistogramResult result;
    result.bin_edges.resize(num_bins + 1);
    result.counts.resize(num_bins, 0);

    for (std::size_t i = 0; i <= num_bins; ++i) {
        result.bin_edges[i] = min_val + static_cast<double>(i) * bin_width;
    }

    for (const auto& x : data) {
        auto bin = static_cast<std::size_t>((x - min_val) / bin_width);
        if (bin >= num_bins) bin = num_bins - 1;  // Clamp the max value
        result.counts[bin]++;
    }

    return result;
}

double l2_norm(const std::vector<double>& data) {
    double sum_sq = 0.0;
    for (const auto& x : data) {
        sum_sq += x * x;
    }
    return std::sqrt(sum_sq);
}

}  // namespace nnviz::statistics
