#pragma once
// ─── Statistical Utilities ──────────────────────────────────────────────────
// Fast running statistics, histogram binning, and distribution metrics.

#include "exports.hpp"
#include <vector>
#include <cstddef>

namespace nnviz::statistics {

/// Compute the arithmetic mean of a data vector.
NNVIZ_API double mean(const std::vector<double>& data);

/// Compute the population variance.
NNVIZ_API double variance(const std::vector<double>& data);

/// Compute the population standard deviation.
NNVIZ_API double stddev(const std::vector<double>& data);

/// Compute a histogram with `num_bins` equal-width bins.
/// Returns a pair of (bin_edges, counts).
struct HistogramResult {
    std::vector<double> bin_edges;   // size = num_bins + 1
    std::vector<std::size_t> counts; // size = num_bins
};

NNVIZ_API HistogramResult histogram(const std::vector<double>& data,
                                     std::size_t num_bins = 50);

/// Compute the L2 norm of a vector.
NNVIZ_API double l2_norm(const std::vector<double>& data);

}  // namespace nnviz::statistics
