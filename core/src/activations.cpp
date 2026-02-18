// ─── Activation Function Implementations ────────────────────────────────────
#include "nnviz/activations.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace nnviz::activations {

std::vector<double> relu(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(),
                   [](double v) { return std::max(0.0, v); });
    return result;
}

std::vector<double> sigmoid(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(),
                   [](double v) { return 1.0 / (1.0 + std::exp(-v)); });
    return result;
}

std::vector<double> softmax(const std::vector<double>& x) {
    if (x.empty()) return {};

    // Numerically stable: subtract max before exp
    const double max_val = *std::max_element(x.begin(), x.end());
    std::vector<double> result(x.size());
    double sum = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    for (auto& v : result) {
        v /= sum;
    }
    return result;
}

std::vector<double> gelu(const std::vector<double>& x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr double sqrt_2_over_pi = 0.7978845608028654;
    constexpr double coeff = 0.044715;

    std::vector<double> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(), [](double v) {
        return 0.5 * v * (1.0 + std::tanh(sqrt_2_over_pi * (v + coeff * v * v * v)));
    });
    return result;
}

std::vector<double> leaky_relu(const std::vector<double>& x, double negative_slope) {
    std::vector<double> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(),
                   [negative_slope](double v) {
                       return v >= 0.0 ? v : negative_slope * v;
                   });
    return result;
}

}  // namespace nnviz::activations
