// ─── Network Implementation ─────────────────────────────────────────────────
#include "nnviz/network.hpp"
#include <sstream>

namespace nnviz::nn {

void Network::add_layer(std::size_t input_size, std::size_t output_size, Activation activation) {
    layers_.emplace_back(input_size, output_size, activation);
}

Matrix Network::forward(const Matrix& input) {
    Matrix current = input;
    for (auto& layer : layers_) {
        current = layer.forward(current);
    }
    return current;
}

std::vector<std::size_t> Network::predict(const Matrix& input) {
    Matrix output = forward(input);
    std::vector<std::size_t> predictions(output.rows());
    for (std::size_t i = 0; i < output.rows(); ++i) {
        predictions[i] = output.argmax_row(i);
    }
    return predictions;
}

std::string Network::summary() const {
    std::ostringstream ss;
    ss << "Network Summary (" << layers_.size() << " layers):\n";
    ss << "──────────────────────────────────────────\n";
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const auto& l = layers_[i];
        ss << "  Layer " << i << ": Dense("
           << l.input_size() << " → " << l.output_size()
           << ") | activation=" << activation_to_string(l.activation_type())
           << " | params=" << (l.input_size() * l.output_size() + l.output_size())
           << "\n";
    }
    // Total parameters
    std::size_t total = 0;
    for (const auto& l : layers_) {
        total += l.input_size() * l.output_size() + l.output_size();
    }
    ss << "──────────────────────────────────────────\n";
    ss << "  Total parameters: " << total << "\n";
    return ss.str();
}

}  // namespace nnviz::nn
