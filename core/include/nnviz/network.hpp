#pragma once
// ─── Sequential Neural Network ──────────────────────────────────────────────
// A container for stacking DenseLayers and running forward propagation.

#include "exports.hpp"
#include "matrix.hpp"
#include "dense_layer.hpp"
#include <vector>
#include <memory>

namespace nnviz::nn {

class NNVIZ_API Network {
public:
    Network() = default;

    /// Add a dense layer to the network.
    void add_layer(std::size_t input_size, std::size_t output_size,
                   Activation activation = Activation::ReLU);

    /// Forward propagation through all layers.
    /// @param input Matrix of shape (batch_size, input_features).
    /// @return Output matrix of shape (batch_size, last_layer_output_size).
    Matrix forward(const Matrix& input);

    /// Predict class indices (argmax of output per sample).
    std::vector<std::size_t> predict(const Matrix& input);

    /// Access layers for inspection / weight modification.
    std::vector<DenseLayer>& layers() { return layers_; }
    const std::vector<DenseLayer>& layers() const { return layers_; }

    /// Number of layers.
    std::size_t num_layers() const { return layers_.size(); }

    /// Summary string: layer sizes and activations.
    std::string summary() const;

private:
    std::vector<DenseLayer> layers_;
};

}  // namespace nnviz::nn
