// ─── pybind11 Bindings for nnviz_core ───────────────────────────────────────
// Exposes the C++ core library to Python as the `nnviz_core` module.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "nnviz/linalg.hpp"
#include "nnviz/statistics.hpp"
#include "nnviz/activations.hpp"
#include "nnviz/matrix.hpp"
#include "nnviz/dense_layer.hpp"
#include "nnviz/network.hpp"
#include "nnviz/loss.hpp"

namespace py = pybind11;
using namespace nnviz::nn;

PYBIND11_MODULE(nnviz_core, m) {
    m.doc() = "nnviz C++ core — linear algebra, statistics, activations, and neural network engine";

    // ── Linear Algebra (Eigen-backed) ───────────────────────────────────────
    auto m_linalg = m.def_submodule("linalg", "Matrix operations (Eigen-backed)");

    m_linalg.def("matmul", &nnviz::linalg::matmul,
                 py::arg("a"), py::arg("b"),
                 "Matrix multiplication: C = A * B");

    m_linalg.def("transpose", &nnviz::linalg::transpose,
                 py::arg("m"),
                 "Transpose a matrix");

    m_linalg.def("frobenius_norm", &nnviz::linalg::frobenius_norm,
                 py::arg("m"),
                 "Compute the Frobenius norm of a matrix");

    m_linalg.def("flatten", &nnviz::linalg::flatten,
                 py::arg("m"),
                 "Flatten a matrix to a 1-D list (row-major)");

    m_linalg.def("singular_values", &nnviz::linalg::singular_values,
                 py::arg("m"),
                 "Compute the singular values of a matrix");

    // ── Statistics ──────────────────────────────────────────────────────────
    auto m_stats = m.def_submodule("statistics", "Statistical computations");

    m_stats.def("mean", &nnviz::statistics::mean,
                py::arg("data"),
                "Arithmetic mean of a data vector");

    m_stats.def("variance", &nnviz::statistics::variance,
                py::arg("data"),
                "Population variance");

    m_stats.def("stddev", &nnviz::statistics::stddev,
                py::arg("data"),
                "Population standard deviation");

    m_stats.def("l2_norm", &nnviz::statistics::l2_norm,
                py::arg("data"),
                "L2 norm of a vector");

    m_stats.def("histogram",
                [](const std::vector<double>& data, std::size_t num_bins) {
                    auto result = nnviz::statistics::histogram(data, num_bins);
                    py::dict d;
                    d["bin_edges"] = result.bin_edges;
                    d["counts"] = result.counts;
                    return d;
                },
                py::arg("data"), py::arg("num_bins") = 50,
                "Compute a histogram with equal-width bins");

    // ── Activations ─────────────────────────────────────────────────────────
    auto m_act = m.def_submodule("activations", "Vectorized activation functions");

    m_act.def("relu", &nnviz::activations::relu,
              py::arg("x"), "Element-wise ReLU");
    m_act.def("sigmoid", &nnviz::activations::sigmoid,
              py::arg("x"), "Element-wise sigmoid");
    m_act.def("softmax", &nnviz::activations::softmax,
              py::arg("x"), "Softmax over vector (numerically stable)");
    m_act.def("gelu", &nnviz::activations::gelu,
              py::arg("x"), "Element-wise GELU approximation");
    m_act.def("leaky_relu", &nnviz::activations::leaky_relu,
              py::arg("x"), py::arg("negative_slope") = 0.01,
              "Element-wise Leaky ReLU");

    // ═══════════════════════════════════════════════════════════════════════
    //  Neural Network Engine
    // ═══════════════════════════════════════════════════════════════════════
    auto m_nn = m.def_submodule("nn", "C++ Neural Network Engine");

    // ── Matrix ──────────────────────────────────────────────────────────────
    py::class_<Matrix>(m_nn, "Matrix")
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t, double>(),
             py::arg("rows"), py::arg("cols"), py::arg("fill") = 0.0)
        .def(py::init<std::size_t, std::size_t, const std::vector<double>&>(),
             py::arg("rows"), py::arg("cols"), py::arg("data"))
        .def_static("zeros", &Matrix::zeros)
        .def_static("ones", &Matrix::ones)
        .def_static("random", &Matrix::random,
                     py::arg("rows"), py::arg("cols"),
                     py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
        .def_static("xavier", &Matrix::xavier)
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def_property_readonly("size", &Matrix::size)
        .def("__getitem__", [](const Matrix& m, std::pair<std::size_t, std::size_t> idx) {
            return m(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix& m, std::pair<std::size_t, std::size_t> idx, double val) {
            m(idx.first, idx.second) = val;
        })
        .def("__add__", [](const Matrix& a, const Matrix& b) { return a + b; })
        .def("__sub__", [](const Matrix& a, const Matrix& b) { return a - b; })
        .def("__mul__", [](const Matrix& a, const Matrix& b) { return a * b; })
        .def("__mul__", [](const Matrix& a, double s) { return a * s; })
        .def("__rmul__", [](const Matrix& a, double s) { return s * a; })
        .def("hadamard", &Matrix::hadamard)
        .def("transpose", &Matrix::transpose)
        .def("sum", &Matrix::sum)
        .def("mean", &Matrix::mean)
        .def("max_val", &Matrix::max_val)
        .def("argmax_row", &Matrix::argmax_row)
        .def("add_row_vector", &Matrix::add_row_vector)
        .def("data", [](const Matrix& m) { return m.data(); })
        .def("print", &Matrix::print, py::arg("label") = "")
        .def("__repr__", [](const Matrix& m) {
            return "Matrix(" + std::to_string(m.rows()) + ", " +
                   std::to_string(m.cols()) + ")";
        });

    // ── Activation Enum ─────────────────────────────────────────────────────
    py::enum_<Activation>(m_nn, "Activation")
        .value("NONE", Activation::None)
        .value("RELU", Activation::ReLU)
        .value("SIGMOID", Activation::Sigmoid)
        .value("TANH", Activation::Tanh)
        .value("SOFTMAX", Activation::Softmax)
        .export_values();

    // ── DenseLayer ──────────────────────────────────────────────────────────
    py::class_<DenseLayer>(m_nn, "DenseLayer")
        .def(py::init<std::size_t, std::size_t, Activation>(),
             py::arg("input_size"), py::arg("output_size"),
             py::arg("activation") = Activation::ReLU)
        .def("forward", &DenseLayer::forward)
        .def_property_readonly("input_size", &DenseLayer::input_size)
        .def_property_readonly("output_size", &DenseLayer::output_size)
        .def_property_readonly("activation_type", &DenseLayer::activation_type)
        .def("get_pre_activation", &DenseLayer::get_pre_activation)
        .def("get_output", &DenseLayer::get_output)
        .def("weights", static_cast<const Matrix& (DenseLayer::*)() const>(&DenseLayer::weights))
        .def("bias", static_cast<const Matrix& (DenseLayer::*)() const>(&DenseLayer::bias));

    // ── Network ─────────────────────────────────────────────────────────────
    py::class_<Network>(m_nn, "Network")
        .def(py::init<>())
        .def("add_layer", &Network::add_layer,
             py::arg("input_size"), py::arg("output_size"),
             py::arg("activation") = Activation::ReLU)
        .def("forward", &Network::forward)
        .def("predict", &Network::predict)
        .def("num_layers", &Network::num_layers)
        .def("summary", &Network::summary)
        .def("__repr__", [](const Network& n) {
            return "Network(" + std::to_string(n.num_layers()) + " layers)";
        });

    // ── Loss Functions ──────────────────────────────────────────────────────
    m_nn.def("mse_loss", &mse_loss, py::arg("predictions"), py::arg("targets"),
             "Compute Mean Squared Error loss");
    m_nn.def("mse_gradient", &mse_gradient, py::arg("predictions"), py::arg("targets"),
             "Gradient of MSE loss w.r.t. predictions");
    m_nn.def("cross_entropy_loss", &cross_entropy_loss,
             py::arg("predictions"), py::arg("targets"),
             "Compute cross-entropy loss");
}
