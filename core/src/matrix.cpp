// ─── Matrix Implementation ──────────────────────────────────────────────────
#include "nnviz/matrix.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace nnviz::nn {

// ── Constructors ────────────────────────────────────────────────────────────

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, const std::vector<double>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument(
            "Matrix data size (" + std::to_string(data.size()) +
            ") doesn't match dimensions (" + std::to_string(rows) + "x" +
            std::to_string(cols) + ")");
    }
}

// ── Factory Methods ─────────────────────────────────────────────────────────

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, double mean, double stddev) {
    Matrix m(rows, cols);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(mean, stddev);
    for (auto& v : m.data_) v = dist(gen);
    return m;
}

Matrix Matrix::xavier(std::size_t rows, std::size_t cols) {
    double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
    Matrix m(rows, cols);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto& v : m.data_) v = dist(gen);
    return m;
}

// ── Element Access ──────────────────────────────────────────────────────────

double& Matrix::operator()(std::size_t r, std::size_t c) {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
}

double Matrix::operator()(std::size_t r, std::size_t c) const {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
}

// ── Arithmetic Operators ────────────────────────────────────────────────────

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix addition: dimension mismatch");
    }
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix subtraction: dimension mismatch");
    }
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Matrix multiply: incompatible (" +
            std::to_string(rows_) + "x" + std::to_string(cols_) + ") * (" +
            std::to_string(other.rows_) + "x" + std::to_string(other.cols_) + ")");
    }
    Matrix result(rows_, other.cols_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            double a_ik = data_[i * cols_ + k];
            for (std::size_t j = 0; j < other.cols_; ++j) {
                result.data_[i * other.cols_ + j] += a_ik * other.data_[k * other.cols_ + j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix += : dimension mismatch");
    }
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix -= : dimension mismatch");
    }
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (auto& v : data_) v *= scalar;
    return *this;
}

// ── Element-wise Operations ─────────────────────────────────────────────────

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Hadamard product: dimension mismatch");
    }
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Matrix Matrix::apply(std::function<double(double)> fn) const {
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = fn(data_[i]);
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result.data_[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return result;
}

// ── Reductions ──────────────────────────────────────────────────────────────

double Matrix::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Matrix::mean() const {
    if (data_.empty()) return 0.0;
    return sum() / static_cast<double>(data_.size());
}

double Matrix::max_val() const {
    if (data_.empty()) return 0.0;
    return *std::max_element(data_.begin(), data_.end());
}

std::size_t Matrix::argmax_row(std::size_t row) const {
    assert(row < rows_);
    auto start = data_.begin() + static_cast<long>(row * cols_);
    auto end = start + static_cast<long>(cols_);
    return static_cast<std::size_t>(std::distance(start, std::max_element(start, end)));
}

// ── Broadcasting ────────────────────────────────────────────────────────────

Matrix Matrix::add_row_vector(const Matrix& row_vec) const {
    if (row_vec.rows() != 1 || row_vec.cols() != cols_) {
        throw std::invalid_argument("add_row_vector: vector must be (1 x cols)");
    }
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result.data_[i * cols_ + j] = data_[i * cols_ + j] + row_vec.data_[j];
        }
    }
    return result;
}

// ── Debug ───────────────────────────────────────────────────────────────────

void Matrix::print(const std::string& label) const {
    if (!label.empty()) std::cout << label << ":\n";
    std::cout << "[" << rows_ << " x " << cols_ << "]\n";
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << data_[i * cols_ + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// ── Non-member Operators ────────────────────────────────────────────────────

Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}

}  // namespace nnviz::nn
