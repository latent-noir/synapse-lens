#pragma once
// ─── Matrix Class ───────────────────────────────────────────────────────────
// A lightweight 2D matrix for the C++ neural network engine.
// Not a replacement for Eigen — this is a pedagogical, self-contained
// implementation suitable for portfolio demonstration.

#include "exports.hpp"
#include <vector>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <cassert>
#include <cmath>

namespace nnviz::nn {

class NNVIZ_API Matrix {
public:
    // ── Constructors ────────────────────────────────────────────────────
    Matrix();
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0);
    Matrix(std::size_t rows, std::size_t cols, const std::vector<double>& data);

    // ── Factory Methods ─────────────────────────────────────────────────
    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix ones(std::size_t rows, std::size_t cols);
    static Matrix random(std::size_t rows, std::size_t cols,
                         double mean = 0.0, double stddev = 1.0);
    static Matrix xavier(std::size_t rows, std::size_t cols);

    // ── Shape ───────────────────────────────────────────────────────────
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    // ── Element Access ──────────────────────────────────────────────────
    double& operator()(std::size_t r, std::size_t c);
    double  operator()(std::size_t r, std::size_t c) const;

    // ── Raw Data ────────────────────────────────────────────────────────
    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

    // ── Arithmetic Operators ────────────────────────────────────────────
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;  // Matrix multiply
    Matrix operator*(double scalar) const;
    Matrix operator+(double scalar) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

    // ── Element-wise Operations ─────────────────────────────────────────
    Matrix hadamard(const Matrix& other) const;
    Matrix apply(std::function<double(double)> fn) const;
    Matrix transpose() const;

    // ── Reductions ──────────────────────────────────────────────────────
    double sum() const;
    double mean() const;
    double max_val() const;
    std::size_t argmax_row(std::size_t row) const;

    // ── Broadcasting ────────────────────────────────────────────────────
    /// Add a row vector (1 x cols) to every row of this matrix
    Matrix add_row_vector(const Matrix& row_vec) const;

    // ── Debug ───────────────────────────────────────────────────────────
    void print(const std::string& label = "") const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;  // Row-major storage
};

// Non-member operators
NNVIZ_API Matrix operator*(double scalar, const Matrix& m);

}  // namespace nnviz::nn
