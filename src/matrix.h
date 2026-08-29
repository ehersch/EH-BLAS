#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <optional>
#include <string>
#include <tuple>
#include <cstddef>

class Matrix {
  public:
    std::vector<double> data;
    size_t rows;
    size_t cols;

    Matrix(const std::vector<std::vector<double>>& mat);
    Matrix(size_t rows, size_t cols, double init = 0.0);

    double& at(size_t r, size_t c);
    const double& at(size_t r, size_t c) const;

    void print() const;
    static void print(const Matrix& mat);

    static std::optional<Matrix> matmul(const Matrix& mat_a, const Matrix& mat_b);

    bool operator==(const Matrix& other) const;

    bool approx_equal(const Matrix& other, double rtol=1e-05, double atol=1e-08) const;

    static std::optional<Matrix> matmul_parallel(const Matrix& mat_a, const Matrix& mat_b);

    std::optional<Matrix> operator*(const Matrix& other) const;

    Matrix operator-(const Matrix& other) const;

    static std::optional<Matrix> matmul_blocked(const Matrix& mat_a, const Matrix& mat_b);

    std::string to_string() const;

    std::tuple<double, double, double> compare_times(const Matrix& other) const;

    Matrix transpose() const;
};

#endif
