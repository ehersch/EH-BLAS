#include "matrix.h"
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <omp.h>
#include <tuple>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <cmath>

Matrix::Matrix(const std::vector<std::vector<double>>& mat) {
  if (mat.empty()) {
    rows = 0;
    cols = 0;
    return;
  }
  rows = mat.size();
  cols = mat[0].size();
  data.resize(rows * cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      at(i, j) = mat[i][j];
    }
  }
}

Matrix::Matrix(size_t rows, size_t cols, double init)
    : data(rows * cols, init), rows(rows), cols(cols) {}

double& Matrix::at(size_t r, size_t c) {
  return data[r * cols + c];
}

const double& Matrix::at(size_t r, size_t c) const {
  return data[r * cols + c];
}

void Matrix::print() const {
  print(*this);
}

std::string to_string_helper(const Matrix& mat) {
  std::string output = "---\n";

  for (size_t r = 0; r < mat.rows; r++) {
    std::string row = "";
    for (size_t c = 0; c < mat.cols - 1; c++) {
      row = row + std::to_string(mat.at(r, c)) + ", ";
    }
    row = row + std::to_string(mat.at(r, mat.cols - 1));
    output = output + row + "\n";
  }
  output = output + "---";
  return output;
}

void Matrix::print(const Matrix& mat) {
  std::string mat_str = to_string_helper(mat);
  std::cout << mat_str << std:: endl;
}

Matrix Matrix::transpose() const {
  Matrix result(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result.at(j, i) = at(i, j);
    }
  }
  return result;
}

std::optional<Matrix> Matrix::matmul(const Matrix& mat_a, const Matrix& mat_b) {
  size_t n = mat_a.rows;
  size_t k = mat_a.cols;

  if (mat_b.rows != k) {
    throw std::runtime_error("Matrix dimensions do not match.");
  }

  size_t m = mat_b.cols;
  Matrix C(n, m);

  for (size_t row = 0; row < n; row++) {
    for (size_t col = 0; col < m; col++) {
      double sum = 0.0;
      for (size_t i = 0; i < k; i++) {
        sum += mat_a.at(row, i) * mat_b.at(i, col);
      }
      C.at(row, col) = sum;
    }
  }

  return C;
}

bool Matrix::operator==(const Matrix& other) const {
  return approx_equal(other);
}

bool Matrix::approx_equal(const Matrix& other, double rtol, double atol) const {
  if (rows != other.rows || cols != other.cols)
    return false;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      double a = at(i, j);
      double b = other.at(i, j);
      if (std::abs(a - b) > (atol + rtol * std::abs(b)))
        return false;
    }
  }
  return true;
}

double dot(const std::vector<double>& x, const std::vector<double>& y) {
  int n = x.size();
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += x[i] * y[i];
  }
  return res;
}

std::optional<Matrix> Matrix::matmul_parallel(const Matrix& mat_a, const Matrix& mat_b) {
  size_t n = mat_a.rows;
  size_t k = mat_a.cols;

  if (mat_b.rows != k) {
    throw std::runtime_error("Matrix dimensions do not match.");
  }

  size_t m = mat_b.cols;
  Matrix C(n, m);

  #pragma omp parallel for collapse(2)
  for (size_t row = 0; row < n; row++) {
    for (size_t col = 0; col < m; col++) {
      double sum = 0.0;
      for (size_t i = 0; i < k; i++) {
        sum += mat_a.at(row, i) * mat_b.at(i, col);
      }
      C.at(row, col) = sum;
    }
  }

  return C;
}

std::optional<Matrix> Matrix::operator*(const Matrix& other) const {
  return Matrix::matmul_blocked(*this, other);
}

Matrix Matrix::operator-(const Matrix& other) const {
  Matrix C(rows, cols);

  #pragma omp parallel for
  for (size_t i = 0; i < data.size(); i++) {
    C.data[i] = data[i] - other.data[i];
  }

  return C;
}

std::optional<Matrix> Matrix::matmul_blocked(const Matrix& mat_a, const Matrix& mat_b) {
    size_t n = mat_a.rows;
    size_t k = mat_a.cols;

    if (mat_b.rows != k)
        throw std::runtime_error("Matrix dimensions do not match.");

    size_t m = mat_b.cols;

    size_t block = 64;

    Matrix C(n, m);

    const double* A_data = mat_a.data.data();
    const double* B_data = mat_b.data.data();
    double* C_data = C.data.data();
    const size_t A_cols = mat_a.cols;
    const size_t B_cols = mat_b.cols;
    const size_t C_cols = C.cols;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < n; ii += block) {
        for (size_t jj = 0; jj < m; jj += block) {
            for (size_t kk = 0; kk < k; kk += block) {

                size_t i_max = std::min(ii + block, n);
                size_t k_max = std::min(kk + block, k);
                size_t j_max = std::min(jj + block, m);

                for (size_t i = ii; i < i_max; i++) {
                    double* c_row = C_data + i * C_cols;
                    for (size_t kk2 = kk; kk2 < k_max; kk2++) {
                        double a_val = A_data[i * A_cols + kk2];
                        const double* b_row = B_data + kk2 * B_cols;
                        #pragma omp simd
                        for (size_t j = jj; j < j_max; j++) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }

            }
        }
    }

    return C;
}

std::string Matrix::to_string() const {
  return to_string_helper(*this);
}

std::tuple<double, double, double> Matrix::compare_times(const Matrix& other) const {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration;

  auto t1 = high_resolution_clock::now();
  auto C = Matrix::matmul(*this, other);
  auto t2 = high_resolution_clock::now();

  auto C_1 = Matrix::matmul_parallel(*this, other);
  auto t3 = high_resolution_clock::now();

  auto C_3 = Matrix::matmul_blocked(*this, other);
  auto t4 = high_resolution_clock::now();

  duration<double, std::milli> basic_time = t2 - t1;
  duration<double, std::milli> parallel_time = t3 - t2;
  duration<double, std::milli> blocked_time = t4 - t3;

  return std::make_tuple(basic_time.count(), parallel_time.count(), blocked_time.count());
}
