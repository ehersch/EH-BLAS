#include "matrix.h"
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <omp.h>
#include <string>
#include <tuple>
#include <cassert>
#include <chrono>
#include <algorithm>


// Constructor taking a 2D vector
Matrix::Matrix(const std::vector<std::vector<double>>& mat) : M(mat) {}

void Matrix::print() const {
  print(*this);
}

std::string to_string_helper(const Matrix& mat) {
  auto& M = mat.M;

  int n = M.size();
  int m = M[0].size();

  std::string output = "---\n";

  for(int r=0; r < n; r += 1) {
    std::string row = "";
    for(int c = 0; c < m - 1; c += 1) {
      row = row + std::to_string(M[r][c]) + ", ";
    }
    row = row + std::to_string(M[r][m-1]);
    output = output + row + "\n";
  }
  output = output + "---";
  return output;
}

void Matrix::print(const Matrix& mat) {
  std::string mat_str = to_string_helper(mat);
  std::cout << mat_str << std:: endl;
}

std::optional<Matrix> Matrix::matmul(const Matrix& mat_a, const Matrix& mat_b) {
  std::vector<std::vector<double>> A = mat_a.M;
  std::vector<std::vector<double>> B = mat_b.M;
  size_t n = A.size();
  size_t k = A[0].size();

  if (B.size() != k) {
    throw std::runtime_error("Matrix dimensions do not match.");
  }

  size_t m = B[0].size();
  
  std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));
  for (size_t row = 0; row < n; row++) {
    for (size_t col = 0; col < m; col++) {
      for (size_t i = 0; i < k; i++) {
        C[row][col] += A[row][i] * B[i][col];
      }
    }
  }

  return Matrix(C);
}

// add const bc doesnt alter member's internal state
bool Matrix::operator==(const Matrix& other) const {
  return approx_equal(other);
}

bool Matrix::approx_equal(const Matrix& other, double rtol, double atol) const {
  if (M.size() != other.M.size() || M[0].size() != other.M[0].size())
    return false;

  int n = M.size();
  int m = M[0].size();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      double a = M[i][j];
      double b = other.M[i][j];
      // absolute + relative tolerance
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

// https://courses.grainger.illinois.edu/cs484/sp2020/6_merged.pdf

std::optional<Matrix> Matrix::matmul_parallel(const Matrix& mat_a, const Matrix& mat_b) {
  std::vector<std::vector<double>> A = mat_a.M;
  std::vector<std::vector<double>> B = mat_b.M;
  size_t n = A.size();
  size_t k = A[0].size();

  if (B.size() != k) {
    throw std::runtime_error("Matrix dimensions do not match.");
  }

  size_t m = B[0].size();
  std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));

  //omp_set_num_threads(8);
  #pragma omp parallel for collapse(2)
  for (size_t row = 0; row < n; row++) {
    for (size_t col = 0; col < m; col++) {
      double sum = 0.0;
      for (size_t i = 0; i < k; i++) {
        sum += A[row][i] * B[i][col];
      }
      C[row][col] = sum;
    }
  }


  return Matrix(C);
}

std::optional<Matrix> Matrix::operator*(const Matrix& other) const {
  return Matrix::matmul_blocked(*this, other);
}

Matrix Matrix::operator-(const Matrix& other) const {
  int n = (int)M.size();
  int m = (int)M[0].size();

  std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));

  #pragma omp parallel for collapse(2)

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      C[i][j] = M[i][j] - other.M[i][j];
    }
  }

  return Matrix(C);
}


std::optional<Matrix> Matrix::matmul_blocked(const Matrix& mat_a, const Matrix& mat_b) {
    // auto& just gets the type automatically and binds a reference to it
    const auto& A = mat_a.M;
    const auto& B = mat_b.M;

    size_t n = A.size();
    size_t k = A[0].size();

    if (B.size() != k)
        throw std::runtime_error("Matrix dimensions do not match.");

    size_t m = B[0].size();

    size_t block = 64;  // good for Apple M-series CPUs

    std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));

    //parallelize only (ii, jj) to avoid race conditions
    #pragma omp parallel for collapse(2) schedule(static)

    for (size_t ii = 0; ii < n; ii += block) {
        for (size_t jj = 0; jj < m; jj += block) {
            for (size_t kk = 0; kk < k; kk += block) {

                size_t i_max = std::min(ii + block, n);
                size_t k_max = std::min(kk + block, k);
                size_t j_max = std::min(jj + block, m);

                for (size_t i = ii; i < i_max; i++) {
                    for (size_t kk2 = kk; kk2 < k_max; kk2++) {
                        double a_val = A[i][kk2];
                        for (size_t j = jj; j < j_max; j++) {
                            C[i][j] += a_val * B[kk2][j];
                        }
                    }
                }

            }
        }
    }

    return Matrix(C);
}

std::string Matrix::to_string() const {
  auto& mat = *this;
  std::string mat_str = to_string_helper(mat);
  return mat_str;
}

std::tuple<double, double, double> Matrix::compare_times(const Matrix& other) const {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  // Time naive matmul
  auto t1 = high_resolution_clock::now();
  auto C = Matrix::matmul(*this, other);
  auto t2 = high_resolution_clock::now();

  // Time parallel matmul
  auto C_1 = Matrix::matmul_parallel(*this, other);
  auto t3 = high_resolution_clock::now();

  // Time blocked matmul
  auto C_3 = Matrix::matmul_blocked(*this, other);
  auto t4 = high_resolution_clock::now();

  duration<double, std::milli> basic_time = t2 - t1;
  duration<double, std::milli> parallel_time = t3 - t2;
  duration<double, std::milli> blocked_time = t4 - t3;

  return std::make_tuple(basic_time.count(), parallel_time.count(), blocked_time.count());
}