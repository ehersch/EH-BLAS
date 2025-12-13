#include "matrix.h"
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <omp.h>
#include <string>

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
  int n = A.size();
  int k = A[0].size();

  if (B.size() != k) {
    throw std::runtime_error("Matrix dimensions do not match.");
  }

  int m = B[0].size();
  
  std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < m; col++) {
      for (int i = 0; i < k; i++) {
        C[row][col] += A[row][i] * B[i][col];
      }
    }
  }

  return Matrix(C);
}

// add const bc doesnt alter member's internal state
bool Matrix::operator==(const Matrix& other) const {
  return approx_equal(other, 1e-12);
}

bool Matrix::approx_equal(const Matrix& other, double tol) const {
  if (M.size() != other.M.size() || M[0].size() != other.M[0].size())
    return false;

  int n = M.size();
  int m = M[0].size();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      double diff = std::abs(M[i][j] - other.M[i][j]);
      if (diff > tol)
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
  int n = A.size();
  int k = A[0].size();

  if (B.size() != k) {
    throw std::runtime_error("Matrix dimensions do not match.");
  }

  int m = B[0].size();
  std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));

  //omp_set_num_threads(8);
  #pragma omp parallel for collapse(3)

  // change the order of i and col to increase hit rate
  // i.e. we reuse A[row][i] so keep it cached
  for (int row = 0; row < n; row++) {
    for (int i = 0; i < k; i++) {
      for (int col = 0; col < m; col++) {
        C[row][col] += A[row][i] * B[i][col];
      }
    }
  }

  return Matrix(C);
}

std::optional<Matrix> Matrix::operator*(const Matrix& other) const {
  return Matrix::matmul_blocked(*this, other);
}

std::optional<Matrix> Matrix::matmul_blocked(const Matrix& mat_a, const Matrix& mat_b) {
    // auto& just gets the type automatically and binds a reference to it
    const auto& A = mat_a.M;
    const auto& B = mat_b.M;

    int n = A.size();
    int k = A[0].size();

    if (B.size() != k)
        throw std::runtime_error("Matrix dimensions do not match.");

    int m = B[0].size();

    int block = 64;  // good for Apple M-series CPUs

    std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));

    #pragma omp parallel for collapse(3) schedule(static)

    for (int ii = 0; ii < n; ii += block) {
        for (int kk = 0; kk < k; kk += block) {
            for (int jj = 0; jj < m; jj += block) {

                int i_max = std::min(ii + block, n);
                int k_max = std::min(kk + block, k);
                int j_max = std::min(jj + block, m);

                for (int i = ii; i < i_max; i++) {
                    for (int kk2 = kk; kk2 < k_max; kk2++) {
                        double a_val = A[i][kk2];
                        for (int j = jj; j < j_max; j++) {
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