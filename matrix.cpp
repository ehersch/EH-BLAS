#include "matrix.h"
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <omp.h>

// Constructor taking a 2D vector
Matrix::Matrix(const std::vector<std::vector<double>>& mat) : M(mat) {}

void Matrix::print() const {
  print(*this);
}

void Matrix::print(const Matrix& mat) {
  int rows = mat.M.size();
  int cols = mat.M[0].size();

  std::cout << "---" << std:: endl;

  for (int r = 0; r < rows; r++) {
    std::string cur_row = "";
    for (int c = 0; c < cols; c++) {
      cur_row += std::to_string(mat.M[r][c]) + ", ";
    }
    if (!cur_row.empty()) {
      cur_row = cur_row.substr(0, cur_row.size() - 2);
    }
    std::cout << cur_row << std::endl;
  }
  std::cout << "---" << std:: endl;
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
  return M == other.M;
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
  #pragma omp parallel for collapse(2)

  for (int row = 0; row < n; row++) {
    for (int col = 0; col < m; col++) {
      for (int i = 0; i < k; i++) {
        C[row][col] += A[row][i] * B[i][col];
      }
    }
  }

  return Matrix(C);
}

std::optional<Matrix> Matrix::operator*(const Matrix& other) const {
  return Matrix::matmul(*this, other);
}
