#include "utils.h"
#include <vector>
#include <optional>
#include <tuple>
#include <cmath>
#include <algorithm>

static void swap_rows(Matrix& mat, size_t r1, size_t r2) {
  for (size_t j = 0; j < mat.cols; j++) {
    std::swap(mat.at(r1, j), mat.at(r2, j));
  }
}

Matrix transpose(const Matrix& mat) {
  return mat.transpose();
}

Matrix identity(int n) {
  Matrix M(n, n);
  for (int i = 0; i < n; i++) {
    M.at(i, i) = 1.0;
  }
  return M;
}

LUResult LU(const Matrix& M) {
  int n = static_cast<int>(M.rows);

  Matrix L = identity(n);
  Matrix P = identity(n);
  Matrix U = M;

  for (int k = 0; k < n - 1; k++) {
    int pivot = k;
    double max_val = std::abs(U.at(k, k));

    for (int i = k + 1; i < n; ++i) {
      if (std::abs(U.at(i, k)) > max_val) {
        max_val = std::abs(U.at(i, k));
        pivot = i;
      }
    }

    if (pivot != k) {
      swap_rows(U, k, pivot);
      swap_rows(P, k, pivot);

      for (int j = 0; j < k; ++j) {
        std::swap(L.at(k, j), L.at(pivot, j));
      }
    }

    for (int i = k + 1; i < n; i++) {
      L.at(i, k) = U.at(i, k) / U.at(k, k);
      for (int j = k; j < n; ++j) {
        U.at(i, j) -= L.at(i, k) * U.at(k, j);
      }
    }
  }

  return {L, U, P};
}

std::vector<double> forward_substitution(
    const Matrix& M,
    const std::vector<double>& b
) {
    int n = static_cast<int>(b.size());
    std::vector<double> x(n);

    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += M.at(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / M.at(i, i);
    }
    return x;
}

std::vector<double> back_substitution(
  const Matrix& M,
  const std::vector<double>& b
) {
  int n = static_cast<int>(M.rows);
  std::vector<double> x(n);

  for (int i = n - 1; i >= 0; i--) {
    double sum = 0;
    for (int j = i + 1; j < n; j++) {
      sum += M.at(i, j) * x[j];
    }
    x[i] = (b[i] - sum) / M.at(i, i);
  }
  return x;
}

std::optional<Matrix> inverse(const Matrix& mat) {
  int n = static_cast<int>(mat.rows);

  auto lu = LU(mat);
  Matrix L = lu.L;
  Matrix U = lu.U;
  Matrix P = lu.P;

  Matrix U_inv(n, n);

  for (int i = 0; i < n; i++) {
    std::vector<double> e_i(n);
    e_i[i] = 1;
    std::vector<double> u_i = back_substitution(U, e_i);
    for (int j = 0; j < n; j++) {
      U_inv.at(j, i) = u_i[j];
    }
  }

  Matrix L_inv(n, n);

  for (int i = 0; i < n; i++) {
    std::vector<double> e_i(n);
    e_i[i] = 1;
    std::vector<double> l_i = forward_substitution(L, e_i);
    for (int j = 0; j < n; j++) {
      L_inv.at(j, i) = l_i[j];
    }
  }

  return Matrix::matmul_blocked(U_inv, Matrix::matmul_blocked(L_inv, P).value());
}
