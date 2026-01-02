#include "utils.h"
#include <vector>
#include <optional>
#include <tuple>

Matrix transpose(const Matrix& mat) {
  auto& A = mat.M;

  int n = A.size();
  int m = A[0].size();

  std::vector<std::vector<double>> B(m, std::vector<double>(n, 0));

  # pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      B[j][i] = A[i][j];
    }
  }
  return Matrix(B);
}

Matrix identity(int n) {
  // returns an nxn identity matrix
  std::vector<std::vector<double>> M(n, std::vector<double>(n, 0));

  for (int i = 0; i < n; i++) {
    M[i][i] = 1;
  }
  return Matrix(M);
}

LUResult LU(const Matrix& M) {
  // Computes the LU decomposition (partial pivoting)
  // https://www.cs.cornell.edu/courses/cs4220/2022sp/lec/2022-02-11.pdf
  std::vector<std::vector<double>> A = M.M;
  int n = A.size(); // only square matrices are invertible

  auto L = identity(n).M;
  auto P = identity(n).M;

  std::vector<std::vector<double>> U_copy = A; // this effectively takes a copy of A by assigment
  auto U = Matrix(U_copy).M;

  for (int k = 0; k < n - 1; k++) {
    int pivot = k;
    double max_val = std::abs(U[k][k]);

    for (int i = k + 1; i < n; ++i) {
      if (std::abs(U[i][k]) > max_val) {
        max_val = std::abs(U[i][k]);
        pivot = i;
      }
    }

    if (pivot != k) {
      std::swap(U[k], U[pivot]);
      std::swap(P[k], P[pivot]);

      // swap only first k columns of L
      for (int j = 0; j < k; ++j) {
        std::swap(L[k][j], L[pivot][j]);
      }
    }
    
    // perform elimination
    for (int i = k+1; i < n; i++) {
      L[i][k] = U[i][k] / U[k][k];
      for (int j = k; j < n; ++j) {
        U[i][j] -= L[i][k] * U[k][j];
      }
    }
  }

  return {Matrix(L), Matrix(U), Matrix(P)};
}

std::vector<double> forward_substitution(
    const Matrix& M,
    const std::vector<double>& b
) {
    std::vector<std::vector<double>> A = M.M;
    int n = b.size();
    std::vector<double> x(n); // allocates a vector of n 0.0s

    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
}

// Always use const T& for read-only inputs
std::vector<double> back_substitution(
  const Matrix& M,
  const std::vector<double>& b
) {
  std::vector<std::vector<double>> A = M.M;
  int n = A.size();
  std::vector<double> x(n);

  for (int i = n - 1; i >= 0; i--) {
    double sum = 0;
    for (int j = i + 1; j < n; j++) {
      sum += A[i][j] * x[j];
    }
    x[i] = (b[i] - sum) / A[i][i];
  }
  return x;
}


std::optional<Matrix> inverse(const Matrix& mat) {
  auto& A = mat.M;
  int n = A.size();

  auto lu = LU(A);
  Matrix L = lu.L;
  Matrix U = lu.U;
  Matrix P = lu.P;
  // // PA = LU
  // // A = P^-1 LU
  // // A = P^T LU since the permutation matrix is orthogonal
  // // A^-1 = U^-1 L^-1 P

  // auto& U_inv = back_substitution(U);
  // to solve for U_inv, solve for each column of U_inv
  // U * U[:,i]^-1 = e_i (the ith standard basis)
  // each of these is solved with back-substitution
  std::vector<std::vector<double>> U_inv_vec(n, std::vector<double>(n, 0));

  for (int i = 0; i < n ; i++) {
    std::vector<double> e_i(n);
    e_i[i] = 1;
    std::vector<double> u_i = back_substitution(U, e_i);
    for (int j = 0; j < n; j++) {
      U_inv_vec[j][i] = u_i[j];
    }
  }
  Matrix U_inv = Matrix(U_inv_vec);

  std::vector<std::vector<double>> L_inv_vec(n, std::vector<double>(n, 0));

  for (int i = 0; i < n ; i++) {
    std::vector<double> e_i(n);
    e_i[i] = 1;
    std::vector<double> l_i = forward_substitution(L, e_i);
    for (int j = 0; j < n; j++) {
      L_inv_vec[j][i] = l_i[j];
    }
  }
  Matrix L_inv = Matrix(L_inv_vec);
  // matmul_blocked is a static Matrix member, so it must be called as Matrix::matmul_blocked
  return Matrix::matmul_blocked(U_inv, Matrix::matmul_blocked(L_inv, P).value());
}