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
  size_t n = A.size();
  size_t k = A[0].size();

  if (B.size() != k) {
    throw std::runtime_error("Matrix dimensions do not match.");
  }

  size_t m = B[0].size();
  std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));

  //omp_set_num_threads(8);
  #pragma omp parallel for collapse(3)

  // change the order of i and col to increase hit rate
  // i.e. we reuse A[row][i] so keep it cached
  for (size_t row = 0; row < n; row++) {
    for (size_t i = 0; i < k; i++) {
      for (size_t col = 0; col < m; col++) {
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

    size_t n = A.size();
    size_t k = A[0].size();

    if (B.size() != k)
        throw std::runtime_error("Matrix dimensions do not match.");

    size_t m = B[0].size();

    size_t block = 64;  // good for Apple M-series CPUs

    std::vector<std::vector<double>> C(n, std::vector<double>(m, 0));

    #pragma omp parallel for collapse(3) schedule(static)

    for (size_t ii = 0; ii < n; ii += block) {
        for (size_t kk = 0; kk < k; kk += block) {
            for (size_t jj = 0; jj < m; jj += block) {

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

Matrix Matrix::transpose() const {
  auto& A = this->M;

  int n = A.size();
  int m = A[0].size();

  std::vector<std::vector<double>> B(m, std::vector<double>(n, 0));

  # pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      B[i][j] = A[j][i];
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
    const Matrix& L,
    const std::vector<double>& b
) {
    int n = b.size();
    std::vector<double> y(n); // allocates a vector of n 0.0s

    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= L.M[i][j] * y[j];
        }
        // no division: L(i,i) = 1
    }
    return y;
}


Matrix back_substitution(const Matrix& M) {
  std::vector<std::vector<double>> A = M.M;
  int n = A.size();
  return Matrix(A);
}


Matrix Matrix::inverse() const {
  auto& A = (*this).M;

  // (L, U, P) = LU(A);
  // // PA = LU
  // // A = P^-1 LU
  // // A = P^T LU since the permutation matrix is orthogonal
  // // A^-1 = U^-1 L^-1 P

  // auto& U_inv = back_substitution(U);
  // auto& L_inv = forward_substitution(L);

  // return matmul_blocked(U_inv, matmul_blocked(_inv, P));

  return Matrix(A);
}

Matrix inverse(const Matrix& A) {
  return A.inverse();
}