#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "matrix.h"

Matrix random_matrix(int rows, int cols) {
  static std::mt19937 generator(42);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::vector<std::vector<double>> mat(rows, std::vector<double>(cols, 0));
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      mat[r][c] = distribution(generator);
    }
  }
  return Matrix(mat);
}

double bench_once(int n, int runs) {
  using clock = std::chrono::high_resolution_clock;
  Matrix A = random_matrix(n, n);
  Matrix B = random_matrix(n, n);

  // warmup
  Matrix::matmul_blocked(A, B);

  double total_ms = 0.0;
  for (int r = 0; r < runs; r++) {
    auto t1 = clock::now();
    auto C = Matrix::matmul_blocked(A, B);
    auto t2 = clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    (void)C;
  }
  return total_ms / runs;
}

double bench_variant(int n, int runs, const char* variant) {
  using clock = std::chrono::high_resolution_clock;
  Matrix A = random_matrix(n, n);
  Matrix B = random_matrix(n, n);

  if (std::string(variant) == "naive") {
    Matrix::matmul(A, B);
  } else if (std::string(variant) == "parallel") {
    Matrix::matmul_parallel(A, B);
  } else {
    Matrix::matmul_blocked(A, B);
  }

  double total_ms = 0.0;
  for (int r = 0; r < runs; r++) {
    auto t1 = clock::now();
    if (std::string(variant) == "naive") {
      Matrix::matmul(A, B);
    } else if (std::string(variant) == "parallel") {
      Matrix::matmul_parallel(A, B);
    } else {
      Matrix::matmul_blocked(A, B);
    }
    auto t2 = clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
  }
  return total_ms / runs;
}

int main() {
  const int runs = 5;
  const int sizes[] = {256, 512, 1024};

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "variant,size,avg_ms\n";
  for (int n : sizes) {
    std::cout << "naive," << n << "," << bench_variant(n, runs, "naive") << "\n";
    std::cout << "parallel," << n << "," << bench_variant(n, runs, "parallel") << "\n";
    std::cout << "blocked," << n << "," << bench_variant(n, runs, "blocked") << "\n";
  }
  return 0;
}
