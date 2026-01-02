#include <vector>
#include <string>
#include <iostream>
#include "matrix.h"
#include "utils.h"
#include <random>
#include <ctime>

double max_abs_entry(const Matrix& M) {
  double maxv = 0.0;
  for (const auto& row : M.M)
    for (double v : row)
      maxv = std::max(maxv, std::abs(v));
  return maxv;
}

Matrix random_matrix(int n) {
  static std::mt19937 generator(std::random_device{}());  // seed ONCE per call
  // https://en.cppreference.com/w/cpp/numeric/random/random_device.html
  // https://cplusplus.com/reference/random/mt19937/
  std::normal_distribution<double> distribution(0,1);
  std::vector<std::vector<double>> mat(n, std::vector<double>(n, 0));

  for (int r = 0; r < n; r++) {
    for (int c = 0; c < n; c++) {
      mat[r][c] = distribution(generator);
    }
  }

  return Matrix(mat);
}

void test_identity(int n) {
  Matrix A = random_matrix(n);
  Matrix A_inv = inverse(A).value();
  Matrix I = identity(A.M.size());
  Matrix I_approx = (A * A_inv).value();
  std::cout << "Inverse of size " << n << " correct: " << std::boolalpha << 
  I.approx_equal(I_approx) << std::endl;

  Matrix I_approx_2 = (A * A_inv).value();
  Matrix err = I_approx_2 - identity(n);

  std::cout << "max |I - A*A_inv| = "
            << max_abs_entry(err) << std::endl;
}

int main() {
  std::vector<std::vector<double>> A = {{1,2},{3,4}};

  Matrix mat_A = Matrix(A);
  Matrix B = transpose(mat_A);
  std::cout << B.to_string() << std::endl;

  auto lu = LU(mat_A);
    
  lu.L.print();
  lu.U.print();
  lu.P.print();

  auto PA = (lu.P * mat_A).value();
  auto LU_mat = (lu.L * lu.U).value();

  bool is_accurate = PA.approx_equal(LU_mat);
  std::cout << "LU decomposition correct: " << std::boolalpha << is_accurate << std::endl;

  Matrix A_inv = inverse(mat_A).value();
  A_inv.print();
  Matrix I = identity(2);
  Matrix I_approx = (mat_A * A_inv).value();
  std::cout << "Inverse correct: " << std::boolalpha << 
  I.approx_equal(I_approx) << std::endl;

  test_identity(5);
}