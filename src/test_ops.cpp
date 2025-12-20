#include <vector>
#include <string>
#include <iostream>
#include "matrix.h"
#include <random>

Matrix random_matrix(int n) {
  std::default_random_engine generator;
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
  Matrix A_inv = A.inverse().value();
  Matrix I = Matrix::identity(A.M.size());
  Matrix I_approx = (A * A_inv).value();
  std::cout << "Inverse of size " << n << " correct: " << std::boolalpha << 
  I.approx_equal(I_approx) << std::endl;
}

int main() {
  std::vector<std::vector<double>> A = {{1,2},{3,4}};

  Matrix mat_A = Matrix(A);
  Matrix B = mat_A.transpose();
  std::cout << B.to_string() << std::endl;

  auto lu = LU(mat_A);
    
  lu.L.print();
  lu.U.print();
  lu.P.print();

  auto PA = (lu.P * mat_A).value();
  auto LU_mat = (lu.L * lu.U).value();

  bool is_accurate = PA.approx_equal(LU_mat);
  std::cout << "LU decomposition correct: " << std::boolalpha << is_accurate << std::endl;

  Matrix A_inv = mat_A.inverse().value();
  A_inv.print();
  Matrix I = Matrix::identity(2);
  Matrix I_approx = (mat_A * A_inv).value();
  std::cout << "Inverse correct: " << std::boolalpha << 
  I.approx_equal(I_approx) << std::endl;

  test_identity(100);
}