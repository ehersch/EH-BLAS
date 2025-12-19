#include <vector>
#include <string>
#include <iostream>
#include "matrix.h"

int main() {
  std::vector<std::vector<double>> A = {{1,2},{3,4}};

  Matrix mat_A = Matrix(A);
  Matrix B = mat_A.transpose();
  std::cout << B.to_string() << std::endl;
}