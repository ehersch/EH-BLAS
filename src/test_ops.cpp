#include <vector>
#include <string>
#include <iostream>
#include "matrix.h"

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
}