#include <iostream>
#include <vector>
#include "matrix.h"
#include <cassert>
#include <chrono>
#include <random>
#include <ctime>

Matrix random_matrix(int rows, int cols) {
  static std::mt19937 generator(std::random_device{}());  // seed ONCE per call
  std::normal_distribution<double> distribution(0,1);
  std::vector<std::vector<double>> mat(rows, std::vector<double>(cols, 0));

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      mat[r][c] = distribution(generator);
    }
  }

  return Matrix(mat);
}

int main() {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  // blocking works with various dims
  int N = 899;
  int M = 723;

  Matrix A = random_matrix(N,N);
  Matrix B = random_matrix(N,N);

  // Matrix A(a);
  // //A.print(); // dynamic printing of matrix

  // Matrix B(b);
  //Matrix::print(B); // static printing of matrix

  // Find the product of A@B

  auto t1 = high_resolution_clock::now();
  auto C = Matrix::matmul(A, B); // have to use auto bc could be error
  auto t2 = high_resolution_clock::now();

  //other alternative
  Matrix C_1 = Matrix::matmul_parallel(A, B).value();
  auto t3 = high_resolution_clock::now();

  duration<double, std::milli> basic_time = t2 - t1;
  duration<double, std::milli> parallel_time = t3 - t2;

  //With bloxking
  Matrix C_2 = Matrix::matmul_blocked(A, B).value();
  auto t4 = high_resolution_clock::now();

  duration<double, std::milli> blocked_time = t4 - t3;
  //if (C) Matrix::print(*C); // must use dereference C to get value

  //Matrix::print(C_1);

  //std::vector<std::vector<double>> result = {{33, 36, 39}, {114, 126, 138}};

  // The use of == below includes an implicit conversion to pass Matrix(result)
  //bool res_bool = (*C == result);

  //std::cout << std::boolalpha << "C result: " << res_bool << std::endl;

  //std::cout << std::boolalpha << "C result: " <<  (C_1 == result) << std::endl;
  //A @ B; // alternate support of matrix multiplication

  std::cout << "Basic count: " << basic_time.count() << "ms\n";
  std::cout << "Parallel time: " << parallel_time.count() << "ms\n";
  std::cout << "Blocked time: " << blocked_time.count() << "ms\n";

  bool res_bool = (C_1 == C.value() and C_2 == C.value());

  std::cout << "All correct: " << res_bool << std::endl;

  return 0;
}