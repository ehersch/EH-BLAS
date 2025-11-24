#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <optional>

class Matrix {
  private:
    std::vector<std::vector<double>> M; // use std::vector for dynamic 2D storage
  
  public:
    Matrix(const std::vector<std::vector<double>>& mat);

    void print() const;
    static void print(const Matrix& mat);

    static std::optional<Matrix> matmul(const Matrix& mat_a, const Matrix& mat_b);

    bool operator==(const Matrix& other) const;

    /* 
      The operator above works without the trailing const because both operands are non-const and the compiler allows modifying 'this' if needed. 
      However, using (Matrix& other) fails since it can’t bind a const or temporary Matrix (like the one created in (*C == result)) to a non-const reference.
    */

    static std::optional<Matrix> matmul_parallel(const Matrix& mat_a, const Matrix& mat_b);


    std::optional<Matrix> operator*(const Matrix& other) const;
};

#endif