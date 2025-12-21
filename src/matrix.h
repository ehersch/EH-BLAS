#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <optional>
#include <string>
#include <tuple>

class Matrix {
  private:
  
  public:
    std::vector<std::vector<double>> M; // use std::vector for dynamic storage

    Matrix(const std::vector<std::vector<double>>& mat);

    void print() const;
    static void print(const Matrix& mat);

    static std::optional<Matrix> matmul(const Matrix& mat_a, const Matrix& mat_b);

    bool operator==(const Matrix& other) const;

    /* 
      The operator above works without the trailing const because both operands are non-const and the compiler allows modifying 'this' if needed. 
      However, using (Matrix& other) fails since it can’t bind a const or temporary Matrix (like the one created in (*C == result)) to a non-const reference.
    */

    bool approx_equal(const Matrix& other, double rtol=1e-05, double atol=1e-08) const;

    static std::optional<Matrix> matmul_parallel(const Matrix& mat_a, const Matrix& mat_b);

    std::optional<Matrix> operator*(const Matrix& other) const;

    Matrix operator-(const Matrix& other) const;

    static std::optional<Matrix> matmul_blocked(const Matrix& mat_a, const Matrix& mat_b);

    std::string to_string() const;

    std::tuple<double, double, double> compare_times(const Matrix& other) const;

    // doesn't modify Matrix.M, so is const
    Matrix transpose() const;

    // static function
    static Matrix identity(int n);

    std::optional<Matrix> inverse() const;
};

struct LUResult {
    Matrix L;
    Matrix U;
    Matrix P;
};

LUResult LU(const Matrix& A);

#endif