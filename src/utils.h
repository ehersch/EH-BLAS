#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <optional>
#include <string>
#include <tuple>
#include "matrix.h"

Matrix transpose(const Matrix& mat);

// static function
Matrix identity(int n);

struct LUResult {
    Matrix L;
    Matrix U;
    Matrix P;
};

LUResult LU(const Matrix& A);

std::vector<double> forward_substitution(
    const Matrix& M,
    const std::vector<double>& b
);

std::vector<double> back_substitution(
  const Matrix& M,
  const std::vector<double>& b
);

std::optional<Matrix> inverse(const Matrix& mat);

#endif