#include "least_squares.h"

#include <vector>
#include <optional>
#include <string>
#include <tuple>


OLSRegression::OLSRegression(const std::vector<std::vector<double>>& X_in, const std::vector<double>& y_in)
  : X(X_in), y(y_in) {}

OLSRegression::RegressionSolution OLSRegression::solve() const {
  Matrix X_mat(X);

  // Build y as an (n x 1) column matrix
  std::vector<std::vector<double>> y_col(y.size(), std::vector<double>(1));
  for (size_t i = 0; i < y.size(); ++i) y_col[i][0] = y[i];
  Matrix y_mat(y_col); // (n x 1)

  Matrix Xt = transpose(X_mat);            // (d x n)
  Matrix XtX = (Xt * X_mat).value();                 // (d x d)
  Matrix XtX_inv = inverse(XtX).value();           // (d x d)
  Matrix w = ((XtX_inv * Xt).value() * y_mat).value();         // (d x 1)

  return RegressionSolution{w};
}