#ifndef LEAST_SQUARES_H
#define LEAST_SQUARES_H

#include "matrix.h"
#include "utils.h"
#include <vector>
#include <optional>
#include <string>
#include <tuple>

class OLSRegression {
  private:
    std::vector<std::vector<double>> X;
    std::vector<double> y;
  
  public:
    OLSRegression(const std::vector<std::vector<double>>& X_in, const std::vector<double>& y_in);

    struct RegressionSolution {
      Matrix w;
    };

    RegressionSolution solve() const;
};

#endif