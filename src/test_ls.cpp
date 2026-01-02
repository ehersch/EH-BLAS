#include "least_squares.h"
#include <iostream>

int main() {
    // Example: y = 1 + 2x
    std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {1.0, 2.0},
        {1.0, 3.0},
        {1.0, 4.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0};

    OLSRegression reg(X, y);
    auto sol = reg.solve();

    // Print weights (depends on your Matrix API)
    sol.w.print(); // or std::cout << sol.w.to_string() << "\n";

     // Example 2: y approx 1 + 2x
    std::vector<std::vector<double>> X_1 = {
        {1.0, 1.01},
        {1.1, 2.1},
        {1.0, 3.05},
        {1.0, 4.07}
    };
    std::vector<double> y_1 = {3.01, 5.2, 7.1, 9.0};

    OLSRegression reg_approx(X_1, y_1);
    auto sol_1 = reg_approx.find_l2_residual();

    std::cout << "Residual: " << sol_1 << "\n";

    return 0;
}
