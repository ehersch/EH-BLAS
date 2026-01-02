#include "least_squares.h"
#include <iostream>

int main() {
    // Example: y ≈ 1 + 2x
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

    return 0;
}
