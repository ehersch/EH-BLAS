#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "matrix.h"
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(emath, m) {
    m.doc() = "pybind11 matrix wrapper";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<const std::vector<std::vector<double>>&>(),
         "Construct Matrix from list of lists")
        
         // The explicit static_cast disambiguates the overloaded Matrix::print method so pybind11 knows exactly which function signature to bind.
        .def("print",
            static_cast<void (Matrix::*)() const>(&Matrix::print),
            "Print matrix object")

        .def("__str__",
            &Matrix::to_string,
            "String representation")

        .def("__repr__",
            &Matrix::to_string,
            "String representation")

        // Static method: Matrix.matmul(a, b)
        .def(
            "matmul_naive",
            &Matrix::matmul,
            "Naive matrix multiplication"
        )

        .def(
            "matmul_parallel",
            &Matrix::matmul_parallel,
            "Parallel matrix multiplication"
        )

        .def(
            "matmul_blocked",
            &Matrix::matmul_blocked,
            "Blocked matrix multiplication"
        )

        .def("approx_equal",
            &Matrix::approx_equal,
            py::arg("other"),
            py::arg("tol") = 1e-6,
            "Approximately equal."
        )

        .def(
            "__eq__",
            // lambda function
            [](const Matrix &A, const Matrix &B) {
                return A == B;
            },
            py::is_operator()
        )

        .def("__matmul__",
            &Matrix::matmul_blocked,
            "Blocked matrix multiplication"
        )

        .def("compare_times",
            &Matrix::compare_times,
            "Compares runtime of naive vs blocked matmul."
        )

        .def("transpose", &Matrix::transpose, "Transpose matrix");
}