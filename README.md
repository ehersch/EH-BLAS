# Matrix multiplication, inversion, factorizations from scratch in C++

This project recreates core linear algebra routines—matrix multiplication, inversion, QR factorization, and gradient-based solvers—to understand the optimizations that make libraries like NumPy or SciPy efficient. The implementations power both command-line experiments and Python bindings, enabling end-to-end regression workflows (OLS, MAP, least squares) without external dependencies.

Check out my corresponding Medium post here! https://medium.com/@herschethan/c-matrix-multiplication-optimization-e9818ac2593e

## Project overview

- Start from a naive matrix multiply and progressively add cache-friendly blocking and OpenMP parallelism to quantify the performance impact of each optimization.
- Extend the codebase with matrix inverse, QR factorization, and gradient descent so least-squares problems can be solved purely with in-house routines.
- Surface the functionality to Python via pybind11 for interactive experimentation.

## Build the native binaries

All commands assume Homebrew’s `g++-14` is available.

### Enable OpenMP parallelism

```bash
g++-14 -O3 -fopenmp main.cpp matrix.cpp -o main
```

Running the binary produced above executes the CLI-backed experiments with true multi-threaded matrix multiplication.

## Python bindings

### pybind11 (recommended)

```bash
g++-14 -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  emath.cpp matrix.cpp \
  -Wl,-undefined,dynamic_lookup \
  -o emath$(python3-config --extension-suffix)
```

## Validate the Python module

```python
import emath
```

If the import succeeds without errors, the build finished correctly.

## References

- https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa
- https://math.stackexchange.com/questions/3070297/what-is-the-best-algorithm-to-find-the-inverse-of-matrix-a
- https://medium.com/@cj.ptsz/parallelized-blocked-matrix-multiplication-using-openmp-97a4bc620a47
- https://www.boost.org/doc/libs/1_59_0/libs/python/doc/tutorial/doc/html/index.html#python.quickstart
- https://github.com/pybind/pybind11

## Some C++ fundamentals with call by reference

I use call by reference a lot `(const Matrix& M)` for example. Here is the rule of thumb:

`T` -> copy
`T&` -> same object (mutable)
`const T&` -> same object (read-only)
`T*` -> same object, may be null

Essentially, Calling `T` itself will make a copy. If we just want to read (and not alter values), we can pass in a reference (the actual object and not a copy) with `T&`. The `const` part is a promise enforced by the compiler: “This function will not modify this object.” If the function tries to modify it, the compiler errors out.

This sums it up well: https://www.geeksforgeeks.org/cpp/cpp-functions-pass-by-reference/
