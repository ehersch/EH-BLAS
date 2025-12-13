# Matrix multiplication, inversion, factorizations from scratch in C++

They might be incredibly simple, and the most basic ML models (if you could call them that), but regression can be incredibly powerful. We've all solved cool problems with regression and may understand the theory, but it's a strong exercise to implement this from scratch, using a lower-level language like C++ for proper performance.

I have been a user of SciPy, NumPy, and SKLearn, but I will make sure I truly understand the implementations of basic features in these packages. In this project, I begin by implementing matrix multiplication (using a naive approach, then leveraging optimizations and showing the performance improvements). Then, I will implement other basic matrix functions, such as funding the inverse of a matrix or performing the QR factorization, which will be useful for least squares problems.

I will also implement gradient descent. These tools will come together to solve basic linear regression using OLS regression, a fundamental statistical approach using MLE (also MAP), and finally generalizing for any least squares problems.

First start with matrix multiplication.

## \_\_\_

## To have real parallelism

Must compile with `g++-14 -o3 -fopenmp main.cpp matrix.cpp -o main`

## To Compile with Boost.Python Integration

Use the following command to build a Python extension module (`mult_demo.so`):

```bash
/opt/homebrew/bin/g++-14 -O3 -fopenmp -shared -fPIC \
    boost_interface.cpp matrix.cpp \
    -I/opt/homebrew/include \
    -I/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Headers \
    -L/opt/homebrew/lib \
    -L/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib \
    -lboost_python314 \
    -lpython3.14 \
    -o boost_interface.so
```

The following command actually works

```
g++-14 -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  emath.cpp matrix.cpp \
  -Wl,-undefined,dynamic_lookup \
  -o emath$(python3-config --extension-suffix)
```

To test, then run `python3` in terminal, then

```
import example
example.test()
```

## References

https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa

https://math.stackexchange.com/questions/3070297/what-is-the-best-algorithm-to-find-the-inverse-of-matrix-a

https://medium.com/@cj.ptsz/parallelized-blocked-matrix-multiplication-using-openmp-97a4bc620a47

https://www.boost.org/doc/libs/1_59_0/libs/python/doc/tutorial/doc/html/index.html#python.quickstart

CONSIDER THIS: https://github.com/pybind/pybind11
