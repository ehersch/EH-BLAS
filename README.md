# Matrix multiplication, inversion, factorizations from scratch in C++

They might be incredibly simple, and the most basic ML models (if you could call them that), but regression can be incredibly powerful. We've all solved cool problems with regression and may understand the theory, but it's a strong exercise to implement this from scratch, using a lower-level language like C++ for proper performance.

I have been a user of SciPy, NumPy, and SKLearn, but I will make sure I truly understand the implementations of basic features in these packages. In this project, I begin by implementing matrix multiplication (using a naive approach, then leveraging optimizations and showing the performance improvements). Then, I will implement other basic matrix functions, such as funding the inverse of a matrix or performing the QR factorization, which will be useful for least squares problems.

I will also implement gradient descent. These tools will come together to solve basic linear regression using OLS regression, a fundamental statistical approach using MLE (also MAP), and finally generalizing for any least squares problems.

First start with matrix multiplication.

## \_\_\_

## To have real parallelism

Must compile with '''g++-14 -fopenmp main.cpp matrix.cpp -o main'''


## References

https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa

https://math.stackexchange.com/questions/3070297/what-is-the-best-algorithm-to-find-the-inverse-of-matrix-a