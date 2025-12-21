# Why `approx_equal` Uses Absolute + Relative Tolerance

This document explains **why floating-point equality cannot be tested using a
single absolute tolerance**, and why this codebase uses a combination of
**absolute and relative tolerance** in `Matrix::approx_equal`.

This explanation came directly out of debugging matrix multiplication,
LU decomposition, and matrix inversion, where mathematically correct
algorithms initially appeared “wrong” due to how floating-point error works.

---

## The Problem We Are Solving

We want to answer a deceptively simple question:

> Are two floating-point numbers `a` and `b` effectively equal, given rounding
> error?

A naive approach is: `|a - b| <= tol`

This looks reasonable — but it is **incorrect** for floating-point arithmetic.

---

## Why Absolute Tolerance Alone Is Wrong

Floating-point numbers do **not** have uniform absolute precision.
Their error scales with the **magnitude of the number**.

Using a single absolute tolerance leads to both false failures and false passes,
depending on scale.

---

### Case 1: Values Near Zero (Too Strict)

```
a = 1e-12
b = 0
|a - b| = 1e-12
```

This difference is just rounding noise.  
However, with `tol = 1e-13`, an absolute-only check fails.

**Problem:**  
Absolute tolerance over-penalizes small values.

This matters because many matrix entries (e.g., off-diagonals) are expected to
be zero up to numerical noise.

---

### Case 2: Large Values (Still Too Strict)

```
a = 1e8
b = 1e8 + 1
|a - b| = 1
```

Here:

- absolute error = 1
- relative error = 1e-8

Numerically, these values are extremely close.

**Problem:**  
Absolute tolerance ignores scale.

---

### Case 3: Mixed Scales (Matrices)

In matrix algorithms:

- diagonal entries may be ~1
- off-diagonal entries should be ~0
- intermediate values can be very large or very small

A single absolute tolerance cannot correctly judge all entries at once.

---

## Floating-Point Error Is Relative

IEEE double-precision floating point guarantees approximately:

> **16 decimal digits of relative precision**

This means:

- larger values naturally incur larger absolute error
- smaller values require absolute tolerance
- correctness depends on _relative scale_, not fixed thresholds

So the real question is:

> Is the difference small **relative to the magnitude of the values involved**?

---

## The Correct Comparison Rule

To handle all regimes correctly, we use:

`|a - b| <= tol × max(1.0, |a|, |b|)`

This combines **absolute tolerance** and **relative tolerance** in a single,
stable rule.

---

## How This Rule Works

### Near Zero → Absolute Tolerance

If both `a` and `b` are small:

`max(1.0, |a|, |b|) = 1.0`

The condition reduces to:

`|a - b| <= tol`

This:

- avoids dividing by tiny numbers
- prevents exploding relative error
- correctly handles values that should be zero

This behavior is critical for:

- off-diagonal matrix entries
- residuals
- sparsity patterns

---

### Large Values → Relative Tolerance

If either value is large:

`allowed error ≈ tol × magnitude `

This:

- allows small relative error
- tolerates larger absolute differences for large values
- matches how floating-point hardware behaves

---

## Why This Matters for Matrix Algorithms

Matrix multiplication, LU decomposition, and inversion involve:

- O(n³) floating-point operations
- accumulation of rounding error
- results whose scale varies by entry

Using absolute tolerance alone causes:

- false failures for correct results
- confusion between numerical noise and real bugs

Using absolute + relative tolerance:

- adapts per entry
- matches numerical reality
- enables meaningful correctness checks

---

## Why This Fixed the Inverse Test

When checking:

`I ≈ A × A⁻¹`

we expect:

- diagonal entries ≈ 1 ± ε
- off-diagonal entries ≈ 0 ± ε

With absolute-only tolerance:

- tiny rounding noise caused failures
- results appeared “numerically wrong”

With combined tolerance:

- diagonal entries are checked relatively
- off-diagonal entries are checked absolutely
- the comparison matches mathematical intent

After fixing unrelated race conditions in matrix multiplication, the residual
became:

`max |I − A*A⁻¹| ≈ 5e-14`

This is an **excellent result** for:

- LU factorization
- explicit matrix inversion
- n = 100

---

## This Is Standard Practice

This approach is not ad-hoc. It is used by:

- NumPy (`allclose`)
- SciPy
- LAPACK test routines
- Eigen
- MATLAB

For example, NumPy uses:

`|a - b| <= atol + rtol × |b|`

---

## How to Interpret Results

| Behavior                        | Interpretation                |
| ------------------------------- | ----------------------------- |
| Same small error every run      | Correct                       |
| Error ~1e-14 to 1e-12           | Excellent                     |
| Error ~1e-6 to 1e-4             | Acceptable for large matrices |
| Error ~1 or varies between runs | Bug (often a race condition)  |

---

## One-Sentence Takeaway

Floating-point error scales with magnitude, so numerical equality must be checked
using a combination of absolute tolerance (near zero) and relative tolerance
(for large values), not a single absolute threshold.

---

**Bottom line:**  
This comparison reflects how floating-point arithmetic actually works and is
required for any serious numerical linear algebra code.
