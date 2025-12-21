# Numerical & Parallel Debugging Notes (Matrix Library)

This document explains several subtle bugs that were encountered while
implementing matrix multiplication, LU decomposition, and matrix inversion.
These issues were **not obvious** during early testing and only surfaced
at larger matrix sizes or under parallel execution.

The goal of this file is to explain **why these bugs were easy to miss**
and **how to avoid them in the future**.

---

## 1. Why These Bugs Were Hard to Catch

### 1.1 Small Tests Hide Big Problems

Most early tests used:

- small matrices (2×2, 5×5)
- deterministic inputs
- low thread counts

For small sizes:

- race conditions are less likely to trigger
- partial sums are small
- floating-point error does not amplify

As a result, **broken parallel code can appear correct**.

---

### 1.2 Floating-Point Noise Can Mask Real Bugs

When numerical code is wrong, developers often assume:

> “This is just floating-point error.”

That assumption is sometimes correct — but **sometimes it hides real bugs**.

Key warning sign:

- error is **large** (O(1))
- error **changes between runs**

This indicates **data races or memory corruption**, not numerical instability.

---

### 1.3 Deterministic Code Is Easier to Trust

Once race conditions were removed:

- results became deterministic
- numerical error stabilized (~5e-14)
- correctness became verifiable

**Non-determinism is a red flag in numerical code.**

---

## 2. Race Condition in Parallel Matrix Multiplication

### 2.1 The Core Issue

In matrix multiplication, many loops perform **accumulation**:

```cpp
C[i][j] += A[i][k] * B[k][j];
```

### 2.2 Why `collapse(3)` Was Incorrect

`collapse(n)` is only safe when:

- Each iteration writes to disjoint memory
- In blocked matrix multiplication:
- different `k` blocks contribute to the same `C[i][j]` parallelizing over k causes data races

This led to:

- wildly varying results between runs
- errors on the order of 1–10 (not floating-point noise)

### 2.3 Correct Parallelization Strategy

Parallelize only over output tiles, never over accumulation:

```cpp
#pragma omp parallel for collapse(2)
for (ii)
  for (jj)
    for (kk)   // must remain serial
      C += ...
```

This ensures:

- each (ii, jj) tile of C is owned by one thread
- accumulation is deterministic
- no synchronization is required

---

## 3. Race Condition in Naive Parallel MatMul

This code was also incorrect:

```cpp
#pragma omp parallel for collapse(3)
for (row)
  for (i)
    for (col)
      C[row][col] += ...
```

Problem:

- `i` is the accumulation index
- parallelizing over it causes races on `C[row][col]`

Correct version:

```cpp
#pragma omp parallel for collapse(2)
for (row)
  for (col) {
    double sum = 0;
    for (i)
      sum += ...
    C[row][col] = sum;
  }
```

---

## 4. Transpose Bug (Indexing Error)

4.1 The Bug
`B[i][j] = A[j][i];   // incorrect`

This:

- swaps indices incorrectly
- causes out-of-bounds access for non-square matrices
- silently corrupts memory

### 4.2 Correct Transpose

`B[j][i] = A[i][j];`

---

## 5. Why the Inverse Looked “Numerically Wrong”

Initially:

`max |I - A*A⁻¹| was ~1–10` results varied between runs

This suggested numerical instability — but was actually:

- broken matrix multiplication
- data races
- incorrect accumulation
