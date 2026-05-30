# TODO: Tiled Matmul Optimization Roadmap

This project already has the right high-level direction for fast matrix
multiplication:

- naive matmul for correctness and comparison
- OpenMP parallel matmul
- blocked/tiled matmul for cache locality

The next optimizations should focus on making the innermost blocked loop easier
for the CPU to vectorize and reducing memory-layout overhead.

## 1. Add SIMD to the inner tile loop

Current blocked matmul structure:

```cpp
for (size_t i = ii; i < i_max; i++) {
    for (size_t kk2 = kk; kk2 < k_max; kk2++) {
        double a_val = A[i][kk2];
        for (size_t j = jj; j < j_max; j++) {
            C[i][j] += a_val * B[kk2][j];
        }
    }
}
```

The innermost loop is the important part:

```cpp
C[i][j] += a_val * B[kk2][j];
```

Conceptually, this is a vector operation over a chunk of a row:

```text
C_row_chunk += scalar * B_row_chunk
```

That is closer to a scaled vector add / fused multiply-add than plain vector
addition. This is motivated by Nathan's comment about C++ `vadd` vector addition.

Try adding an OpenMP SIMD hint:

```cpp
for (size_t i = ii; i < i_max; i++) {
    for (size_t kk2 = kk; kk2 < k_max; kk2++) {
        double a_val = A[i][kk2];

        #pragma omp simd
        for (size_t j = jj; j < j_max; j++) {
            C[i][j] += a_val * B[kk2][j];
        }
    }
}
```

Then compile with strong optimization:

```bash
g++-14 -O3 -march=native -fopenmp src/run_results.cpp src/matrix.cpp -o main
```

If `-march=native` causes portability issues, keep `-O3 -fopenmp` for the
portable build and use `-march=native` only for local benchmarking.

## 2. Ask the compiler whether it vectorized

For GCC, build with vectorization reports:

```bash
g++-14 -O3 -march=native -fopenmp \
  -fopt-info-vec-optimized -fopt-info-vec-missed \
  src/run_results.cpp src/matrix.cpp -o main
```

Look for the inner `j` loop in `matmul_blocked`. The ideal result is that GCC
reports the loop was vectorized. If it says vectorization was missed, the reason
usually points to memory aliasing, layout, or control-flow issues.

## 3. Benchmark every optimization

After each change, compare:

- naive matmul
- parallel matmul
- blocked matmul
- blocked matmul with SIMD hint

Use matrix sizes large enough for optimization to matter, such as:

- 256 x 256
- 512 x 512
- 1024 x 1024

Small matrices can be noisy because thread startup and allocation overhead can
dominate the actual computation.

## 4. Avoid copying input matrices in hot paths

The naive and parallel methods currently copy the matrix storage:

```cpp
std::vector<std::vector<double>> A = mat_a.M;
std::vector<std::vector<double>> B = mat_b.M;
```

Prefer references:

```cpp
const auto& A = mat_a.M;
const auto& B = mat_b.M;
```

The blocked implementation already does this. Applying the same pattern to the
other matmul variants makes timing comparisons more honest and avoids measuring
unnecessary copies.

## 5. Tune block size experimentally

The blocked implementation uses:

```cpp
size_t block = 64;
```

That is a reasonable starting point, especially on Apple Silicon, but the best
value depends on CPU cache sizes, matrix shape, compiler, and thread count.

Try benchmarking:

- 16
- 32
- 64
- 96
- 128

Eventually, expose block size as a parameter or constant so benchmarks can sweep
over it without editing the source each time.

## 6. Consider transposing B

Naive matmul accesses `B[i][col]`, which walks down a column. In row-major
storage, that is cache-unfriendly.

Blocked matmul improves this by iterating across `B[kk2][j]`, which walks across
a row. That is already good.

For non-blocked matmul, or for alternate kernels, consider precomputing:

```cpp
B_T[col][i] = B[i][col];
```

Then dot products can read both `A[row]` and `B_T[col]` contiguously.

This may help the naive/parallel versions more than the current blocked version.

## 7. Move from vector-of-vectors to flat contiguous storage

The biggest structural improvement would be changing `Matrix` storage from:

```cpp
std::vector<std::vector<double>> M;
```

to something like:

```cpp
std::vector<double> data;
size_t rows;
size_t cols;
```

with indexing:

```cpp
data[i * cols + j]
```

Why this helps:

- all matrix data is contiguous
- fewer heap allocations
- better cache behavior
- easier SIMD vectorization
- less pointer chasing
- easier interop with BLAS-style APIs later

This is a larger refactor because constructors, Python bindings, tests, and
utility functions all expect `std::vector<std::vector<double>>`. Do this after
the smaller loop-level optimizations.

## 8. Add a dedicated microkernel later

The current tiled algorithm updates one row of `C` at a time. A more advanced
BLAS-like implementation uses a small register-blocked microkernel, for example:

- compute a 4 x 4 or 8 x 4 chunk of `C`
- keep partial sums in CPU registers
- use SIMD/FMA instructions heavily
- write back to memory once per output chunk

This is more complex, but it is the path toward serious BLAS-like performance.

Do this only after:

- correctness tests are solid
- benchmark harness is stable
- storage is flat/contiguous
- basic compiler vectorization has been measured

## Suggested order

1. Add `#pragma omp simd` to the blocked inner `j` loop.
2. Build with `-O3 -march=native -fopenmp`.
3. Check GCC vectorization reports.
4. Benchmark blocked matmul before/after.
5. Remove input copies from naive and parallel matmul.
6. Sweep block sizes.
7. Refactor storage to a flat contiguous buffer.
8. Consider register-blocked microkernels.

## Main takeaway

Do use the idea, but think of it as vectorized fused multiply-add rather than
plain vector addition. The useful operation in tiled matmul is:

```text
C tile row += A scalar * B tile row
```

That maps naturally to SIMD and is exactly where the current blocked
implementation should be optimized next.
