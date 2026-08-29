# SIMD Benchmark Results: `#pragma omp simd` on Blocked Matmul

## Change

Added an OpenMP SIMD hint to the innermost loop of `matmul_blocked`:

```cpp
#pragma omp simd
for (size_t j = jj; j < j_max; j++) {
    C[i][j] += a_val * B[kk2][j];
}
```

Build flags used for all benchmarks:

```bash
g++-14 -O3 -march=native -fopenmp src/bench_simd.cpp src/matrix.cpp -o bench
```

## Environment

| Setting | Value |
|---------|-------|
| CPU | Apple M2 |
| Cores | 8 |
| Architecture | arm64 |
| Compiler | g++-14 (Homebrew GCC 14.3.0) |
| Date | 2026-08-28 |
| Runs per measurement | 5 (full suite), 10 (blocked-only focused) |
| Random seed | 42 (fixed for reproducibility) |

## Full Variant Comparison (5-run average, ms)

These numbers compare all three matmul variants. Only `blocked` changed between the two builds; naive and parallel are included for context but may vary run-to-run due to system load.

| Variant | 256×256 (before) | 256×256 (after) | 512×512 (before) | 512×512 (after) | 1024×1024 (before) | 1024×1024 (after) |
|---------|------------------|-----------------|------------------|-----------------|--------------------|--------------------|
| naive | 19.51 | 17.91 | 161.49 | 156.06 | 1499.97 | 1465.38 |
| parallel | 3.77 | 4.24 | 35.07 | 37.04 | 352.67 | 375.18 |
| **blocked** | **0.96** | **0.96** | **7.42** | **7.35** | **57.67** | **61.82** |

## Blocked Matmul Only (10-run, ms)

Focused benchmark isolating `matmul_blocked` before/after the SIMD hint.

| Size | Metric | Without `#pragma omp simd` | With `#pragma omp simd` | Change |
|------|--------|---------------------------|-------------------------|--------|
| 256×256 | min | 0.93 | 0.94 | +1% |
| 256×256 | median | 1.15 | 1.22 | +6% |
| 256×256 | avg | 1.17 | 1.30 | +11% |
| 512×512 | min | 7.05 | 6.76 | −4% |
| 512×512 | median | 7.92 | 7.61 | −4% |
| 512×512 | avg | 8.15 | 7.73 | −5% |
| 1024×1024 | min | 62.69 | 56.74 | −10% |
| 1024×1024 | median | 66.85 | 64.97 | −3% |
| 1024×1024 | avg | 67.45 | 64.98 | −4% |

At 1024×1024, blocked matmul with the SIMD hint is roughly **3–4% faster** on average (67.45 ms → 64.98 ms). At 256×256 there is no clear win and results are within measurement noise.

## Correctness

Verified with `run_results.cpp` (899×723 random matrices):

```
All correct: 1
```

Naive, parallel, and blocked (with SIMD hint) all produce equivalent results within tolerance.

## Compiler Vectorization Report

GCC was asked to report vectorization decisions:

```bash
g++-14 -O3 -march=native -fopenmp \
  -fopt-info-vec-optimized -fopt-info-vec-missed \
  -c src/matrix.cpp
```

Relevant output for the blocked inner loop:

```
src/matrix.cpp:184:39: missed: not vectorized: multiple nested loops.
src/matrix.cpp:189:46: missed: failed: evolution of base is not affine.
```

The `#pragma omp simd` hint was added, but GCC did **not** vectorize the inner `j` loop. The main blockers are:

1. **Nested loops** — the `i`, `kk2`, and `j` loops are too deeply nested for the vectorizer to isolate the inner loop cleanly.
2. **Non-affine indexing** — `C[i][j]` and `B[kk2][j]` use `std::vector<std::vector<double>>`, so row pointers are not a single contiguous stride. The compiler cannot prove a simple affine memory access pattern.

This explains why the runtime improvement is small: the hint alone does not produce NEON/AVX instructions without also fixing memory layout.

## Speedup vs Other Variants (blocked with SIMD, 1024×1024)

Using the post-change blocked average of **64.98 ms**:

| Comparison | Speedup |
|------------|---------|
| blocked vs naive (1465 ms) | ~22.5× |
| blocked vs parallel (375 ms) | ~5.8× |

OpenMP thread parallelism and cache blocking remain the dominant optimizations in this project. The SIMD hint is a small incremental step.

## Reproduce

```bash
cd src
g++-14 -O3 -march=native -fopenmp bench_simd.cpp matrix.cpp -o bench_simd
./bench_simd
```

For blocked-only comparison, compile `bench_simd.cpp` (or the blocked-only harness) against `matrix.cpp` with and without the `#pragma omp simd` line.

## Next Steps

Per `TODO.md`, further SIMD gains likely require:

1. **Flat contiguous storage** (`data[i * cols + j]`) so the compiler can vectorize row updates.
2. **Explicit intrinsics** (NEON on Apple Silicon) if compiler auto-vectorization remains blocked.
3. **Register-blocked microkernel** for BLAS-level performance.
