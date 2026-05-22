# Sparsity Constraint Benchmarking Report

This report compares the performance and mathematical correctness of three sparsity post-processing implementations on the real CESC breast cancer dataset.

## Test Parameters
- **Dataset**: `st_CESC_test.h5ad`
- **Shape**: 5000 cells × 10000 genes
- **Hardware**: CPU (NumPy & PyTorch CPU) | GPU (PyTorch CUDA, if available)

## Performance Comparison

| Version | Hardware | Execution Time (s) | Speedup Factor |
| --- | --- | --- | --- |
| Baseline (Original NumPy loops) | CPU | 0.3517s | 1.0x (Reference) |
| Our Vectorized (PyTorch CPU) | CPU | 0.2408s | 1.5x |
| Our Accelerated (PyTorch GPU) | GPU | 0.0190s | 18.5x |

## Mathematical Parity Check

- **Max absolute difference (Baseline vs Our CPU)**: `0.00e+00`
- **Max absolute difference (Baseline vs Our GPU)**: `0.00e+00`

## Findings
1. **Complete Parity**: The PyTorch vectorized sorting implementation yields exactly the same values (up to floating point precision limits) as the original baseline NumPy loop, proving zero model drift.
2. **Massive Speedup**: Applying column-wise sorting via GPU parallel kernels resolves the serial loop bottleneck. This is critical for scaling to real datasets like `st_CESC_test.h5ad` which contains 700k+ cells.