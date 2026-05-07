# Rebuttal Main Results Review

This file collects the main quantitative results from the three rebuttal experiments:

- Query-level batching sweep at fixed `n_train = 32768`
- Tensor-Core ablation for Flash-SD-KDE
- Streaming vs full materialization at the large `65536 / 8192` case

Source artifacts:

- `paper/figures/table_rebuttal_query_batching_16d.md`
- `paper/figures/table_rebuttal_tensorcore_ablation_16d.md`
- `paper/figures/table_rebuttal_materialization_case_16d.md`
- `paper/figures/runtime/rebuttal_query_batching_16d.pdf`
- `paper/figures/runtime/rebuttal_tensorcore_ablation_16d.pdf`
- `paper/figures/runtime/rebuttal_materialization_case_16d.pdf`

## 1. Query-Level Batching Sweep

Setup:

- `n_train = 32768`
- `n_test in {4, 16, 64, 256, 1024, 4096, 16384}`
- Baselines: exact Torch, `torch.compile`, PyKeOps

Main takeaway:

- Flash-SD-KDE remains faster across the full query-batching range.
- Runtime rises only modestly from `1.91 ms` to `2.75 ms` as `n_test` grows by `4096x`.
- Speedups remain large throughout: `52.54x-57.54x` vs Torch, `15.69x-17.53x` vs `torch.compile`, and `6.11x-8.38x` vs PyKeOps.

| n_train | n_test | Torch (ms) | torch.compile (ms) | PyKeOps (ms) | Flash-SD-KDE (ms) | Torch/Flash | Compile/Flash | PyKeOps/Flash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 4 | 100.97 | 32.44 | 15.64 | 1.92 | 52.54x | 16.88x | 8.14x |
| 32768 | 16 | 101.00 | 32.41 | 15.43 | 1.91 | 52.78x | 16.94x | 8.06x |
| 32768 | 64 | 103.27 | 33.44 | 15.48 | 1.91 | 53.99x | 17.48x | 8.09x |
| 32768 | 256 | 104.09 | 33.68 | 16.11 | 1.92 | 54.19x | 17.53x | 8.38x |
| 32768 | 1024 | 106.22 | 33.69 | 15.81 | 1.97 | 54.04x | 17.14x | 8.04x |
| 32768 | 4096 | 116.81 | 35.97 | 15.85 | 2.11 | 55.32x | 17.04x | 7.51x |
| 32768 | 16384 | 158.18 | 43.14 | 16.79 | 2.75 | 57.54x | 15.69x | 6.11x |

## 2. Tensor-Core Ablation

Setup:

- `n_test = n_train / 8`
- `n_train in {2048, 4096, 8192, 16384, 32768, 65536}`
- Comparison: Flash-SD-KDE with Tensor Cores vs a copied no-Tensor-Core kernel path

Main takeaway:

- Tensor Cores become important once the problem is moderately large.
- At `n_train = 32768`, Tensor Cores give a `4.94x` speedup.
- At `n_train = 65536`, Tensor Cores give a `4.80x` speedup.
- The largest relative gain appears around `n_train = 32768-65536`, while the smallest sizes show little difference.

| n_train | n_test | Flash-SD-KDE Tensor Core (ms) | Flash-SD-KDE no Tensor Core (ms) | no-TC / TC speedup | max abs delta | rel-L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 256 | 0.30 | 0.31 | 1.04x | 7.516e-10 | 2.270e-02 |
| 4096 | 512 | 0.28 | 0.35 | 1.24x | 1.350e-09 | 2.420e-02 |
| 8192 | 1024 | 0.29 | 0.87 | 2.99x | 2.171e-09 | 2.585e-02 |
| 16384 | 2048 | 0.66 | 2.83 | 4.27x | 3.996e-09 | 2.787e-02 |
| 32768 | 4096 | 2.12 | 10.48 | 4.94x | 7.691e-09 | 2.983e-02 |
| 65536 | 8192 | 9.30 | 44.59 | 4.80x | 1.353e-08 | 3.243e-02 |

## 3. Streaming vs Full Materialization at 65536 / 8192

Setup:

- Single large case: `n_train = 65536`, `n_test = 8192`
- Four-way comparison:
  - Streamed Flash + Tensor Cores
  - Streamed Flash + No Tensor Cores
  - Full materialization + Tensor Cores
  - Full materialization + No Tensor Cores
- The materialized variants explicitly build the full `n_train x n_train` train kernel matrix and the `n_test x n_train` query kernel matrix.

Main takeaway:

- This gives a much clearer memory story than the earlier fused/de-fused workspace ablation.
- At the `65536 / 8192` case, the streamed Flash path is far faster and vastly more memory-efficient than full matrix materialization.
- Streamed Flash reduces peak extra allocation by about `1009x` relative to the materialized baseline.
- Streamed Flash is `55.66x` faster than materialization when Tensor Cores are enabled, and `12.47x` faster when Tensor Cores are disabled.
- Tensor Cores matter strongly within the streamed path (`4.51x` speedup), but almost not at all within the materialized path (`1.01x`), which supports the claim that the materialized baseline is dominated by matrix construction / memory cost.

### Full Metrics

| Method | Runtime (ms) | Peak Extra Alloc (MB) | Peak Extra Reserved (MB) | Explicit Materialized Kernel Peak (MB) |
| --- | ---: | ---: | ---: | ---: |
| Streamed Flash + Tensor Cores | 10.99 | 16.25 | 0.00 | 0.00 |
| Streamed Flash + No Tensor Cores | 49.52 | 16.25 | 0.00 | 0.00 |
| Full Materialization + Tensor Cores | 611.63 | 16400.50 | 16404.00 | 16384.00 |
| Full Materialization + No Tensor Cores | 617.49 | 16400.50 | 16404.00 | 16384.00 |

### Derived Comparisons

| Quantity | Value |
| --- | ---: |
| Streamed / Materialized speedup (Tensor Cores) | 55.66x |
| Streamed / Materialized speedup (No Tensor Cores) | 12.47x |
| Streamed / Materialized memory reduction (Tensor Cores) | 1009.26x |
| Streamed / Materialized memory reduction (No Tensor Cores) | 1009.26x |
| Tensor Core speedup within streamed path | 4.51x |
| Tensor Core speedup within materialized path | 1.01x |

### Correctness Checks

| Comparison | max abs delta | rel-L2 |
| --- | ---: | ---: |
| Materialized vs streamed (Tensor Cores) | 1.337e-08 | 3.248e-02 |
| Materialized vs streamed (No Tensor Cores) | 1.734e-12 | 1.445e-06 |

## Suggested One-Paragraph Summary

Across all rebuttal experiments, Flash-SD-KDE remains consistently strong on the RTX A6000. In the query-batching sweep at fixed `n_train = 32768`, Flash-SD-KDE stays between `1.91 ms` and `2.75 ms` as `n_test` ranges from `4` to `16384`, yielding `52.54x-57.54x` speedups over exact Torch, `15.69x-17.53x` over `torch.compile`, and `6.11x-8.38x` over PyKeOps. A separate Tensor-Core ablation shows that Tensor Cores account for a large share of runtime acceleration at larger scales, reaching `4.94x` speedup at `n_train = 32768` and `4.80x` at `n_train = 65536`. Finally, the single large-case streaming-vs-materialization experiment gives a much clearer memory-efficiency story: at `n_train = 65536`, `n_test = 8192`, streamed Flash is `55.66x` faster than full materialization with Tensor Cores and `12.47x` faster without them, while reducing peak extra allocation by about `1009x` in both settings, showing that the memory advantage comes from streaming rather than Tensor-Core use.
