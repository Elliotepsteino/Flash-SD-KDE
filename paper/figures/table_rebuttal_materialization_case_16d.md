## Full Metrics

| Method | Runtime (ms) | Peak Extra Alloc (MB) | Peak Extra Reserved (MB) | Explicit Materialized Kernel Peak (MB) |
| --- | ---: | ---: | ---: | ---: |
| Streamed Flash + Tensor Cores | 10.99 | 16.25 | 0.00 | 0.00 |
| Streamed Flash + No Tensor Cores | 49.52 | 16.25 | 0.00 | 0.00 |
| Full Materialization + Tensor Cores | 611.63 | 16400.50 | 16404.00 | 16384.00 |
| Full Materialization + No Tensor Cores | 617.49 | 16400.50 | 16404.00 | 16384.00 |

## Derived Comparisons

| Quantity | Value |
| --- | ---: |
| Streamed / Materialized speedup (Tensor Cores) | 55.66x |
| Streamed / Materialized speedup (No Tensor Cores) | 12.47x |
| Streamed / Materialized memory reduction (Tensor Cores) | 1009.26x |
| Streamed / Materialized memory reduction (No Tensor Cores) | 1009.26x |
| Tensor Core speedup within streamed path | 4.51x |
| Tensor Core speedup within materialized path | 1.01x |

## Correctness Checks

| Comparison | max abs delta | rel-L2 |
| --- | ---: | ---: |
| Materialized vs streamed (Tensor Cores) | 1.337e-08 | 3.248e-02 |
| Materialized vs streamed (No Tensor Cores) | 1.734e-12 | 1.445e-06 |

Notes:
- This experiment uses the single large case `n_train=65536`, `n_test=8192`.
- `Peak Extra Alloc` and `Peak Extra Reserved` are measured above the resident input-tensor footprint.
- The materialized variants explicitly build the full `n_train x n_train` train kernel matrix and the `n_test x n_train` query kernel matrix.
