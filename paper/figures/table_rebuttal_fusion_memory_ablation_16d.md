## Full Metrics

| n_train | n_test | Method | Runtime (ms) | Peak Extra Alloc (MB) | Peak Extra Reserved (MB) | Explicit Workspace Peak (MB) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | 256 | Fused + Tensor Cores | 0.42 | 0.51 | 0.00 | 0.00 |
| 2048 | 256 | Fused + No Tensor Cores | 0.35 | 0.51 | 0.00 | 0.00 |
| 2048 | 256 | De-fused + Tensor Cores | 0.45 | 0.51 | 0.00 | 0.13 |
| 2048 | 256 | De-fused + No Tensor Cores | 0.41 | 0.51 | 0.00 | 0.13 |
| 4096 | 512 | Fused + Tensor Cores | 0.44 | 1.02 | 0.00 | 0.00 |
| 4096 | 512 | Fused + No Tensor Cores | 0.44 | 1.02 | 0.00 | 0.00 |
| 4096 | 512 | De-fused + Tensor Cores | 0.46 | 1.05 | 0.00 | 0.53 |
| 4096 | 512 | De-fused + No Tensor Cores | 0.43 | 1.05 | 0.00 | 0.53 |
| 8192 | 1024 | Fused + Tensor Cores | 0.35 | 2.03 | 2.00 | 0.00 |
| 8192 | 1024 | Fused + No Tensor Cores | 0.90 | 2.03 | 2.00 | 0.00 |
| 8192 | 1024 | De-fused + Tensor Cores | 0.38 | 3.16 | 22.00 | 2.12 |
| 8192 | 1024 | De-fused + No Tensor Cores | 0.94 | 3.16 | 22.00 | 2.12 |
| 16384 | 2048 | Fused + Tensor Cores | 0.72 | 4.06 | 4.00 | 0.00 |
| 16384 | 2048 | Fused + No Tensor Cores | 3.12 | 4.06 | 4.00 | 0.00 |
| 16384 | 2048 | De-fused + Tensor Cores | 0.75 | 10.56 | 24.00 | 8.50 |
| 16384 | 2048 | De-fused + No Tensor Cores | 3.16 | 10.56 | 24.00 | 8.50 |
| 32768 | 4096 | Fused + Tensor Cores | 2.20 | 8.12 | 0.00 | 0.00 |
| 32768 | 4096 | Fused + No Tensor Cores | 12.88 | 8.12 | 0.00 | 0.00 |
| 32768 | 4096 | De-fused + Tensor Cores | 2.28 | 38.12 | 32.00 | 34.00 |
| 32768 | 4096 | De-fused + No Tensor Cores | 12.66 | 38.12 | 32.00 | 34.00 |
| 65536 | 8192 | Fused + Tensor Cores | 11.87 | 16.25 | 0.00 | 0.00 |
| 65536 | 8192 | Fused + No Tensor Cores | 49.53 | 16.25 | 0.00 | 0.00 |
| 65536 | 8192 | De-fused + Tensor Cores | 11.65 | 144.25 | 128.00 | 136.00 |
| 65536 | 8192 | De-fused + No Tensor Cores | 49.20 | 144.25 | 128.00 | 136.00 |

## Derived Comparisons

| n_train | n_test | Fused/De-fused Speedup (TC) | Fused/De-fused Speedup (No TC) | Fused/De-fused Memory Reduction (TC) | Fused/De-fused Memory Reduction (No TC) | TC Speedup Within Fused | TC Speedup Within De-fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 256 | 1.08x | 1.19x | 1.00x | 1.00x | 0.83x | 0.92x |
| 4096 | 512 | 1.06x | 0.97x | 1.03x | 1.03x | 1.01x | 0.93x |
| 8192 | 1024 | 1.06x | 1.04x | 1.55x | 1.55x | 2.56x | 2.50x |
| 16384 | 2048 | 1.05x | 1.01x | 2.60x | 2.60x | 4.33x | 4.19x |
| 32768 | 4096 | 1.04x | 0.98x | 4.69x | 4.69x | 5.86x | 5.56x |
| 65536 | 8192 | 0.98x | 0.99x | 8.88x | 8.88x | 4.17x | 4.22x |

Notes:
- `Peak Extra Alloc` and `Peak Extra Reserved` are measured above the resident input-tensor footprint.
- `Explicit Workspace Peak` counts only the intentionally materialized global-memory workspaces in the de-fused implementations; fused variants report `0.00` by construction.
- Fused/De-fused ratios are `de_fused / fused`, so values above `1.00x` mean the fused path is better.
