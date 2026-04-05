| n_train | n_test | Torch (ms) | torch.compile (ms) | PyKeOps (ms) | Non-fused impl (ms) | Flash-SD-KDE (ms) | Torch/Flash | Compile/Flash | PyKeOps/Flash | Non-fused/Flash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 4 | 100.96 | 31.06 | 16.07 | 2.77 | 1.91 | 52.87x | 16.27x | 8.42x | 1.45x |
| 32768 | 16 | 101.03 | 32.27 | 15.44 | 3.17 | 1.92 | 52.72x | 16.84x | 8.05x | 1.65x |
| 32768 | 64 | 101.39 | 32.76 | 16.19 | 2.99 | 1.91 | 52.98x | 17.12x | 8.46x | 1.56x |
| 32768 | 256 | 102.06 | 33.93 | 15.91 | 2.62 | 1.91 | 53.42x | 17.76x | 8.33x | 1.37x |
| 32768 | 1024 | 104.67 | 33.38 | 15.96 | 4.32 | 1.97 | 53.13x | 16.94x | 8.10x | 2.20x |
| 32768 | 4096 | 115.12 | 34.97 | 16.14 | 15.22 | 2.12 | 54.32x | 16.50x | 7.61x | 7.18x |
| 32768 | 16384 | 157.63 | 42.90 | 16.90 | 58.15 | 2.79 | 56.56x | 15.40x | 6.06x | 20.86x |

Lower is better for runtime. Speedup columns are `baseline_runtime / flash_runtime`,
so values above `1.00x` mean Flash-SD-KDE is faster.

The `Non-fused impl` row is the separate-pass Laplace-corrected implementation
(`kde_eval_linearized_nonfused`), included as an implementation baseline rather than
an exact SD-KDE estimator.
