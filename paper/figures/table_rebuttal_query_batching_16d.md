| n_train | n_test | Torch (ms) | torch.compile (ms) | PyKeOps (ms) | Flash-SD-KDE (ms) | Torch/Flash | Compile/Flash | PyKeOps/Flash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 4 | 100.97 | 32.44 | 15.64 | 1.92 | 52.54x | 16.88x | 8.14x |
| 32768 | 16 | 101.00 | 32.41 | 15.43 | 1.91 | 52.78x | 16.94x | 8.06x |
| 32768 | 64 | 103.27 | 33.44 | 15.48 | 1.91 | 53.99x | 17.48x | 8.09x |
| 32768 | 256 | 104.09 | 33.68 | 16.11 | 1.92 | 54.19x | 17.53x | 8.38x |
| 32768 | 1024 | 106.22 | 33.69 | 15.81 | 1.97 | 54.04x | 17.14x | 8.04x |
| 32768 | 4096 | 116.81 | 35.97 | 15.85 | 2.11 | 55.32x | 17.04x | 7.51x |
| 32768 | 16384 | 158.18 | 43.14 | 16.79 | 2.75 | 57.54x | 15.69x | 6.11x |

Lower is better for runtime. Speedup columns are `baseline_runtime / flash_runtime`,
so values above `1.00x` mean Flash-SD-KDE is faster.
