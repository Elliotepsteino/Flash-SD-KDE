| n_train | n_test | Flash-SD-KDE Tensor Core (ms) | Flash-SD-KDE no Tensor Core (ms) | no-TC / TC speedup | max abs delta | rel-L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 256 | 0.30 | 0.31 | 1.04x | 7.516e-10 | 2.270e-02 |
| 4096 | 512 | 0.28 | 0.35 | 1.24x | 1.350e-09 | 2.420e-02 |
| 8192 | 1024 | 0.29 | 0.87 | 2.99x | 2.171e-09 | 2.585e-02 |
| 16384 | 2048 | 0.66 | 2.83 | 4.27x | 3.996e-09 | 2.787e-02 |
| 32768 | 4096 | 2.12 | 10.48 | 4.94x | 7.691e-09 | 2.983e-02 |
| 65536 | 8192 | 9.30 | 44.59 | 4.80x | 1.353e-08 | 3.243e-02 |

Lower is better for runtime. The `no-TC / TC speedup` column is
`runtime_without_tensorcores / runtime_with_tensorcores`, so values above `1.00x`
mean the Tensor-Core path is faster.
