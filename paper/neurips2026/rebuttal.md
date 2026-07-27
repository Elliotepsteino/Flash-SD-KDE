# NeurIPS 2026 Rebuttal — Submission 33138: Flash-SD-KDE

---

## General Response (all reviewers and AC)

We thank the reviewers for their careful reading. Following the reviews, we ran four new experiments on the same RTX A6000 workstation used in the paper; all will be added to the appendix of the revision:

1. **Numerical fidelity of TF32 (R1):** pointwise and metric-level error of the TF32 Tensor-Core path against an FP64 reference, plus a user-selectable FP32-IEEE mode that our kernels already support.
2. **Pass-level latency breakdown and TF32-enabled baselines (R2):** per-pass timings for eager PyTorch (FP32 and TF32-enabled), `torch.compile`, and Flash-SD-KDE, with profiler kernel tables and peak-memory measurements that attribute the speedup.
3. **Dimension sweep (R3):** runtimes and a Tensor-Core on/off ablation at $d \in \{16, 32, 64, 128\}$.
4. **Heavy-tailed non-Gaussian oracle study (R4):** KDE vs. Flash-SD-KDE vs. Flash-Laplace-KDE on a Student-$t_3$ mixture, with negative-mass diagnostics.

Headline findings: (i) enabling Tensor Cores in the cuBLAS-backed eager baseline leaves its runtime unchanged (112.9 ms with TF32 off vs. 113.0 ms with TF32 on), because materialized $O(n^2)$ elementwise traffic — not the GEMM — dominates it; the speedup of Flash-SD-KDE comes from the dataflow reordering, with Tensor Cores contributing a further $\approx 5\times$ on top (paper Table 5). (ii) TF32 leaves all statistical accuracy metrics unchanged to 4 significant digits, and an exact FP32-IEEE mode is a one-flag fallback. (iii) The implementation extends to $d$ up to 128 and remains the fastest exact method at every dimension ($2.5$–$26\times$ over a TF32-enabled eager baseline). (iv) On heavy tails, full SD-KDE robustly improves on KDE at every $n$, while the Laplace surrogate is best at small $n$ and degrades at large $n$, which sharpens the recommendation of when to use each estimator.

---

## Response to Reviewer 39iH

Thank you for your detailed review.

**Q1 (what is different from a standard cuBLAS GEMM implementation):** A cuBLAS-based implementation is exactly our eager PyTorch baseline: it must materialize the $n \times n$ distance and kernel matrices in global memory and run separate memory-bound elementwise kernels (norms, `exp`, reductions) between GEMM calls, because cuBLAS cannot consume a fused producer or apply a fused epilogue of this form. Our Triton kernel fuses the norm/`exp`/accumulation work into the GEMM tile loop, so the $O(n^2)$ intermediates never leave registers/shared memory and global traffic stays linear in $n$. This is visible in memory, not just time — peak extra allocation of the score pass at $n_{\text{train}}=32{,}768$:

| Implementation | Peak extra memory (score pass) |
|---|---|
| Eager PyTorch (cuBLAS GEMM + elementwise kernels) | 12.3 GB |
| `torch.compile` (fuses elementwise, still materializes $\Phi$ for cuBLAS) | 4.10 GB |
| Flash-SD-KDE (streaming, ours) | **8.1 MB** |

Paper Table 4 isolates the runtime effect of this streaming design: $55.7\times$ at $n=65{,}536$ against an otherwise-identical materialized implementation.

**Q2 (CUDA cores are also applicable):** Agreed, and we measure precisely this ablation: paper Table 5 runs the *identical* streamed kernel with Tensor-Core tiling disabled (FP32 CUDA-core `tl.dot`). The Tensor-Core path is $4.94\times$ faster at $n=32{,}768$ and $4.80\times$ at $n=65{,}536$. So the decomposition of the total speedup is: streaming/fusion accounts for the bulk, and Tensor Cores contribute a further $\approx 5\times$ at scale — both are needed to reach 2.4 ms.

**Q3 (impact of TF32 precision):** TF32 rounds only the `tl.dot` *inputs* to a 10-bit mantissa; storage and accumulation remain FP32, and no data is cast to FP16. New experiment (R1), $d=16$, $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, against an FP64 reference of the full SD-KDE pipeline:

| Variant | Max rel. err. vs FP64 | Mean rel. err. vs FP64 |
|---|---|---|
| Flash-SD-KDE, TF32 | $8.9\times10^{-2}$ | $3.2\times10^{-2}$ |
| Flash-SD-KDE, FP32-IEEE mode | $5.5\times10^{-6}$ | $1.1\times10^{-6}$ |
| Eager PyTorch FP32 | $1.1\times10^{-5}$ | $1.6\times10^{-6}$ |

For calibration, the *statistical* error of the estimator itself at this $n$ (mean relative deviation of the FP64 SD-KDE from the true density, dominated by estimation variance in 16-D) is $1.26$, i.e., $\approx 40\times$ larger than the mean TF32 perturbation, and the debiased sample positions move by at most $1.1\%$ of $h$. At the metric level the two modes are indistinguishable: mean squared error against the oracle density is $2.16487\times10^{-8}$ (TF32) vs. $2.16484\times10^{-8}$ (FP32-IEEE). Our kernels already expose `precision_mode="fp32_ieee"` for bit-comparable-to-FP32 results at roughly the no-Tensor-Core runtime of Table 5; we will state this prominently in the revision.

**Q4 (roofline analysis):** Section 4.1 contains the quantitative roofline analysis; we will add the figure to the appendix. Summary for the score kernel at $n=32{,}768$ on the A6000:

| Quantity | Value |
|---|---|
| Arithmetic intensity, tile model | 72 FLOPs/byte |
| Arithmetic intensity, Nsight-measured | $\approx 95$ FLOPs/byte |
| FP32 roofline knee | $\approx 50$ FLOPs/byte |
| TF32 Tensor-Core roofline knee | $\approx 200$ FLOPs/byte |
| Nsight "Speed of Light" SM / L1 throughput | $\approx 68\%$ / $\approx 90\%$ |
| Sustained fraction of absolute TC peak (Fig. 3, $n \geq 131$k) | $\approx 25$–$27\%$ ($\approx 40$ TFLOP/s) |

The kernel sits above the FP32 roof and below the pure-TC balance point, exactly as expected for a mixed GEMM + FP32-scalar workload; this is why we stopped low-level tuning at $\approx 68\%$ SM throughput.

**Limitation (numerical accuracy with Tensor Cores):** Addressed quantitatively in Q3 above; the fidelity study and the FP32-IEEE mode will become a new appendix section.

**Limitation (broader ML applications):** The paper already evaluates three real, non-synthetic workloads (Appendix A.7): Statlog Shuttle (recovers 807/879 true outliers vs. 610 for the best published baseline, in 0.60 s), ALOI (0.09 s), and KDD-Cup 99 HTTP with $n=620{,}098$ (ROC-AUC 0.92 in 7.27 s). These are exactly the density/OOD-scoring workloads where a fast exact estimator matters. Applications such as diffusion-model components are out of scope for a systems paper about making exact SD-KDE practical, but we note (Appendix C.4) that the same kernels are drop-in for KDE-primitive pipelines such as KDE-based attention approximations and embedding-space OOD scoring.

---

## Response to Reviewer 3kmd

Thank you for your thorough and positive review.

**Q1 (performance as $d$ approaches 64 or 128, Tensor-Core utilization vs. $d$):** The bandwidth $h$ enters only through the scalar $\exp(-r^2/2h^2)$, so growing $h$ with $d$ does not affect the GEMM structure; $d$ is the reduction (K) dimension of the GEMM, and the arithmetic-intensity model of Section 4.1 gives $I_d(k) \sim C(d)\,k$ with $C(d)$ increasing in $d$ — larger $d$ makes the kernel *more* compute-bound, not less. New experiment (R3), $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, generic padded-tile kernel with a per-$d$ launch-parameter sweep (as in Appendix A.4), Tensor-Core on/off within the identical kernel, and the eager PyTorch baseline with TF32 *enabled*:

| $d$ | Flash-SD-KDE (ms) | Flash, Tensor Cores off (ms) | TC speedup | PyTorch eager, TF32 on (ms) | Flash speedup |
|---|---|---|---|---|---|
| 16 | 4.32 | 14.29 | $3.31\times$ | 112.8 | $26.1\times$ |
| 32 | 12.85 | 19.39 | $1.51\times$ | 113.9 | $8.9\times$ |
| 64 | 24.43 | 38.42 | $1.57\times$ | 115.1 | $4.7\times$ |
| 128 | 47.17 | 75.66 | $1.60\times$ | 116.4 | $2.5\times$ |

Flash-SD-KDE remains the fastest exact method at every $d$. Its runtime grows roughly linearly in $d$ from 32 to 128 as the FLOP model predicts, while the baseline is nearly flat in $d$ because its materialized $O(n^2)$ memory traffic (independent of $d$) dominates — which is also why the ratio narrows at large $d$ while remaining above $2\times$. Two conservative-side notes: these numbers use the *generic* padded kernel, and at $d=16$ the specialized kernel of Table 1 is a further $2\times$ faster (2.2 ms), so per-$d$ specialization (block-K tiling, launch shapes) should widen these margins; the same applies to the Tensor-Core ablation column, where the generic kernel at $d \geq 32$ has not yet received the per-$d$ tuning that produced the $4.94\times$ TC gain of Table 5. We will add this sweep to the appendix.

We believe the sweep above and the precision analysis below cover the two items you flagged as score-relevant (trade-offs at $d>32$; behavior under reduced-precision data types).

**Q2 (compute-bound with lower-precision types):** We do not cast to FP16/BF16: inputs and accumulation are FP32 and TF32 rounds only the `tl.dot` inputs. On numerical impact (new experiment R1, FP64 reference, $n=32{,}768$): TF32 perturbs the density pointwise by $3.2\%$ on average, $\approx 40\times$ below the statistical error of the estimator at the same $n$ (mean relative deviation from the oracle: $126\%$), and the oracle MSE is unchanged to 4 significant digits ($2.16487\times10^{-8}$ vs. $2.16484\times10^{-8}$). A `precision_mode="fp32_ieee"` flag gives bit-comparable-to-FP32 results (max rel. err. $5.5\times10^{-6}$) at the no-Tensor-Core runtime of Table 5.

**Q3 (Laplace surrogate vs. full SD-KDE on non-Gaussian, heavy-tailed distributions):** New experiment (R4): 1-D two-component Student-$t_3$ mixture (infinite fourth moment), Silverman bandwidth, ISE against the true density over 5 seeds ($\times 10^3$, mean $\pm$ std):

| $n$ | KDE | Flash-SD-KDE | Flash-Laplace-KDE |
|---|---|---|---|
| 1,024 | $10.02 \pm 1.48$ | $5.35 \pm 1.15$ | $\mathbf{2.95 \pm 0.84}$ |
| 4,096 | $5.71 \pm 0.24$ | $2.32 \pm 0.16$ | $\mathbf{1.38 \pm 0.16}$ |
| 16,384 | $2.88 \pm 0.25$ | $\mathbf{0.94 \pm 0.13}$ | $1.23 \pm 0.13$ |
| 65,536 | $1.99 \pm 0.06$ | $\mathbf{0.86 \pm 0.06}$ | $2.53 \pm 0.18$ |

Full SD-KDE improves on KDE by $1.9$–$3.1\times$ in ISE at *every* $n$, so the estimator we accelerate is robust to heavy tails. The Laplace surrogate is the most accurate at small-to-moderate $n$ but degrades at large $n$: its signed tail correction $(1 + d/2 - \|x\|^2/2h^2)$ amplifies variance where $t_3$ places many distant samples, and $t_3$ lacks the fourth moments that fourth-order kernels exploit asymptotically. This matches its framing as a fast surrogate: when tails are unknown, use full SD-KDE — which is exactly what Flash-SD-KDE makes cheap. We will add this study and recommendation to the revision.

**W2 (negative density values):** We now quantify this. On the heavy-tailed benchmark above — a worst case for the signed correction — the integrated negative mass of Flash-Laplace-KDE is at most $2.2\times10^{-4}$ (i.e., $0.02\%$ of unit mass, at $n=1{,}024$) and decreases in $n$ over the first three sizes ($8.6\times10^{-5}$ at $n=16{,}384$). For downstream integration, clipping at zero changes the estimate by at most this mass, and the full SD-KDE path is nonnegative by construction and available at 2.4 ms for $n=32$k, so the surrogate is never forced on a user.

---

## Response to Reviewer YaGf

Thank you for your insightful review.

**W1 (novelty of the GPU techniques):** The individual techniques (tiling, streaming, mixed precision) are indeed known — as they were for FlashAttention, whose contribution was showing that reordering one specific primitive unlocks a new capability regime. Our contributions are estimator-specific and not obtainable from generic tooling: (i) the algebraic reordering $\sum_j (x_i - x_j)\varphi_{ij} = x_i \sum_j \varphi_{ij} - (\Phi X)_i$, which turns the empirical-score numerator into two GEMMs; (ii) the streaming formulation that makes *exact* SD-KDE run at $n \approx 10^6$ on one GPU for the first time; and (iii) Flash-Laplace-KDE, a new fused surrogate estimator with the same leading-order bias correction and no score pass. The strongest evidence that generic tooling does not get there: `torch.compile` — which has tiling, fusion, and Tensor Cores available — lands at 34.3 ms on the Table 1 workload (25.7 ms under PyTorch 2.9 in our rerun below) vs. our 2.2–2.4 ms, a $12$–$14\times$ gap, and we show below why.

**W2 (baseline Tensor-Core usage; compiled graph of `torch.compile`):** In Table 1 the PyTorch baselines run at PyTorch's default matmul precision (TF32 disabled, i.e., FP32 CUDA-core GEMMs); we will state this explicitly. We have now rerun the eager baseline with TF32 *enabled* (R2): 113.0 ms vs. 112.9 ms with TF32 off — no improvement. The profiler explains why: with TF32 on, cuBLAS does dispatch a Tensor-Core kernel (`cutlass_80_tensorop_s1688gemm`), but all GEMMs combined are only $11\%$ of the baseline's CUDA time; $\approx 89\%$ is materialized elementwise/reduction traffic (`vectorized_elementwise_kernel`, `reduce_kernel`). For `torch.compile` (TF32 on), the compiled score graph contains five kernels: two small norm/epilogue Triton kernels, one large fused subtract-clamp-`exp`-sum reduction ($50\%$ of kernel time), and two cuBLAS GEMMs ($XX^\top$ as TF32 tensorop, $25\%$; $\Phi X$ as `ampere_sgemm`, $25\%$). Inductor must *materialize* $\Phi$ — a 4.10 GB buffer, exactly one $32{,}768^2$ FP32 matrix — because a cuBLAS GEMM cannot consume a fused producer. Our kernel keeps that matrix in registers/shared memory (8.1 MB peak extra). We will include the full trace and this analysis in the revision.

**Q1 (latency breakdown):** New experiment (R2), $d=16$, $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, interleaved min-of-30 on the paper's A6000 workstation:

| Method | Score pass (ms) | KDE pass (ms) | Total (ms) | Score share |
|---|---|---|---|---|
| PyTorch eager, FP32 (Table 1 baseline) | 101.0 | 11.9 | 112.9 | 89% |
| PyTorch eager, TF32 enabled | 100.9 | 12.1 | 113.0 | 89% |
| `torch.compile`, TF32 enabled | 24.0 | 1.7 | 25.7 | 93% |
| Flash-SD-KDE | 1.93 | 0.28 | 2.21 | 87% |

(The eager and Flash rows reproduce Table 1's 113.3 ms and 2.4 ms; `torch.compile` improved from 34.3 ms to 25.7 ms under the newer PyTorch 2.9 used for the rerun, and remains $12\times$ slower than Flash-SD-KDE.) This confirms the Nsight finding in the paper that the score pass is $\approx 90$–$95\%$ of end-to-end runtime for every implementation.

**Q2 (numerical impact of dtype casting):** There is no dtype cast: inputs are stored FP32, accumulation is FP32, and TF32 only rounds the `tl.dot` inputs' mantissas. Against an FP64 reference (R1, $n=32{,}768$): mean pointwise density error $3.2\%$, which is $\approx 40\times$ below the estimator's statistical error at the same $n$ (mean relative deviation from the true density: $126\%$); oracle-MSE agrees with the exact mode to 4 significant digits ($2.16487\times10^{-8}$ vs. $2.16484\times10^{-8}$). An exact `precision_mode="fp32_ieee"` flag is available (max rel. err. $5.5\times10^{-6}$ vs. FP64) at the cost of forgoing Tensor Cores ($\approx 5\times$ slower at scale, Table 5).

**Q3 (fusing the score and KDE passes):** It is possible, but Amdahl's law caps the benefit: the score pass is $90$–$95\%$ of runtime (R2 above and Appendix A.6), so fusing the two passes saves at most $\approx 5$–$10\%$ for nontrivial code complexity, which is why we kept the two-kernel structure. Where fusion does pay is *within* a pass: Flash-Laplace-KDE is exactly the fully fused single-pass variant (correction applied inside the same tile loop), and it is $2$–$5\times$ faster than the non-fused Laplace implementation across $n$ (Figure 7) while matching its accuracy.
