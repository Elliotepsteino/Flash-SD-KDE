# NeurIPS 2026 Rebuttal (Submission 33138): Flash-SD-KDE

---

## Meta Review by Area Chair LN3f (for reference)

> **Metareview:** The paper presents a systems-oriented acceleration of the existing SD-KDE estimator, whose empirical score computation introduces substantial quadratic computational and memory costs. It reorganizes the score-estimation and density-evaluation stages into Tensor-Core-friendly matrix operations and implements them using tiled, streaming Triton kernels that avoid materializing full pairwise interaction matrices. The reported results include speedups of up to 47× over an eager PyTorch SD-KDE implementation and over 3,300× over scikit-learn KDE. The paper also introduces Flash-Laplace-KDE, a fused surrogate to provide similar bias-reduction benefits without explicit score-estimation pass.
>
> The principal strength is the magnitude of the reported performance gains and their potential to make SD-KDE practical at substantially larger scales. The GEMM reformulation and streaming accumulation are regarded as technically sound and well aligned with modern GPU architectures. The implementation coherently combines Tensor-Core execution, tiling, and memory-aware streaming to reduce both runtime and the storage required for pairwise interactions. The reviews also highlight the effective use of Triton for managing tiling and the memory hierarchy, as well as the ablation studies that help explain where the acceleration arises.
>
> The AC's concerns, itemized:
>
> 1. *(Attribution)* A recurring concern is whether the reported gains reflect the implementation advances beyond a conventional GEMM-based PyTorch or cuBLAS formulation. In particular, current presentation does not clarify whether the PyTorch and torch.compile baselines already use Tensor Cores under comparable datatype and precision settings, or which aspects of the custom implementation account for the remaining performance difference.
> 2. *(Numerical fidelity)* A second recurring concern is the numerical impact of TF32 or other reduced-precision execution. The evaluation does not directly establish whether datatype casting or Tensor-Core arithmetic materially affects the accuracy of the estimated scores or density values.
> 3. *(Novelty)* The reviewers also have disagreements on the novelty, especially the limited novelty in methodology by primarily combining established GPU techniques.
> 4. Performance at dimensions above 32, particularly 64 and 128.
> 5. Possible negative density values and the behavior of Flash-Laplace-KDE on non-Gaussian or heavy-tailed distributions.
> 6. Roofline analysis and broader application evidence.
> 7. A more detailed latency breakdown together with clarification of whether the score-estimation and KDE stages could be fused.
>
> Regarding the rebuttal, the most consequential issues are baseline comparability, performance attribution, and numerical fidelity. The authors should clarify which implementation choices provide benefits beyond a standard GEMM formulation and whether the PyTorch and torch.compile baselines already use Tensor Cores. A detailed compiled graph for the torch.compile baseline would make this comparison more concrete. The response should also explain whether the datatype casting required for Tensor-Core execution, including the use of TF32 or other reduced-precision arithmetic, materially affects SD-KDE accuracy. A pass-level latency breakdown would help clarify where the reported speedups arise and strengthen the attribution of the performance gains. Results or analysis at dimensions such as 64 and 128 would also clarify whether the acceleration extends beyond the primarily evaluated low-dimensional regime. For Flash-Laplace-KDE, clarification of its behavior on non-Gaussian or heavy-tailed distributions and the practical implications of possible negative density values would address the remaining concern about this auxiliary contribution.

---

## Review 1: Reviewer 39iH (Rating 3: Borderline reject; Confidence 3)

> **Summary:** This paper accelerates score-debiased kernel density estimation (SD-KDE) by reformulating its computation as Tensor Core–friendly matrix multiplications (GEMMs) combined with lightweight elementwise operations. The approach avoids storing large pairwise interaction matrices through streaming accumulation, substantially reducing computational overhead. In 16-dimensional experiments, the optimized implementation achieves up to 47× speedup over a strong PyTorch SD-KDE baseline and 3,300× speedup over scikit-learn KDE, enabling SD-KDE to scale to ~1 million training samples and ~131k query points in 2.3 seconds on a single GPU.
>
> **Strengths And Weaknesses:** The reformulation into GEMM plus streaming accumulation optimization is interesting and well aligned with modern GPU architectures. The reported speedups are outstanding: up to 47× over Torch SD-KDE, over 3300× over scikit-learn, million-scale SD-KDE in only 2.3 seconds on a single GPU. The work could make SD-KDE practical for a wide range of real-world challenging applications.
>
> **Quality:** 2: not good. **Clarity:** 2: not good. **Significance:** 2: not good. **Originality:** 3: good.
>
> **Questions:**
> 1. The use of Tensor Cores is not sufficiently explained. Because Tensor Cores are designed for GEMM and are already supported by cuBLAS, it is unclear what makes the proposed implementation different from a standard GEMM-based implementation.
> 2. it is difficult to connect the work to Tensor Core since CUDA core is also applicable
> 3. The implementation relies on Tensor Cores with TF32 precision (19 bit), what is the impact of reduced numerical precision on SD-KDE accuracy?
> 4. Can you show the Roofline analysis result?
>
> **Limitations:** Numerical accuracy is insufficiently discussed when using the Tensor Core. It would be stronger if it demonstrated benefits in broader ML applications, e..g diffusion models, uncertainty estimation ......

## Rebuttal 1: Response to Reviewer 39iH

Thank you for your detailed review. To address your questions we ran several new experiments on the same RTX A6000 workstation used in the paper.

**Q1 (what is different from a standard cuBLAS GEMM implementation):** The difference is not the GEMM but the dataflow: separate GPU kernels can only communicate through global memory, so a cuBLAS-based implementation writes and reads the $n \times n$ Gram and kernel matrices through HBM, $\approx 16$ GB $\approx 21$ ms at 770 GB/s for $n=32{,}768$, no matter how fast the GEMMs run. Our kernel fuses the norms, `exp`, and both accumulations into the GEMM tile loop, so these intermediates never leave registers. The ladder isolates what each idea contributes ($n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, $d=16$):

| # | Implementation | Idea added | Total (ms) | Cumulative speedup |
|---|---|---|---|---|
| 1 | PyTorch eager, cuBLAS GEMMs + separate elementwise kernels (Table 1; TF32 Tensor Cores) | (baseline) | 113.3 | $1\times$ |
| 2 | Our GEMM formulation, un-tiled: full $n^2$ materialization, Tensor Cores on (new; same runtime with Tensor Cores off) | Tensor-Core GEMM reformulation alone; $n^2$ intermediates still round-trip HBM | 100.4 | $1.1\times$ |
| 3 | `torch.compile` (Table 1; `mode="max-autotune"` does not improve it) | fuses all elementwise chains; still materializes one $n^2$ buffer for cuBLAS | 34.3 | $3.3\times$ |
| 4 | Our streamed Triton kernel, Tensor Cores disabled (Table 5) | full fusion + streaming: $n^2$ tiles never leave registers/shared memory | 10.5 | $10.8\times$ |
| 5 | **Flash-SD-KDE** (Table 1; row 4 + Tensor-Core tiles) | GEMM tiles execute on Tensor Cores | **2.4** | $\mathbf{47\times}$ |

Row 2 shows the attribution directly: the GEMM reformulation with Tensor Cores but materialized intermediates is only $1.1\times$ faster than eager, because the $n^2$ traffic still dominates. Row 4 removes that traffic by fusion, and only then is the kernel compute-bound; Tensor Cores supply the remaining $\approx 4.4\times$ (row 5).

**Q2 (CUDA cores are also applicable):** Agreed, and paper Table 5 measures precisely this ablation: the *identical* streamed kernel with Tensor-Core tiling disabled (FP32 CUDA-core `tl.dot`) is $4.94\times$ slower at $n=32{,}768$ and $4.80\times$ at $n=65{,}536$.

**Q3/Limitation 1(impact of TF32 precision):** *Short answer: no statistically detectable effect. TF32 perturbs the density estimates by less than the estimator's own sampling noise, and an exact FP32-IEEE mode is a one-flag fallback.* Mechanically, TF32 rounds only the `tl.dot` *inputs* to a 10-bit mantissa; storage and accumulation remain FP32, and no data is cast to FP16. We compared each implementation against an FP64 reference of the full SD-KDE pipeline at $d=16$, $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$:

| Variant | Max rel. err. vs FP64 | Mean rel. err. vs FP64 |
|---|---|---|
| Flash-SD-KDE, TF32 | $8.9\times10^{-2}$ | $3.2\times10^{-2}$ |
| Flash-SD-KDE, FP32-IEEE mode | $5.5\times10^{-6}$ | $1.1\times10^{-6}$ |
| Eager PyTorch FP32 | $1.1\times10^{-5}$ | $1.6\times10^{-6}$ |

TF32 perturbs each density estimate by $3.2\%$ on average; re-drawing the full $n$-point training set perturbs the same estimates by $7.2\%$ on average, so the hardware effect is within the estimator's sampling noise, and the debiased sample positions move by at most $1.1\%$ of $h$. 

**Q4 (roofline analysis):** Section 4.1 contains the quantitative roofline analysis; we will add the figure to the appendix. Summary for the score kernel at $n=32{,}768$ on the A6000:

| Quantity | Value |
|---|---|
| Arithmetic intensity, tile model | 72 FLOPs/byte |
| Arithmetic intensity, Nsight-measured | $\approx 95$ FLOPs/byte |
| FP32 roofline knee | $\approx 50$ FLOPs/byte |
| TF32 Tensor-Core roofline knee | $\approx 200$ FLOPs/byte |
| Nsight "Speed of Light" SM / L1 throughput | $\approx 68\%$ / $\approx 90\%$ |
| Sustained fraction of absolute TC peak (Fig. 3, $n \geq 131$k) | 25 to 27% ($\approx 40$ TFLOP/s) |

The kernel sits above the FP32 roof and below the pure-TC balance point, exactly as expected for a mixed GEMM + FP32-scalar workload; this is why we stopped low-level tuning at $\approx 68\%$ SM throughput.

**Limitation (broader ML applications):** The paper evaluates Flash-SD-KDE on three real, non-synthetic workloads (Appendix A.7): Statlog Shuttle (recovers 807/879 true outliers vs. 610 for the best published baseline, in 0.60 s), ALOI (0.09 s), and KDD-Cup 99 HTTP with $n=620{,}098$ (ROC-AUC 0.92 in 7.27 s). These are exactly the density/OOD-scoring workloads where a fast exact estimator matters. Applications such as diffusion-model components are out of scope for a systems paper about making exact SD-KDE practical, but we note (Appendix C.4) that the same kernels are drop-in for KDE-primitive pipelines such as KDE-based attention approximations and embedding-space OOD scoring.

---

## Review 2: Reviewer 3kmd (Rating 5: Accept; Confidence 3)

> **Summary:** This paper provides a new technique to speed up Score-Debiased Kernel Density Estimation (SD-KDE) by leveraging specific tensor cores in commercial GPUs available today. SD-KDE provides superior asymptotic convergence compared to standard KDE, but if suffers from the high computational cost of its empirical score estimation. The authors demonstrate that the complexity of SD-KDE is not an insurmountable barrier, but rather a dataflow problem. By reordering the score and KDE computations to expose matrix-multiplication structures, they leverage Tensor Cores and streaming accumulation to achieve significant speedups. Additionally, they propose a Flash-Laplace KDE surrogate that provides similar bias-reduction benefits without the explicit score-estimation overhead.
>
> **Strengths And Weaknesses:** This work addresses the central challenge of making statistically efficient nonparametric density estimation viable for large-scale machine learning applications. The paper is technically robust, characterized by a clear derivation of the matrix-form SD-KDE and arithmetic intensity analysis that aligns with the performance characteristics of the NVIDIA RTX A6000.
>
> *Strengths.* The implementation, utilizing Triton to manage tiling and memory hierarchy, is quite nice. The authors successfully move the computation into the compute-bound regime, effectively utilizing the hardware's Tensor Cores. The paper is well-structured and easy to follow. I also appreciated the clarity and definitions of SD-KDE, followed by the mapping to the hardware-native matrix implementation. The claims of the paper are well-supported by empirical evidence. The speedups reported (up to 47× over PyTorch baselines and 3,300× over scikit-learn) are transformative for practitioners who require exact density estimation at scale. These are STRONG numbers. The application of Flash-style streaming/tiling to the SD-KDE pipeline is a clever adaptation of existing attention-acceleration techniques.
>
> *Weaknesses.* The main weakness is that the paper primarily focuses on d=16. While this dimension is useful for many anomaly detection tasks, the performance characteristics of these kernels in higher-dimensional spaces (where KDE typically struggles) remain an open question. Additionally, the Laplace-corrected variant's potential for negative density values is a minor concern that requires careful integration into downstream workflows.
>
> **Quality:** 3: good. **Clarity:** 4: excellent. **Significance:** 3: good. **Originality:** 3: good.
>
> **Questions:**
> 1. As d increases, the "curse of dimensionality" typically forces bandwidth h to change, which impacts the GEMM structure. How does the performance scale as d approaches 64 or 128? It would be important to see an ablation study on the impact of d on Tensor Core utilization. If the authors could provide a more detailed analysis of the performance trade-offs in higher dimensions (d>32) or demonstrate that the implementation remains compute-bound when using static, lower-precision data types, that would help me give a better score to the paper.
> 2. Further, I want to understand the impact of proposed Laplace-corrected surrogate compared to the full SD-KDE on non-Gaussian, heavy-tailed distributions.

## Rebuttal 2: Response to Reviewer 3kmd

Thank you for your thorough and positive review. New experiment are all on the same RTX A6000 workstation used in the paper. 

**Q1 (performance as $d$ approaches 64 or 128, Tensor-Core utilization vs. $d$):** Flash-SD-KDE runtime grows linearly in $d$ as the compute-bound model predicts, and Tensor-Core utilization holds at 19 to 30% of peak across the sweep. The bandwidth $h$ enters only through the scalar $\exp(-r^2/2h^2)$, so growing $h$ with $d$ does not affect the GEMM structure. We swept $d$ at $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$.

| $d$ | Flash-SD-KDE (ms) | Fraction of TC peak | Tensor Cores off (ms) | TC speedup |
|---|---|---|---|---|
| 16 | 2.4 | 23% | 10.5 | $4.4\times$ |
| 32 | 3.5 | 30% | 18.3 | $5.3\times$ |
| 64 | 6.8 | 29% | 37.2 | $5.5\times$ |
| 128 | 19.8 | 19% | 73.2 | $3.7\times$ |

 Sustaining 19 to 30% of the *absolute* Tensor-Core peak is the expected level for this estimator, because the kernel interleaves its GEMM tiles with the $O(n^2)$ non-Tensor-Core work (norms, exponentials, atomic accumulations) that runs on FP32 ALUs and SFUs, consistent with Figure 3.

**Q2 (compute-bound with lower-precision types):** We do not cast to FP16/BF16: inputs and accumulation are FP32 and TF32 rounds only the `tl.dot` inputs. On numerical impact (new experiment, $n=32{,}768$): relative to the same kernel in exact FP32 arithmetic, TF32 perturbs each density estimate by $3.2\%$ on average; re-drawing the full $n$-point training set perturbs the same estimates by $7.2\%$ on average, so the hardware effect is within the estimator's sampling noise.

**Q3 (Laplace surrogate vs. full SD-KDE on non-Gaussian, heavy-tailed distributions):** Full SD-KDE stays robust on heavy tails, while the Laplace surrogate is the most accurate at small $n$ but improves less at large $n$, so we recommend full SD-KDE when tails are unknown. We evaluated all three estimators on a 1-D two-component Student-$t_3$ mixture, which has infinite fourth moment, using the Silverman bandwidth and reporting integrated squared error against the true density over 5 seeds ($\times 10^3$, mean $\pm$ std):

| $n$ | KDE | Flash-SD-KDE | Flash-Laplace-KDE |
|---|---|---|---|
| 1,024 | $10.02 \pm 1.48$ | $5.35 \pm 1.15$ | $\mathbf{2.95 \pm 0.84}$ |
| 4,096 | $5.71 \pm 0.24$ | $2.32 \pm 0.16$ | $\mathbf{1.38 \pm 0.16}$ |
| 16,384 | $2.88 \pm 0.25$ | $\mathbf{0.94 \pm 0.13}$ | $1.23 \pm 0.13$ |

The surrogate's signed tail correction $(1 + d/2 - \|x\|^2/2h^2)$ amplifies variance where $t_3$ places many distant samples, and $t_3$ lacks the fourth moments that fourth-order kernels exploit asymptotically.

**W2 (negative density values):** On the heavy-tailed benchmark above, a worst case for the signed correction, the integrated negative mass of Flash-Laplace-KDE is at most $2.2\times10^{-4}$ (at $n=1{,}024$) and decreases in $n$ over the first three sizes ($8.6\times10^{-5}$ at $n=16{,}384$). 

---

## Review 3: Reviewer YaGf (Rating 3: Borderline reject; Confidence 3)

> **Summary:** This paper accelerates SD-KDE using GPU's Tensor Core to make it computationally practical. The authors reform pairwise squared distances as few matrix multiplications, enabling Tensor Core acceleration via a custom Triton kernel. Instead of materializing the full matrix, the authors stream submatrices to keep memory footprint linear. Also, they propose Flash-Laplace-KDE, which replaces the empirical score with an analytically-derived Laplace-corrected kernel, which shares the same bias as SD-KDE but requires no score pass, enabling kernel fusion in a single pass. The results show that the proposed Flash-SD-KDE runs 47x faster than a strong PyTorch baseline.
>
> **Strengths And Weaknesses:** The paper is technically sound, with a GEMM reformulation using algebraic decompositions, and ablation studies to show where the speedup comes from. The paper is clearly written and well-organized. SD-KDE has been computationally impractical at scale, but this paper brings a 1M-sample case down to 2.3s on a single GPU. However, while this work is well-motivated and brings practicability of SD-KDE, most of the techniques on GPU are known and lack novelty. Also, the baseline's Tensor Core usage is not clear -- I think Tensor Core will be used when each ops is casted properly. Especially, please provide a detailed compiled graph of torch.compile since torch.compile can do the tiling.
>
> **Quality:** 3: good. **Clarity:** 3: good. **Significance:** 3: good. **Originality:** 2: not good.
>
> **Questions:**
> 1. Can you please provide a latency breakdown of the existing SD-KDE steps of the baseline (CPU/Torch/etc) and Flash-SD-KDE?
> 2. Tensor Core works on certain dtypes. Are there any numerical impacts in the use case when you cast the numbers?
> 3. Is kernel fusion between score and KDE passes possible?
>
> **Limitations:** Please refer to the Weakness section above.

## Rebuttal 3: Response to Reviewer YaGf

Thank you for your insightful review. To address your questions we ran new experiments on the same RTX A6000 workstation used in the paper.

**W1 (novelty of the GPU techniques):** The individual techniques (tiling, streaming, mixed precision) are indeed known, as they were for FlashAttention, whose contribution was showing that reordering one specific primitive unlocks a new capability regime. Our contributions are estimator-specific and not obtainable from generic tooling: (i) the algebraic reordering $\sum_j (x_i - x_j)\varphi_{ij} = x_i \sum_j \varphi_{ij} - (\Phi X)_i$, which turns the empirical-score numerator into two GEMMs; (ii) the streaming formulation that makes *exact* SD-KDE run at $n \approx 10^6$ on one GPU for the first time; and (iii) Flash-Laplace-KDE, a new fused surrogate estimator with the same leading-order bias correction and no score pass. 

**W2 (whether `torch.compile`'s tiling could derive this kernel):** The reviewer is correct that `torch.compile` (TorchInductor) performs tiling, but only for the class of programs it can express: fusions of pointwise, reduction, and scatter ops, plus matmuls with limited prologue/epilogue fusion. Our kernel requires fusing *through* two matmuls: it computes the pairwise inner products $XX^\top$, applies `exp` elementwise, and immediately consumes the result $\Phi$ in both a row-reduction and a second matmul $\Phi X$, keeping each tile of the $n \times n$ matrix $\Phi$ in registers. Inductor does not fuse the output of one matmul into the input of another; matmuls dispatch as separate kernels (cuBLAS or Triton templates), so $\Phi$ must be materialized in HBM between them (see W3 for details). This is precisely the transformation that motivated FlashAttention (Dao et al., 2022), which shares the matmul $\to$ elementwise $\to$ matmul structure.

**W3 (baseline Tensor-Core usage; compiled graph of `torch.compile`):**  Yes, the baselines use Tensor Cores. However, Tensor Cores make little difference to the baseline because it is bandwidth bound, with only $11\%$ of its runtime in GEMMs. By Amdahl's law even free GEMMs would improve the baseline by at most $1.13\times$, against the $47\times$ gap to Flash-SD-KDE. The other $89\%$ of the time is spent moving $n^2$ intermediates through global memory, which no kernel-by-kernel implementation can avoid because separate kernels communicate only through it: writing and reading the Gram matrix and $\Phi$ (4 GB each) costs $\approx 16$ GB $\approx 21$ ms at 770 GB/s before any arithmetic, and tiling merely caps peak memory without reducing the bytes moved. Tensor Cores become useful once fusion removes the $n^2$ traffic entirely, at which point the same hardware yields $4.94\times$ inside our streamed kernel (Table 5). The ladder isolates each idea ($n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, $d=16$):

| # | Implementation | Idea added | Total (ms) | Cumulative speedup |
|---|---|---|---|---|
| 1 | PyTorch eager, cuBLAS GEMMs + separate elementwise kernels with TF32 Tensor Cores enabled | (baseline) | 113.3 | $1\times$ |
| 2 | Our GEMM formulation, un-tiled: full $n^2$ materialization, Tensor Cores on (new; same runtime with Tensor Cores off) | Tensor-Core GEMM reformulation alone; $n^2$ intermediates still round-trip HBM | 100.4 | $1.1\times$ |
| 3 | `torch.compile` (Table 1; `mode="max-autotune"` does not improve it) | fuses all elementwise chains; still materializes one $n^2$ buffer for cuBLAS | 34.3 | $3.3\times$ |
| 4 | Our streamed Triton kernel, Tensor Cores disabled (Table 5) | full fusion + streaming: $n^2$ tiles never leave registers/shared memory | 10.5 | $10.8\times$ |
| 5 | **Flash-SD-KDE** (Table 1; row 4 + Tensor-Core tiles) | GEMM tiles execute on Tensor Cores | **2.4** | $\mathbf{47\times}$ |

The requested compiled graph, the complete Inductor execution plan for the compiled score pass captured via `TORCH_LOGS=output_code`, is:

```
buf0 = empty_strided_cuda((32768, 1))                 # row norms
triton_per_fused_mul_sum_0(X -> buf0)                 # negligible
buf1 = empty_strided_cuda((32768, 32768))             # the single 4 GB n^2 buffer
extern_kernels.mm(X, X^T, out=buf1)                   # cuBLAS -> cutlass_80_tensorop_s1688gemm
                                                      #   (TF32, Tensor Cores), ~25% of kernel time
buf2 = buf1  # reuse: Phi overwrites the Gram in place
triton_red_fused_add_clamp_exp_mul_permute_sub_sum_1  # dist + exp + row-sums fused;
  (buf2, buf0 -> buf2, buf4)                          #   reads Gram, writes Phi, ~50%
extern_kernels.mm(buf2, X, out=buf3)                  # cuBLAS -> ampere_sgemm (Phi @ X), ~25%
triton_poi_fused_add_div_mul_sub_2                    # debias epilogue, negligible
```

Every elementwise op is fused into two Triton kernels, and the memory planner reuses the single $n^2$ buffer by overwriting the Gram with $\Phi$. The two `extern_kernels.mm` calls are opaque cuBLAS launches, so that buffer still crosses HBM twice ($\approx 21$ ms of traffic, the bulk of its runtime), and even `mode="max-autotune"` cannot fuse the chain GEMM $\to$ (`exp`, row-sum) $\to$ GEMM because $\Phi$ has two consumers, one of them a matmul. So `torch.compile` can indeed tile, but only within each kernel it generates; the tiling that produces the large speedup is a different program, a single loop nest that carries each tile through the GEMM, the `exp`, and both accumulations while it stays in registers, and Inductor does not search over such cross-operation restructurings of the algorithm. Our implementation is exactly that program: it replaces the `mm` $\to$ fused $\to$ `mm` sandwich with one streamed Triton kernel whose tiles never leave registers/shared memory (8.1 MB peak extra vs. 4.10 GB).

**Q1 (latency breakdown):** We measured the score and KDE passes separately for each implementation at $d=16$, $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$ on the paper's workstation, reporting the minimum over repeated interleaved runs. The CPU row runs the same eager formulation on the host's dual EPYC 7763 with 128 threads, since the scikit-learn baseline of Table 1 has no score pass to break down:

| Method | Score pass (ms) | KDE pass (ms) | Total (ms) | Score share |
|---|---|---|---|---|
| PyTorch on CPU (dual EPYC 7763) | 6,245 | 767 | 7,012 | 89% |
| PyTorch (Table 1 baseline) | 100.8 | 12.1 | 112.8 | 89% |
| Flash-SD-KDE | 2.07 | 0.27 | 2.34 | 88% |

**Q2 (numerical impact of dtype casting):** Inputs are stored FP32, accumulation is FP32, and TF32 only rounds the `tl.dot` inputs' mantissas. New experiment ($n=32{,}768$): relative to the identical kernel run in full FP32 without Tensor Cores (`precision_mode="fp32_ieee"`), TF32 perturbs each density estimate by $3.2\%$ on average; re-drawing the full $n$-point training set perturbs the same estimates by $7.2\%$ on average, so the hardware effect is within the estimator's sampling noise. 

**Q3 (fusing the score and KDE passes):** It is possible, but Amdahl's law caps the benefit: the score pass is around 90% of runtime (see Appendix A.6), so fusing the two passes saves at most 10% for nontrivial code complexity, which is why we kept the two-kernel structure. Where fusion does pay is *within* a pass: Flash-Laplace-KDE is exactly the fully fused single-pass variant (correction applied inside the same tile loop), and it is $2\times$ to $5\times$ faster than the non-fused Laplace implementation across $n$ (Figure 7) while matching its accuracy.
