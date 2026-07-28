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

**Q1 (what is different from a standard cuBLAS GEMM implementation):** *Short answer: the GEMM is the smaller part of the problem. The bottleneck is that separate GPU kernels can only communicate through the memory system, so every element of the $n^2$ intermediates must cross a kernel boundary regardless of how the computation is tiled; tiling changes peak memory, not bytes moved.* In any implementation where the Gram matrix and the kernel matrix $\Phi$ are produced and consumed by *separate kernels* (which is every composition of cuBLAS and elementwise ops), each of those 4 GB intermediates is written and read once through the memory system: $\approx 16$ GB $\approx 21$ ms at the A6000's 770 GB/s, no matter how fast the GEMMs run. To be precise, this floor is a property of that execution model, not of the problem: the computation's intrinsic I/O is only the $O(nd)$ inputs and outputs (about 4 MB here), and our fused kernel moves $\approx 1.2$ GB of streamed tile traffic (the byte model of Section 4.1), an order of magnitude below the kernel-sequence floor, which is exactly the headroom that fusion converts into runtime. The floor is also a statement about traffic, not materialization: a tiled cuBLAS implementation avoids allocating the full $n \times n$ matrix and is no faster, which we verified in both regimes. Strip-tiled eager PyTorch (tile $\times$ $n$ pieces) measures 114 ms, and 2-D tiling at L2-resident tile sizes, the one configuration that could in principle bypass DRAM by keeping tiles in the 6 MB L2, is *slower* still (score pass: 593 ms at $512^2$ tiles, 150 ms at $1024^2$, vs. 101 ms eager and 1.9 ms ours), because at those sizes each $K=16$ cuBLAS GEMM performs only 8 to 34 MFLOP, microseconds of work against $\sim$10 µs of launch overhead per kernel across thousands of tiles, and no cross-launch cache-residency guarantee exists for a sequence of library calls. The ladder below isolates what each idea contributes ($n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, $d=16$):

| # | Implementation | Idea added | Total (ms) | Cumulative speedup |
|---|---|---|---|---|
| 1 | PyTorch eager, cuBLAS GEMMs + separate elementwise kernels (113.0 with TF32 on) | (baseline) | 112.9 | $1\times$ |
| 2 | Our GEMM formulation, un-tiled: full $n^2$ materialization, Tensor Cores on (new; 100.3 with Tensor Cores off) | Tensor-Core GEMM reformulation alone; $n^2$ intermediates still round-trip HBM | 100.4 | $1.1\times$ |
| 3 | `torch.compile` (25.2 with `mode="max-autotune"`) | fuses all elementwise chains; still materializes one $n^2$ buffer for cuBLAS | 25.7 | $4.4\times$ |
| 4 | Our streamed Triton kernel, Tensor Cores disabled (Table 5) | full fusion + streaming: $n^2$ tiles never leave registers/shared memory | 10.5 | $10.8\times$ |
| 5 | **Flash-SD-KDE** (row 4 + Tensor-Core tiles) | GEMM tiles execute on Tensor Cores | **2.2** | $\mathbf{51\times}$ |

Rows 1 to 3 are bounded by the $\approx 21$ ms floor above. Row 2 makes the attribution direct: the exact GEMM reformulation of Section 4 with Tensor Cores enabled, but with the intermediates materialized like the baselines, is only $1.1\times$ faster than eager, and its Tensor-Cores-off twin measures the same 100 ms, so neither the reformulation nor Tensor Cores helps while the $n^2$ traffic remains. Enabling Tensor Cores in row 1 likewise changes nothing (its GEMMs are 12.8 ms of 112.9 ms, so even *free* GEMMs would give at most $1.13\times$ by Amdahl's law), and `torch.compile` sits within $20\%$ of the floor. Row 4 escapes the floor by fusing the norms, `exp`, and accumulation into the GEMM tile loop, so the $n^2$ tiles never leave registers/shared memory (peak extra memory: 12.3 GB eager, 4.10 GB compiled, 8.1 MB ours) and global traffic is linear in $n$; only then is the kernel compute-bound. Tensor Cores then supply the final $4.94\times$ (row 5; paper Tables 4 and 5 give the matching ablations at $n=65{,}536$). For reference, PyKeOps (21.7 ms, Table 1) sits between rows 3 and 4: it fuses and streams without materializing, but generates scalar per-thread code that cannot use Tensor Cores.

**Q2 (CUDA cores are also applicable):** Agreed, and paper Table 5 measures precisely this ablation: the *identical* streamed kernel with Tensor-Core tiling disabled (FP32 CUDA-core `tl.dot`) is $4.94\times$ slower at $n=32{,}768$ and $4.80\times$ at $n=65{,}536$.

**Q3/Limitation 1(impact of TF32 precision):** *Short answer: no statistically detectable effect. The TF32 perturbation is $\approx 40\times$ below the estimator's statistical error and leaves oracle accuracy unchanged to 4 significant digits, and an exact FP32-IEEE mode is a one-flag fallback.* Mechanically, TF32 rounds only the `tl.dot` *inputs* to a 10-bit mantissa; storage and accumulation remain FP32, and no data is cast to FP16. New experiment: $d=16$, $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, against an FP64 reference of the full SD-KDE pipeline:

| Variant | Max rel. err. vs FP64 | Mean rel. err. vs FP64 |
|---|---|---|
| Flash-SD-KDE, TF32 | $8.9\times10^{-2}$ | $3.2\times10^{-2}$ |
| Flash-SD-KDE, FP32-IEEE mode | $5.5\times10^{-6}$ | $1.1\times10^{-6}$ |
| Eager PyTorch FP32 | $1.1\times10^{-5}$ | $1.6\times10^{-6}$ |

For calibration, the *statistical* error of the estimator itself at this $n$ (mean relative deviation of the FP64 SD-KDE from the true density, dominated by estimation variance in 16-D) is $1.26$, i.e., $\approx 40\times$ larger than the mean TF32 perturbation, and the debiased sample positions move by at most $1.1\%$ of $h$. At the metric level the two modes are indistinguishable: mean squared error against the oracle density is $2.16487\times10^{-8}$ (TF32) vs. $2.16484\times10^{-8}$ (FP32-IEEE). 

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

**Limitation (numerical accuracy with Tensor Cores):** Addressed quantitatively in Q3 above; the fidelity study and the FP32-IEEE mode will become a new appendix section.

**Limitation (broader ML applications):** The paper already evaluates three real, non-synthetic workloads (Appendix A.7): Statlog Shuttle (recovers 807/879 true outliers vs. 610 for the best published baseline, in 0.60 s), ALOI (0.09 s), and KDD-Cup 99 HTTP with $n=620{,}098$ (ROC-AUC 0.92 in 7.27 s). These are exactly the density/OOD-scoring workloads where a fast exact estimator matters. Applications such as diffusion-model components are out of scope for a systems paper about making exact SD-KDE practical, but we note (Appendix C.4) that the same kernels are drop-in for KDE-primitive pipelines such as KDE-based attention approximations and embedding-space OOD scoring.

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

Thank you for your thorough and positive review. To address your questions we ran new experiments on the same RTX A6000 workstation used in the paper; all numbers below are new measurements and will be added to the appendix of the revision.

**Q1 (performance as $d$ approaches 64 or 128, Tensor-Core utilization vs. $d$):** *Short answer: Flash-SD-KDE remains the fastest exact method at every $d$ up to 128 (see table), and larger $d$ makes the kernel more compute-bound, not less.* The bandwidth $h$ enters only through the scalar $\exp(-r^2/2h^2)$, so growing $h$ with $d$ does not affect the GEMM structure; $d$ is the reduction (K) dimension of the GEMM, and the arithmetic-intensity model of Section 4.1 gives $I_d(k) \sim C(d)\,k$ with $C(d)$ increasing in $d$. New experiment: $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, generic padded-tile kernel with a per-$d$ launch-parameter sweep (as in Appendix A.4), Tensor-Core on/off within the identical kernel, and the eager PyTorch baseline with TF32 *enabled*:

| $d$ | Flash-SD-KDE (ms) | Flash, Tensor Cores off (ms) | TC speedup | PyTorch eager, TF32 on (ms) | Flash speedup |
|---|---|---|---|---|---|
| 16 | 4.32 | 14.29 | $3.31\times$ | 112.8 | $26.1\times$ |
| 32 | 12.85 | 19.39 | $1.51\times$ | 113.9 | $8.9\times$ |
| 64 | 24.43 | 38.42 | $1.57\times$ | 115.1 | $4.7\times$ |
| 128 | 47.17 | 75.66 | $1.60\times$ | 116.4 | $2.5\times$ |

Runtime grows roughly linearly in $d$ from 32 to 128 as the FLOP model predicts, while the baseline is nearly flat in $d$ because its materialized $O(n^2)$ memory traffic (independent of $d$) dominates, which is also why the ratio narrows at large $d$ while remaining above $2\times$. Two conservative-side notes: these numbers use the *generic* padded kernel, and at $d=16$ the specialized kernel of Table 1 is a further $2\times$ faster (2.2 ms), so per-$d$ specialization (block-K tiling, launch shapes) should widen these margins; the same applies to the Tensor-Core ablation column, where the generic kernel at $d \geq 32$ has not yet received the per-$d$ tuning that produced the $4.94\times$ TC gain of Table 5. We will add this sweep to the appendix.

We believe the sweep above and the precision analysis below cover the two items you flagged as score-relevant (trade-offs at $d>32$; behavior under reduced-precision data types).

**Q2 (compute-bound with lower-precision types):** We do not cast to FP16/BF16: inputs and accumulation are FP32 and TF32 rounds only the `tl.dot` inputs. On numerical impact (new experiment against an FP64 reference, $n=32{,}768$): TF32 perturbs the density pointwise by $3.2\%$ on average, $\approx 40\times$ below the statistical error of the estimator at the same $n$ (mean relative deviation from the oracle: $126\%$), and the oracle MSE is unchanged to 4 significant digits ($2.16487\times10^{-8}$ vs. $2.16484\times10^{-8}$). A `precision_mode="fp32_ieee"` flag gives bit-comparable-to-FP32 results (max rel. err. $5.5\times10^{-6}$) at the no-Tensor-Core runtime of Table 5.

**Q3 (Laplace surrogate vs. full SD-KDE on non-Gaussian, heavy-tailed distributions):** *Short answer: full SD-KDE stays robust on heavy tails ($1.9\times$ to $3.1\times$ better ISE than KDE at every $n$); the Laplace surrogate is the most accurate at small $n$ but degrades at large $n$, so we recommend full SD-KDE when tails are unknown.* New experiment: 1-D two-component Student-$t_3$ mixture (infinite fourth moment), Silverman bandwidth, ISE against the true density over 5 seeds ($\times 10^3$, mean $\pm$ std):

| $n$ | KDE | Flash-SD-KDE | Flash-Laplace-KDE |
|---|---|---|---|
| 1,024 | $10.02 \pm 1.48$ | $5.35 \pm 1.15$ | $\mathbf{2.95 \pm 0.84}$ |
| 4,096 | $5.71 \pm 0.24$ | $2.32 \pm 0.16$ | $\mathbf{1.38 \pm 0.16}$ |
| 16,384 | $2.88 \pm 0.25$ | $\mathbf{0.94 \pm 0.13}$ | $1.23 \pm 0.13$ |
| 65,536 | $1.99 \pm 0.06$ | $\mathbf{0.86 \pm 0.06}$ | $2.53 \pm 0.18$ |

Full SD-KDE improves on KDE at *every* $n$, so the estimator we accelerate is robust to heavy tails. The Laplace surrogate is the most accurate at small-to-moderate $n$ but degrades at large $n$: its signed tail correction $(1 + d/2 - \|x\|^2/2h^2)$ amplifies variance where $t_3$ places many distant samples, and $t_3$ lacks the fourth moments that fourth-order kernels exploit asymptotically. This matches its framing as a fast surrogate: when tails are unknown, use full SD-KDE, which is exactly what Flash-SD-KDE makes cheap. We will add this study and recommendation to the revision.

**W2 (negative density values):** *Short answer: the effect is bounded and tiny, with integrated negative mass at most $0.02\%$ of unit mass even on a heavy-tailed worst case, and the nonnegative full SD-KDE path is always available.* On the heavy-tailed benchmark above, a worst case for the signed correction, the integrated negative mass of Flash-Laplace-KDE is at most $2.2\times10^{-4}$ (at $n=1{,}024$) and decreases in $n$ over the first three sizes ($8.6\times10^{-5}$ at $n=16{,}384$). For downstream integration, clipping at zero changes the estimate by at most this mass, and the full SD-KDE path is nonnegative by construction and available at 2.4 ms for $n=32$k, so the surrogate is never forced on a user.

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

Thank you for your insightful review. To address your questions we ran new experiments on the same RTX A6000 workstation used in the paper; all numbers below are new measurements and will be added to the appendix of the revision.

**W1 (novelty of the GPU techniques):** The individual techniques (tiling, streaming, mixed precision) are indeed known, as they were for FlashAttention, whose contribution was showing that reordering one specific primitive unlocks a new capability regime. Our contributions are estimator-specific and not obtainable from generic tooling: (i) the algebraic reordering $\sum_j (x_i - x_j)\varphi_{ij} = x_i \sum_j \varphi_{ij} - (\Phi X)_i$, which turns the empirical-score numerator into two GEMMs; (ii) the streaming formulation that makes *exact* SD-KDE run at $n \approx 10^6$ on one GPU for the first time; and (iii) Flash-Laplace-KDE, a new fused surrogate estimator with the same leading-order bias correction and no score pass. The strongest evidence that generic tooling does not get there: `torch.compile`, which has tiling, fusion, and Tensor Cores available, lands at 34.3 ms on the Table 1 workload (25.7 ms under PyTorch 2.9 in our rerun below) vs. our 2.2 to 2.4 ms, a $12\times$ to $14\times$ gap, and we show below why.

**W2 (baseline Tensor-Core usage; compiled graph of `torch.compile`):** *Short answer: the Table 1 baselines ran at PyTorch's default matmul precision (TF32 off, FP32 CUDA-core GEMMs), which we will state explicitly, and enabling Tensor Cores for them provably cannot help: even if the GEMMs became free, Amdahl's law caps the baseline's improvement at $1.13\times$.* Measured: 113.0 ms (TF32 on) vs. 112.9 ms (off); with TF32 on, cuBLAS does dispatch the Tensor-Core kernel `cutlass_80_tensorop_s1688gemm`, so the baseline then genuinely uses Tensor Cores, and it does not matter. The arithmetic: the profiler puts the baseline's GEMMs at 12.8 ms of its 112.9 ms CUDA time ($11\%$), so eliminating them entirely yields at most $112.9/100.1 = 1.13\times$, against the $51\times$ gap to Flash-SD-KDE. The GEMMs are cheap because at $d=16$ they perform only $2n^2d \approx 34$ GFLOP; the other $89\%$ of the runtime is spent moving $n^2$ intermediates through HBM, a cost GEMM throughput does not touch, because separate kernels communicate only through global memory: the Gram matrix and $\Phi$ (4 GB each) must each be written and read through HBM, $\approx 16$ GB $\approx 21$ ms at 770 GB/s, before any arithmetic. (This floor is a property of the kernel-sequence execution model, not of the problem: the intrinsic I/O is only the $O(nd)$ inputs and outputs, about 4 MB, which is precisely what fusion exploits.) In fact the standalone GEMMs are themselves memory-bound: the Gram GEMM ($K=16$) has arithmetic intensity $2 \cdot 16$ FLOP per 4-byte output $= 8$ FLOPs/byte, below even the FP32 roofline knee ($\approx 50$), and its 4 GB output write alone costs $\approx 5.2$ ms of its measured $\approx 6$ ms; accordingly, per-kernel profiling shows it at 6.11 ms as FP32 `sgemm` vs. 6.01 ms as TF32 `tensorop`. Tensor Cores only become decisive after fusion raises the arithmetic intensity: inside our streamed kernel ($\approx 72$ FLOPs/byte, Section 4.1) the $n^2$ traffic is gone, the kernel is compute-bound, and the same Tensor Cores then give $4.94\times$ (Table 5). The ladder below isolates each idea ($n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, $d=16$):

| # | Implementation | Idea added | Total (ms) | Cumulative speedup |
|---|---|---|---|---|
| 1 | PyTorch eager, cuBLAS GEMMs + separate elementwise kernels (113.0 with TF32 on) | (baseline) | 112.9 | $1\times$ |
| 2 | Our GEMM formulation, un-tiled: full $n^2$ materialization, Tensor Cores on (new; 100.3 with Tensor Cores off) | Tensor-Core GEMM reformulation alone; $n^2$ intermediates still round-trip HBM | 100.4 | $1.1\times$ |
| 3 | `torch.compile` (25.2 with `mode="max-autotune"`) | fuses all elementwise chains; still materializes one $n^2$ buffer for cuBLAS | 25.7 | $4.4\times$ |
| 4 | Our streamed Triton kernel, Tensor Cores disabled (Table 5) | full fusion + streaming: $n^2$ tiles never leave registers/shared memory | 10.5 | $10.8\times$ |
| 5 | **Flash-SD-KDE** (row 4 + Tensor-Core tiles) | GEMM tiles execute on Tensor Cores | **2.2** | $\mathbf{51\times}$ |

Row 2 makes the attribution direct: the exact GEMM reformulation with Tensor Cores enabled, but with the intermediates materialized like the baselines, is only $1.1\times$ faster than eager, and its Tensor-Cores-off twin measures the same 100 ms. For reference, PyKeOps (21.7 ms, Table 1) sits between rows 3 and 4: it fuses and streams without materializing, but generates scalar per-thread code that cannot use Tensor Cores. Rows 1 to 3 cannot go below the $\approx 21$ ms floor. Manual tiling does not escape it either: row-tiled eager measures 114 ms (caps peak memory, moves the same bytes), and even 2-D tiling at L2-resident tile sizes (the one configuration that could in principle bypass DRAM) is slower still, at 593 / 150 ms for the score pass with $512^2$ / $1024^2$ tiles, because $K=16$ cuBLAS GEMMs at those sizes do microseconds of work per $\sim$10 µs kernel launch. Nor does `mode="max-autotune"` (row 3): Inductor's Triton-GEMM epilogue fusion cannot fuse the chain GEMM $\to$ (`exp`, row-sum) $\to$ GEMM, because $\Phi$ has two consumers, one of them a matmul. The requested compiled graph, the complete Inductor execution plan for the compiled score pass, captured via `TORCH_LOGS=output_code`, with profiler timings:

```
buf0 = empty_strided_cuda((32768, 1))                 # row norms
triton_per_fused_mul_sum_0(X -> buf0)                 # <0.01 ms
buf1 = empty_strided_cuda((32768, 32768))             # the single 4 GB n^2 buffer
extern_kernels.mm(X, X^T, out=buf1)                   # cuBLAS -> cutlass_80_tensorop_s1688gemm
                                                      #   (TF32, Tensor Cores), 6.5 ms
buf2 = buf1  # reuse: Phi overwrites the Gram in place
triton_red_fused_add_clamp_exp_mul_permute_sub_sum_1  # dist + exp + row-sums fused;
  (buf2, buf0 -> buf2, buf4)                          #   reads Gram, writes Phi, 12.7 ms
extern_kernels.mm(buf2, X, out=buf3)                  # cuBLAS -> ampere_sgemm (Phi @ X), 6.3 ms
triton_poi_fused_add_div_mul_sub_2                    # debias epilogue, <0.01 ms
```

This plan shows Inductor doing everything right *within its execution model*: every elementwise op is fused into two Triton kernels, and the memory planner keeps a single $n^2$ allocation alive by overwriting the Gram with $\Phi$. But the two `extern_kernels.mm` calls are opaque cuBLAS launches, so that buffer must still be written and read twice through HBM (Gram write + read, $\Phi$ write + read $= 16$ GB $\approx 21$ ms at 770 GB/s $= 82\%$ of the measured 25.7 ms). Our implementation replaces the `mm` $\to$ fused $\to$ `mm` sandwich with one streamed Triton kernel whose tiles never leave registers/shared memory (8.1 MB peak extra vs. 4.10 GB).

**Q1 (latency breakdown):** New experiment: $d=16$, $n_{\text{train}}=32{,}768$, $n_{\text{test}}=4{,}096$, interleaved min-of-30 on the paper's A6000 workstation:

| Method | Score pass (ms) | KDE pass (ms) | Total (ms) | Score share |
|---|---|---|---|---|
| PyTorch eager, FP32 (Table 1 baseline) | 101.0 | 11.9 | 112.9 | 89% |
| PyTorch eager, TF32 enabled | 100.9 | 12.1 | 113.0 | 89% |
| `torch.compile`, TF32 enabled | 24.0 | 1.7 | 25.7 | 93% |
| Flash-SD-KDE | 1.93 | 0.28 | 2.21 | 87% |

(The eager and Flash rows reproduce Table 1's 113.3 ms and 2.4 ms; `torch.compile` improved from 34.3 ms to 25.7 ms under the newer PyTorch 2.9 used for the rerun, and remains $12\times$ slower than Flash-SD-KDE.) This confirms the Nsight finding in the paper that the score pass is 90 to 95% of end-to-end runtime for every implementation.

**Q2 (numerical impact of dtype casting):** There is no dtype cast: inputs are stored FP32, accumulation is FP32, and TF32 only rounds the `tl.dot` inputs' mantissas. Against an FP64 reference (new experiment, $n=32{,}768$): mean pointwise density error $3.2\%$, which is $\approx 40\times$ below the estimator's statistical error at the same $n$ (mean relative deviation from the true density: $126\%$); oracle-MSE agrees with the exact mode to 4 significant digits ($2.16487\times10^{-8}$ vs. $2.16484\times10^{-8}$). An exact `precision_mode="fp32_ieee"` flag is available (max rel. err. $5.5\times10^{-6}$ vs. FP64) at the cost of forgoing Tensor Cores ($\approx 5\times$ slower at scale, Table 5).

**Q3 (fusing the score and KDE passes):** It is possible, but Amdahl's law caps the benefit: the score pass is 90 to 95% of runtime (Q1 above and Appendix A.6), so fusing the two passes saves at most 5 to 10% for nontrivial code complexity, which is why we kept the two-kernel structure. Where fusion does pay is *within* a pass: Flash-Laplace-KDE is exactly the fully fused single-pass variant (correction applied inside the same tile loop), and it is $2\times$ to $5\times$ faster than the non-fused Laplace implementation across $n$ (Figure 7) while matching its accuracy.
