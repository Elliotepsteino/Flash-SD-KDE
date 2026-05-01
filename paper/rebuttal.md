Flash-SD-KDE: Accelerating SD-KDE with Tensor Cores
Download PDF
Elliot L Epstein, Rajat Vadiraj Dwaraknath, John Winnicki, Thanawat Sornwanee
24 Jan 2026 (modified: 30 Apr 2026)
Submitted to ICML 2026
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
BibTeX
CC BY-SA 4.0
Verify Author List: I have double-checked the author list and understand that additions and removals will not be allowed after the abstract submission deadline.
TL;DR: By exploiting Tensor Cores, we make score-debiased kernel density estimation practical at large scale
Abstract:
Score-debiased kernel density estimation (SD-KDE) achieves improved asymptotic convergence rates over classical KDE, but its use of an empirical score has made it significantly slower in practice. We show that by re-ordering the SD-KDE computation to expose matrix-multiplication structure, Tensor Cores can be used to accelerate the GPU implementation. On a 32k-sample 16-dimensional problem, our approach runs up to 
 faster than a strong SD-KDE GPU baseline and 
 faster than scikit-learn’s KDE. On a larger 1M-sample 16-dimensional task evaluated on 131k queries, SD-KDE completes in 
 on a single GPU, making score-debiased density estimation practical at previously infeasible scales.

Supplementary Material:  zip
Primary Area: General Machine Learning->Hardware and Software
Keywords: Kernel Density Estimation, Score Debiasing, Tensor Cores, SD-KDE, GPU Acceleration
Ethics Agreement: I certify that all co-authors of this work have read and are committed to adhering to the Call for Papers, Author Instructions, Research Ethics, and Peer-review Ethics.
LLM Policy: This submission allows Policy B.
Proceedings-only Option: If this paper is accepted, the authors tentatively plan to present it in person at the conference (as a poster and, if selected, as an oral).
Reciprocal Reviewing Status: This submission is NOT exempt from the Reciprocal Reviewing requirement. (We expect most submissions to fall in this category.)
Reciprocal Reviewing Author:  John Winnicki
Submission Number: 32966
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
16 / 16 replies shown
Add:
Paper Decision
Decisionby Program Chairs30 Apr 2026, 09:19 (modified: 30 Apr 2026, 11:45)Program Chairs, AuthorsRevisions
Decision: Reject
Comment:
The paper demonstrates an optimized implementation for score-debiased kernel density estimation (SD-KDE), a method recently proposed as an improvement to vanilla KDE. It shows the components of the method can be cast as matrix multiplication operations which are very efficient and optimized on modern hardware.

This paper received borderline scores and mostly with uncertain confidence attestations. Reviewers remained split at the end of the discussion, and consensus was not reached, though opinions in general were not strong in either direction and all reviewers ended up providing borderline recommendations (on whichever side). Consequently, the decision is based on the reviewers' helpful input and discussion as well as on my own reading of the paper.

Taking all into account, I regretfully recommend to not accept the paper for the following reasons.

The first and primary reason is concern about the relevance and interest of the work to this venue. The paper is focused on engineering and seems like a more natural fit to a system-aligned venue. In an ML venue such work could be deemed justified if it addressed the acceleration of core technique in ML that would be of wide interest to the community. In this case, the target of heavy engineering work is a very recently proposed method that has not yet gained the level of wide interest that would merit highlighting an dedicated implementational effort on it in a core ML venue.

Another reason is that the rebuttal contained a rather substantial volume of new experiments, code revisions and discussion of more related work to be added, which the authors have pledged to include in the revised version. While I do not doubt they would, the volume of planned revisions seems to me on the border of what could pass without an additional full round of review.

A final reason (which is perhaps more a suggestion for improvement) is that the manuscript reads like a technical report, diving straight into the details without spending time in the introduction distilling the main ideas and insights that drive and enable its results. That would have made the paper more accessible and appealing to a wide ML readership like the present venue targets.

Importantly, all reviewers as well as I acknowledge that the paper has substantive technical merit for the problem it chooses to undertake and that the effort it represents is rigorous and meticulous, and the authors are commended for their committed engagement in the discussion.

Official Review of Submission32966 by Reviewer yVUm
Official Reviewby Reviewer yVUm18 Mar 2026, 15:10 (modified: 08 Apr 2026, 04:46)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer yVUmRevisions
Summary:
This paper presents Flash-SD-KDE, a GPU-accelerated method that makes score-debiased kernel density estimation (SD-KDE) practical at scale. The authors reformulate both KDE evaluation and the costly score computation into matrix multiplications, enabling efficient execution on Tensor Cores with a streaming, memory-efficient design. They also propose a Laplace-corrected KDE as a cheaper approximation that preserves the main bias-reduction benefits of SD-KDE.

Strengths And Weaknesses:
Strengths
The paper is built on a solid theoretical foundation, with clear motivation for why SD-KDE is statistically preferable to classical KDE.
The proposed implementation delivers very large empirical speedups, especially relative to non-fused and standard baselines, which makes the practical value of the work clear.
Weakness
The overall performance still appears to be meaningfully limited by non-Tensor-Core operations, so the extent to which the method is truly Tensor-Core-driven should be characterized more carefully.
The claimed benefit of fusion is not yet fully convincing. The paper would benefit from a clearer operator-level breakdown of the baseline (i.e., where time is spent), rather than analyzing the fully fused (mixed) operations only. In particular, it is unclear whether a non-fused (or partially fused) design could still exploit batching or tiling effectively. Does the observed gains primarily come from op fusion that shows reduced memory traffic per tile? or mostly from increased Tensor Core utilization?
The baseline implementations appear relatively weak. Stronger compiler- or system-level baselines (e.g., torch.compile or other optimized fused-kernel approaches) should be included to better contextualize the reported speedups. In particular, the advantages of fusing Tensor Core operations with non–Tensor Core operations are not clearly demonstrated.
Please note that FlashAttention is particularly effective because it can be further optimized with techniques such as batching to increase arithmetic intensity, and its core computations are fully Tensor Core–friendly maximizing computation throughput.

Minor comments
There seems to be an inconsistency in the arithmetic intensity analysis, particularly in the denominator of 
. The scaling discussion suggests that both FLOPs and bytes are 
, so the resulting arithmetic intensity should be approximately constant in 
, but the derivation appears to imply otherwise.
What is relationship between Emp-SD-KDE, SD-KDE, and Flash-SD-KDE?
Soundness: 2: fair
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:
Is it possible to further approximate or reformulate the computation to alleviate bottlenecks from non–Tensor Core operations (e.g., exponentials, reductions)?
Do the observed performance gains primarily stem from operator fusion (e.g., reduced memory traffic per tile), or from increased Tensor Core utilization?
Can the proposed kernel be effectively batched across multiple queries or training sets, and if so, how would this impact performance and utilization?
While Flash-Laplace-KDE achieves the best performance across the reported experiments, it consistently shows lower error than Flash-SD-KDE. This raises the question of whether there are regimes where Flash-SD-KDE is actually necessary or provides clear advantages.
Limitations:
Yes.

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:
The authors have demonstrated validity across a range of configurations, and from a systems perspective, I think the approach is practical and useful. However, when considering broader applicability to real-world use cases, I do not feel sufficiently confident in my expertise to fully assess this aspect. I will therefore maintain my overall score of 3.

Rebuttal by Authors
Rebuttalby Authors (Elliot L Epstein, John Winnicki, Thanawat Sornwanee, Rajat Vadiraj Dwaraknath)30 Mar 2026, 23:50 (modified: 31 Mar 2026, 07:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for the constructive feedback.

W1(non-Tensor-Core bottlenecks):
We agree that "Tensor-Core driven" is the right characterization. Section 3 rewrites both KDE evaluation and the score numerator into GEMM structure, while already noting that the remaining work consists of norms, broadcasted additions, exponentials, and atomics. The same section's roofline analysis explicitly explains that the effective balance lies between the Tensor-Core and FP32 roofs because the kernel mixes Tensor-Core GEMMs with scalar work, and the optimization section likewise notes that nominal Tensor-Core peak is not attainable for exactly this reason. We will sharpen this wording in the abstract and performance discussion.

W2/Q2(fusion vs Tensor Cores):
While the gains come from both the biggest is from strong Tensor-Core utilization. Tensor-Core tiling accelerates the GEMM portions, while streaming/fusion avoids materializing full pairwise matrices and reduces launch and memory traffic. The fusion of the Score and the KDE step give a rather modest gain as around 90% of the total time is spent in the Score step.

W3(stronger baselines):
We now include stronger exact baselines across the Figure 1 operating points, including extending the PyKeOps baseline to a the full sweep of problem sizes and augmenting the torch baseline with torch.compile. Flash-SD-KDE remains the fastest exact method throughout the sweep; at n_train=32768, it runs in 2.13 ms versus 16.45 ms for PyKeOps, 34.25 ms for torch.compile, 112.84 ms for eager Torch, and 8117.25 ms for scikit-learn. For completeness, the full sweep is:

n_train	n_test	sklearn KDE (ms)	SD-KDE Torch (ms)	SD-KDE Torch compile (ms)	SD-KDE PyKeOps (ms)	Flash-SD-KDE (ms)
2048	256	30.7	1.1	6.0	1.9	0.7
4096	512	132.6	2.2	2.9	2.5	0.5
8192	1024	509.7	7.5	5.0	3.6	0.5
16384	2048	1895.7	29.0	11.3	6.7	1.1
32768	4096	7292.7	113.6	34.3	21.7	2.8
W4(arithmetic-intensity derivation):
In the 16-D tile-aware model, both FLOPs and bytes scale as O(k^2), so the arithmetic intensity is approximately constant. The O(k) scaling appears only in the separate 1-D appendix model, which uses a different streaming byte-count argument. We will rewrite this section so these two analyses cannot be conflated.

W5(Emp-SD-KDE vs SD-KDE vs Flash-SD-KDE):
Thank you for flagging this. SD-KDE / Emp-SD-KDE refer to the exact reference estimator, while Flash-SD-KDE is our accelerated implementation of that estimator. We will make sure the notation is consistent in the final revision.

Q1(non-Tensor-Core approximations):
Flash-Laplace-KDE can be viewed as a way to acheive this goal over Flash-SD-KDE, since it removes the explicit score pass while retaining the same leading bias correction. More aggressive approximations to exponentials or score computation are plausible future work, but the paper's current contribution is exact SD-KDE acceleration plus this first-order surrogate.

Q3(batching):
Batching across queries for the same density is used by default to speed up the computation, greatly improving the performance of the code. While most KDE use cases focus on a single training set we did not initially focus on batching across training sets, but this is an interesting extension.

Q4(when Flash-SD-KDE is needed):
Flash-SD-KDE remains important whenever one wants the exact SD-KDE estimator rather than the Laplace surrogate, and especially whenever a proper nonnegative density is required. We now also quantify the signed behavior of Flash-Laplace-KDE, the fraction of negative estimates is around 1%. This makes the tradeoff explicit: Flash-Laplace-KDE is a fast and approximate surrogate, while Flash-SD-KDE is the recommended choice when validity as a density matters.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer yVUm
Rebuttal Acknowledgementby Reviewer yVUm04 Apr 2026, 01:41Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (b) Partially resolved - I have follow-up questions for the authors.
Reasons:
Thanks to the authors for the detailed clarification.

I acknowledge that the paper is based on a clear idea: reformulating SD-KDE into GEMM and streaming it improves GPU utilization, leading to substantial speedups. The empirical results support this claim with comparisons against multiple baselines. However, I still have concerns regarding soundness and adaptability suspecting whether these benefits generalize across different GPUs and scenarios.

This mostly stems from the evaluation lacking detailed breakdowns and systematic analysis, leaving some uncertainty about robustness and practical applicability. Such lack of analysis also contrasts with prior Flash-style approaches, where memory/IO savings are typically quantified more explicitly.

Generalization across hardware. The evaluation is limited to a single GPU (RTX A6000) and relies on hardware-specific assumptions (e.g., SFU/FP32 ratio). Without a clear performance breakdown, it is difficult to attribute the gains to specific components or assess how they would translate across architectures with different characteristics (e.g., special-function throughput).

Query-level batching. The experiments fix 
, making it difficult to isolate the effect of query-level batching. Evaluating scenarios with fixed or small 
 or imbalanced settings would clarify behavior in more realistic configurations, where non-Tensor-Core operations (e.g., exponentials, reductions) may play a larger role.

Memory efficiency (fusion/streaming). While streaming and fusion are claimed to reduce memory overhead, there is no quantitative analysis or comparison against a non-fused baseline. In this draft, this benefit appears conflated with gains from Tensor-Core operations. Explicitly characterizing memory savings would strengthen this aspect.

 Replying to Rebuttal Acknowledgement by Reviewer yVUm
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors (Elliot L Epstein, John Winnicki, Thanawat Sornwanee, Rajat Vadiraj Dwaraknath)05 Apr 2026, 16:42 (modified: 05 Apr 2026, 16:46)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Comment:
Thank you for the careful follow-up.

W1 (Generalization Across Hardware)
We reran the rebuttal runtime sweep on a Tesla V100-PCIE-16GB using the same n_test = n_train / 8 setting, with the runtimes measured in milliseconds (ms):

n_train	n_test	sklearn KDE	SD-KDE Torch	SD-KDE Torch compile	SD-KDE PyKeOps	Flash-SD-KDE
2048	256	51.60	0.55	65.20	3.29	0.51
4096	512	270.94	1.78	63.74	4.15	0.62
8192	1024	810.39	6.51	54.25	5.51	1.19
16384	2048	3362.95	25.31	59.88	11.19	3.92
The V100 table shows that the same overall advantage persists on a different GPU family, and in the final revision we will expand this to include additional GPU types.

W2 (Query-Level Batching)
We fixed n_train = 32768 and swept n_test into a large range of values:

n_train	n_test	Torch (ms)	torch.compile (ms)	PyKeOps (ms)	Flash-SD-KDE (ms)
32768	4	100.97	32.44	15.64	1.92
32768	16	101.00	32.41	15.43	1.91
32768	64	103.27	33.44	15.48	1.91
32768	256	104.09	33.68	16.11	1.92
32768	1024	106.22	33.69	15.81	1.97
32768	4096	116.81	35.97	15.85	2.13
32768	16384	158.18	43.14	16.79	2.75
This shows that the gains persist throughout the batching sweep rather than depending on large test set batches.

W3 (Memory Efficiency / Streaming vs Tensor Cores)
First, to isolate streaming itself, we compared the streamed implementation against a full-materialization baseline at the n_train = 65536, n_test = 8192 size. The materialized baseline explicitly forms the full n_train x n_train train kernel matrix and the n_test x n_train query kernel matrix:

Method	Runtime (ms)	Peak Alloc (MB)
Streamed Flash + Tensor Cores	10.99	16.25
Streamed Flash + No Tensor Cores	49.52	16.25
Full Materialization + Tensor Cores	611.63	16400.50
This corresponds to roughly 1000x lower peak GPU memory allocation for the streamed implementation than for the full-materialization baseline.

Second, to isolate Tensor-Core utilization within the streamed Flash implementation, we compared Flash-SD-KDE against a copied no-Tensor-Core kernel path:

n_train	n_test	Flash-SD-KDE Tensor Core (ms)	Flash-SD-KDE no Tensor Core (ms)	no-TC / TC speedup
2048	256	0.30	0.31	1.04x
4096	512	0.28	0.35	1.24x
8192	1024	0.29	0.87	2.99x
16384	2048	0.66	2.83	4.27x
32768	4096	2.13	10.48	4.94x
65536	8192	9.30	44.59	4.80x
We do not use a fused score-plus-KDE kernel in the current implementation, but rather separate score and KDE kernels. In our experiments, the score kernel accounts for 90% of the end-to-end runtime, so fusing the score and KDE stages had only a minimal effect on total runtime, while making the code substantially more complex and less modular.

Thank you again for the insightful questions which have helped make the analysis in the paper more robust.

Official Review of Submission32966 by Reviewer 4RYf
Official Reviewby Reviewer 4RYf14 Mar 2026, 19:32 (modified: 06 Apr 2026, 21:03)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 4RYfRevisions
Summary:
Summary:

Score-debiased KDE (SD-KDE) from Epstein et al. (2025) achieves better asymptotic convergence than standard KDE at the cost of an extra quadratic pass for the empirical score. This paper shows that both the KDE evaluation and the score numerator can be expressed as matrix multiplications, enabling Tensor Core acceleration via a Triton kernel. The paper also introduces a Laplace-corrected KDE as a fast approximation that avoids the score pass entirely, with the same leading bias reduction. Claimed speedups are 40x over a PyTorch SD-KDE baseline and 2800x over sklearn at d=16, n=32k.

While the speedups seem real and I can appreciate the engineering -- this paper is far from being complete. My expertise is more on the statistical side of kernel methods and I am not a GPU systems expert, so take my hardware comments lightly. And the statistical scope concerns are independent of that:

The entire paper is about d=16. The Tensor Core approach requires d to be a multiple of 16, and all experiments are at d=16. This is a very narrow setting -- kernel density estimation in practice often runs at d=50-500 (e.g., post-processing MCMC chains). What happens at d=32 or d=64? The claim that this makes SD-KDE "practical at previously infeasible scales" is hard to evaluate without results outside d=16. This should be a limitation in the abstract, not buried as future work.

The Laplace-corrected KDE (Section 5) is not new statistically -- higher-order bias corrections via kernel modification are well-studied (Fan & Hu 1992, Jones & Signorini 1997, both cited). The contribution is identifying it as a fast surrogate for SD-KDE and implementing it in a fused kernel. But there is no variance or MISE analysis for Laplace-KDE, only the bias order. The O(h^4) bias improvement could be swamped by higher variance, especially since the estimator can be negative.

The Laplace-KDE can be negative. The paper tracks this as a "diagnostic" but never quantifies how often or how bad it is. For a density estimator, non-negativity is often a hard requirement. The MISE/MIAE comparisons in Figures 2-3 are on the "signed density," which is not the same as a proper density estimation comparison. In Figure 2, Emp-SD-KDE actually has lower MIAE than the Laplace variants -- so the headline contribution is dominated by a method that has the negativity problem.

The paper completely omits a long line of KDE work as well as recent work on compression schemes like Kernel thinning (Dwivedi & Mackey 2021) and several of its variants + follow-ups -- without a proper discussion of these line of work, the paper is incomplete.

Minor:

Figure 5 shows utilization below 30% of Tensor Core peak even at 1M samples, which is oddly low for something described as "firmly compute-bound." More explanation needed.
PyKeOps comparison (Table 1) is at a single (n=32k, ntest=4k) data point. Should be compared across the sweep in Figure 1.
Two separate contributions -- Tensor Core SD-KDE and Laplace-KDE -- are mixed together but never clearly integrated. Is Laplace-KDE also Tensor Core accelerated? It seems so, but the paper is not explicit.
Strengths And Weaknesses:
written in summary

Soundness: 1: poor
Presentation: 1: poor
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
written in summary

Limitations:
yes, nothing major

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:
I thank the authors for the detailed response and the additional data provided. I acknowledge the following updates to my previous claims:

Dimensionality: The code update to support 
 via padding resolves the primary usability concern.
Baselines: The performance sweep against torch.compile and PyKeOps provides the necessary context for the claimed speedups.
Laplace-KDE: The quantification of negative values (approx. 1%) and the clarification on its use as a surrogate are noted.
Hardware Utilization: The roofline analysis clarifies that the SFU/FP32 bottleneck for transcendental functions is the limiting factor.
Related Work: I acknowledge the commitment to include literature on Kernel Thinning and compression schemes.
I expect the authors to definitely include these new experiments and technical clarifications in the final revision. I will raise my score to 4.

Rebuttal by Authors
Rebuttalby Authors (Elliot L Epstein, John Winnicki, Thanawat Sornwanee, Rajat Vadiraj Dwaraknath)31 Mar 2026, 00:59 (modified: 31 Mar 2026, 07:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for the detailed and careful critique.

W1(high-dimensional scope beyond d=16)
The implementation was intentionally focused on d=16, because this is both a common moderate-dimensional problem size and a particularly natural fit for Tensor-Core tiling. Many higher-dimensional density-estimation pipelines are also reduced to this range before KDE, for example through PCA or related representation reduction, so this is not an artificial setting. The paper's claim is therefore not uniform acceleration for arbitrary d, but exact SD-KDE acceleration in the regime where Tensor-Core execution is most effective. The implementation now supports arbitrary feature dimension by padding to a multiple of 16, we will revise the paper so this design choice and scope are stated explicitly.

W2(Laplace correction not statistically new)
The Laplace correction is not new as a statistical kernel construction, the contribution is instead the identification of this correction as the appropriate first-order surrogate for SD-KDE in our setting, its connection to SD-KDE's leading bias reduction, and its fused GPU realization. In our view, that is a substantive contribution, not merely a restatement of prior higher-order KDE theory.

W3(Laplace-KDE theory and negative densities)
The main result of the paper is Flash-SD-KDE, i.e. the accelerated implementation of the exact SD-KDE estimator. Flash-Laplace-KDE is presented as an additional surrogate that trades exactness and non-negativity for lower cost; it is not the method on which the paper's core claim rests. For applications that require a proper density, there is no need to rely on the Laplace surrogate: Flash-SD-KDE is the more appropriate method precisely because it preserves non-negativity while retaining the same exact estimator. At the same time, we do not think Flash-Laplace-KDE is unsupported. The paper gives its leading-bias connection to SD-KDE and evaluates it directly by oracle MISE/MIAE against known densities, which is the relevant empirical criterion here. During rebuttal we also quantified the signed behavior explicitly: across the sweep, the fraction of negative estimates is low, around 0.75% to 1.14%, while the negative-mass fraction is around 6.5% to 10.7%. We will report these diagnostics prominently and clarify in the paper that Flash-Laplace-KDE is a fast signed surrogate, whereas Flash-SD-KDE is the default recommendation whenever a valid density is required.

W4(missing literature context and completeness)
While the paper will benefit from discussing the approximate-KDE and compression literature more fully, those lines of work primarily approximate classical KDE, whereas our paper accelerates exact SD-KDE. They are therefore important context and useful future comparison points, but they are not direct substitutes for the contribution studied here. The core contribution remains a working exact SD-KDE GPU implementation, with clear speedups over strong exact baselines and direct oracle error evaluation. We will sharpen the positioning so this exact-versus-approximate distinction is clearer in the revision.

M1(utilization below 30% vs compute-bound)
We do not see a contradiction here. Compute-bound does not imply near-peak Tensor-Core utilization when the kernel mixes Tensor-Core GEMMs with scalar work such as norms, exponentials, and atomics. As the paper's roofline discussion explains, the relevant operating point lies between the Tensor-Core roof and the FP32 / SFU roof, which is exactly why the kernel can be compute-bound while remaining below nominal Tensor-Core peak utilization.

M2(PyKeOps only at one point)
A sweep is the right comparison, and we now include it together with a torch.compile exact baseline:

n_train	n_test	sklearn KDE (ms)	SD-KDE Torch (ms)	SD-KDE Torch compile (ms)	SD-KDE PyKeOps (ms)	Flash-SD-KDE (ms)
2048	256	30.7	1.1	6.0	1.9	0.7
4096	512	132.6	2.2	2.9	2.5	0.5
8192	1024	509.7	7.5	5.0	3.6	0.5
16384	2048	1895.7	29.0	11.3	6.7	1.1
32768	4096	7292.7	113.6	34.3	21.7	2.8
Lower is better. Measurements use the 16-D Gaussian-mixture benchmark with n_test = n_train / 8 and exclude one-time warmup / JIT costs.

M3(SD-KDE and Laplace-KDE are mixed together)
Flash-Laplace-KDE is also Tensor-Core accelerated and fused, and it is introduced precisely as a surrogate obtained by simplifying the exact SD-KDE pipeline. We will revise the presentation so the exact path and the surrogate path are separated more explicitly, while keeping their relationship clear.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer 4RYf
Rebuttal Acknowledgementby Reviewer 4RYf03 Apr 2026, 14:31Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:
I appreciate the authors for the detailed response and the additional data provided. I acknowledge the following updates to my previous claims:

Dimensionality: The code update to support 
 via padding resolves the primary usability concern.
Baselines: The performance sweep against torch.compile and PyKeOps provides the necessary context for the claimed speedups.
Laplace-KDE: The quantification of negative values (approx. 1%) and the clarification on its use as a surrogate are noted.
Hardware Utilization: The roofline analysis clarifies that the SFU/FP32 bottleneck for transcendental functions is the limiting factor.
Related Work: I acknowledge the commitment to include literature on Kernel Thinning and compression schemes.
I expect the authors to definitely include these new experiments and technical clarifications in the final revision. I will raise my score to 4.

 Replying to Rebuttal Acknowledgement by Reviewer 4RYf
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors (Elliot L Epstein, John Winnicki, Thanawat Sornwanee, Rajat Vadiraj Dwaraknath)05 Apr 2026, 17:02Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thank you for the updated positive assessment of the work and for the useful suggestions. We will incorporate these additional experiments, technical clarifications, and related-work updates in the final revision.

Official Review of Submission32966 by Reviewer Njy1
Official Reviewby Reviewer Njy113 Mar 2026, 15:30 (modified: 06 Apr 2026, 21:03)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Njy1Revisions
Summary:
This paper proposes an optimized GPU implementation of SD-KDE, a new kernel density method that was introduced a few months ago at NeurIPS. KDE is an important computational intrinsic that lies at the core of many important ML methods because it allows the data-generating distribution to be approximated from samples. This has broad applications - for example, self-attention can be written as a weighted KDE sum, KDE is useful to compute divergences between datasets, etc. Unfortunately, KDE suffers from the curse of dimensionality, in the sense that we require 
 data points to attain a fixed-error approximation to the data-generating distribution. This is problematic because the time complexity of exact-KDE is 
 (which makes it 
 if we desire a fixed error).

SD-KDE improves the error dependence to 
, but has a quadratic-cost preprocessing step that doubles the cost of the query. This paper presents a very strong GPU engineering / systems effort to make the cost of SD-KDE feasible in practice for datasets that are up to a few million points in size. The paper contains experiments on synthetic data drawn from Gaussian mixture distributions, showing that the GPU-accelerated SD-KDE has a better tradeoff between error (MISE) and computation time than baselines.

Strengths And Weaknesses:
Summary
Overall, I am a bit conflicted about this paper. The research direction is to improve the MISE of the density estimates that can be achieved in practice, and I feel that this is a good direction. There has been a large amount of algorithm/theory work on this problem in the last 10 years, but most of the work has focused on the fast/approximate computation of the KDE sum itself (not on alternatives to the KDE sum, such as SD-KDE, that might offer better MISE scaling). There is also some strong engineering work in the paper and the implementation will surely be useful to anyone who needs to compute exact KDE results on a few million points in relatively low dimensions.

On the other hand, the practical utility is not clear. While KDE is a strong focus of ML research in recent years, it is because of the theoretical / practical implications for non-KDE workloads. It is somewhat hard to find applications that need to directly compute the KDE, and it is still harder to find modern applications that need to do this in fewer than 100 dimensions. The SD-KDE paper, which this paper builds on, is interesting for theoretical reasons because it offers an asymptotic error reduction on a classical ML problem. The case for this paper is less clear.

In summary, I feel this paper is a very well-executed engineering effort in a good direction, but which operates in a setting that is rarely encountered in practice. I also worry that in practical settings, the best approach may require a combination of algorithmic work and GPU acceleration. Details below.

Strengths:
Strong engineering work and good presentation.
The research direction is good, and it is especially nice to see work that quantifies the MISE against a known distribution (most work just shows approximation errors of the KDE, which are a bit artificial since they do not relate back to the actual density).
Weaknesses:
There are not any algorithmic innovations - this paper reads as an engineering effort to build a GPU library that computes the SD-KDE efficiently. While there is a strong precedent of engineering / systems papers at ICML that accelerate an algorithm, they are typically most useful when the method is widely used (e.g., flash attention accelerates nearly any model that uses attention).
This is not true for KDE - the biggest reason why the ML community is presently interested in fast KDE algorithms is because they can often be parlayed into better algorithms for something else. For example, an approximate KDE algorithm with sub-quadratic runtime can be turned into an approximate attention algorithm, and we have seen this happen multiple times: the KDEFormer is based on a theoretical KDE improvement at FOCS 2020, polynomial kernel sketching methods were later turned into PolySketchFormer, the RACE KDE algorithm was used to implement RACE attention, etc. This kind of reuse does not seem possible for this paper, since the implementation is specifically for density estimation.

Limited, synthetic experiments. The paper currently only contains experiments at relatively small scale (
) and low dimension 
). Most modern applications are in 
, and I feel the paper would be significantly stronger if it were to show that the GPU version of SD-KDE improves a real, concrete density estimation application (this would also address weakness #1). If the authors are looking for some, I would recommend checking out applications where people need to compute distances / divergences between datasets. For example, this shows up in LLM pre-training data mixtures, where people want to understand whether a new dataset or training example is similar / dissimilar to datasets already used for training - see this paper for discussion. It also shows up in heterogenous federated learning for similar reasons. Perhaps a concrete application can be found in one of these areas.

Missing literature context - there is a whole line of work to break the curse of dimensionality in traditional KDE using hashing and randomized algorithm approximations. It would be interesting to compare against these algorithms, since they represent the state of the art in algorithmic KDE acceleration.

The seminal work in this area is the "hashing-based estimator" (HBE) framework by Charikar and Siminelakis at FOCS'17, which at a high level computes the kernel sum by (1) hashing all of the points in the dataset into LSH buckets, (2) hashing the query into the same buckets, (3) computing the kernel values for all values in the buckets where the query lands, and (4) repeating this process several times to refine the estimate. The intuition is that since LSH sends nearby points to the same buckets, and kernel values are large only for nearby points, we can get most of the mass in the kernel sum by sampling from the LSH buckets.

There have been several improvements to the original HBE work, initially focused on improving the space requirements by sampling ("Space and time efficient KDE" at NeurIPS'19) and getting the scheme to work in practice ("Rehashing kernel evaluation" at ICML'19), with a later paper that extends the scheme to more general classes of kernels ("Multi-resolution hashing" at FOCS'20). One unavoidable source of inefficiency with the HBE idea is that for radial kernels, we may have points with similar kernel contributions go to different LSH buckets (e.g., because they are on opposite sides of the query - imagine a situation where the query is the midpoint between two data clusters). This leads to redundant kernel value calculations. The most recent progress on this method (Charikar 2020), addresses this issue by counting the number of points in spherical shells around the query. This newer method (1) uniformly downsamples the dataset into subsets of exponentially-smaller sizes - one for each spherical shell radius, (2) creates an LSH index for each subset, and (3) at query-time, looks through each bucket to find the number of points in the shell radius.

Ultimately, I feel that the best way to implement fast, practical SD-KDE may be a combination of algorithmic and engineering work. For example, the original SD-KDE paper claims that the method is empirically robust to small variations in the score. If that is the case, could the score be computed in a sub-quadratic way using any of the fast-approximate-KDE algorithms? Could a rigorous MISE bound be established for scores that are computed using these approximations?

Soundness: 4: excellent
Presentation: 4: excellent
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
Is there an example of a real application that would be accelerated by this technique? And, if so, does the GPU SD-KDE work on this application?
How does SD-KDE compare against sparse, approximate KDE algorithms on realistic (i.e. high-dimensional) workloads? The approximate algorithms run on CPU, but have to do significantly less computation.
Can the existing fast-KDE work be combined with SD-KDE in some way, e.g., by using approximate algorithms to get the score before doing the full 
 computation on GPU?
Limitations:
yes

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Rebuttal by Authors
Rebuttalby Authors (Elliot L Epstein, John Winnicki, Thanawat Sornwanee, Rajat Vadiraj Dwaraknath)31 Mar 2026, 02:57 (modified: 31 Mar 2026, 07:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for the thoughtful and well-informed feedback.

W1(primary contribution is engineering rather than algorithmic)
The paper's main contribution is indeed in ML systems. We think that is substantive here because SD-KDE is already a statistically attractive estimator; the barrier was computational, not conceptual. Making the exact estimator practical at meaningful scale is therefore not merely an implementation detail, but the key step needed to make SD-KDE usable in practice. It may also matter beyond direct density estimation: once KDE-family primitives become fast enough, they can serve as building blocks inside larger pipelines and potentially future KDE-based neural architectures.

Q1/W2(practical utility, experimental scope, and real data)
We do not think the practical utility is as narrow as the review suggests. KDE is a real workload in settings where one directly needs density values, scores, or density-derived similarities and divergences, including embedding-space dataset comparison, data-mixture analysis, anomaly / OOD scoring, and related nonparametric inference pipelines. Concretely, these include tasks such as measuring whether a candidate dataset is similar to an existing corpus, identifying atypical or out-of-distribution examples in an embedding space, or selecting informative low-density points in data-constrained pipelines.

We also believe the synthetic oracle benchmarks are the correct primary evaluation for the paper's central claim because they directly measure MISE / MIAE against known densities. To complement this, we ran a higher-dimensional embedding-space benchmark in which SD-KDE were used to distinguish in-distribution MNIST examples from OOD Fashion-MNIST examples after PCA reduction to 64D. In that setting, exact SD-KDE still performed well, achieving ROC AUC 0.9279 at n_train=4000.

We also ran a small real-data selection pilot in embedding space: using 384D MiniLM embeddings of 19,983 instruction-tuning examples, we scored points with both an approximate KDE baseline and Flash-SD-KDE, and then sampled 1,000 training examples with probability proportional to inverse estimated density, favoring rarer examples in a data-constrained regime. In this pilot, the approximate KDE scoring step took 0.74 s, while Flash-SD-KDE took 1.02 s. We view this pilot as preliminary application evidence and as an end-to-end feasibility check for using Flash-SD-KDE as an embedding-space selection primitive. We will include the full experimental details in the final revision.

Q2/W3(approximate-KDE literature, comparison, and hybrid directions)
The goal of this paper is to speed up SD-KDE itself, precisely because SD-KDE has stronger statistical convergence properties than classical KDE. Approximate-KDE methods are therefore not a drop-in replacement for the contribution studied here: they typically approximate classical KDE and thus change the estimator, rather than making exact SD-KDE practical. This is also why we would position them as complementary context rather than as like-for-like baselines for the exact SD-KDE contribution. That said, this literature should be discussed more fully, and we will expand the related work to include HBE, rehashing / multi-resolution hashing, and kernel thinning / compression. More broadly, hybrid approximate-score plus hardware-accelerated evaluation directions are plausible and interesting, and a fast exact SD-KDE implementation is a useful building block for such work rather than being in tension with it.

Q3(can fast approximate KDE ideas help SD-KDE)
Conceptually yes. In future work we are exploring using non-uniform FFT to further speed up the scaling for the KDE and Score computation. Flash-Laplace-KDE already removes the explicit score pass as a first surrogate step, and approximate score computation is a natural next dire

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer Njy1
Rebuttal Acknowledgementby Reviewer Njy103 Apr 2026, 21:24Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (b) Partially resolved - I have follow-up questions for the authors.
Reasons:
Thank you for the rebuttal - I have a few follow-up comments. At this time, I am inclined to maintain my rating.

W1(primary contribution is engineering rather than algorithmic)

Q1/W2(practical utility, experimental scope, and real data)

Most of these KDE applications are in much higher dimensions than those where SD-KDE has been validated. It would be great to back up the claims of practical utility (which I feel are required for an ML systems paper, in a way that is not required for a theory paper) with citations or references to problems that SD-KDE could directly address. Ideally, the paper would contain experiments on real datasets that show this utility.

I do want to be clear that I appreciate the effort during the rebuttal to show results on higher-dimension datasets, and I also do agree that ML systems work to improve the empirical cost-vs-MISE tradeoff is worthwhile. I think the rebuttal experiments are a good start, but (in my opinion) a bit more evidence is needed to put this paper over the bar.

For example, when we wish see if a candidate is similar to an existing corpus of documents, we typically do it by analyzing text embeddings. These embeddings have dimensions in the high 100s (and more recently, in the 1000s). This dimensionality goes beyond what was considered in the original paper, and we really would need to see a study that scales both the dimensionality and the data size to show the practical utility here.

Model Name	Standard Dimensions	Configurable Range / Features
**OpenAI text-embedding-3-large	3072	Configurable (via Matryoshka Representation Learning) down to 256
Gemini Embedding	3072	Configurable down to 256
Qwen3-Embedding-8B	4096	User-defined (32 – 4096)
EmbeddingGemma-300M	768	Configurable to (128, 256, 512)
many more	-	-
Anomaly detection: It is good to see that a max-likelihood classifier based on SD-KDE works on Fashion MNIST (albeit with the PCA reduction). However, I do not find this result convincing enough, because Fashion-MNIST is known to be very easy and intrinsically low-dimensional. For example, dimension reduction to a single pixel is enough to get about 70 - 90% classification accuracy.

In an ideal world, the paper would run anomaly detection on something more challenging, such as the datasets in "Arrays of (locality-sensitive) Count Estimators (ACE): High-Speed Anomaly Detection via Cache Lookups." That paper also contains baseline numbers for non-KDE methods that give an idea of what competitive performance looks like, so that we can see whether SD-KDE is useful compared to other anomaly detection tasks. Fashion-MNIST is also quite small - it would be good to see results on something much larger (such as the Yandex Text-to-Image dataset, which has 1 billion 200-dimensional vectors from either images or text). The MiniLM experiment is much more realistic in dimension, even if it is inconclusive (because we cannot argue that the higher-cost SD-KDE also leads to higher downstream quality).

Approximate-KDE methods are therefore not a drop-in replacement for the contribution studied here: they typically approximate classical KDE and thus change the estimator, rather than making exact SD-KDE practical. This is also why we would position them as complementary context rather than as like-for-like baselines for the exact SD-KDE contribution.

If the goal is to optimize the "cost-vs-MISE" tradeoff, then I argue approximate KDE methods and exact KDE methods are indeed drop-in replacements for one another in practice. This is particularly true when we use PCA to reduce the dimension of the input to SD-KDE, as is done here.

MISE tradeoffs: I would expect for approximate KDE methods to have a lower cost (but probably higher MISE) and SD-KDE to have a higher cost (but probably lower MISE). To know for sure would require that we make a cost-vs-error scatterplot, where each point represents a tuning configuration of a method (e.g., PCA reduction to X dimensions, then SD-KDE, or some hyperparameter configuration for approximate KDE). This would permit us to compare SD-KDE against all of the other methods in use and to see whether there is some regime where it is the best practical choice.

If SD-KDE had Pareto optimality over approximate KDE methods in the low-MISE regime of the tradeoff, it would be a very interesting result that strongly motivates the development of a GPU-accelerated system (since the approximate methods are typically CPU-only). That result, combined with good anomaly detection results on more challenging/realistic anomaly detection data, would make a much stronger case to accept this paper.

 Replying to Rebuttal Acknowledgement by Reviewer Njy1
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors (Elliot L Epstein, John Winnicki, Thanawat Sornwanee, Rajat Vadiraj Dwaraknath)06 Apr 2026, 14:45Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thank you for the thoughtful follow-up.

Q1 (Higher Dimensional Real-Data Evidence)
In the final revision we will add more citations to concrete settings where Flash-SD-KDE can have a large utility.

As a natural scale-up of the current embedding-space selection pilot, the final revision will use a pool of 500k embedded Tulu-3 training examples, sample 100k examples from that pool for SFT, and fine-tune Qwen-3-0.6B on those 100k sampled examples. We will evaluate by held-out perplexity and OLMES task performance, and compare three selection strategies: uniform random sampling, inverse-density sampling with the hashing-based approximate KDE baseline, and inverse-density sampling with Flash-SD-KDE. We will report both selection/runtime cost and downstream metrics.

Q2 (Anomaly Detection)
To address the request for a harder anomaly-detection benchmark, we compared Flash-SD-KDE directly against the methods reported in ACE [1] using the same anomaly-detection protocol. The non Flash-SD-KDE rows are from the paper [1], while the Flash-SD-KDE rows are our measurements on the RTX A6000. The main takeaways are that Flash-SD-KDE gives the strongest anomaly-recovery result on Shuttle with the fastest runtime, remains competitive on ALOI while running much faster than ACE. The full tables are shown below.

Statlog Shuttle (n=34987, outliers=879, d=9)

Method	Outliers Reported	Correctly Reported	Outliers Missed	Execution Time (s)	Runtime / ACE runtime
ACE	6763	273	606	0.81s	1x
LOF	4356	381	498	14.12s	17.4x
kNN	4897	493	386	12.35s	15.2x
kNNW	5264	610	269	13.54s	16.7x
LoOP	6145	201	678	14.51s	17.9x
LDOF	6433	330	549	16.42s	20.3x
ODIN	9775	375	504	12.21s	15.1x
KDEOS	12630	314	565	11.73s	14.5x
COF	9133	280	599	13.45s	16.6x
LDF	9809	375	504	19.93s	24.6x
INFLO	4488	183	696	14.03s	17.3x
FastVOA	8532	271	608	235.10s	290.2x
Flash-SD-KDE	5130	807	72	0.60s	0.74x
Object Images (ALOI) (n=50000, outliers=1508, d=27)

Method	Outliers Reported	Correctly Reported	Outliers Missed	Execution Time (s)	Runtime / ACE runtime
ACE	7216	340	1168	1.26s	1x
LOF	4476	519	989	72.31s	57.4x
kNN	5428	447	1061	63.27s	50.2x
kNNW	5558	329	1508	89.96s	71.4x
LoOP	5121	253	1179	59.97s	47.6x
LDOF	7501	470	1038	60.39s	47.9x
ODIN	10110	162	1346	72.69s	57.6x
KDEOS	9515	404	1104	55.89s	44.36x
COF	8746	284	1224	81.74s	64.9x
LDF	9133	301	1207	60.51s	48.0x
INFLO	10328	420	1088	72.13s	57.2x
FastVOA	8931	319	1189	291.10s	231.0x
Flash-SD-KDE	7250	437	1071	0.09s	0.07x
We were not able to reproduce the KDD-Cup99 HTTP benchmark exactly from the ACE paper text alone, because the public KDD99 HTTP variants we found did not match the setting reported in the paper [1]: n=596853, outliers=1055, d=36. Instead, we ran Flash-SD-KDE on a public KDD99 HTTP point-anomaly benchmark release at similar scale (n=620098, outliers=1052, d=29). Although it is not the exact ACE subset, it still shows strong large-scale performance, with total runtime 7.27s, ROC-AUC=0.9200. For reference, ACE reports 23.33s on its original KDD-Cup99 HTTP subset.

Q3 (Cost-Performance Pareto Frontier)
Flash-SD-KDE occupies a favorable part of the practical performance-cost frontier relative to approximate alternatives. The anomaly detection results above show that it is competitive on both quality and runtime at realistic scales, and on Shuttle, clearly stronger than ACE on the task itself.

This is consistent with our oracle results in Figure 2: Flash-SD-KDE achieves a substantial MISE improvement over standard Silverman KDE, more than an order of magnitude in the reported setting, while also running an order of magnitude faster than the best baseline.

Taken together, these results support the following picture: by having a faster convergence rate than standard KDE, Flash-SD-KDE achieves substantial MISE improvements while remaining competitive on runtime with approximate alternatives at large scale. We expect approximate methods with linear scaling to be faster in the very largest regimes, and we will include experiments at that scale in the final revision.

Reference
[1] Chen Luo and Anshumali Shrivastava. 2018. "Arrays of (locality-sensitive) Count Estimators (ACE): Anomaly Detection on the Edge." In WWW 2018: The 2018 Web Conference, April 23–27, 2018, Lyon, France. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3178876.3186056

Official Review of Submission32966 by Reviewer v3qz
Official Reviewby Reviewer v3qz12 Mar 2026, 12:23 (modified: 06 Apr 2026, 21:03)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer v3qzRevisions
Summary:
This paper introduces Flash SD-KDE, a hardware-aware approach designed to resolve the computational bottlenecks of score debiased kernel density estimation (SD-KDE). The authors reformulate the SD-KDE computation into a matrix-multiplication structure and utilize streaming accumulation, allowing the algorithm to maximize GPU Tensor Core efficiency without prohibitive memory traffic. Further, to provide a faster alternative, the paper also introduces a fused Laplace-corrected estimator that achieves similar leading order bias reduction without the expensive empirical score pass.

Strengths And Weaknesses:
Strengths

The paper provides a clear analytical model of arithmetic intensity to prove that their 16-D formulation is compute-bound rather than memory-bound.
The empirical results strongly support the claims, demonstrating up to 40x speedups over the PyTorch baseline and 2,800x speedups over scikit-learn.
Weakness

The paper leaves the extension to dimensions that are not multiples of 16 to future work, somewhat limiting the immediate generalizability of the optimal results.
Soundness: 3: good
Presentation: 3: good
Significance: 4: excellent
Originality: 3: good
Key Questions For Authors:
The paper states that the Laplace-corrected KDE 
 is a fourth-order kernel that can yield negative density values in the tails. In downstream machine learning tasks that require valid probability distributions (e.g., evaluating log-likelihoods or generating samples), how do you propose practitioners handle these negative values?

Limitations:
NA

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 1: Your assessment is an educated guess. The submission is not in your area, or the submission was difficult to understand. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Rebuttal by Authors
Rebuttalby Authors (Elliot L Epstein, John Winnicki, Thanawat Sornwanee, Rajat Vadiraj Dwaraknath)31 Mar 2026, 01:04 (modified: 31 Mar 2026, 07:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for the positive assessment and the important question.

W1(generalizability beyond multiples of 16):
We have updated the code to handle dimensions that are not a multiple of 16 by padding the data to reach a dimension that's a multiple of 16, allowing an efficient computation.

Q1(handling negative Laplace densities):
For applications that require a valid density, such as likelihood evaluation or sampling, Flash-SD-KDE is the better recommendation because it preserves non-negativity. Flash-Laplace-KDE should be viewed as a fast surrogate when that tradeoff is acceptable.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer v3qz
Rebuttal Acknowledgementby Reviewer v3qz06 Apr 2026, 23:18Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:
I thank the authors for addressing the queries.