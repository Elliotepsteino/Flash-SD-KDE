# NeurIPS 2026 rebuttal experiments

Scripts and raw results backing the numbers in `paper/neurips2026/rebuttal.md`.
All runs use an RTX A6000 and the repo kernels; run from the repo root with
`PYTHONPATH=. python experiments/neurips2026_rebuttal/<script>.py`.

- `exp1_tf32_fidelity.py` — TF32 vs FP32-IEEE vs eager-FP32 pointwise error against an FP64 SD-KDE reference, plus the statistical-error scale vs the oracle density (16-D GMM, n=32,768 / 4,096).
- `exp2_breakdown.py` — score/KDE pass latency breakdown for eager PyTorch (FP32 and TF32-enabled), `torch.compile` (TF32), and Flash-SD-KDE; interleaved min-of-30; also dumps profiler kernel tables for the eager baselines.
- `exp3_dsweep.py` — d in {16, 32, 64, 128} sweep with a per-(d, precision) launch-parameter mini-sweep; Flash TF32, Flash FP32-IEEE (no Tensor Cores), and eager PyTorch with TF32 enabled.
- `exp4_heavytail.py` — 1-D Student-t3 mixture oracle study: ISE/IAE and integrated negative mass for KDE, Flash-SD-KDE, Flash-Laplace-KDE over 5 seeds.
- `exp5_compile_profile.py` — metric-level TF32 vs IEEE check, peak-extra-memory of the score pass (eager / compiled / Flash), and the `torch.compile` kernel profile.

Timing scripts (`exp2`, `exp3`) require an otherwise-idle GPU; accuracy scripts
(`exp1`, `exp4`, and the metric/memory parts of `exp5`) are load-insensitive.
- `exp6_tiled_torch.py` — can pure PyTorch close the gap: manually tiled eager (row-chunked), torch.compile default and mode="max-autotune", vs Flash-SD-KDE; all with TF32 enabled.
- `exp7_output_code.py` / `exp7_inductor_call_plan.txt` — TORCH_LOGS=output_code capture of the Inductor execution plan for the compiled score pass (quoted in the YaGf response).
- `exp9_tiled2d.py` — 2-D tiled cuBLAS score pass at L2-resident tile sizes: tests whether small tiles can bypass the HBM-traffic floor (they cannot; launch-bound).
- `exp10_materialization_32k.json` — full-materialization ablation of our kernel at n=32,768 (ladder row 2), via `python -m experiments.runtime.benchmark_rebuttal_16d_materialization_case --n-train 32768 --n-test 4096`.
- `exp2_results_papertorch271.json` — the exp2 breakdown rerun in the paper conda env (torch 2.7.1+cu118); numbers quoted in the YaGf Q1 table.
- `exp11_table1_harness.json` — rerun of the paper Table 1 harness (n=32,768, paper env): shows the torch.compile mean is contaminated by occasional recompilations (mean 80.2 / min 27.2 / std 74.9 ms over 3 repeats); steady state ~25.6 ms. Table 1 compile (and likely PyKeOps) entries should be refreshed for the revision.
