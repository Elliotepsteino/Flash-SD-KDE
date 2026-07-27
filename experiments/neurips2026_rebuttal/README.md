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
