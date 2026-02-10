# Error Suite A100 16D

This suite evaluates Flash-SD-KDE in 16D using log-density comparisons, statistical
metrics against ground truth, and Pareto analysis.

## Single run

```bash
python -m experiments.error_suite_a100_16d.run --config configs/error_suite_a100_16d/default.yaml
```

## Sweep

```bash
python -m experiments.error_suite_a100_16d.sweep --config configs/error_suite_a100_16d/grid_pareto_16d.yaml
```

## Oracle error (16D MoG)

```bash
python -m experiments.error_suite_a100_16d.sweep --config configs/error_suite_a100_16d/grid_oracle_mog_16d.yaml
```

Generates `fig_oracle_error_vs_n_16d.pdf/png` and `oracle_mise_miae_vs_n_16d.pdf/png` with KDE, fused Laplace,
non-fused Laplace, and Emp-SD-KDE.

Outputs land in `file_storage/error_suite_a100_16d/<timestamp>/`.
