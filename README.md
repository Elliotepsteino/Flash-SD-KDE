# Flash-SD-KDE (v2)

This repo contains the v2 refactor of Flash SD-KDE with split-K kernels, explicit
precision modes, and a config-driven benchmark pipeline for MNIST vs Fashion-MNIST
OOD detection in PCA-16 space.

## Layout

- `flash_sd_kde/` — public API wrappers, configs, utilities, and references.
- `kernels/` — Triton kernels (split-K, symmetric atomic, reductions).
- `benchmarks/` — config-driven benchmark entrypoints (no CLI args).
- `plots/` — config-driven plotting + image grid generators.
- `tests/` — pytest suites (small + large).
- `file_storage/` — benchmark outputs and artifacts.

## Environment setup (uv)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you need a specific CUDA-enabled PyTorch build, install it before the
requirements and then re-run `uv pip install -r requirements.txt`.

## Running the benchmark

Edit `benchmarks/mnist_fashion_pca16_ood_config.py` to change seeds, sizes,
or backend variants (all backends run by default). Then run:

```bash
make bench.mnist_ood
```

Outputs land in `file_storage/benchmarks/mnist_fashion_pca16_ood/<run_id>/`.

## Plotting and grids

Edit `plots/plot_mnist_fashion_ood_config.py` or
`plots/save_density_ranked_grids_config.py`, then:

```bash
make plot
```

## Paper plots (validation)

To snapshot the current paper figures and regenerate what the repo can produce,
run:

```bash
make full_paper
```

This creates `file_storage/paper_plots/<timestamp>/baseline` (a copy of
`paper/figures`) and `file_storage/paper_plots/<timestamp>/generated` (newly
generated plots).

To run the experiments and then regenerate *everything needed for the paper*,
including the 16D oracle sweep, run:

```bash
make full_paper_experiments_plots
```

This target is GPU-heavy and can take a long time. It writes only the plots
used by the paper to `file_storage/paper_plots/<timestamp>/generated` (not
`paper/figures`).

## Paper figure commands (in order)

The list below follows the LaTeX figure order in the paper. For each figure,
replace `<ts>` with a timestamp of your choice (e.g., `20260210_120000`) to keep
outputs together in `file_storage/paper_plots/<ts>/generated`.

1. **Figure 1** — 16D runtime comparison (`runtime_16d_kde_sdkde.pdf`)
```bash
make run.nd_runtime_sweep PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> LEGACY_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
2. **Figure 2** — 16D oracle error (`fig_oracle_error_vs_n_16d.pdf/png`)
```bash
make oracle_16d_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```
3. **Figure 3** — 1D oracle error (`fig_oracle_error_vs_n_1d.pdf/png`)
```bash
make toy_1d_oracle_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```
4. **Figure 4** — 1D fused vs non‑fused runtime (`fig_fused_vs_nonfused_runtime.pdf/png`)
```bash
make toy_1d_oracle_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```
5. **Figure 5** — 16D utilization (`util_16d_sdkde_tensorcore.pdf`)
```bash
make run.triton_sd_kde_nd PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> LEGACY_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
6. **Figure 6** — 1D runtime appendix (`runtime_1d_kde_sdkde.pdf`)
```bash
make run.sweep PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> LEGACY_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
7. **Figure 7** — 1D utilization appendix (`util_1d_empirical_sdkde.pdf`)
```bash
make run.sweep PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> LEGACY_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
8. **Appendix oracle plot** — 16D oracle MISE/MIAE (`oracle_mise_miae_vs_n_16d.pdf/png`)
```bash
make oracle_16d_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

To generate all figures in one go, use:

```bash
make full_paper_experiments_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

## Legacy sweep scripts (deprecated)

The root-level `run_*.sh` scripts are deprecated in this refactor. We keep
Makefile targets with matching names for compatibility; they exit with a
message pointing you to the supported `bench.*` and `plot.*` targets.

```bash
make run.sweep
make run.nd_runtime_sweep
make run.triton_scaling
make run.triton_sd_kde_nd
```

## Tests

```bash
make test.small
make test.large
```

The large suite assumes CUDA is available; tests will skip if not.
