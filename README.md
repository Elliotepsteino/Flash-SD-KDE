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

This target is GPU-heavy and can take a long time. It writes all regenerated
plots to `file_storage/paper_plots/<timestamp>/generated` (not `paper/figures`).

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
