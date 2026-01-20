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
or backends. Then run:

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

## Tests

```bash
make test.small
make test.large
```

The large suite assumes CUDA is available; tests will skip if not.
