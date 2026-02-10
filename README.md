# Flash-SD-KDE: Accelerating SD-KDE with Tensor Cores

Official code release for the paper `Flash-SD-KDE: Accelerating SD-KDE with Tensor Cores`.

![Flash-SD-KDE 16D Runtime Banner](https://raw.githubusercontent.com/Elliotepsteino/Flash-SD-KDE/main/paper/figures/runtime/runtime_16d_kde_sdkde.png)

## Quick Links

- Experiment reproduction guide: `EXPERIMENTS.md`
- Paper source: `paper/main.tex`
- Minimal runnable API demo: `example.py`

## Repository Layout

- `flash_sd_kde/`: public API, estimator wrapper, config helpers, and references
- `kernels/`: Triton kernel implementations
- `benchmarks/`: benchmark entrypoints
- `plots/`: plotting scripts
- `experiments/`: runtime and error-suite experiment pipelines
- `tests/`: pytest suites
- `file_storage/`: generated artifacts and experiment outputs

## Development Setup (Source Install)

```bash
git clone https://github.com/Elliotepsteino/Flash-SD-KDE.git
cd Flash-SD-KDE
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

python -c "from flash_sd_kde import FlashSDKDE; print('ok')"
```

## Install from PyPI

```bash
pip install flash-sd-kde
```

If you need a specific CUDA-enabled PyTorch build, install it first, then run:

```bash
uv pip install -r requirements.txt
```

## sklearn-Style API

`flash_sd_kde` exposes a sklearn-style estimator wrapper:

```python
import numpy as np
from flash_sd_kde import FlashSDKDE

X_train = np.random.randn(4096, 16).astype(np.float32)
X_query = np.random.randn(1024, 16).astype(np.float32)

est = FlashSDKDE(mode="kde", bandwidth="silverman", device="cuda")
est.fit(X_train)
log_density = est.score_samples(X_query)
```

To run the end-to-end demo (including a quick 16D timing comparison against sklearn KDE):

```bash
.venv/bin/python example.py
```

## Core Kernel APIs

The low-level CUDA/Triton kernels live under `kernels.flash_sd_kde`.

```python
gaussian_kde_triton_nd(data, queries, bandwidth, block_m=64, block_n=64,
                       num_warps=4, num_stages=2, device="cuda", synchronize=True):
"""Evaluate 16D Gaussian KDE on CUDA using Tensor-Core-friendly Triton kernels.
Arguments:
    data: (n_train, 16) training samples.
    queries: (n_query, 16) query samples.
    bandwidth: positive scalar KDE bandwidth.
    block_m, block_n: query/data tile sizes.
    num_warps, num_stages: Triton launch parameters.
    device: CUDA device.
    synchronize: whether to synchronize before returning.
Return:
    out: (n_query,) tensor of KDE density values.
"""
```

```python
emp_score_16d_flash_sd_kde(data, bandwidth, block_m=64, block_n=2048,
                           num_warps=2, num_stages=2, device="cuda", synchronize=True):
"""Compute empirical-score accumulators for SD-KDE debiasing in 16D.
Arguments:
    data: (n_train, 16) training samples.
    bandwidth: positive scalar KDE bandwidth.
    block_m, block_n: query/data tile sizes.
    num_warps, num_stages: Triton launch parameters.
    device: CUDA device.
    synchronize: whether to synchronize before returning.
Return:
    pdf_sum: (n_train,) tensor.
    weighted_sum: (n_train, 16) tensor.
"""
```

```python
empirical_sd_kde_triton_nd(data, bandwidth, block_m=64, block_n=2048,
                           num_warps=2, num_stages=2, device="cuda",
                           return_tensor=False, synchronize=True):
"""Run one-step empirical SD-KDE debiasing in 16D.
Arguments:
    data: (n_train, 16) training samples.
    bandwidth: positive scalar KDE bandwidth.
    return_tensor: return CUDA tensor if True, else numpy array.
    synchronize: whether to synchronize before returning.
Return:
    debiased_data: (n_train, 16), tensor or numpy array.
    bandwidth: scalar bandwidth used.
"""
```

## Paper Reproduction

For full reproduction commands in paper figure order, see:

- `EXPERIMENTS.md`

One-command full run:

```bash
make full_paper_experiments_plots
```
