from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Any

import numpy as np
import torch

from experiments.error_suite_a100_16d.truth import GaussianMixtureDiag16D, GaussianSingle16D, GroundTruthDist
from globals import ND_FEATURES


def _weights_from_config(weights: Any, n_components: int) -> np.ndarray:
    if weights is None or weights == "uniform":
        return np.full(n_components, 1.0 / n_components, dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape[0] != n_components:
        raise ValueError("weights length must match n_components")
    arr = arr / arr.sum()
    return arr


def make_truth_dist(config: dict[str, Any], *, seed: int) -> GroundTruthDist:
    data_cfg = config.get("data", {})
    dataset = str(data_cfg.get("dataset", "gm_diag_16d"))
    if int(data_cfg.get("dim", ND_FEATURES)) != ND_FEATURES:
        raise ValueError(f"expected dim=16, got {data_cfg.get('dim')}")

    params = dict(data_cfg.get("distribution_params", {}) or {})

    if dataset == "gm_diag_16d":
        n_components = int(params.get("n_components", 8))
        component_std = params.get("component_std", 1.0)
        mean_scale = float(params.get("mean_scale", 2.0))
        weights = _weights_from_config(params.get("weights", "uniform"), n_components)
        rng = np.random.default_rng(seed)
        means = rng.normal(size=(n_components, ND_FEATURES)).astype(np.float64) * mean_scale
        comp_std = np.asarray(component_std, dtype=np.float64)
        return GaussianMixtureDiag16D(means=means, component_std=comp_std, weights=weights)

    if dataset == "gaussian_single_16d":
        mean = np.asarray(params.get("mean", np.zeros(ND_FEATURES)), dtype=np.float64)
        std = np.asarray(params.get("std", np.ones(ND_FEATURES)), dtype=np.float64)
        return GaussianSingle16D(mean=mean, std=std)

    raise ValueError(f"unsupported dataset: {dataset}")


def make_dataset(
    config: dict[str, Any], *, seed: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, GroundTruthDist, dict[str, Any]]:
    data_cfg = config.get("data", {})
    n_train = int(data_cfg.get("n_train", 0))
    n_test = int(data_cfg.get("n_test", 0))
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")

    standardize = bool(data_cfg.get("standardize", True))

    truth = make_truth_dist(config, seed=seed)

    train = truth.sample(n_train, seed=seed, device=device, dtype=dtype)
    test = truth.sample(n_test, seed=seed + 1, device=device, dtype=dtype)

    stats: dict[str, Any] = {"standardize": standardize}

    if standardize:
        mean = train.mean(dim=0, keepdim=True)
        std = train.std(dim=0, keepdim=True)
        std = torch.where(std <= 0, torch.ones_like(std), std)
        train = (train - mean) / std
        test = (test - mean) / std
        stats.update({"mean": mean.squeeze().cpu().numpy().tolist(), "std": std.squeeze().cpu().numpy().tolist()})
        truth = truth.standardized(mean.cpu().numpy().squeeze(), std.cpu().numpy().squeeze())

    return train, test, truth, stats
