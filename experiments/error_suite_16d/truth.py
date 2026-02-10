from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from globals import ND_FEATURES


class GroundTruthDist:
    def sample(self, n: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError

    def true_log_density(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def standardized(self, mean: np.ndarray, std: np.ndarray) -> "GroundTruthDist":
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianMixtureDiag16D(GroundTruthDist):
    means: np.ndarray
    component_std: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.means.shape[1] != ND_FEATURES:
            raise ValueError(f"expected means shape (*, {ND_FEATURES})")
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if self.weights.shape[0] != self.means.shape[0]:
            raise ValueError("weights length must match number of components")

    def _component_std_matrix(self) -> np.ndarray:
        if self.component_std.ndim == 0:
            return np.full_like(self.means, float(self.component_std))
        if self.component_std.ndim == 1:
            if self.component_std.shape[0] == ND_FEATURES:
                return np.broadcast_to(self.component_std[None, :], self.means.shape)
            if self.component_std.shape[0] == self.means.shape[0]:
                return np.broadcast_to(self.component_std[:, None], self.means.shape)
        if self.component_std.shape == self.means.shape:
            return self.component_std
        raise ValueError("component_std has incompatible shape")

    def sample(self, n: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        comp_idx = rng.choice(self.weights.shape[0], size=n, p=self.weights)
        means = self.means[comp_idx]
        stds = self._component_std_matrix()[comp_idx]
        samples = rng.normal(size=means.shape).astype(np.float64)
        samples = samples * stds + means
        return torch.as_tensor(samples, device=device, dtype=dtype)

    def true_log_density(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float64)
        means = torch.as_tensor(self.means, dtype=torch.float64, device=x.device)
        stds = torch.as_tensor(self._component_std_matrix(), dtype=torch.float64, device=x.device)
        weights = torch.as_tensor(self.weights, dtype=torch.float64, device=x.device)

        diff = (x[None, :, :] - means[:, None, :]) / stds[:, None, :]
        log_det = torch.sum(torch.log(stds), dim=-1)
        quad = 0.5 * torch.sum(diff * diff, dim=-1)
        log_norm = -0.5 * ND_FEATURES * math.log(2.0 * math.pi) - log_det[:, None]
        log_probs = log_norm - quad + torch.log(weights)[:, None]
        return torch.logsumexp(log_probs, dim=0)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "gm_diag_16d",
            "n_components": int(self.weights.shape[0]),
            "component_std": self.component_std.tolist() if isinstance(self.component_std, np.ndarray) else float(self.component_std),
            "means": self.means.tolist(),
            "weights": self.weights.tolist(),
        }

    def standardized(self, mean: np.ndarray, std: np.ndarray) -> "GaussianMixtureDiag16D":
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
        new_means = (self.means - mean) / std
        comp_std = self._component_std_matrix() / std
        return GaussianMixtureDiag16D(means=new_means, component_std=comp_std, weights=self.weights)


@dataclass(frozen=True)
class GaussianSingle16D(GroundTruthDist):
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape[0] != ND_FEATURES:
            raise ValueError(f"expected mean length {ND_FEATURES}")
        if self.std.shape[0] != ND_FEATURES:
            raise ValueError(f"expected std length {ND_FEATURES}")

    def sample(self, n: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        samples = rng.normal(size=(n, ND_FEATURES)).astype(np.float64)
        samples = samples * self.std + self.mean
        return torch.as_tensor(samples, device=device, dtype=dtype)

    def true_log_density(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float64)
        mean = torch.as_tensor(self.mean, dtype=torch.float64, device=x.device)
        std = torch.as_tensor(self.std, dtype=torch.float64, device=x.device)
        diff = (x - mean) / std
        quad = 0.5 * torch.sum(diff * diff, dim=-1)
        log_det = torch.sum(torch.log(std))
        log_norm = -0.5 * ND_FEATURES * math.log(2.0 * math.pi) - log_det
        return log_norm - quad

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "gaussian_single_16d",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def standardized(self, mean: np.ndarray, std: np.ndarray) -> "GaussianSingle16D":
        new_mean = (self.mean - mean) / std
        new_std = self.std / std
        return GaussianSingle16D(mean=new_mean, std=new_std)
