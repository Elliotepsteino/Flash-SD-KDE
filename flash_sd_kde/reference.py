from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import math
from typing import Tuple

import numpy as np

from globals import DEFAULT_EPS, ND_FEATURES


def silverman_bandwidth_1d(data: np.ndarray) -> float:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("data must be 1D for silverman_bandwidth_1d.")
    n = x.size
    if n == 0:
        raise ValueError("data must contain at least one element.")
    std_dev = float(np.std(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    sigma = min(std_dev, iqr / 1.34)
    if sigma <= 0:
        sigma = 1.0
    return float(0.9 * sigma * n ** (-1.0 / 5.0))


def silverman_bandwidth_nd(data: np.ndarray) -> float:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("data must be 2D for silverman_bandwidth_nd.")
    n, d = x.shape
    if n == 0:
        raise ValueError("data must contain at least one element.")
    std_per_dim = np.std(x, axis=0)
    iqr_per_dim = np.subtract(*np.percentile(x, [75, 25], axis=0)) / 1.34
    sigma = np.minimum(std_per_dim, iqr_per_dim)
    sigma = float(np.mean(sigma))
    if sigma <= 0:
        sigma = 1.0
    return float(0.9 * sigma * n ** (-1.0 / (d + 4)))


def kde_eval_1d_numpy(queries: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    x = np.asarray(queries, dtype=np.float32)
    d = np.asarray(data, dtype=np.float32)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if d.size == 0:
        raise ValueError("data must contain at least one element.")
    diff = (x[:, None] - d[None, :]) / np.float32(bandwidth)
    phi = np.exp(-0.5 * diff * diff)
    inv_norm = 1.0 / (math.sqrt(2.0 * math.pi) * d.size * bandwidth)
    return inv_norm * phi.sum(axis=1)


def kde_eval_1d_linearized_numpy(queries: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    """Linearized Emp-SD-KDE approximation using K - (h^2/2) ΔK for 1D."""
    x = np.asarray(queries, dtype=np.float32)
    d = np.asarray(data, dtype=np.float32)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if d.size == 0:
        raise ValueError("data must contain at least one element.")
    diff = (x[:, None] - d[None, :]) / np.float32(bandwidth)
    scaled = diff * diff
    phi = np.exp(-0.5 * scaled)
    phi = phi * (1.0 + 0.5 - 0.5 * scaled)
    inv_norm = 1.0 / (math.sqrt(2.0 * math.pi) * d.size * bandwidth)
    return inv_norm * phi.sum(axis=1)


def kde_eval_nd_numpy(queries: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    x = np.asarray(queries, dtype=np.float32)
    d = np.asarray(data, dtype=np.float32)
    if x.ndim != 2 or d.ndim != 2:
        raise ValueError("queries and data must be 2D arrays.")
    if x.shape[1] != d.shape[1]:
        raise ValueError("queries and data must share feature dimension.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if d.shape[0] == 0:
        raise ValueError("data must contain at least one element.")
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    x_norm = (x * x).sum(axis=1, keepdims=True)
    d_norm = (d * d).sum(axis=1, keepdims=True).T
    dist = x_norm + d_norm - 2.0 * (x @ d.T)
    dist = np.maximum(dist, 0.0)
    phi = np.exp(-0.5 * dist * inv_h2)
    dim = d.shape[1]
    norm = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * (bandwidth ** dim) * d.shape[0])
    return norm * phi.sum(axis=1)


def kde_eval_nd_linearized_numpy(queries: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    """Linearized Emp-SD-KDE approximation using K - (h^2/2) ΔK for ND."""
    x = np.asarray(queries, dtype=np.float32)
    d = np.asarray(data, dtype=np.float32)
    if x.ndim != 2 or d.ndim != 2:
        raise ValueError("queries and data must be 2D arrays.")
    if x.shape[1] != d.shape[1]:
        raise ValueError("queries and data must share feature dimension.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if d.shape[0] == 0:
        raise ValueError("data must contain at least one element.")
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    x_norm = (x * x).sum(axis=1, keepdims=True)
    d_norm = (d * d).sum(axis=1, keepdims=True).T
    dist = x_norm + d_norm - 2.0 * (x @ d.T)
    dist = np.maximum(dist, 0.0)
    scaled = dist * inv_h2
    phi = np.exp(-0.5 * scaled)
    dim = d.shape[1]
    phi = phi * (1.0 + 0.5 * dim - 0.5 * scaled)
    norm = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * (bandwidth ** dim) * d.shape[0])
    return norm * phi.sum(axis=1)


def empirical_score_nd_numpy(
    data: np.ndarray,
    bandwidth: float,
    *,
    eps: float = DEFAULT_EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("data must be 2D.")
    if x.shape[1] != ND_FEATURES:
        raise ValueError(f"expected {ND_FEATURES} features, got {x.shape[1]}.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    x_norm = (x * x).sum(axis=1, keepdims=True)
    dist = x_norm + x_norm.T - 2.0 * (x @ x.T)
    dist = np.maximum(dist, 0.0)
    phi = np.exp(-0.5 * dist * inv_h2)
    pdf_sum = phi.sum(axis=1)
    weighted = phi @ x
    score = (weighted / (pdf_sum[:, None] + eps) - x) * inv_h2
    return pdf_sum, weighted, score


def empirical_sd_kde_transform_nd_numpy(
    data: np.ndarray,
    bandwidth: float,
    *,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    _, _, score = empirical_score_nd_numpy(data, bandwidth, eps=eps)
    delta = 0.5 * (bandwidth ** 2)
    return np.asarray(data, dtype=np.float32) + delta * score
