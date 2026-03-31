import math
from typing import Tuple

import numpy as np

from globals import DEFAULT_EPS, ND_FEATURES

mixture_params_list = [
    {"pi": 0.4, "mu1": -2, "sigma1": 0.5, "mu2": 2, "sigma2": 1.0},
    {"pi": 0.3, "mu1": -2, "sigma1": 0.4, "mu2": 4, "sigma2": 1.5},
    {"pi": 0.5, "mu1": 0, "sigma1": 0.4, "mu2": 1.5, "sigma2": 1.5},
]


def sample_from_mixture(n: int, params: dict[str, float]) -> np.ndarray:
    pi_ = params["pi"]
    z = np.random.rand(n) < pi_
    x_samps = np.zeros(n)
    x_samps[z] = np.random.normal(params["mu1"], params["sigma1"], size=z.sum())
    x_samps[~z] = np.random.normal(params["mu2"], params["sigma2"], size=(~z).sum())
    return x_samps


def silverman_bandwidth(data: np.ndarray) -> float:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("data must be 1D for silverman_bandwidth.")
    n = x.size
    if n == 0:
        raise ValueError("data must contain at least one element.")
    std_dev = float(np.std(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    sigma = min(std_dev, iqr / 1.34)
    if sigma <= 0:
        sigma = 1.0
    return float(0.9 * sigma * n ** (-1.0 / 5.0))


def silverman_bandwidth_1d(data: np.ndarray) -> float:
    return silverman_bandwidth(data)


def kde_pdf_eval_base(x_points: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    x_points = np.asarray(x_points)
    data = np.asarray(data)
    m = x_points.size
    n = data.size
    z = (x_points.reshape(m, 1) - data.reshape(1, n)) / bandwidth
    pdf_mat = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
    return pdf_mat.mean(axis=1) / bandwidth


_INV_SQRT_2PI_F32 = np.float32(1.0 / np.sqrt(2.0 * np.pi))


def kde_pdf_eval(
    x_points: np.ndarray,
    data: np.ndarray,
    bandwidth: float,
    block_size: int = 8192,
) -> np.ndarray:
    x = np.asarray(x_points, dtype=np.float32)
    d = np.asarray(data, dtype=np.float32)
    h = np.float32(bandwidth)

    m = x.size
    n = d.size

    inv_h = np.float32(1.0) / h
    inv_2h2 = np.float32(0.5) * inv_h * inv_h

    out = np.zeros(m, dtype=np.float32)

    for start in range(0, n, block_size):
        db = d[start : start + block_size]
        diff = x[:, None] - db[None, :]
        out += np.exp(-(diff * diff) * inv_2h2).sum(axis=1)

    out *= (_INV_SQRT_2PI_F32 * inv_h / np.float32(n))
    return out


def kde_score_eval(
    x_points: np.ndarray,
    data: np.ndarray,
    bandwidth: float,
    *,
    block_size: int = 4096,
    eps: float = 1e-12,
) -> np.ndarray:
    """Evaluate the empirical KDE score d/dx log p_hat(x) at the given points."""
    x = np.asarray(x_points, dtype=np.float32)
    d = np.asarray(data, dtype=np.float32)
    h = np.float32(bandwidth)

    if h <= 0:
        raise ValueError("bandwidth must be positive for KDE score evaluation.")

    m = x.size
    n = d.size
    if n == 0:
        raise ValueError("data must contain at least one point.")

    z_sum = np.zeros(m, dtype=np.float32)
    dz_sum = np.zeros(m, dtype=np.float32)

    inv_h = np.float32(1.0) / h

    for start in range(0, n, block_size):
        db = d[start : start + block_size]
        diff = (x[:, None] - db[None, :]) * inv_h
        phi = np.exp(-0.5 * diff * diff)

        z_sum += phi.sum(axis=1)
        dz_sum += (-diff * phi).sum(axis=1)

    scale_pdf = _INV_SQRT_2PI_F32 / (n * h)
    scale_deriv = _INV_SQRT_2PI_F32 / (n * h * h)

    pdf_vals = scale_pdf * z_sum
    deriv_vals = scale_deriv * dz_sum
    return deriv_vals / (pdf_vals + eps)


def one_step_debiased_data_emp_kde(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """One-step empirical SD-KDE debiasing using KDE-derived scores."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        raise ValueError("x must contain at least one element.")

    std_dev = np.std(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    sigma = min(std_dev, iqr / 1.34)

    h = 0.4 * sigma * n ** (-1 / 9)
    delta = 0.5 * (h ** 2)

    s_hat = kde_score_eval(x, x, h)
    x_deb = x + delta * s_hat
    return x_deb, h


def sample_gaussian_mixture_16d(
    n: int,
    *,
    pi: float = 0.5,
    mean_offset: float = 1.5,
    sigma1: float = 0.5,
    sigma2: float = 1.0,
) -> np.ndarray:
    """Sample an isotropic two-component Gaussian mixture in 16 dimensions."""
    if not (0.0 < pi < 1.0):
        raise ValueError("pi must lie in (0, 1).")
    d = 16
    z = np.random.rand(n) < pi
    samples = np.empty((n, d), dtype=np.float32)
    n1 = int(z.sum())
    n2 = n - n1
    if n1:
        samples[z] = np.random.normal(
            loc=-mean_offset, scale=sigma1, size=(n1, d)
        ).astype(np.float32, copy=False)
    if n2:
        samples[~z] = np.random.normal(
            loc=mean_offset, scale=sigma2, size=(n2, d)
        ).astype(np.float32, copy=False)
    return samples


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
