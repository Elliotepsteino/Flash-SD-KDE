"""Lightweight KDE utilities that avoid heavy plotting dependencies."""

from __future__ import annotations

import numpy as np
from typing import Tuple


mixture_params_list = [
    {'pi': 0.4, 'mu1': -2, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 1.0},
    {'pi': 0.3, 'mu1': -2, 'sigma1': 0.4, 'mu2':  4, 'sigma2': 1.5},
    {'pi': 0.5, 'mu1':  0, 'sigma1': 0.4, 'mu2':  1.5, 'sigma2': 1.5},
]


def sample_from_mixture(n: int, params: dict[str, float]) -> np.ndarray:
    pi_ = params['pi']
    z = np.random.rand(n) < pi_
    x_samps = np.zeros(n)
    x_samps[z] = np.random.normal(params['mu1'], params['sigma1'], size=z.sum())
    x_samps[~z] = np.random.normal(params['mu2'], params['sigma2'], size=(~z).sum())
    return x_samps


def silverman_bandwidth(data: np.ndarray) -> float:
    n = len(data)
    std_dev = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma = min(std_dev, iqr / 1.34)
    return 0.9 * sigma * n ** (-1 / 5)


def kde_pdf_eval_base(x_points: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    x_points = np.asarray(x_points)
    data = np.asarray(data)
    M = x_points.size
    N = data.size
    z = (x_points.reshape(M, 1) - data.reshape(1, N)) / bandwidth
    pdf_mat = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
    return pdf_mat.mean(axis=1) / bandwidth



_INV_SQRT_2PI_F32 = np.float32(1.0 / np.sqrt(2.0 * np.pi))

def kde_pdf_eval(x_points: np.ndarray,
                            data: np.ndarray,
                            bandwidth: float,
                            block_size: int = 8192) -> np.ndarray:
    # Ensure contiguous float32
    x = np.asarray(x_points, dtype=np.float32)
    d = np.asarray(data, dtype=np.float32)
    h = np.float32(bandwidth)

    M = x.size
    N = d.size

    inv_h = np.float32(1.0) / h
    inv_2h2 = np.float32(0.5) * inv_h * inv_h  # 1/(2h^2)

    out = np.zeros(M, dtype=np.float32)

    # Block over data to avoid MxN temporaries
    for start in range(0, N, block_size):
        db = d[start:start + block_size]  # (B,)
        # diff shape (M, B), but B is small enough to fit cache
        diff = x[:, None] - db[None, :]
        out += np.exp(-(diff * diff) * inv_2h2).sum(axis=1)

    out *= (_INV_SQRT_2PI_F32 * inv_h / np.float32(N))
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

    M = x.size
    N = d.size
    if N == 0:
        raise ValueError("data must contain at least one point.")

    z_sum = np.zeros(M, dtype=np.float32)
    dz_sum = np.zeros(M, dtype=np.float32)

    inv_h = np.float32(1.0) / h

    for start in range(0, N, block_size):
        db = d[start:start + block_size]
        diff = (x[:, None] - db[None, :]) * inv_h
        phi = np.exp(-0.5 * diff * diff)

        z_sum += phi.sum(axis=1)
        dz_sum += (-diff * phi).sum(axis=1)

    scale_pdf = _INV_SQRT_2PI_F32 / (N * h)
    scale_deriv = _INV_SQRT_2PI_F32 / (N * h * h)

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
