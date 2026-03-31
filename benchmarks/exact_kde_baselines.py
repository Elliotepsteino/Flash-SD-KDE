from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch


def torch_exact_log_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    q_norm = (queries * queries).sum(dim=1, keepdim=True)
    d_norm = (train * train).sum(dim=1, keepdim=True).T
    dot = queries @ train.T
    dist = torch.clamp(q_norm + d_norm - 2.0 * dot, min=0.0)
    exponent = -0.5 * dist * inv_h2
    dim = train.shape[1]
    log_norm = -math.log(train.shape[0]) - dim * math.log(bandwidth) - 0.5 * dim * math.log(2.0 * math.pi)
    return torch.logsumexp(exponent, dim=1) + log_norm


def torch_exact_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    return torch.exp(torch_exact_log_kde_nd(train, queries, bandwidth))


def torch_exact_sd_debias_nd(
    train: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    x_norm = (train * train).sum(dim=1, keepdim=True)
    gram = train @ train.T
    dist = torch.clamp(x_norm + x_norm.T - 2.0 * gram, min=0.0)
    phi = torch.exp(-0.5 * dist * inv_h2)
    phi_sum = phi.sum(dim=1, keepdim=True)
    weighted = phi @ train
    score = (weighted / (phi_sum + 1e-12) - train) * inv_h2
    return train + 0.5 * (bandwidth ** 2) * score


def torch_exact_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    debiased = torch_exact_sd_debias_nd(train, bandwidth)
    return torch_exact_kde_nd(debiased, queries, bandwidth)


def torch_exact_log_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    debiased = torch_exact_sd_debias_nd(train, bandwidth)
    return torch_exact_log_kde_nd(debiased, queries, bandwidth)


def time_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float, np.ndarray]:
    for _ in range(max(warmup, 0)):
        _ = fn()
        torch.cuda.synchronize(device)

    values = None
    times_ms: list[float] = []
    for _ in range(max(repeats, 1)):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        values = fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    assert values is not None
    arr = np.asarray(times_ms, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), values.detach().cpu().numpy()
