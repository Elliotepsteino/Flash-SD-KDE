from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import math
from typing import Any

import numpy as np
import torch

from experiments.error_suite_a100_16d.numerics import logmeanexp


def silverman_diag_bandwidth(data: torch.Tensor) -> float:
    if data.ndim != 2:
        raise ValueError("data must be 2D")
    x = data.detach().cpu().numpy().astype(np.float64)
    n, d = x.shape
    if n == 0:
        raise ValueError("data must be non-empty")
    std = np.std(x, axis=0)
    iqr = (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)) / 1.34
    sigma = np.minimum(std, iqr)
    sigma = float(np.mean(sigma))
    if sigma <= 0:
        sigma = 1.0
    return float(0.9 * sigma * n ** (-1.0 / (d + 4)))


def reference_log_density(
    samples: torch.Tensor,
    queries: torch.Tensor,
    *,
    bandwidth: float,
    chunk_cfg: dict[str, Any],
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = samples.device

    samples_t = samples.to(device=device, dtype=dtype, copy=False).contiguous()
    queries_t = queries.to(device=device, dtype=dtype, copy=False).contiguous()

    if samples_t.ndim != 2 or queries_t.ndim != 2:
        raise ValueError("samples and queries must be 2D")

    n_samples = samples_t.shape[0]
    n_queries = queries_t.shape[0]
    dim = samples_t.shape[1]

    if n_samples == 0 or n_queries == 0:
        raise ValueError("samples and queries must be non-empty")

    q_chunk = int(chunk_cfg.get("query_chunk_size", 0) or n_queries)
    s_chunk = int(chunk_cfg.get("sample_chunk_size", 0) or n_samples)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    log_const = -math.log(n_samples) - dim * math.log(bandwidth) - 0.5 * dim * math.log(2.0 * math.pi)

    outputs = []
    for q_start in range(0, n_queries, q_chunk):
        q_end = min(n_queries, q_start + q_chunk)
        q = queries_t[q_start:q_end]
        q_norm = (q * q).sum(dim=1, keepdim=True)

        logsum = None
        for s_start in range(0, n_samples, s_chunk):
            s_end = min(n_samples, s_start + s_chunk)
            s = samples_t[s_start:s_end]
            s_norm = (s * s).sum(dim=1, keepdim=True).transpose(0, 1)
            dist2 = q_norm + s_norm - 2.0 * (q @ s.transpose(0, 1))
            dist2 = torch.clamp_min(dist2, 0.0)
            exponent = -0.5 * dist2 * inv_h2
            logsum_chunk = torch.logsumexp(exponent, dim=1)
            logsum = logsum_chunk if logsum is None else torch.logaddexp(logsum, logsum_chunk)

        outputs.append(logsum + log_const)

    return torch.cat(outputs, dim=0)


def reference_density(*args, **kwargs) -> torch.Tensor:
    return torch.exp(reference_log_density(*args, **kwargs))


def reference_linearized_density(
    samples: torch.Tensor,
    queries: torch.Tensor,
    *,
    bandwidth: float,
    chunk_cfg: dict[str, Any],
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = samples.device

    samples_t = samples.to(device=device, dtype=dtype, copy=False).contiguous()
    queries_t = queries.to(device=device, dtype=dtype, copy=False).contiguous()

    if samples_t.ndim != 2 or queries_t.ndim != 2:
        raise ValueError("samples and queries must be 2D")

    n_samples = samples_t.shape[0]
    n_queries = queries_t.shape[0]
    dim = samples_t.shape[1]

    if n_samples == 0 or n_queries == 0:
        raise ValueError("samples and queries must be non-empty")

    q_chunk = int(chunk_cfg.get("query_chunk_size", 0) or n_queries)
    s_chunk = int(chunk_cfg.get("sample_chunk_size", 0) or n_samples)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    norm = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * (bandwidth ** dim) * n_samples)

    outputs = []
    for q_start in range(0, n_queries, q_chunk):
        q_end = min(n_queries, q_start + q_chunk)
        q = queries_t[q_start:q_end]
        q_norm = (q * q).sum(dim=1, keepdim=True)

        acc = None
        for s_start in range(0, n_samples, s_chunk):
            s_end = min(n_samples, s_start + s_chunk)
            s = samples_t[s_start:s_end]
            s_norm = (s * s).sum(dim=1, keepdim=True).transpose(0, 1)
            dist2 = q_norm + s_norm - 2.0 * (q @ s.transpose(0, 1))
            dist2 = torch.clamp_min(dist2, 0.0)
            scaled = dist2 * inv_h2
            phi = torch.exp(-0.5 * scaled)
            phi = phi * (1.0 + 0.5 * dim - 0.5 * scaled)
            chunk_sum = torch.sum(phi, dim=1)
            acc = chunk_sum if acc is None else acc + chunk_sum

        outputs.append(norm * acc)

    return torch.cat(outputs, dim=0)


def reference_linearized_log_density(*args, eps: float = 1e-30, **kwargs) -> torch.Tensor:
    density = reference_linearized_density(*args, **kwargs)
    return torch.log(torch.clamp(density, min=eps))
