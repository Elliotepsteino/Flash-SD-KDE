from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Any

import torch


def logmeanexp(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    return max_val.squeeze(dim) + torch.log(torch.mean(torch.exp(x - max_val), dim=dim))


def log_abs_diff_exp(loga: torch.Tensor, logb: torch.Tensor) -> torch.Tensor:
    max_log = torch.maximum(loga, logb)
    min_log = torch.minimum(loga, logb)
    return max_log + torch.log1p(-torch.exp(min_log - max_log))


def exp_diagnostics(logp: torch.Tensor) -> dict[str, Any]:
    dtype = logp.dtype if logp.is_floating_point() else torch.float32
    finfo = torch.finfo(dtype)
    log_min = torch.log(torch.tensor(finfo.tiny, device=logp.device, dtype=logp.dtype))
    log_max = torch.log(torch.tensor(finfo.max, device=logp.device, dtype=logp.dtype))
    underflow = (logp < log_min).float().mean().item()
    overflow = (logp > log_max).float().mean().item()
    return {
        "log_min": float(log_min.item()),
        "log_max": float(log_max.item()),
        "underflow_frac": float(underflow),
        "overflow_frac": float(overflow),
    }
