"""(a) Metric-level TF32 vs IEEE check: does TF32 change oracle-error metrics?
(b) torch.compile profile + peak-memory evidence of n^2 materialization.
"""
import json
import math

import numpy as np
import torch

from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd, gaussian_kde_triton_nd
from kernels.flash_sd_kde.padded_nd import (
    empirical_sd_kde_triton_padded_nd,
    gaussian_kde_triton_padded_nd,
)

DEV = "cuda"
N_TRAIN, N_TEST = 32768, 4096
EPS = 1e-12


def true_pdf_16d(x, pi=0.5, mean_offset=1.5, s1=0.5, s2=1.0):
    d = x.shape[1]
    def comp(mu, s):
        z = ((x - mu) ** 2).sum(1) / (s * s)
        return torch.exp(-0.5 * z) / (((2 * math.pi) ** (d / 2)) * (s**d))
    return pi * comp(-mean_offset, s1) + (1 - pi) * comp(mean_offset, s2)


def torch_score(tr, h):
    inv_h2 = 1.0 / (h * h)
    xn = (tr * tr).sum(1, keepdim=True)
    d2 = torch.clamp(xn + xn.T - 2.0 * (tr @ tr.T), min=0)
    phi = torch.exp(-0.5 * d2 * inv_h2)
    return tr + 0.5 * h * h * ((phi @ tr) / (phi.sum(1, keepdim=True) + EPS) - tr) * inv_h2


def main():
    np.random.seed(0)
    xtr = sample_gaussian_mixture_16d(N_TRAIN)
    xte = sample_gaussian_mixture_16d(N_TEST)
    h = silverman_bandwidth_nd(xtr)
    tr = torch.as_tensor(xtr, device=DEV)
    te = torch.as_tensor(xte, device=DEV)
    p_true = true_pdf_16d(torch.as_tensor(xte, dtype=torch.float64, device=DEV))

    out = {}
    for mode in ("fast_tf32", "fp32_ieee"):
        deb_np, _ = empirical_sd_kde_triton_padded_nd(tr, h, device=DEV, precision_mode=mode)
        deb = torch.as_tensor(deb_np, device=DEV)
        p = gaussian_kde_triton_padded_nd(deb, te, h, device=DEV, precision_mode=mode).double()
        out[mode] = {
            "mean_rel_err_vs_oracle": float((torch.abs(p - p_true) / p_true).mean()),
            "mean_sq_err_vs_oracle": float(((p - p_true) ** 2).mean()),
        }
    print(json.dumps(out, indent=2))
    with open("exp5_metric_level.json", "w") as f:
        json.dump(out, f, indent=2)

    # Peak-memory evidence: eager score vs compiled score vs flash score
    torch.backends.cuda.matmul.allow_tf32 = True
    mem = {}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    _ = torch_score(tr, h)
    torch.cuda.synchronize()
    mem["eager_score_peak_extra_MB"] = (torch.cuda.max_memory_allocated() - base) / 2**20

    c_score = torch.compile(torch_score)
    _ = c_score(tr, h)  # warmup/compile
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    _ = c_score(tr, h)
    torch.cuda.synchronize()
    mem["compile_score_peak_extra_MB"] = (torch.cuda.max_memory_allocated() - base) / 2**20

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    _ = empirical_sd_kde_triton_nd(tr, h, return_tensor=True, synchronize=True)
    mem["flash_score_peak_extra_MB"] = (torch.cuda.max_memory_allocated() - base) / 2**20

    print(json.dumps(mem, indent=2))
    with open("exp5_memory.json", "w") as f:
        json.dump(mem, f, indent=2)

    # Profiler: top CUDA kernels for the compiled score pass
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _ = c_score(tr, h)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(table)
    with open("exp5_profile_compile.txt", "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
