"""Pass-level latency breakdown (score vs KDE) and TF32-enabled Torch baselines.

Methods: eager Torch FP32 (paper baseline), eager Torch with TF32 enabled,
torch.compile with TF32 enabled, Flash-SD-KDE (specialized 16-D kernels).
Timing: interleaved rounds, min over repeats (robust to shared-GPU bursts).
Also captures a torch.profiler kernel table for the eager baselines.
"""
import json
import math
import time

import numpy as np
import torch

from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd, gaussian_kde_triton_nd

DEV = "cuda"
N_TRAIN, N_TEST = 32768, 4096
EPS = 1e-12
REPEATS = 30


def torch_score(tr, h):
    inv_h2 = 1.0 / (h * h)
    xn = (tr * tr).sum(1, keepdim=True)
    d2 = torch.clamp(xn + xn.T - 2.0 * (tr @ tr.T), min=0)
    phi = torch.exp(-0.5 * d2 * inv_h2)
    return tr + 0.5 * h * h * ((phi @ tr) / (phi.sum(1, keepdim=True) + EPS) - tr) * inv_h2


def torch_kde(deb, te, h):
    inv_h2 = 1.0 / (h * h)
    dn = (deb * deb).sum(1, keepdim=True).T
    q2 = torch.clamp((te * te).sum(1, keepdim=True) + dn - 2.0 * (te @ deb.T), min=0)
    d = deb.shape[1]
    norm = 1.0 / (((2 * math.pi) ** (d / 2)) * (h**d) * deb.shape[0])
    return norm * torch.exp(-0.5 * q2 * inv_h2).sum(1)


def flash_score(tr, h):
    deb, _ = empirical_sd_kde_triton_nd(tr, h, return_tensor=True, synchronize=False)
    return deb


def flash_kde(deb, te, h):
    return gaussian_kde_triton_nd(deb, te, h, synchronize=False)


def set_tf32(on: bool):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def timed(fn, *args):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


def main():
    np.random.seed(0)
    xtr = sample_gaussian_mixture_16d(N_TRAIN)
    xte = sample_gaussian_mixture_16d(N_TEST)
    h = silverman_bandwidth_nd(xtr)
    tr = torch.as_tensor(xtr, device=DEV)
    te = torch.as_tensor(xte, device=DEV)

    set_tf32(True)
    c_score = torch.compile(torch_score)
    c_kde = torch.compile(torch_kde)
    # compile warmup
    deb = c_score(tr, h)
    _ = c_kde(deb, te, h)
    torch.cuda.synchronize()

    methods = {
        "torch_fp32": (torch_score, torch_kde, False),
        "torch_tf32": (torch_score, torch_kde, True),
        "compile_tf32": (c_score, c_kde, True),
        "flash": (flash_score, flash_kde, True),
    }

    # warmup all
    for name, (fs, fk, tf32) in methods.items():
        set_tf32(tf32)
        d = fs(tr, h)
        _ = fk(d, te, h)
    torch.cuda.synchronize()

    times = {name: {"score": [], "kde": []} for name in methods}
    for _ in range(REPEATS):
        for name, (fs, fk, tf32) in methods.items():
            set_tf32(tf32)
            t_s, deb = timed(fs, tr, h)
            t_k, _ = timed(fk, deb, te, h)
            times[name]["score"].append(t_s)
            times[name]["kde"].append(t_k)

    results = {"n_train": N_TRAIN, "n_test": N_TEST, "repeats": REPEATS}
    for name, tt in times.items():
        s = float(np.min(tt["score"]))
        k = float(np.min(tt["kde"]))
        results[name] = {
            "score_ms_min": s,
            "kde_ms_min": k,
            "total_ms_min": s + k,
            "score_share": s / (s + k),
            "score_ms_mean": float(np.mean(tt["score"])),
            "kde_ms_mean": float(np.mean(tt["kde"])),
        }
    print(json.dumps(results, indent=2))
    with open("exp2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Profiler evidence: which CUDA kernels dominate the eager baselines
    from torch.profiler import ProfilerActivity, profile

    for label, tf32 in (("eager_fp32", False), ("eager_tf32", True)):
        set_tf32(tf32)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            d = torch_score(tr, h)
            _ = torch_kde(d, te, h)
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=8)
        print(f"\n==== profiler {label} ====\n{table}")
        with open(f"exp2_profile_{label}.txt", "w") as f:
            f.write(table)


if __name__ == "__main__":
    main()
