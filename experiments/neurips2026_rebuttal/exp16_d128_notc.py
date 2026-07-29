"""Targeted no-TC (fp32_ieee) tune for the d=128 ablation cell."""
import json
import time

import numpy as np
import torch

from flash_sd_kde.reference import silverman_bandwidth_nd
from kernels.flash_sd_kde.padded_nd import (
    empirical_sd_kde_triton_padded_nd,
    gaussian_kde_triton_padded_nd,
)

N, NT, D = 32768, 4096, 128
rng = np.random.default_rng(0)
z = rng.random(N) < 0.5
xtr = np.empty((N, D), dtype=np.float32)
xtr[z] = rng.normal(-1.5, 0.5, size=(int(z.sum()), D))
xtr[~z] = rng.normal(1.5, 1.0, size=(N - int(z.sum()), D))
xte = rng.normal(0, 1.5, size=(NT, D)).astype(np.float32)
h = silverman_bandwidth_nd(xtr)
tr = torch.as_tensor(xtr, device="cuda")
te = torch.as_tensor(xte, device="cuda")


def run(cfg):
    bm, bn, bk, w, s = cfg
    deb, _ = empirical_sd_kde_triton_padded_nd(
        tr, h, device="cuda", precision_mode="fp32_ieee", block_m=bm, block_n=bn,
        block_k=bk, num_warps=w, num_stages=s, return_tensor=True, synchronize=False,
    )
    return gaussian_kde_triton_padded_nd(
        deb, te, h, device="cuda", precision_mode="fp32_ieee", block_m=128,
        block_n=128, block_k=16, num_warps=4, num_stages=1, synchronize=False,
    )


def tmin(cfg, reps=8):
    run(cfg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(cfg)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


best = (None, 1e9)
for cfg in [
    (32, 128, 32, 4, 2), (32, 128, 16, 4, 2), (32, 256, 32, 4, 2),
    (32, 128, 32, 4, 1), (64, 128, 32, 4, 2), (32, 128, 64, 4, 2),
    (32, 256, 16, 4, 2), (16, 128, 32, 4, 2),
]:
    try:
        t = tmin(cfg)
    except Exception:
        print(cfg, "infeasible", flush=True)
        continue
    print(cfg, f"{t:.2f} ms", flush=True)
    if t < best[1]:
        best = (cfg, t)
print(json.dumps({"best_cfg": best[0], "no_tc_ms_min": best[1]}))
