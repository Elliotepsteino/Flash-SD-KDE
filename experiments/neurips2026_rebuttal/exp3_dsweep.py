"""Dimension sweep d in {16, 32, 64, 128}: Flash (TF32), Flash (no-TC / IEEE),
eager Torch with TF32 enabled. n_train=32768, n_test=4096.

Reports min-of-repeats runtime, speedups, and model-FLOP effective throughput.
"""
import json
import math
import time

import numpy as np
import torch

from flash_sd_kde.reference import silverman_bandwidth_nd
from kernels.flash_sd_kde.padded_nd import (
    empirical_sd_kde_triton_padded_nd,
    gaussian_kde_triton_padded_nd,
)

DEV = "cuda"
N_TRAIN, N_TEST = 32768, 4096
EPS = 1e-12
REPEATS = 20
DIMS = [16, 32, 64, 128]


def sample_mixture(n, d, rng):
    z = rng.random(n) < 0.5
    out = np.empty((n, d), dtype=np.float32)
    n1 = int(z.sum())
    out[z] = rng.normal(-1.5, 0.5, size=(n1, d)).astype(np.float32)
    out[~z] = rng.normal(1.5, 1.0, size=(n - n1, d)).astype(np.float32)
    return out


def torch_sd_kde(tr, te, h):
    inv_h2 = 1.0 / (h * h)
    xn = (tr * tr).sum(1, keepdim=True)
    d2 = torch.clamp(xn + xn.T - 2.0 * (tr @ tr.T), min=0)
    phi = torch.exp(-0.5 * d2 * inv_h2)
    deb = tr + 0.5 * h * h * ((phi @ tr) / (phi.sum(1, keepdim=True) + EPS) - tr) * inv_h2
    dn = (deb * deb).sum(1, keepdim=True).T
    q2 = torch.clamp((te * te).sum(1, keepdim=True) + dn - 2.0 * (te @ deb.T), min=0)
    d = tr.shape[1]
    norm = 1.0 / (((2 * math.pi) ** (d / 2)) * (h**d) * tr.shape[0])
    return norm * torch.exp(-0.5 * q2 * inv_h2).sum(1)


def flash_sd_kde(tr, te, h, mode, cfg):
    block_n, num_warps = cfg
    deb, _ = empirical_sd_kde_triton_padded_nd(
        tr,
        h,
        device=DEV,
        precision_mode=mode,
        block_m=64,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=2,
        return_tensor=True,
        synchronize=False,
    )
    return gaussian_kde_triton_padded_nd(
        deb, te, h, device=DEV, precision_mode=mode, synchronize=False
    )


def tune_flash(tr, te, h, mode):
    """Mini launch-parameter sweep per (d, mode), mirroring the paper's A.4 sweep."""
    best, best_t = None, float("inf")
    for block_n in (64, 128, 256, 512, 1024):
        for num_warps in (2, 4):
            cfg = (block_n, num_warps)
            try:
                flash_sd_kde(tr, te, h, mode, cfg)
                torch.cuda.synchronize()
                ts = [timed(lambda: flash_sd_kde(tr, te, h, mode, cfg)) for _ in range(3)]
            except Exception:
                continue
            t = min(ts)
            if t < best_t:
                best, best_t = cfg, t
    if best is None:
        raise RuntimeError(f"no feasible launch config for mode={mode}")
    return best


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def model_flops(d, k):
    return (4 * d + 12 + d / 4 + 1.5) * k * k


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(0)
    rows = []
    for d in DIMS:
        xtr = sample_mixture(N_TRAIN, d, rng)
        xte = sample_mixture(N_TEST, d, rng)
        h = silverman_bandwidth_nd(xtr)
        tr = torch.as_tensor(xtr, device=DEV)
        te = torch.as_tensor(xte, device=DEV)

        cfg_tf32 = tune_flash(tr, te, h, "fast_tf32")
        cfg_ieee = tune_flash(tr, te, h, "fp32_ieee")
        fns = {
            "flash_tf32": lambda: flash_sd_kde(tr, te, h, "fast_tf32", cfg_tf32),
            "flash_ieee_no_tc": lambda: flash_sd_kde(tr, te, h, "fp32_ieee", cfg_ieee),
            "torch_tf32": lambda: torch_sd_kde(tr, te, h),
        }
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()

        times = {name: [] for name in fns}
        for _ in range(REPEATS):
            for name, fn in fns.items():
                times[name].append(timed(fn))

        row = {"d": d, "bandwidth": h, "cfg_tf32": cfg_tf32, "cfg_ieee": cfg_ieee}
        for name, ts in times.items():
            row[name + "_ms_min"] = float(np.min(ts))
            row[name + "_ms_mean"] = float(np.mean(ts))
        row["speedup_vs_torch_tf32"] = row["torch_tf32_ms_min"] / row["flash_tf32_ms_min"]
        row["tc_speedup"] = row["flash_ieee_no_tc_ms_min"] / row["flash_tf32_ms_min"]
        row["flash_model_tflops"] = model_flops(d, N_TRAIN) / (row["flash_tf32_ms_min"] * 1e-3) / 1e12
        rows.append(row)
        print(json.dumps(row))
    with open("exp3_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
