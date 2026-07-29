"""d-sweep with a proper per-d launch-parameter search for the padded kernels.

Stage 1: coarse sweep over (block_m, block_n, block_k) at warps=4, stages=2,
         separately for the score pass and the KDE pass.
Stage 2: refine (num_warps, num_stages) on the best block config.
Final:   interleaved min-of-20 timing of Flash (TF32), Flash (IEEE / no-TC),
         and eager PyTorch with TF32 enabled.
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


def run_score(tr, h, mode, cfg):
    bm, bn, bk, w, s = cfg
    return empirical_sd_kde_triton_padded_nd(
        tr, h, device=DEV, precision_mode=mode, block_m=bm, block_n=bn, block_k=bk,
        num_warps=w, num_stages=s, return_tensor=True, synchronize=False,
    )[0]


def run_kde(deb, te, h, mode, cfg):
    bm, bn, bk, w, s = cfg
    return gaussian_kde_triton_padded_nd(
        deb, te, h, device=DEV, precision_mode=mode, block_m=bm, block_n=bn,
        block_k=bk, num_warps=w, num_stages=s, synchronize=False,
    )


def timed(fn, reps=2):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def bk_candidates(d):
    dim_pad = ((d + 15) // 16) * 16
    return sorted({bk for bk in (16, 32, 64, dim_pad) if bk <= dim_pad and dim_pad % bk == 0})


def tune(run_fn, d, label):
    best, best_t = None, float("inf")
    for bm in (32, 64, 128):
        for bn in (128, 256, 512, 1024):
            for bk in bk_candidates(d):
                # stages=2 first; large tiles that exceed shared memory get a
                # stages=1 fallback (large block_n cuts atomic traffic).
                for s in (2, 1):
                    cfg = (bm, bn, bk, 4, s)
                    try:
                        t = timed(lambda: run_fn(cfg))
                    except Exception:
                        continue
                    if t < best_t:
                        best, best_t = cfg, t
                    break
    if best is None:
        raise RuntimeError(f"no feasible config for {label}")
    bm, bn, bk, _, s0 = best
    for w in (2, 4, 8):
        for s in {1, 2, 3, 4, s0}:
            cfg = (bm, bn, bk, w, s)
            try:
                t = timed(lambda: run_fn(cfg))
            except Exception:
                continue
            if t < best_t:
                best, best_t = cfg, t
    print(f"  tuned {label}: cfg={best} ({best_t:.2f} ms)", flush=True)
    return best


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(0)
    rows = []
    for d in DIMS:
        print(f"=== d={d}", flush=True)
        xtr = sample_mixture(N_TRAIN, d, rng)
        xte = sample_mixture(N_TEST, d, rng)
        h = silverman_bandwidth_nd(xtr)
        tr = torch.as_tensor(xtr, device=DEV)
        te = torch.as_tensor(xte, device=DEV)

        cfgs = {}
        for mode in ("fast_tf32", "fp32_ieee"):
            score_cfg = tune(lambda c: run_score(tr, h, mode, c), d, f"score/{mode}")
            deb = run_score(tr, h, mode, score_cfg)
            kde_cfg = tune(lambda c: run_kde(deb, te, h, mode, c), d, f"kde/{mode}")
            cfgs[mode] = (score_cfg, kde_cfg)

        def flash(mode):
            sc, kc = cfgs[mode]
            deb = run_score(tr, h, mode, sc)
            return run_kde(deb, te, h, mode, kc)

        fns = {
            "flash_tf32": lambda: flash("fast_tf32"),
            "flash_ieee_no_tc": lambda: flash("fp32_ieee"),
            "torch_tf32": lambda: torch_sd_kde(tr, te, h),
        }
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()
        times = {name: [] for name in fns}
        for _ in range(REPEATS):
            for name, fn in fns.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - t0) * 1e3)

        row = {"d": d, "bandwidth": h,
               "cfg_tf32": cfgs["fast_tf32"], "cfg_ieee": cfgs["fp32_ieee"]}
        for name, ts in times.items():
            row[name + "_ms_min"] = float(np.min(ts))
            row[name + "_ms_mean"] = float(np.mean(ts))
        row["speedup_vs_torch_tf32"] = row["torch_tf32_ms_min"] / row["flash_tf32_ms_min"]
        row["tc_speedup"] = row["flash_ieee_no_tc_ms_min"] / row["flash_tf32_ms_min"]
        row["flash_model_tflops"] = (
            (4 * d + 12 + d / 4 + 1.5) * N_TRAIN * N_TRAIN / (row["flash_tf32_ms_min"] * 1e-3) / 1e12
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open("exp8_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
