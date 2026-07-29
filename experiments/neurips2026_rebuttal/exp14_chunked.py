"""Chunk-streaming general-d score kernel vs the two-pass padded kernel.

Tunes the chunked kernel per (d, precision) and compares min-of-15 score-pass
times against the previously tuned two-pass configurations.
"""
import json
import time

import numpy as np
import torch

from flash_sd_kde.reference import silverman_bandwidth_nd
from kernels.flash_sd_kde.padded_nd import empirical_sd_kde_triton_padded_nd

DEV = "cuda"
N = 32768
DIMS = [16, 32, 64, 128]

# Tuned two-pass score configs (block_m, block_n, block_k, warps, stages)
TWO_PASS = {
    16: {"fast_tf32": (64, 256, 16, 4, 2), "fp32_ieee": (32, 128, 16, 4, 3)},
    32: {"fast_tf32": (64, 256, 32, 4, 1), "fp32_ieee": (32, 256, 16, 4, 2)},
    64: {"fast_tf32": (32, 128, 16, 4, 1), "fp32_ieee": (32, 128, 32, 4, 2)},
    128: {"fast_tf32": (32, 128, 16, 4, 1), "fp32_ieee": (32, 128, 32, 4, 2)},
}


def sample_mixture(n, d, rng):
    z = rng.random(n) < 0.5
    out = np.empty((n, d), dtype=np.float32)
    n1 = int(z.sum())
    out[z] = rng.normal(-1.5, 0.5, size=(n1, d)).astype(np.float32)
    out[~z] = rng.normal(1.5, 1.0, size=(n - n1, d)).astype(np.float32)
    return out


def run_two_pass(tr, h, mode, cfg):
    bm, bn, bk, w, s = cfg
    return empirical_sd_kde_triton_padded_nd(
        tr, h, device=DEV, precision_mode=mode, block_m=bm, block_n=bn, block_k=bk,
        num_warps=w, num_stages=s, return_tensor=True, synchronize=False,
    )[0]


def run_chunked(tr, h, mode, cfg):
    bm, bn, chunk, w, s = cfg
    return empirical_sd_kde_triton_padded_nd(
        tr, h, device=DEV, precision_mode=mode, block_m=bm, block_n=bn,
        num_warps=w, num_stages=s, chunked=True, block_n_chunk=chunk,
        return_tensor=True, synchronize=False,
    )[0]


def timed(fn, reps=3):
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


def tune_chunked(tr, h, mode):
    best, best_t = None, float("inf")
    for bm in (16, 32, 64):
        for bn in (256, 512, 1024, 2048):
            for chunk in (16, 32):
                cfg = (bm, bn, chunk, 4, 1)
                try:
                    t = timed(lambda: run_chunked(tr, h, mode, cfg))
                except Exception:
                    continue
                if t < best_t:
                    best, best_t = cfg, t
    if best is None:
        raise RuntimeError("no feasible chunked config")
    bm, bn, chunk, _, _ = best
    for w in (2, 4, 8):
        for s in (1, 2):
            cfg = (bm, bn, chunk, w, s)
            try:
                t = timed(lambda: run_chunked(tr, h, mode, cfg))
            except Exception:
                continue
            if t < best_t:
                best, best_t = cfg, t
    return best


def final_min(fn, reps=15):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.min(ts))


def main():
    rng = np.random.default_rng(0)
    rows = []
    for d in DIMS:
        xtr = sample_mixture(N, d, rng)
        h = silverman_bandwidth_nd(xtr)
        tr = torch.as_tensor(xtr, device=DEV)
        row = {"d": d}
        for mode in ("fast_tf32", "fp32_ieee"):
            cfg_c = tune_chunked(tr, h, mode)
            row[f"chunked_cfg_{mode}"] = cfg_c
            row[f"chunked_ms_{mode}"] = final_min(lambda: run_chunked(tr, h, mode, cfg_c))
            row[f"twopass_ms_{mode}"] = final_min(
                lambda: run_two_pass(tr, h, mode, TWO_PASS[d][mode])
            )
        row["speedup_tf32"] = row["twopass_ms_fast_tf32"] / row["chunked_ms_fast_tf32"]
        row["tc_speedup_chunked"] = row["chunked_ms_fp32_ieee"] / row["chunked_ms_fast_tf32"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open("exp14_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
