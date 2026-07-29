"""Final end-to-end d-sweep with the chunked score kernel: totals for the Q1 table."""
import json
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
REPEATS = 20

CHUNKED = {
    16: {"fast_tf32": (64, 2048, 32, 2, 1), "fp32_ieee": None},
    32: {"fast_tf32": (64, 2048, 32, 4, 1), "fp32_ieee": None},
    64: {"fast_tf32": (64, 2048, 16, 4, 2), "fp32_ieee": None},
    128: {"fast_tf32": (32, 2048, 32, 4, 2), "fp32_ieee": None},
}
# Best no-TC score configs come from the two-pass kernel (better without TCs at d>=64)
TWO_PASS_IEEE = {
    16: (32, 128, 16, 4, 3),
    32: (32, 256, 16, 4, 2),
    64: (32, 128, 32, 4, 2),
    128: (32, 128, 32, 4, 2),
}
KDE_CFG = {
    16: (64, 128, 16, 4, 2),
    32: (128, 128, 32, 4, 2),
    64: (64, 256, 16, 4, 1),
    128: (128, 128, 16, 4, 1),
}


def sample_mixture(n, d, rng):
    z = rng.random(n) < 0.5
    out = np.empty((n, d), dtype=np.float32)
    n1 = int(z.sum())
    out[z] = rng.normal(-1.5, 0.5, size=(n1, d)).astype(np.float32)
    out[~z] = rng.normal(1.5, 1.0, size=(n - n1, d)).astype(np.float32)
    return out


def flash(tr, te, h, mode, d):
    if mode == "fast_tf32":
        bm, bn, chunk, w, s = CHUNKED[d][mode]
        deb, _ = empirical_sd_kde_triton_padded_nd(
            tr, h, device=DEV, precision_mode=mode, block_m=bm, block_n=bn,
            num_warps=w, num_stages=s, chunked=True, block_n_chunk=chunk,
            return_tensor=True, synchronize=False,
        )
    else:
        bm, bn, bk, w, s = TWO_PASS_IEEE[d]
        deb, _ = empirical_sd_kde_triton_padded_nd(
            tr, h, device=DEV, precision_mode=mode, block_m=bm, block_n=bn,
            block_k=bk, num_warps=w, num_stages=s, return_tensor=True,
            synchronize=False,
        )
    km, kn, kk, kw, ks = KDE_CFG[d]
    return gaussian_kde_triton_padded_nd(
        deb, te, h, device=DEV, precision_mode=mode, block_m=km, block_n=kn,
        block_k=kk, num_warps=kw, num_stages=ks, synchronize=False,
    )


def main():
    rng = np.random.default_rng(0)
    rows = []
    for d in (16, 32, 64, 128):
        xtr = sample_mixture(N_TRAIN, d, rng)
        xte = sample_mixture(N_TEST, d, rng)
        h = silverman_bandwidth_nd(xtr)
        tr = torch.as_tensor(xtr, device=DEV)
        te = torch.as_tensor(xte, device=DEV)

        fns = {
            "flash_tf32": lambda: flash(tr, te, h, "fast_tf32", d),
            "flash_no_tc": lambda: flash(tr, te, h, "fp32_ieee", d),
        }
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()
        times = {k: [] for k in fns}
        for _ in range(REPEATS):
            for k, fn in fns.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times[k].append((time.perf_counter() - t0) * 1e3)

        row = {"d": d}
        for k, ts in times.items():
            row[k + "_ms_min"] = float(np.min(ts))
        row["tc_speedup"] = row["flash_no_tc_ms_min"] / row["flash_tf32_ms_min"]
        flops = (4 * d + 12 + d / 4 + 1.5) * N_TRAIN * N_TRAIN
        row["tflops"] = flops / (row["flash_tf32_ms_min"] * 1e-3) / 1e12
        row["frac_tc_peak"] = row["tflops"] / 155.0
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open("exp15_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
