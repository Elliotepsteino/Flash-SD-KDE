"""2-D tiled cuBLAS score pass: can L2-resident tiles beat the HBM-floor argument?

Tiles the score pass over (query-block x data-block) so each Gram/phi tile is
small enough to (potentially) stay resident in the A6000's 6 MB L2 between the
cuBLAS GEMM and the elementwise kernels. Compares against the eager score pass
and Flash. TF32 enabled.
"""
import json
import time

import numpy as np
import torch

from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd

DEV = "cuda"
N = 32768
EPS = 1e-12

torch.backends.cuda.matmul.allow_tf32 = True


def score_eager(tr, h):
    inv_h2 = 1.0 / (h * h)
    xn = (tr * tr).sum(1, keepdim=True)
    d2 = torch.clamp(xn + xn.T - 2.0 * (tr @ tr.T), min=0)
    phi = torch.exp(-0.5 * d2 * inv_h2)
    return tr + 0.5 * h * h * ((phi @ tr) / (phi.sum(1, keepdim=True) + EPS) - tr) * inv_h2


def score_tiled2d(tr, h, tile):
    inv_h2 = 1.0 / (h * h)
    n, d = tr.shape
    xn = (tr * tr).sum(1, keepdim=True)
    pdf = torch.zeros(n, device=tr.device)
    weighted = torch.zeros_like(tr)
    for i in range(0, n, tile):
        q = tr[i : i + tile]
        qn = xn[i : i + tile]
        pdf_acc = torch.zeros(q.shape[0], device=tr.device)
        w_acc = torch.zeros_like(q)
        for j in range(0, n, tile):
            blk = tr[j : j + tile]
            g = q @ blk.T
            dist = torch.clamp(qn + xn[j : j + tile].T - 2.0 * g, min=0)
            phi = torch.exp(-0.5 * dist * inv_h2)
            pdf_acc += phi.sum(1)
            w_acc += phi @ blk
        pdf[i : i + tile] = pdf_acc
        weighted[i : i + tile] = w_acc
    return tr + 0.5 * h * h * ((weighted / (pdf[:, None] + EPS)) - tr) * inv_h2


def flash_score(tr, h):
    return empirical_sd_kde_triton_nd(tr, h, return_tensor=True, synchronize=False)[0]


def timed_min(fn, reps=10):
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
    np.random.seed(0)
    xtr = sample_gaussian_mixture_16d(N)
    h = silverman_bandwidth_nd(xtr)
    tr = torch.as_tensor(xtr, device=DEV)

    results = {"eager_score": timed_min(lambda: score_eager(tr, h))}
    for tile in (512, 1024, 2048, 4096):
        tile_mb = tile * tile * 4 / 2**20
        reps = 5 if tile <= 1024 else 10
        results[f"tiled2d_{tile} ({tile_mb:.0f} MB/tile)"] = timed_min(
            lambda: score_tiled2d(tr, h, tile), reps=reps
        )
    results["flash_score"] = timed_min(lambda: flash_score(tr, h))
    print(json.dumps(results, indent=2))
    with open("exp9_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
