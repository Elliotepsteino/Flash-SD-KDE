"""Can manually-tiled pure-PyTorch SD-KDE close the gap to Flash-SD-KDE?

Compares, at d=16, n_train=32768, n_test=4096 (score pass + KDE pass):
  - eager full materialization (Table 1 baseline)
  - manually tiled eager (row-chunked, several tile sizes; O(tile*n) memory)
  - torch.compile default
  - torch.compile mode="max-autotune" (Triton GEMM templates + epilogue fusion)
  - Flash-SD-KDE
TF32 enabled for everything.
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
REPEATS = 15

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def sd_kde_eager(tr, te, h):
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


def sd_kde_tiled(tr, te, h, tile):
    """Row-chunked: never allocates more than tile x n. Same math as eager."""
    inv_h2 = 1.0 / (h * h)
    n = tr.shape[0]
    xn = (tr * tr).sum(1, keepdim=True)
    pdf_sum = torch.empty(n, device=tr.device)
    weighted = torch.empty_like(tr)
    for s in range(0, n, tile):
        blk = tr[s : s + tile]
        d2 = torch.clamp((blk * blk).sum(1, keepdim=True) + xn.T - 2.0 * (blk @ tr.T), min=0)
        phi = torch.exp(-0.5 * d2 * inv_h2)
        pdf_sum[s : s + tile] = phi.sum(1)
        weighted[s : s + tile] = phi @ tr
    deb = tr + 0.5 * h * h * ((weighted / (pdf_sum[:, None] + EPS)) - tr) * inv_h2
    dn = (deb * deb).sum(1, keepdim=True).T
    out = torch.empty(te.shape[0], device=te.device)
    for s in range(0, te.shape[0], tile):
        blk = te[s : s + tile]
        q2 = torch.clamp((blk * blk).sum(1, keepdim=True) + dn - 2.0 * (blk @ deb.T), min=0)
        out[s : s + tile] = torch.exp(-0.5 * q2 * inv_h2).sum(1)
    d = tr.shape[1]
    norm = 1.0 / (((2 * math.pi) ** (d / 2)) * (h**d) * n)
    return norm * out


def flash(tr, te, h):
    deb, _ = empirical_sd_kde_triton_nd(tr, h, return_tensor=True, synchronize=False)
    return gaussian_kde_triton_nd(deb, te, h, synchronize=False)


def timed_min(fn, warmup=2, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.min(ts)), float(np.mean(ts))


def main():
    np.random.seed(0)
    xtr = sample_gaussian_mixture_16d(N_TRAIN)
    xte = sample_gaussian_mixture_16d(N_TEST)
    h = silverman_bandwidth_nd(xtr)
    tr = torch.as_tensor(xtr, device=DEV)
    te = torch.as_tensor(xte, device=DEV)

    results = {}
    results["eager_full"] = timed_min(lambda: sd_kde_eager(tr, te, h))
    for tile in (512, 1024, 2048, 4096, 8192):
        results[f"tiled_{tile}"] = timed_min(lambda: sd_kde_tiled(tr, te, h, tile))

    c_default = torch.compile(sd_kde_eager)
    c_default(tr, te, h)
    results["compile_default"] = timed_min(lambda: c_default(tr, te, h))

    try:
        c_ma = torch.compile(sd_kde_eager, mode="max-autotune")
        c_ma(tr, te, h)
        results["compile_max_autotune"] = timed_min(lambda: c_ma(tr, te, h))
    except Exception as exc:  # autotune can be fragile across versions
        results["compile_max_autotune"] = f"failed: {exc}"

    results["flash"] = timed_min(lambda: flash(tr, te, h))

    print(json.dumps(results, indent=2))
    with open("exp6_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
