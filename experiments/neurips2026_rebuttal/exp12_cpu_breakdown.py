"""CPU score/KDE pass breakdown: same eager formulation on the dual EPYC 7763."""
import json
import math
import time

import numpy as np
import torch

from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd

N_TRAIN, N_TEST = 32768, 4096
EPS = 1e-12


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


def main():
    torch.set_num_threads(128)
    np.random.seed(0)
    xtr = sample_gaussian_mixture_16d(N_TRAIN)
    xte = sample_gaussian_mixture_16d(N_TEST)
    h = silverman_bandwidth_nd(xtr)
    tr = torch.as_tensor(xtr)
    te = torch.as_tensor(xte)

    deb = torch_score(tr, h)
    _ = torch_kde(deb, te, h)

    ts_score, ts_kde = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        deb = torch_score(tr, h)
        ts_score.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        _ = torch_kde(deb, te, h)
        ts_kde.append((time.perf_counter() - t0) * 1e3)

    out = {
        "threads": torch.get_num_threads(),
        "score_ms_min": float(np.min(ts_score)),
        "kde_ms_min": float(np.min(ts_kde)),
        "total_ms_min": float(np.min(ts_score) + np.min(ts_kde)),
    }
    print(json.dumps(out, indent=2))
    with open("exp12_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
