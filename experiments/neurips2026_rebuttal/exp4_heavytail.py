"""Heavy-tailed non-Gaussian oracle benchmark: Student-t mixture in 1D.

Compares KDE, Flash-SD-KDE, and Flash-Laplace-KDE on a two-component
Student-t (df=3) mixture. Accuracy-only (contention-safe).
"""
import json
import math

import numpy as np
import torch

from flash_sd_kde.reference import silverman_bandwidth_1d
from kernels.flash_sd_kde.padded_nd import (
    empirical_sd_kde_triton_padded_nd,
    gaussian_kde_triton_padded_nd,
)

DEV = "cuda"
DF = 3.0
COMPS = [(-2.0, 1.0, 0.5), (2.0, 0.5, 0.5)]  # (loc, scale, weight)
GRID = np.linspace(-30.0, 30.0, 12001)
N_LIST = [1024, 4096, 16384, 65536]
SEEDS = [0, 1, 2, 3, 4]


def t_pdf(x, loc, scale, df):
    z = (x - loc) / scale
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2) * scale)
    return c * (1 + z * z / df) ** (-(df + 1) / 2)


def true_pdf(x):
    return sum(w * t_pdf(x, loc, s, DF) for loc, s, w in COMPS)


def sample(n, rng):
    z = rng.random(n)
    out = np.empty(n, dtype=np.float64)
    c0 = z < COMPS[0][2]
    for idx, (loc, s, _w) in enumerate(COMPS):
        mask = c0 if idx == 0 else ~c0
        k = int(mask.sum())
        out[mask] = rng.standard_t(DF, size=k) * s + loc
    return out.astype(np.float32)


def metrics(p_hat, p_ref, grid):
    err = p_hat - p_ref
    ise = float(np.trapz(err * err, grid))
    iae = float(np.trapz(np.abs(err), grid))
    neg = float(np.trapz(np.clip(-p_hat, 0.0, None), grid))
    return ise, iae, neg


def main():
    grid_t = torch.as_tensor(GRID.astype(np.float32)[:, None], device=DEV)
    p_ref = true_pdf(GRID)
    rows = []
    for n in N_LIST:
        acc = {m: {"ise": [], "iae": [], "neg": []} for m in ("kde", "sd_kde", "laplace")}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = sample(n, rng)
            h = silverman_bandwidth_1d(x)
            xt = torch.as_tensor(x[:, None], device=DEV)

            p_kde = gaussian_kde_triton_padded_nd(xt, grid_t, h, device=DEV).cpu().numpy()

            deb_np, _ = empirical_sd_kde_triton_padded_nd(xt, h, device=DEV)
            deb = torch.as_tensor(deb_np, device=DEV)
            p_sd = gaussian_kde_triton_padded_nd(deb, grid_t, h, device=DEV).cpu().numpy()

            p_lc = (
                gaussian_kde_triton_padded_nd(
                    xt, grid_t, h, device=DEV, apply_laplacian_correction=True
                )
                .cpu()
                .numpy()
            )

            for name, p_hat in (("kde", p_kde), ("sd_kde", p_sd), ("laplace", p_lc)):
                ise, iae, neg = metrics(p_hat.astype(np.float64), p_ref, GRID)
                acc[name]["ise"].append(ise)
                acc[name]["iae"].append(iae)
                acc[name]["neg"].append(neg)
        row = {"n": n}
        for name, vals in acc.items():
            row[name] = {
                "ise_mean": float(np.mean(vals["ise"])),
                "ise_std": float(np.std(vals["ise"])),
                "iae_mean": float(np.mean(vals["iae"])),
                "iae_std": float(np.std(vals["iae"])),
                "neg_mass_mean": float(np.mean(vals["neg"])),
            }
        rows.append(row)
        print(json.dumps(row))
    with open("exp4_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
