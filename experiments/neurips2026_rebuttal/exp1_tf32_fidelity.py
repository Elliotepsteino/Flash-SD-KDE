"""TF32 vs FP32-IEEE numerical fidelity of Flash-SD-KDE against an FP64 reference.

16-D Gaussian mixture, n_train=32768, n_test=4096, Silverman bandwidth.
Accuracy-only experiment (contention-safe).
"""
import json
import math

import numpy as np
import torch

from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd, gaussian_kde_triton_nd
from kernels.flash_sd_kde.padded_nd import (
    empirical_sd_kde_triton_padded_nd,
    gaussian_kde_triton_padded_nd,
)

torch.backends.cuda.matmul.allow_tf32 = False

DEV = "cuda"
N_TRAIN, N_TEST = 32768, 4096
EPS = 1e-12


def sd_kde_fp64(tr, te, h, chunk=4096):
    inv_h2 = 1.0 / (h * h)
    xn = (tr * tr).sum(1, keepdim=True)
    pdf_sum = torch.zeros(tr.shape[0], dtype=torch.float64, device=DEV)
    weighted = torch.zeros_like(tr)
    for s in range(0, tr.shape[0], chunk):
        blk = tr[s : s + chunk]
        d2 = torch.clamp((blk * blk).sum(1, keepdim=True) + xn.T - 2.0 * (blk @ tr.T), min=0)
        phi = torch.exp(-0.5 * d2 * inv_h2)
        pdf_sum[s : s + chunk] = phi.sum(1)
        weighted[s : s + chunk] = phi @ tr
    score = (weighted / (pdf_sum[:, None] + EPS) - tr) * inv_h2
    deb = tr + 0.5 * h * h * score
    dn = (deb * deb).sum(1, keepdim=True).T
    out = torch.zeros(te.shape[0], dtype=torch.float64, device=DEV)
    for s in range(0, te.shape[0], chunk):
        blk = te[s : s + chunk]
        d2 = torch.clamp((blk * blk).sum(1, keepdim=True) + dn - 2.0 * (blk @ deb.T), min=0)
        out[s : s + chunk] = torch.exp(-0.5 * d2 * inv_h2).sum(1)
    d = tr.shape[1]
    norm = 1.0 / (((2 * math.pi) ** (d / 2)) * (h**d) * tr.shape[0])
    return deb, norm * out


def torch_sd_kde_fp32(tr, te, h):
    inv_h2 = 1.0 / (h * h)
    xn = (tr * tr).sum(1, keepdim=True)
    d2 = torch.clamp(xn + xn.T - 2.0 * (tr @ tr.T), min=0)
    phi = torch.exp(-0.5 * d2 * inv_h2)
    deb = tr + 0.5 * h * h * ((phi @ tr) / (phi.sum(1, keepdim=True) + EPS) - tr) * inv_h2
    dn = (deb * deb).sum(1, keepdim=True).T
    q2 = torch.clamp((te * te).sum(1, keepdim=True) + dn - 2.0 * (te @ deb.T), min=0)
    d = tr.shape[1]
    norm = 1.0 / (((2 * math.pi) ** (d / 2)) * (h**d) * tr.shape[0])
    return deb, norm * torch.exp(-0.5 * q2 * inv_h2).sum(1)


def true_pdf_16d(x, pi=0.5, mean_offset=1.5, s1=0.5, s2=1.0):
    d = x.shape[1]
    def comp(mu, s):
        z = ((x - mu) ** 2).sum(1) / (s * s)
        return torch.exp(-0.5 * z) / (((2 * math.pi) ** (d / 2)) * (s**d))
    return pi * comp(-mean_offset, s1) + (1 - pi) * comp(mean_offset, s2)


def rel_stats(p, ref):
    r = torch.abs(p.double() - ref) / ref
    return float(r.max()), float(r.mean())


def main():
    np.random.seed(0)
    xtr = sample_gaussian_mixture_16d(N_TRAIN)
    xte = sample_gaussian_mixture_16d(N_TEST)
    h = silverman_bandwidth_nd(xtr)

    tr64 = torch.as_tensor(xtr, dtype=torch.float64, device=DEV)
    te64 = torch.as_tensor(xte, dtype=torch.float64, device=DEV)
    tr32 = torch.as_tensor(xtr, device=DEV)
    te32 = torch.as_tensor(xte, device=DEV)

    deb64, p64 = sd_kde_fp64(tr64, te64, h)

    results = {"n_train": N_TRAIN, "n_test": N_TEST, "bandwidth": h}

    # Statistical error scale: FP64 SD-KDE vs oracle density
    p_true = true_pdf_16d(te64)
    stat = torch.abs(p64 - p_true) / p_true
    results["stat_rel_err_mean"] = float(stat.mean())
    results["stat_rel_err_median"] = float(stat.median())

    # Flash padded, TF32 vs IEEE
    for mode in ("fast_tf32", "fp32_ieee"):
        deb_np, _ = empirical_sd_kde_triton_padded_nd(tr32, h, device=DEV, precision_mode=mode)
        deb = torch.as_tensor(deb_np, device=DEV)
        p = gaussian_kde_triton_padded_nd(deb, te32, h, device=DEV, precision_mode=mode)
        mx, mn = rel_stats(p, p64)
        deb_err = float((deb.double() - deb64).norm(dim=1).max() / h)
        results[f"flash_padded_{mode}"] = {
            "density_rel_err_max": mx,
            "density_rel_err_mean": mn,
            "debias_shift_err_max_over_h": deb_err,
        }

    # Production specialized 16-D kernel (TF32)
    deb, _ = empirical_sd_kde_triton_nd(tr32, h, return_tensor=True)
    p = gaussian_kde_triton_nd(deb, te32, h)
    mx, mn = rel_stats(p, p64)
    results["flash_16d_specialized_tf32"] = {
        "density_rel_err_max": mx,
        "density_rel_err_mean": mn,
        "debias_shift_err_max_over_h": float((deb.double() - deb64).norm(dim=1).max() / h),
    }

    # Eager FP32 Torch baseline (TF32 off), for context
    deb, p = torch_sd_kde_fp32(tr32, te32, h)
    mx, mn = rel_stats(p, p64)
    results["torch_eager_fp32"] = {
        "density_rel_err_max": mx,
        "density_rel_err_mean": mn,
        "debias_shift_err_max_over_h": float((deb.double() - deb64).norm(dim=1).max() / h),
    }

    print(json.dumps(results, indent=2))
    with open("exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
