from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Any

import torch

from experiments.error_suite_16d.numerics import log_abs_diff_exp, logmeanexp


def _safe_float(val: torch.Tensor) -> float:
    return float(val.item())


def compute_log_error_metrics(flash_logp: torch.Tensor, ref_logp: torch.Tensor) -> dict[str, Any]:
    if flash_logp.shape != ref_logp.shape:
        raise ValueError("flash and reference outputs must have matching shapes")

    finite_flash = torch.isfinite(flash_logp)
    finite_ref = torch.isfinite(ref_logp)
    finite_mask = finite_flash & finite_ref

    metrics: dict[str, Any] = {
        "finite_fraction_flash": _safe_float(finite_flash.float().mean()),
        "finite_fraction_ref": _safe_float(finite_ref.float().mean()),
    }

    if not torch.any(finite_mask):
        metrics.update(
            {
                "max_abs_log_err": float("nan"),
                "mean_abs_log_err": float("nan"),
                "rmse_log_err": float("nan"),
                "p95_abs_log_err": float("nan"),
                "bias_log_err": float("nan"),
            }
        )
        return metrics

    diff = flash_logp[finite_mask] - ref_logp[finite_mask]
    abs_diff = diff.abs()

    metrics.update(
        {
            "max_abs_log_err": _safe_float(abs_diff.max()),
            "mean_abs_log_err": _safe_float(abs_diff.mean()),
            "rmse_log_err": _safe_float(torch.sqrt((diff * diff).mean())),
            "p95_abs_log_err": _safe_float(torch.quantile(abs_diff, 0.95)),
            "bias_log_err": _safe_float(diff.mean()),
        }
    )
    return metrics


def compute_statistical_metrics(
    logp_hat: torch.Tensor,
    logp_true: torch.Tensor,
    *,
    ise_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if logp_hat.shape != logp_true.shape:
        raise ValueError("logp_hat and logp_true must have matching shapes")

    finite_mask = torch.isfinite(logp_hat) & torch.isfinite(logp_true)
    if not torch.any(finite_mask):
        return {
            "nll_hat": float("nan"),
            "nll_true": float("nan"),
            "kl_p_to_phat": float("nan"),
            "rmse_logp_err_true": float("nan"),
            "mean_abs_logp_err_true": float("nan"),
        }

    logp_hat = logp_hat[finite_mask]
    logp_true = logp_true[finite_mask]

    diff = logp_hat - logp_true
    metrics = {
        "nll_hat": _safe_float((-logp_hat).mean()),
        "nll_true": _safe_float((-logp_true).mean()),
        "kl_p_to_phat": _safe_float((logp_true - logp_hat).mean()),
        "rmse_logp_err_true": _safe_float(torch.sqrt((diff * diff).mean())),
        "mean_abs_logp_err_true": _safe_float(diff.abs().mean()),
    }

    if ise_cfg and ise_cfg.get("enabled", False):
        log_abs = log_abs_diff_exp(logp_hat, logp_true)
        log_term = 2.0 * log_abs - logp_true
        log_ise = logmeanexp(log_term, dim=0)
        ise = torch.exp(log_ise)
        metrics["log_ise_mc"] = _safe_float(log_ise)
        metrics["ise_mc"] = _safe_float(ise)
        if ise_cfg.get("compute_se", False):
            vals = torch.exp(log_term)
            metrics["ise_mc_se"] = _safe_float(vals.std(unbiased=True) / torch.sqrt(torch.tensor(vals.numel())))

    return metrics


def compute_oracle_error_metrics(logp_hat: torch.Tensor, logp_true: torch.Tensor) -> dict[str, Any]:
    if logp_hat.shape != logp_true.shape:
        raise ValueError("logp_hat and logp_true must have matching shapes")

    finite_mask = torch.isfinite(logp_hat) & torch.isfinite(logp_true)
    if not torch.any(finite_mask):
        return {
            "log_mise_mc": float("nan"),
            "mise_mc": float("nan"),
            "log_miae_mc": float("nan"),
            "miae_mc": float("nan"),
        }

    logp_hat = logp_hat[finite_mask]
    logp_true = logp_true[finite_mask]

    log_abs = log_abs_diff_exp(logp_hat, logp_true)
    log_mise = logmeanexp(2.0 * log_abs - logp_true, dim=0)
    log_miae = logmeanexp(log_abs - logp_true, dim=0)
    return {
        "log_mise_mc": _safe_float(log_mise),
        "mise_mc": _safe_float(torch.exp(log_mise)),
        "log_miae_mc": _safe_float(log_miae),
        "miae_mc": _safe_float(torch.exp(log_miae)),
    }
