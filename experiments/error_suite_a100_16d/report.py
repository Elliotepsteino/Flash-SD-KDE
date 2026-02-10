from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Any


def _format_float(val: Any, *, precision: int = 4) -> str:
    if val is None:
        return ""
    try:
        return f"{float(val):.{precision}g}"
    except (TypeError, ValueError):
        return str(val)


def _top_rows(rows: list[dict[str, Any]], key: str, *, n: int, reverse: bool) -> list[dict[str, Any]]:
    rows = [r for r in rows if r.get(key) is not None]
    rows.sort(key=lambda r: r.get(key, 0.0), reverse=reverse)
    return rows[:n]


def _make_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_float(row.get(key)) if key not in {"config", "label"} else str(row.get(key)) for _, key in columns)
            + " |"
        )
    return lines


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _make_latex_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, str]],
    *,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    col_spec = "".join(align for _, _, align in columns)
    header = " & ".join(_latex_escape(title) for title, _, _ in columns) + r" \\"
    lines = []
    if caption or label:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    for row in rows:
        fields = []
        for _, key, _ in columns:
            val = row.get(key)
            if key in {"config", "label"}:
                fields.append(_latex_escape(str(val)))
            else:
                fields.append(_latex_escape(_format_float(val)))
        lines.append(" & ".join(fields) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(r"\caption{" + _latex_escape(caption) + r"}")
    if label:
        lines.append(r"\label{" + _latex_escape(label) + r"}")
    if caption or label:
        lines.append(r"\end{table}")
    return "\n".join(lines)


def make_latex_tables(*, ok_rows: list[dict[str, Any]], frontier: list[dict[str, Any]]) -> dict[str, str]:
    tables: dict[str, str] = {}
    if ok_rows:
        top_kl = _top_rows(ok_rows, "kl_p_to_phat", n=10, reverse=False)
        tables["top_kl"] = _make_latex_table(
            top_kl,
            [
                ("KL", "kl_p_to_phat", "r"),
                ("Throughput", "throughput_qps", "r"),
                ("dtype", "compute_dtype", "l"),
                ("tf32", "enable_tf32", "l"),
                ("bw", "bandwidth_scale", "r"),
            ],
            caption="Top configurations by KL (lower is better).",
            label="tab:top_kl",
        )
        top_speed = _top_rows(ok_rows, "throughput_qps", n=10, reverse=True)
        tables["top_throughput"] = _make_latex_table(
            top_speed,
            [
                ("Throughput", "throughput_qps", "r"),
                ("KL", "kl_p_to_phat", "r"),
                ("dtype", "compute_dtype", "l"),
                ("tf32", "enable_tf32", "l"),
                ("bw", "bandwidth_scale", "r"),
            ],
            caption="Top configurations by throughput.",
            label="tab:top_throughput",
        )
    if frontier:
        tables["pareto_frontier"] = _make_latex_table(
            frontier[:15],
            [
                ("KL", "kl_p_to_phat", "r"),
                ("Throughput", "throughput_qps", "r"),
                ("dtype", "compute_dtype", "l"),
                ("tf32", "enable_tf32", "l"),
                ("bw", "bandwidth_scale", "r"),
            ],
            caption="Pareto frontier configurations (top 15).",
            label="tab:pareto_frontier",
        )
    return tables


def make_report(
    *,
    meta: dict[str, Any] | None,
    summary: dict[str, Any],
    ok_rows: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    failure_snippet: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Error Suite A100 16D Report")
    lines.append("")

    if meta:
        gpu = meta.get("gpu", {})
        lines.append("## Environment")
        lines.append("")
        sm = gpu.get("gpu_sm")
        sm_str = f", SM {sm}" if sm else ""
        lines.append(f"GPU: {gpu.get('gpu_name')}{sm_str} (VRAM {gpu.get('gpu_total_vram_bytes')})")
        lines.append(f"Torch: {meta.get('torch_version')} | CUDA: {meta.get('cuda_version')}")
        repo = meta.get("repo", {})
        lines.append(f"Git: commit={repo.get('commit')} dirty={repo.get('dirty')}")
        lines.append("")

        data_cfg = meta.get("data_cfg", {})
        truth = meta.get("truth", {})
        if data_cfg or truth:
            lines.append("## Dataset")
            lines.append("")
            lines.append(f"Dataset: {data_cfg.get('dataset')}")
            if data_cfg.get("standardize") is not None:
                lines.append(f"Standardize: {data_cfg.get('standardize')}")
            params = data_cfg.get("distribution_params", {}) if isinstance(data_cfg, dict) else {}
            if truth.get("type") == "gm_diag_16d":
                lines.append(
                    f"GM diag: K={params.get('n_components')}, component_std={params.get('component_std')}, mean_scale={params.get('mean_scale')}"
                )
            elif truth.get("type") == "gaussian_single_16d":
                lines.append("Gaussian single: mean/std from config.")
            lines.append("")

    lines.append("## Summary")
    lines.append("")
    counts = summary.get("counts", {})
    lines.append(
        f"Total: {counts.get('total', 0)}, ok: {counts.get('ok', 0)}, failed: {counts.get('failed', 0)}, skipped: {counts.get('skipped', 0)}"
    )
    lines.append("")

    failures = summary.get("failures", {})
    if failures:
        lines.append("Failures:")
        for k, v in failures.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    if ok_rows:
        lines.append("## Top configs by KL")
        lines.append("")
        top_kl = _top_rows(ok_rows, "kl_p_to_phat", n=10, reverse=False)
        lines.extend(
            _make_table(
                top_kl,
                [
                    ("KL", "kl_p_to_phat"),
                    ("Throughput", "throughput_qps"),
                    ("dtype", "compute_dtype"),
                    ("tf32", "enable_tf32"),
                    ("bw_scale", "bandwidth_scale"),
                ],
            )
        )
        lines.append("")

        lines.append("## Top configs by Throughput")
        lines.append("")
        top_speed = _top_rows(ok_rows, "throughput_qps", n=10, reverse=True)
        lines.extend(
            _make_table(
                top_speed,
                [
                    ("Throughput", "throughput_qps"),
                    ("KL", "kl_p_to_phat"),
                    ("dtype", "compute_dtype"),
                    ("tf32", "enable_tf32"),
                    ("bw_scale", "bandwidth_scale"),
                ],
            )
        )
        lines.append("")

    if frontier:
        lines.append("## Pareto Frontier")
        lines.append("")
        lines.extend(
            _make_table(
                frontier[:15],
                [
                    ("KL", "kl_p_to_phat"),
                    ("Throughput", "throughput_qps"),
                    ("dtype", "compute_dtype"),
                    ("tf32", "enable_tf32"),
                    ("bw_scale", "bandwidth_scale"),
                ],
            )
        )
        lines.append("")

    if failure_snippet:
        lines.append("## Failure Snippets")
        lines.append("")
        lines.append("```")
        lines.append(failure_snippet.rstrip())
        lines.append("```")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Full results: results.csv")
    lines.append("- Pareto frontier: pareto_frontier.json")

    return "\n".join(lines)
