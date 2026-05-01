from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from scipy.io import arff

from flash_sd_kde.estimator import FlashSDKDE
from flash_sd_kde.utils import get_repo_state, make_run_dir, write_json
from globals import FILE_STORAGE_ROOT


ACE_PAPER_URL = "https://arxiv.org/abs/1706.06664"
ACE_SHUTTLE_URL = "https://raw.githubusercontent.com/RUSH-LAB/ace/master/shuttle.csv"
LMU_ALOI_URL = "https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/ALOI.tar.gz"


ACE_PUBLISHED_RESULTS: dict[str, dict[str, Any]] = {
    "shuttle": {
        "display_name": "Statlog Shuttle",
        "n_instances": 34987,
        "n_outliers": 879,
        "dimension": 9,
        "rows": [
            ("ACE", 6763, 273, 606, 0.81, "1x"),
            ("LOF", 4356, 381, 498, 14.12, "17.4x"),
            ("kNN", 4897, 493, 386, 12.35, "15.2x"),
            ("kNNW", 5264, 610, 269, 13.54, "16.7x"),
            ("LoOP", 6145, 201, 678, 14.51, "17.9x"),
            ("LDOF", 6433, 330, 549, 16.42, "20.3x"),
            ("ODIN", 9775, 375, 504, 12.21, "15.1x"),
            ("KDEOS", 12630, 314, 565, 11.73, "14.5x"),
            ("COF", 9133, 280, 599, 13.45, "16.6x"),
            ("LDF", 9809, 375, 504, 19.93, "24.6x"),
            ("INFLO", 4488, 183, 696, 14.03, "17.3x"),
            ("FastVOA", 8532, 271, 608, 235.10, "290.2x"),
        ],
    },
    "aloi": {
        "display_name": "Object Images (ALOI)",
        "n_instances": 50000,
        "n_outliers": 1508,
        "dimension": 27,
        "rows": [
            ("ACE", 7216, 340, 1168, 1.26, "1x"),
            ("LOF", 4476, 519, 989, 72.31, "57.4x"),
            ("kNN", 5428, 447, 1061, 63.27, "50.2x"),
            ("kNNW", 5558, 329, 1508, 89.96, "71.4x"),
            ("LoOP", 5121, 253, 1179, 59.97, "47.6x"),
            ("LDOF", 7501, 470, 1038, 60.39, "47.9x"),
            ("ODIN", 10110, 162, 1346, 72.69, "57.6x"),
            ("KDEOS", 9515, 404, 1104, 55.89, "44.36x"),
            ("COF", 8746, 284, 1224, 81.74, "64.9x"),
            ("LDF", 9133, 301, 1207, 60.51, "48.0x"),
            ("INFLO", 10328, 420, 1088, 72.13, "57.2x"),
            ("FastVOA", 8931, 319, 1189, 291.10, "231.0x"),
        ],
    },
    "kdd_http": {
        "display_name": "KDD-Cup99 HTTP",
        "n_instances": 596853,
        "n_outliers": 1055,
        "dimension": 36,
        "rows": [
            ("ACE", 22160, 406, 649, 23.33, "1x"),
            ("LOF", 13260, 523, 532, 1813.63, "77.7x"),
            ("kNN", 15432, 365, 690, 1483.54, "63.5x"),
            ("kNNW", 14328, 460, 595, 2125.43, "91.1x"),
            ("LoOP", 16578, 396, 659, 1594.54, "68.3x"),
            ("LDOF", 16579, 496, 559, 1674.43, "71.7x"),
            ("ODIN", 18054, 365, 690, 1918.34, "82.2x"),
            ("KDEOS", 21095, 469, 586, 1428.32, "61.2x"),
            ("COF", 20658, 584, 471, 2043.43, "87.5x"),
            ("LDF", 19574, 368, 687, 1485.85, "63.7x"),
            ("INFLO", 25704, 565, 490, 1684.47, "72.2x"),
            ("FastVOA", 29316, 354, 701, 3510.26, "150.4x"),
        ],
    },
}


def _download_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def _load_shuttle_exact() -> tuple[np.ndarray, np.ndarray]:
    raw = _download_bytes(ACE_SHUTTLE_URL).decode("utf-8")
    arr = np.loadtxt(io.StringIO(raw), dtype=np.float32)
    x = np.asarray(arr[:, :-1], dtype=np.float32)
    y = np.asarray(arr[:, -1] != 1.0, dtype=bool)
    return x, y


def _load_aloi_exact() -> tuple[np.ndarray, np.ndarray]:
    payload = _download_bytes(LMU_ALOI_URL)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        member = tf.getmember("ALOI/ALOI_norm.arff")
        with tf.extractfile(member) as handle:
            if handle is None:
                raise RuntimeError("Failed to extract ALOI/ALOI_norm.arff from archive.")
            with io.TextIOWrapper(handle, encoding="utf-8") as text_handle:
                data, _ = arff.loadarff(text_handle)
    names = data.dtype.names
    if names is None:
        raise RuntimeError("ALOI arff has no named columns.")
    feature_names = [name for name in names if name.startswith("att")]
    x = np.column_stack([np.asarray(data[name], dtype=np.float32) for name in feature_names])
    y = np.asarray([str(v, "utf-8") == "yes" for v in data["outlier"]], dtype=bool)
    return x, y


def _flash_row(
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    estimator = FlashSDKDE(mode="sd_kde", device=device)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    estimator.fit(x)
    score_chunks: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        score_chunks.append(estimator.score_samples(x[start:stop]))
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    scores = np.concatenate(score_chunks, axis=0)
    threshold = float(np.mean(scores) - np.std(scores))
    predicted = scores < threshold
    outliers_reported = int(np.sum(predicted))
    correctly_reported = int(np.sum(predicted & y))
    outliers_missed = int(np.sum(y) - correctly_reported)
    return {
        "method": "Flash-SD-KDE",
        "outliers_reported": outliers_reported,
        "correctly_reported": correctly_reported,
        "outliers_missed": outliers_missed,
        "execution_time_seconds": float(elapsed),
        "threshold": threshold,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }


def _fmt_time(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _fmt_speedup(value: float) -> str:
    return f"{value:.1f}x"


def _load_latest_public_kdd_result() -> dict[str, Any] | None:
    root = Path(FILE_STORAGE_ROOT) / "benchmarks" / "kdd99_public_http_njy1"
    if not root.exists():
        return None
    run_dirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    for run_dir in run_dirs:
        json_files = sorted(run_dir.glob("*_results.json"))
        if not json_files:
            continue
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    return None


def _published_ace_time(dataset_key: str) -> float:
    for method, _, _, _, secs, _ in ACE_PUBLISHED_RESULTS[dataset_key]["rows"]:
        if method == "ACE":
            return float(secs)
    raise KeyError(f"No ACE row found for {dataset_key}")


def _render_dataset_table(dataset_key: str, flash_row: dict[str, Any] | None) -> list[str]:
    published = ACE_PUBLISHED_RESULTS[dataset_key]
    lines = [
        f"## {published['display_name']} (n={published['n_instances']}, outliers={published['n_outliers']}, d={published['dimension']})",
        "",
        "| Method | Outliers Reported | Correctly Reported | Outliers Missed | Execution Time (s) | Speed-up with ACE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, reported, correct, missed, secs, speed in published["rows"]:
        lines.append(
            f"| {method} | {reported} | {correct} | {missed} | {_fmt_time(secs)} | {speed} |"
        )
    if flash_row is not None:
        ace_time = _published_ace_time(dataset_key)
        speed = ace_time / float(flash_row["execution_time_seconds"])
        lines.append(
            f"| Flash-SD-KDE | {flash_row['outliers_reported']} | {flash_row['correctly_reported']} | "
            f"{flash_row['outliers_missed']} | {_fmt_time(flash_row['execution_time_seconds'])} | "
            f"{_fmt_speedup(speed)}* |"
        )
    return lines


def _render_dataset_summary() -> list[str]:
    lines = [
        "## Dataset Summary",
        "",
        "| Dataset | Samples (n) | Outliers | Dimension (d) | Flash rerun |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for dataset_key in ["shuttle", "aloi", "kdd_http"]:
        info = ACE_PUBLISHED_RESULTS[dataset_key]
        rerun = "Yes" if dataset_key in {"shuttle", "aloi"} else "Published ACE only"
        lines.append(
            f"| {info['display_name']} | {info['n_instances']} | {info['n_outliers']} | "
            f"{info['dimension']} | {rerun} |"
        )
    return lines


def _render_interpretation(results: dict[str, Any]) -> list[str]:
    shuttle = results["flash_results"].get("shuttle")
    aloi = results["flash_results"].get("aloi")
    public_kdd = results.get("public_kdd_result", {}).get("flash")
    lines = [
        "## Interpretation",
        "",
    ]
    if shuttle is not None:
        lines.append(
            f"- Shuttle helps the case strongly: Flash-SD-KDE recovers `{shuttle['correctly_reported']}` of "
            f"`{ACE_PUBLISHED_RESULTS['shuttle']['n_outliers']}` anomalies, versus `273` for ACE, with fewer missed "
            f"outliers (`{shuttle['outliers_missed']}` vs `606`) and a comparable measured runtime (`{_fmt_time(shuttle['execution_time_seconds'])}` on our GPU)."
        )
    if aloi is not None:
        lines.append(
            f"- ALOI is supportive but less decisive: Flash-SD-KDE improves the number of correctly reported outliers "
            f"(`{aloi['correctly_reported']}` vs `340` for ACE) at a very similar reported-outlier count "
            f"(`{aloi['outliers_reported']}` vs `7216`) and low measured runtime (`{_fmt_time(aloi['execution_time_seconds'])}` on our GPU)."
        )
    if public_kdd is not None:
        lines.append(
            f"- We also ran Flash-SD-KDE on a public KDD99 HTTP benchmark release at a similar scale; although that release does not match the ACE KDD subset exactly, it still shows feasibility at `n={public_kdd['n']}` with `d={public_kdd['d']}`, total runtime `{_fmt_time(public_kdd['total_seconds'])}`, and `ROC-AUC={public_kdd['roc_auc']:.4f}`."
        )
    lines.append(
        "- Overall, these results support practical utility on anomaly-detection benchmarks, but they are still partial evidence because the runtime comparison is cross-hardware and we do not yet have a matched Flash rerun for the ACE KDD benchmark."
    )
    return lines


def _render_public_kdd_section(public_kdd_result: dict[str, Any] | None) -> list[str]:
    if public_kdd_result is None:
        return []
    flash = public_kdd_result["flash"]
    lines = [
        f"## Public KDD99 HTTP Benchmark Release (n={flash['n']}, outliers={flash['n_outliers']}, d={flash['d']})",
        "",
        "This is a separate public KDD99 HTTP point-anomaly benchmark release associated with the",
        "comparative unsupervised anomaly-detection benchmark collection. It does not exactly match",
        "the ACE-paper KDD subset: ACE reports `n=596853`, `outliers=1055`, `d=36`, whereas this",
        f"public release has `n={flash['n']}`, `outliers={flash['n_outliers']}`, `d={flash['d']}`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Fit time (s) | {flash['fit_seconds']:.2f} |",
        f"| Score time (s) | {flash['score_seconds']:.2f} |",
        f"| Total time (s) | {flash['total_seconds']:.2f} |",
        f"| ROC-AUC | {flash['roc_auc']:.4f} |",
        f"| Average Precision | {flash['average_precision']:.6f} |",
        f"| Outliers reported (ACE-style threshold) | {flash['outliers_reported']} |",
        f"| Correctly reported | {flash['correctly_reported']} |",
        f"| Outliers missed | {flash['outliers_missed']} |",
        "",
    ]
    return lines


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# ACE Paper Comparison Report",
        "",
        "## Scope",
        "",
        "This report compares Flash-SD-KDE against the published baselines from",
        "`Arrays of (locality-sensitive) Count Estimators (ACE): High-Speed Anomaly Detection via Cache Lookups`.",
        "",
        f"- ACE paper: `{ACE_PAPER_URL}`",
        f"- Exact Shuttle data source: `{ACE_SHUTTLE_URL}`",
        f"- Exact ALOI bundle source: `{LMU_ALOI_URL}`",
        "",
        "We use the same thresholding rule described in the ACE paper:",
        "report an anomaly whenever the score is below `mean(score) - std(score)`.",
        "For Flash-SD-KDE, the score is the per-point log density returned by the fitted estimator.",
        "",
        "## Notes",
        "",
        "- The ACE paper rows below are copied from Tables 3, 4, and 5 of the paper.",
        "- Flash-SD-KDE runtimes were measured on the current GPU machine, not on the ACE paper's CPU system.",
        "- The `Speed-up with ACE` value for Flash-SD-KDE is therefore cross-hardware and only illustrative.",
        "- We reran Flash-SD-KDE on the exact ACE Shuttle file and on the standard-preprocessed 50,000-example `ALOI_norm.arff` variant from the LMU benchmark bundle, matching the paper's reported ALOI size and dimensionality after preprocessing.",
        "- The ACE KDD-Cup99 HTTP preprocessing does not match a simple public subset, so the ACE table remains published-only here.",
        "- To still provide a large-scale KDD-style datapoint, we also include a separate Flash-SD-KDE run on the public `kdd99-unsupervised-ad.tab` benchmark release and explicitly note its dataset mismatch with the ACE KDD row.",
        "",
    ]

    lines.extend(_render_dataset_summary())
    lines.append("")
    lines.extend(_render_interpretation(results))
    lines.append("")

    for dataset_key in ["shuttle", "aloi", "kdd_http"]:
        lines.extend(_render_dataset_table(dataset_key, results["flash_results"].get(dataset_key)))
        lines.append("")

    lines.extend(_render_public_kdd_section(results.get("public_kdd_result")))

    lines.extend(
        [
            "## Flash-SD-KDE Configuration",
            "",
            f"- Device: `{results['config']['device']}`",
            f"- Batch size: `{results['config']['batch_size']}`",
            f"- Repo commit: `{results['meta']['commit']}`",
            f"- Repo dirty: `{results['meta']['dirty']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Flash-SD-KDE to ACE-paper anomaly benchmarks.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--datasets",
        default="shuttle,aloi",
        help="Comma-separated list drawn from {shuttle,aloi}. KDD paper rows are always included as published reference.",
    )
    parser.add_argument("--output-tag", default="benchmarks/ace_paper_comparison_njy1")
    args = parser.parse_args()

    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    valid = {"shuttle", "aloi"}
    unknown = set(requested) - valid
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {sorted(unknown)}")

    run_dir = make_run_dir(tag=args.output_tag)
    flash_results: dict[str, dict[str, Any]] = {}

    if "shuttle" in requested:
        x, y = _load_shuttle_exact()
        flash_results["shuttle"] = _flash_row(x, y, device=args.device, batch_size=args.batch_size)

    if "aloi" in requested:
        x, y = _load_aloi_exact()
        flash_results["aloi"] = _flash_row(x, y, device=args.device, batch_size=args.batch_size)

    results = {
        "config": {
            "device": args.device,
            "batch_size": args.batch_size,
            "datasets": requested,
            "output_tag": args.output_tag,
        },
        "meta": get_repo_state(),
        "flash_results": flash_results,
        "ace_published_results": ACE_PUBLISHED_RESULTS,
        "public_kdd_result": _load_latest_public_kdd_result(),
    }

    output_stem = Path(args.output_tag).name
    report_path = run_dir / f"{output_stem}_report.md"
    json_path = run_dir / f"{output_stem}_results.json"
    write_json(json_path, results)
    report_path.write_text(_render_report(results), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(f"Wrote JSON to {json_path}")


if __name__ == "__main__":
    main()
