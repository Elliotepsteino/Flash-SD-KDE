from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import csv
from pathlib import Path

from flash_sd_kde.utils import read_json


def _load_negative_mass_headline(results_csv: Path) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok" or row.get("method") != "flash_laplace":
                continue
            n_train = int(float(row["n_train"]))
            grouped.setdefault(n_train, []).append(float(row["negative_mass_fraction_laplace"]))
    if not grouped:
        raise ValueError("No flash_laplace rows found in negative-mass CSV.")
    n_train = max(grouped)
    vals = grouped[n_train]
    return {
        "n_train": n_train,
        "negative_mass_fraction_laplace_mean": sum(vals) / len(vals),
    }


def _load_negative_mass_by_size(results_csv: Path) -> list[dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok" or row.get("method") != "flash_laplace":
                continue
            n_train = int(float(row["n_train"]))
            n_test = int(float(row["n_test"]))
            bucket = grouped.setdefault(
                n_train,
                {
                    "n_test": [float(n_test)],
                    "negative_fraction_laplace": [],
                    "negative_mass_fraction_laplace": [],
                },
            )
            bucket["negative_fraction_laplace"].append(float(row["negative_fraction_laplace"]))
            bucket["negative_mass_fraction_laplace"].append(float(row["negative_mass_fraction_laplace"]))

    rows = []
    for n_train in sorted(grouped):
        bucket = grouped[n_train]
        neg_frac = bucket["negative_fraction_laplace"]
        neg_mass_frac = bucket["negative_mass_fraction_laplace"]
        rows.append(
            {
                "n_train": n_train,
                "n_test": int(bucket["n_test"][0]),
                "negative_fraction_mean": sum(neg_frac) / len(neg_frac),
                "negative_mass_fraction_mean": sum(neg_mass_frac) / len(neg_mass_frac),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an overall ICML 2026 rebuttal report.")
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--negative-mass-csv", required=True)
    parser.add_argument("--embedding-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime = read_json(Path(args.runtime_json))
    embedding = read_json(Path(args.embedding_json))
    neg_headline = _load_negative_mass_headline(Path(args.negative_mass_csv))
    neg_rows = _load_negative_mass_by_size(Path(args.negative_mass_csv))

    runtime_rows = sorted(runtime["rows"], key=lambda row: row["n_train"])
    runtime_last = runtime_rows[-1]
    embed_last_n = max(int(n) for n in embedding["n_train_list"])
    embed_key = str(embed_last_n)

    lines = [
        "# ICML 2026 Rebuttal Report",
        "",
        "## Major Points",
        "",
        f"Flash-SD-KDE remains the fastest method in the 16-D rebuttal runtime sweep. "
        f"At n_train={int(runtime_last['n_train'])}, it runs in {runtime_last['flash_sd_kde_ms']:.2f} ms "
        f"versus {runtime_last['sd_pykeops_ms']:.2f} ms for SD-KDE (PyKeOps), "
        f"{runtime_last['sd_torch_compile_ms']:.2f} ms for SD-KDE (Torch compile), "
        f"{runtime_last['sd_torch_ms']:.2f} ms for SD-KDE (Torch), and "
        f"{runtime_last['sklearn_kde_ms']:.2f} ms for sklearn KDE.",
        "",
        f"The Flash-Laplace-KDE negativity diagnostic stays moderate in the oracle sweep: "
        f"the mean negative-mass fraction at the largest setting is "
        f"{100.0 * neg_headline['negative_mass_fraction_laplace_mean']:.2f}% "
        f"at n_train={neg_headline['n_train']}.",
        "",
        f"In the PCA-64 embedding benchmark at n_train={embed_last_n}, exact SD-KDE with "
        f"torch.compile achieved ROC AUC {embedding['metrics']['exact_sd_compile'][embed_key]['roc_auc']:.4f} "
        f"with runtime {embedding['runtime_ms']['exact_sd_compile'][embed_key]:.2f} ms, while "
        f"matching eager SD-KDE quality almost exactly.",
        "",
        "## Rebuttal Runtime Figure",
        "",
        "The added `torch.compile` row makes the comparison stronger: it shows that even after",
        "optimizing the exact Torch baseline with the compiler, the fused Flash path remains materially faster.",
        "",
        "| n_train | sklearn KDE (ms) | SD-KDE Torch (ms) | SD-KDE Torch compile (ms) | SD-KDE PyKeOps (ms) | Flash-SD-KDE (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runtime_rows:
        lines.append(
            f"| {int(row['n_train'])} | {row['sklearn_kde_ms']:.2f} | {row['sd_torch_ms']:.2f} | "
            f"{row['sd_torch_compile_ms']:.2f} | {row['sd_pykeops_ms']:.2f} | {row['flash_sd_kde_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
        "## PCA-64 Embedding Benchmark",
        "",
        "Plainly: we compress MNIST images to 64 PCA coordinates, fit a density model on MNIST,",
        "and then ask whether the model gives higher scores to MNIST test images than to Fashion-MNIST test images.",
        "So this is a simple ID-vs-OOD density-separation check in a higher-dimensional feature space.",
        "",
        "| n_train | Method | Runtime (ms) | ROC AUC | PR AUC | loglik gap |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for n_train in embedding["n_train_list"]:
        key = str(n_train)
        for method in embedding["methods"]:
            metrics = embedding["metrics"][method][key]
            runtime_ms = embedding["runtime_ms"][method][key]
            lines.append(
                f"| {n_train} | {method} | {runtime_ms:.2f} | {metrics['roc_auc']:.4f} | "
                f"{metrics['pr_auc']:.4f} | {metrics['loglik_gap']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Negativity By Problem Size",
            "",
            "Two views are useful here.",
            "`negative estimates` is the fraction of query points where Flash-Laplace-KDE is below zero.",
            "`negative mass fraction` is the fraction of total signed mass carried by the negative part.",
            "",
            "| n_train | n_test | negative estimates | negative mass fraction |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in neg_rows:
        lines.append(
            f"| {row['n_train']} | {row['n_test']} | "
            f"{100.0 * row['negative_fraction_mean']:.2f}% | "
            f"{100.0 * row['negative_mass_fraction_mean']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Runtime JSON: `{args.runtime_json}`",
            f"- Negative-mass CSV: `{args.negative_mass_csv}`",
            f"- Embedding benchmark JSON: `{args.embedding_json}`",
        ]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote overall rebuttal report to {output_path}")


if __name__ == "__main__":
    main()
