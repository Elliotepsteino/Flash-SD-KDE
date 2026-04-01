from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TASKS = [
    "ifeval::tulu",
    "truthfulqa::tulu",
    "popqa::tulu",
]


def _safe_task_name(task: str) -> str:
    return task.replace("/", "_").replace(":", "_").replace("{", "_").replace("}", "_")


def _load_primary_score(metrics_path: Path, alias: str) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    task_entry = next(task for task in data["tasks"] if task["alias"] == alias)
    return {
        "alias": alias,
        "primary_score": float(task_entry["metrics"]["primary_score"]),
        "num_instances": int(task_entry["num_instances"]),
        "metrics_path": str(metrics_path),
    }


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Quick OLMES Eval",
        "",
        f"- Model: `{results['config']['model_dir']}`",
        f"- GPU: `{results['config']['gpu']}`",
        f"- Limit per task: `{results['config']['limit']}`",
        "",
        "| task | primary_score | num_instances |",
        "| --- | ---: | ---: |",
    ]
    for task in results["tasks"]:
        lines.append(f"| {task['alias']} | {task['primary_score']:.4f} | {task['num_instances']} |")
    lines.extend(
        [
            "",
            f"- Total runtime: `{results['timings']['total_seconds']:.2f}` s",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small OLMES slice for a local model directory.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--olmes-dir", default="/home/epsteine/post-training/olmes")
    parser.add_argument("--uv-path", default="uv")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    olmes_dir = Path(args.olmes_dir).expanduser().resolve()
    uv_resolved = shutil.which(args.uv_path)
    if uv_resolved is None:
        raise FileNotFoundError(f"Could not resolve uv executable from {args.uv_path!r}")

    tasks: list[dict[str, Any]] = []
    overall_start = time.perf_counter()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for task in args.tasks:
        safe_task = _safe_task_name(task)
        task_out_dir = output_dir / safe_task
        task_out_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{safe_task}.log"
        cmd = [
            uv_resolved,
            "run",
            "olmes",
            "--model",
            str(Path(args.model_dir).expanduser().resolve()),
            "--task",
            task,
            "--output-dir",
            str(task_out_dir),
            "--limit",
            str(args.limit),
            "--num-workers",
            "1",
            "--gpus",
            "1",
            "--save-raw-requests",
            "true",
        ]
        with log_path.open("w", encoding="utf-8") as log_handle:
            subprocess.run(
                cmd,
                cwd=str(olmes_dir),
                env=env,
                check=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        metrics_path = task_out_dir / "metrics.json"
        tasks.append(
            {
                **_load_primary_score(metrics_path, task),
                "log_path": str(log_path),
            }
        )

    results = {
        "config": {
            "model_dir": str(Path(args.model_dir).expanduser().resolve()),
            "output_dir": str(output_dir),
            "gpu": args.gpu,
            "limit": args.limit,
            "tasks": args.tasks,
        },
        "timings": {
            "total_seconds": time.perf_counter() - overall_start,
        },
        "tasks": tasks,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    (output_dir / "report.md").write_text(_render_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()
