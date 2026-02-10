from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Any


def _dominates(a: dict[str, Any], b: dict[str, Any], *, x_key: str, y_key: str, minimize_y: bool) -> bool:
    ax = a.get(x_key)
    bx = b.get(x_key)
    ay = a.get(y_key)
    by = b.get(y_key)
    if ax is None or bx is None or ay is None or by is None:
        return False
    if minimize_y:
        return ax >= bx and ay <= by and (ax > bx or ay < by)
    return ax >= bx and ay >= by and (ax > bx or ay > by)


def pareto_frontier(
    points: list[dict[str, Any]],
    *,
    x_key: str = "throughput_qps",
    y_key: str = "kl_p_to_phat",
    minimize_y: bool = True,
) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if _dominates(q, p, x_key=x_key, y_key=y_key, minimize_y=minimize_y):
                dominated = True
                break
        if not dominated:
            frontier.append(p)

    frontier.sort(key=lambda r: r.get(y_key, float("inf")) if minimize_y else -r.get(y_key, float("inf")))
    return frontier


def best_under_accuracy(
    points: list[dict[str, Any]],
    *,
    accuracy_thresholds: list[float],
    error_key: str,
    speed_key: str,
) -> list[dict[str, Any]]:
    results = []
    for threshold in accuracy_thresholds:
        candidates = [p for p in points if p.get(error_key) is not None and p.get(error_key) <= threshold]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r.get(speed_key, 0.0))
        results.append({"threshold": threshold, "best": best})
    return results


def best_under_speed(
    points: list[dict[str, Any]],
    *,
    speed_thresholds: list[float],
    error_key: str,
    speed_key: str,
) -> list[dict[str, Any]]:
    results = []
    for threshold in speed_thresholds:
        candidates = [p for p in points if p.get(speed_key) is not None and p.get(speed_key) >= threshold]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: r.get(error_key, float("inf")))
        results.append({"threshold": threshold, "best": best})
    return results
