import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from globals import FILE_STORAGE_ROOT

try:
    from gitbud.gitbud import get_commit_hash, get_exp_info, get_repo
except ImportError:  # pragma: no cover - optional for package-only usage
    get_commit_hash = None
    get_exp_info = None
    get_repo = None


def _require_gitbud() -> None:
    if get_repo is None or get_commit_hash is None or get_exp_info is None:
        raise RuntimeError(
            "gitbud is required for flash_sd_kde.utils repository helpers. "
            "Install with `pip install gitbud`."
        )


def ensure_repo() -> str:
    _require_gitbud()
    repo = get_repo()
    if repo is None or repo.working_tree_dir is None:
        raise RuntimeError("git repo not found; cannot resolve repo root.")
    return repo.working_tree_dir


def get_repo_state() -> dict[str, Any]:
    if get_repo is None or get_commit_hash is None:
        return {"dirty": None, "commit": None}
    repo = get_repo()
    if repo is None:
        return {"dirty": None, "commit": None}
    return {"dirty": repo.is_dirty(), "commit": get_commit_hash(repo)}


def make_run_dir(*, tag: str) -> Path:
    _require_gitbud()
    repo_root = Path(ensure_repo())
    exp_info = get_exp_info()
    run_dir = repo_root / FILE_STORAGE_ROOT / tag / exp_info
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_env_override(value: Any, env_key: str) -> Any:
    env_val = os.getenv(env_key)
    if env_val is None:
        return value
    return type(value)(env_val)
