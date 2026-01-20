from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from gitbud.gitbud import get_commit_hash, get_exp_info, get_repo

from globals import FILE_STORAGE_ROOT


def ensure_repo() -> str:
    repo = get_repo()
    if repo is None or repo.working_tree_dir is None:
        raise RuntimeError("git repo not found; cannot resolve repo root.")
    return repo.working_tree_dir


def get_repo_state() -> dict[str, Any]:
    repo = get_repo()
    if repo is None:
        return {"dirty": None, "commit": None}
    return {"dirty": repo.is_dirty(), "commit": get_commit_hash(repo)}


def make_run_dir(*, tag: str) -> Path:
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
