# gitbud.md

**Pre-reqs**

* Run inside a **non-bare** Git repo.
* Globals: `TIME_FORMAT`, `FILE_STORAGE_ROOT`.
* Libs: `GitPython`.

---

## API (1-liners)

* `get_repo() -> Repo|None` — find repo (search parents).
* `get_commit_hash(repo) -> str` — `rev-parse HEAD`.
* `is_dirty(repo) -> bool` — working tree dirty?
* `get_time_str() -> str` — `time.strftime(TIME_FORMAT)`.
* `get_exp_info(short_length=8) -> str` — `"{TIME}-{sha8}-dirty={bool}"`.
* `inject_repo_into_sys_path() -> str` — prepend repo root to `sys.path`, return root.
* `get_tree_hash(repo, dir_path) -> str|None` — tree SHA for directory at `HEAD`.
* `branches_pointing_to(repo, commit_sha, *, include_remote=False) -> list[str]` — branch tips equal to `commit_sha`.
* `freeze_notebook(filename, repo, commit_hash, **assignments) -> None`

  * Copies `filename` to:
    `{REPO}/{FILE_STORAGE_ROOT}/notebooks/{COMMIT_TIME}_{BRANCH}_{sha8}/{BRANCH}_{sha8}_{base}`
  * Rewrites in copy:

    * `COMMIT_HASH = "<commit_hash>"`
    * each `name = ...` in `assignments` → `name = "<value>"` (stringified).

---

## Gotchas

* `freeze_notebook` **requires** a `COMMIT_HASH = ...` line in source.
* All `assignments` keys must exist as `name = ...` lines.
* Uses first **local** branch pointing to commit (`branches[0]`); detached/no-branch → error.
* `inject_repo_into_sys_path()` returns `repo_root`; if no repo, it prints and returns nothing.

---

