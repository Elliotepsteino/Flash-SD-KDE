# SLACK_RESULTS: How to send results to Slack (Codex-safe)

This document explains how automated tools / LLMs (like Codex) should use the
`slack-results` Python package.

Goal: **send experiment results (plots in png, PDFs, etc.) to Slack** in a consistent,
LLM-safe way, with **one simple function**.

---

## 1) TL;DR usage

Always use this exact pattern:

```python
from slack_results import send_results_to_slack

send_results_to_slack(
    "SHORT SUMMARY OF THE RESULTS HERE",
    [
        # IMPORTANT: these paths are RELATIVE TO THE GIT REPO ROOT,
        # not the current working directory.
        "file_storage/runs/42/loss.png",
        "file_storage/runs/42/trajectory.gif",
        "file_storage/runs/42/report.pdf",
    ],
)
```

- **MUST NOT** call `slack_sdk` directly.
- **MUST NOT** create channels manually (the helper handles it).
- **MUST NOT** retry in a loop if Slack fails.
* **MUST NOT** try to guess repo roots, tokens, or fix configuration issues.
- If Slack fails, **MUST** continue the run without Slack and surface the error.

---

## 2) What to put in the Slack summary 

The `summary` string can be multi-line. Use it to make the message terse but informative.
The primary goal is to use this as a means to share important results in visual form (so png plots, gifs, tables)

- **MUST** include:
  - repo state (dirty or not, nothing else) (retrieved via `gitbud.get_repo().is_dirty()` ONLY, do not use another mechanism).
  - generated visual artifacts, like plots, gifs or pdfs, or even tables.
  - descriptive text depending on what is asked

- Also include informative, terse, bulleted summary of results, conclusions, inferences, any sus things noticed, and next steps.

- **MUST NOT** include:
  - raw data dumps, csvs, parquets, jsons, etc. unless *EXPLICITLY* asked for,
  - large text blobs.
  - file paths (especially long ones)
  - commit hashes or subtree hashes, unless *EXPLICITLY* asked for.


## 3) What `send_results_to_slack` does

Signature:

```python
from slack_results import send_results_to_slack

send_results_to_slack(summary: str, files: list[str | pathlib.Path]) -> None
```

Behavior:

1. **Find the current git repo via `gitbud.get_repo()`**
   * If there is no non-bare git repo: it raises `SlackSendError`.
   * Repo root = `repo.working_tree_dir`.
   * Repo name = basename of the repo root directory.

2. **Map repo name → Slack channel**
   * Normalized channel name:
     * lowercase
     * non-alphanumeric chars → `-`
     * must start with a letter/number (otherwise prefixed with `proj-`)
     * max length 80 characters
   * If the channel exists → reuse it.
   * If not → create a new public channel with that name.

3. **Interpret file paths relative to the repo root**
   * For each entry in `files`:
     * If it is an **absolute path**, use it as-is.
     * Otherwise, treat it as **relative to the git repo root**, *not* the current working directory.
       * Example: if repo root is `/home/rajat/workspace/metric-graphs`,
         `"out/run42/loss.png"` → `/home/rajat/workspace/metric-graphs/out/run42/loss.png`.
   * If a file does not exist:
     * It is **skipped with a warning** (printed to stdout).
     * The rest of the files are still uploaded.

5. **Credentials**
   * Uses the environment variable `SLACK_BOT_TOKEN` as the Slack bot token.

---

## 4) How to think about errors (for LLMs / tools)

`send_results_to_slack` may raise `SlackSendError` in several situations:

1. **No git repo / gitbud cannot find a repo**
   * Meaning: `gitbud.get_repo()` returned `None` or failed, or the repo does not have
     a valid `working_tree_dir`.
   * Interpretation: This is a logic / environment bug.
   * **MUST NOT**:
     * try random `cd` commands,
     * guess alternative directories,
     * retry blindly.
   * **MUST**:
     * skip Slack notifications for this run,
     * surface the error message to the user if possible.

2. **`SLACK_BOT_TOKEN` is missing**
   * Meaning: the environment is not configured with a bot token.
   * **MUST NOT**:
     * invent or guess a token,
     * hard-code tokens.
   * **MUST**:
     * treat this as configuration,
     * skip Slack and report the problem.

3. **Slack API failure**
   * For example: insufficient scopes, channel naming policy, etc.
   * The error message includes `error`, `needed`, `provided`.
   * **MUST NOT**:
     * retry in a loop,
     * change channel names or scopes on your own.
   * **MUST**:
     * skip Slack for this run,
     * report the error text to the user.
---
