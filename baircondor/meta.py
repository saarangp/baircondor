"""Generate meta.json for reproducibility."""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def write_meta(
    run_dir: Path,
    repo_dir: Path,
    jobname: str,
    mode: str,
    command: list[str],
    resources: dict,
    conda: dict,
) -> Path:
    data = {
        "user": _get_user(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_dir": str(repo_dir),
        "run_dir": str(run_dir),
        "jobname": jobname,
        "mode": mode,
        "command": command,
        "resources": {k: v for k, v in resources.items() if v is not None},
        "conda": {k: v for k, v in conda.items() if v is not None},
        "git": _git_info(repo_dir),
    }
    path = run_dir / "meta.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


# ── helpers ──────────────────────────────────────────────────────────────────


def _get_user() -> str:
    import os

    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def _git_info(repo_dir: Path) -> dict:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return {"is_repo": True, "commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"is_repo": False}
