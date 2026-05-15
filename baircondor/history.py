"""History JSONL log: append entries at submit time, read for history/last."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path.home() / ".local" / "share" / "baircondor" / "history.jsonl"

_STATUS_MAP = {
    "1": "idle",
    "2": "running",
    "3": "removed",
    "4": "done",
    "5": "held",
    "6": "running",  # transferring output
    "7": "held",  # suspended
}


def append_entry(
    run_dir: Path,
    jobname: str,
    cluster_id: str | None,
    gpus: int,
    command: list[str],
    user: str,
    history_file: Path = HISTORY_FILE,
) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": user,
        "jobname": jobname,
        "run_dir": str(run_dir),
        "cluster_id": cluster_id,
        "gpus": gpus,
        "command": command,
    }
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_entries(
    n: int = 3,
    user: str | None = None,
    history_file: Path = HISTORY_FILE,
) -> list[dict]:
    if not history_file.exists():
        return []
    entries = []
    for line in history_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if user is None or entry.get("user") == user:
            entries.append(entry)
    return list(reversed(entries))[:n]


def get_last_dirs(
    n: int = 1,
    user: str | None = None,
    history_file: Path = HISTORY_FILE,
) -> list[Path]:
    return [Path(e["run_dir"]) for e in get_entries(n=n, user=user, history_file=history_file)]


def get_job_status(cluster_id: str | None, timeout: float = 3.0) -> str:
    if cluster_id is None:
        return "?"
    try:
        result = subprocess.run(
            ["condor_q", str(cluster_id), "-format", "%d\n", "JobStatus"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        code = result.stdout.strip()
        if code:
            return _STATUS_MAP.get(code, "?")

        result = subprocess.run(
            ["condor_history", str(cluster_id), "-format", "%d\n", "JobStatus", "-match", "1"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        code = result.stdout.strip()
        return _STATUS_MAP.get(code, "?")
    except subprocess.TimeoutExpired:
        return "?"
