"""History JSONL log: append entries at submit time, read for history/last."""

from __future__ import annotations

import json
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path.home() / ".local" / "share" / "baircondor" / "history.jsonl"

STATUS_COLORS = {
    "idle": "yellow",
    "running": "green",
    "done": "dim green",
    "failed": "red",
    "held": "red",
    "removed": "dim red",
}

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
    try:
        lines = history_file.read_text().splitlines()
    except FileNotFoundError:
        return []
    entries: deque[dict] = deque(maxlen=n)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if user is None or entry.get("user") == user:
            entries.append(entry)
    return list(reversed(entries))


def get_last_dirs(
    n: int = 1,
    user: str | None = None,
    history_file: Path = HISTORY_FILE,
) -> list[Path]:
    return [Path(e["run_dir"]) for e in get_entries(n=n, user=user, history_file=history_file)]


def get_job_info(cluster_id: str | None, timeout: float = 3.0) -> tuple[str, str]:
    """(status, short execute-host name) for a cluster id; ("?", "") when unknown."""
    if cluster_id is None:
        return "?", ""
    try:
        result = subprocess.run(
            ["condor_q", str(cluster_id), "-af", "JobStatus", "RemoteHost"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        line = result.stdout.strip()
        if not line:
            result = subprocess.run(
                [
                    "condor_history",
                    str(cluster_id),
                    "-af",
                    "JobStatus",
                    "LastRemoteHost",
                    "-match",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            line = result.stdout.strip()
        return _parse_job_info(line)
    except (subprocess.TimeoutExpired, OSError):
        return "?", ""


def _parse_job_info(line: str) -> tuple[str, str]:
    parts = line.split()
    if not parts:
        return "?", ""
    status = _STATUS_MAP.get(parts[0], "?")
    host = ""
    if len(parts) > 1 and parts[1] != "undefined":
        # RemoteHost looks like slot1@hostname.domain — keep just the short hostname
        host = parts[1].split("@")[-1].split(".")[0]
    return status, host


def get_job_status(cluster_id: str | None, timeout: float = 3.0) -> str:
    return get_job_info(cluster_id, timeout)[0]
