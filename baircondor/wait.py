"""Block until a condor job leaves the queue, the hold-aware and blip-proof way.

Two lessons from earlier campaigns are baked in:
- a held job (JobStatus 5) never leaves the queue, so a watcher that only looks for
  queue exit sits forever; we stop and print HoldReason instead;
- the schedd occasionally answers condor_q with nothing for a moment, so one empty
  answer is not an exit; we need a condor_history record before believing it.
"""

from __future__ import annotations

import subprocess
import sys
import time

from rich.console import Console
from rich.markup import escape

from .config import get_user

_console = Console(stderr=True, soft_wrap=True)
_PREFIX = f"[dim]{escape('[baircondor]')}[/dim]"

EXIT_UNKNOWN = 2
EXIT_HELD = 3
EXIT_REMOVED = 4
EXIT_TIMEOUT = 5

_NAMES = {
    1: "idle",
    2: "running",
    3: "removed",
    4: "completed",
    5: "held",
    6: "transferring",
    7: "suspended",
}


def run_wait(args) -> None:
    cluster = args.cluster
    if cluster == "last":
        from .history import HISTORY_FILE, get_entries

        entries = get_entries(n=1, user=get_user(), history_file=HISTORY_FILE)
        cluster = entries[0].get("cluster_id") if entries else None
        if not cluster:
            sys.exit("error: no recent submission with a cluster id in your history")
    code = wait_for_cluster(str(cluster), interval=args.interval, timeout=args.timeout)
    sys.exit(code)


def wait_for_cluster(cluster: str, interval: int = 30, timeout: int | None = None) -> int:
    """Return 0 on clean exit, the job's exit code, or one of the EXIT_* codes."""
    start = time.monotonic()
    empties = 0
    seen = None
    while True:
        q = query_queue(cluster)
        if q is not None:
            empties = 0
            outcome = _outcome(q, cluster)
            if outcome is not None:
                return outcome
            if q["status"] != seen:
                _log(f"cluster {cluster}: {_NAMES.get(q['status'], q['status'])}")
                seen = q["status"]
        else:
            empties += 1
            h = query_history(cluster)
            if h is not None:
                outcome = _outcome(h, cluster)
                return outcome if outcome is not None else 0
            if empties >= 3 and seen is None:
                _log(f"cluster {cluster}: not in the queue and not in condor_history", "red")
                return EXIT_UNKNOWN
            if empties >= 3:
                _log(
                    f"cluster {cluster}: queue empty {empties}x with no history record yet; still waiting"
                )
        if timeout is not None and time.monotonic() - start > timeout:
            _log(f"cluster {cluster}: timeout after {timeout}s", "red")
            return EXIT_TIMEOUT
        time.sleep(interval)


def _outcome(info: dict, cluster: str) -> int | None:
    status = info["status"]
    if status == 5:
        _log(
            f"cluster {cluster}: HELD. HoldReason: {info.get('hold_reason') or '(none given)'}",
            "red",
        )
        _log("not releasing or resubmitting; read the run dir's stderr.txt and ask the user", "red")
        return EXIT_HELD
    if status == 3:
        _log(f"cluster {cluster}: removed", "red")
        return EXIT_REMOVED
    if status == 4:
        code = info.get("exit_code")
        if code is None:
            _log(f"cluster {cluster}: completed without an exit code (killed by a signal?)", "red")
            return 1
        _log(f"cluster {cluster}: completed, exit {code}", "green" if code == 0 else "red")
        return min(code, 255)
    return None


def query_queue(cluster: str) -> dict | None:
    return parse_status(_run(["condor_q", cluster, "-af:t", "JobStatus", "ExitCode", "HoldReason"]))


def query_history(cluster: str) -> dict | None:
    return parse_status(
        _run(
            [
                "condor_history",
                cluster,
                "-af:t",
                "JobStatus",
                "ExitCode",
                "HoldReason",
                "-match",
                "1",
            ]
        )
    )


def parse_status(text: str) -> dict | None:
    """First job's (status, exit_code, hold_reason) from -af:t output; None when empty."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    if not line:
        return None
    parts = line.split("\t")
    status = int(parts[0]) if parts[0].isdigit() else None
    if status is None:
        return None
    exit_code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
    hold = parts[2] if len(parts) > 2 and parts[2] != "undefined" else None
    return {"status": status, "exit_code": exit_code, "hold_reason": hold}


def _run(cmd: list[str], timeout: float = 30) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _log(msg: str, style: str | None = None) -> None:
    text = escape(msg)
    _console.print(f"{_PREFIX} [{style}]{text}[/{style}]" if style else f"{_PREFIX} {text}")
