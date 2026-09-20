"""Block until a condor job leaves the queue, the hold-aware and blip-proof way.

Two lessons from earlier campaigns are baked in:
- a held job (JobStatus 5) never leaves the queue, so a watcher that only looks for
  queue exit sits forever; we stop and print HoldReason instead;
- the schedd occasionally answers condor_q with nothing for a moment, so one empty
  answer is not an exit; we need a condor_history record before believing it.
"""

from __future__ import annotations

import sys
import time

from .config import get_user
from .console import log
from .history import JOB_STATUS, condor_out

EXIT_UNKNOWN = 2
EXIT_HELD = 3
EXIT_REMOVED = 4
EXIT_TIMEOUT = 5

_TERMINAL = {3, 4, 5}


def run_wait(args) -> None:
    cluster = args.cluster
    if cluster == "last":
        from .history import HISTORY_FILE, get_entries

        entries = get_entries(n=20, user=get_user(), history_file=HISTORY_FILE)
        cluster = next((e["cluster_id"] for e in entries if e.get("cluster_id")), None)
        if not cluster:
            sys.exit("error: no recent batch submission with a cluster id in your history")
    sys.exit(wait_for_cluster(str(cluster), interval=args.interval, timeout=args.timeout))


def wait_for_cluster(cluster: str, interval: int = 30, timeout: int | None = None) -> int:
    """Return 0 on clean exit, the job's exit code, or one of the EXIT_* codes."""
    start = time.monotonic()
    empties = 0
    seen = None
    while True:
        info = query_queue(cluster)
        if info is None:
            empties += 1
            info = query_history(cluster)
        else:
            empties = 0
        if info and info["status"] in _TERMINAL:
            return _finish(info, cluster)
        if info and info["status"] != seen:
            log(f"cluster {cluster}: {JOB_STATUS.get(info['status'], info['status'])}")
            seen = info["status"]
        if empties >= 3:
            if seen is None:
                log(f"cluster {cluster}: not in the queue and not in condor_history", style="red")
                return EXIT_UNKNOWN
            log(
                f"cluster {cluster}: queue empty {empties}x with no history record yet; still waiting"
            )
        if timeout is not None and time.monotonic() - start > timeout:
            log(f"cluster {cluster}: timeout after {timeout}s", style="red")
            return EXIT_TIMEOUT
        time.sleep(interval)


def _finish(info: dict, cluster: str) -> int:
    status = info["status"]
    if status == 5:
        log(
            f"cluster {cluster}: HELD. HoldReason: {info.get('hold_reason') or '(none given)'}",
            style="red",
        )
        log(
            "not releasing or resubmitting; read the run dir's stderr.txt and ask the user",
            style="red",
        )
        return EXIT_HELD
    if status == 3:
        log(f"cluster {cluster}: removed", style="red")
        return EXIT_REMOVED
    code = info.get("exit_code")
    if code is None:
        log(f"cluster {cluster}: completed without an exit code (killed by a signal?)", style="red")
        return 1
    log(f"cluster {cluster}: completed, exit {code}", style="green" if code == 0 else "red")
    return min(code, 255)


def query_queue(cluster: str) -> dict | None:
    return parse_status(
        condor_out(["condor_q", cluster, "-af:t", "JobStatus", "ExitCode", "HoldReason"])
    )


def query_history(cluster: str) -> dict | None:
    return parse_status(
        condor_out(
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
    lines = text.strip().splitlines()
    if not lines:
        return None
    parts = lines[0].split("\t")
    if not parts[0].isdigit():
        return None
    exit_code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
    hold = parts[2] if len(parts) > 2 and parts[2] != "undefined" else None
    return {"status": int(parts[0]), "exit_code": exit_code, "hold_reason": hold}
