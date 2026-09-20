"""GPU audit against the per-server cap: who holds each GPU, and how many you could take.

The lab rule is at most CAP GPUs per user per server, counting condor claims AND direct
processes. Neither source alone is enough: condor does not see direct runs (and on some
hosts manages only a subset of the GPUs), while nvidia-smi does not see a condor claim
whose job has not started work yet. This joins the two on GPU UUID.
"""

from __future__ import annotations

import json
import subprocess
import sys

from .config import get_user
from .submit import _get_submit_host

CAP = 3

_STATUS_NAMES = {"1": "idle", "2": "running", "5": "held", "6": "transferring", "7": "suspended"}


def run_gpus(args) -> None:
    me = get_user()
    local_host = _get_submit_host()
    machine = (getattr(args, "machine", None) or local_host).split(".")[0]
    is_local = machine.lower() == local_host.split(".")[0].lower()

    slots = parse_slots(_run(_slots_cmd(machine)))
    gpus = parse_nvidia_gpus(_run(_NVIDIA_GPUS_CMD)) if is_local else []
    apps = parse_nvidia_apps(_run(_NVIDIA_APPS_CMD)) if is_local else []
    owners = process_owners([a["pid"] for a in apps]) if apps else {}
    queue = parse_queue(
        _run(["condor_q", me, "-af:t", "ClusterId", "RequestGpus", "JobStatus", "RemoteHost"])
    )

    if not slots and not gpus:
        raise ValueError(
            f"no condor slots match '{machine}' and no local nvidia-smi output; "
            "check the machine name with: condor_status -af Machine TotalGpus GPUs_DeviceName"
        )

    report = audit(machine, slots, gpus, apps, owners, queue, me, need=getattr(args, "need", 1))
    report["local"] = is_local
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print(format_audit(report))
    if report["would_exceed"]:
        sys.exit(1)


# ── data collection ───────────────────────────────────────────────────────────

_NVIDIA_GPUS_CMD = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]
_NVIDIA_APPS_CMD = [
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
    "--format=csv,noheader,nounits",
]


def _slots_cmd(machine: str) -> list[str]:
    return [
        "condor_status",
        "-af:t",
        "Name",
        "SlotType",
        "State",
        "RemoteUser",
        "AssignedGPUs",
        "-constraint",
        f'regexp("^{machine}", Machine, "i")',
    ]


def _run(cmd: list[str], timeout: float = 20) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def process_owners(pids: list[str]) -> dict[str, str]:
    out = _run(["ps", "-o", "pid=,user=", "-p", ",".join(pids)])
    owners = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            owners[parts[0]] = parts[1]
    return owners


# ── parsers (pure, tested with canned output) ─────────────────────────────────


def _prefix(uuid: str) -> str:
    """'GPU-f47e8274-1cde-...' and 'GPU-f47e8274' both become 'f47e8274'."""
    return uuid.strip().removeprefix("GPU-")[:8]


def parse_slots(text: str) -> list[dict]:
    slots = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        name, slot_type, state, user, gpus = parts[:5]
        slots.append(
            {
                "name": name.split("@")[0],
                "type": slot_type,
                "state": state,
                "user": None if user in ("undefined", "") else user.split("@")[0],
                "gpus": (
                    []
                    if gpus in ("undefined", "")
                    else [_prefix(g) for g in gpus.split(",") if g.strip()]
                ),
            }
        )
    return slots


def parse_nvidia_gpus(text: str) -> list[dict]:
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "idx": int(parts[0]),
                "uuid": _prefix(parts[1]),
                "name": parts[2],
                "mem_used": int(parts[3]),
                "mem_total": int(parts[4]),
            }
        )
    return gpus


def parse_nvidia_apps(text: str) -> list[dict]:
    apps = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        apps.append(
            {
                "pid": parts[0],
                "uuid": _prefix(parts[1]),
                "mem": parts[2],
                "name": parts[3].rsplit("/", 1)[-1],
            }
        )
    return apps


def parse_queue(text: str) -> list[dict]:
    jobs = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        host = parts[3] if len(parts) > 3 and parts[3] != "undefined" else ""
        jobs.append(
            {
                "cluster": parts[0],
                "gpus": int(parts[1]) if parts[1].isdigit() else 0,
                "status": _STATUS_NAMES.get(parts[2], parts[2]),
                "host": host.split("@")[-1].split(".")[0],
            }
        )
    return jobs


# ── the audit itself ──────────────────────────────────────────────────────────


def audit(
    machine: str,
    slots: list[dict],
    gpus: list[dict],
    apps: list[dict],
    owners: dict[str, str],
    queue: list[dict],
    me: str,
    need: int = 1,
) -> dict:
    managed: set[str] = set()
    claims: dict[str, tuple[str, str]] = {}
    for s in slots:
        if s["type"] != "Dynamic":
            managed.update(s["gpus"])
        if s["user"] and s["type"] != "Partitionable":
            for g in s["gpus"]:
                claims[g] = (s["user"], s["name"])

    rows = []
    if gpus:
        for g in gpus:
            procs = [a for a in apps if a["uuid"] == g["uuid"]]
            rows.append(
                _row(
                    g["idx"], g["uuid"], g["uuid"] in managed, claims.get(g["uuid"]), procs, owners
                )
            )
    else:
        for uuid in sorted(managed):
            rows.append(_row("-", uuid, True, claims.get(uuid), [], {}))

    direct = sum(1 for r in rows if r["state"] == "busy" and me in r["owners"])
    condor = sum(1 for r in rows if r["state"] == "claimed" and r["owner"] == me)
    idle = sum(j["gpus"] for j in queue if j["status"] == "idle")
    total = direct + condor + idle
    free = [r for r in rows if r["state"] == "free"]
    return {
        "machine": machine,
        "cap": CAP,
        "managed": sorted(managed),
        "rows": rows,
        "you": {"direct": direct, "condor": condor, "idle_in_queue": idle, "total": total},
        "free": [{"idx": r["idx"], "uuid": r["uuid"], "via": r["via"]} for r in free],
        "need": need,
        "would_exceed": total + need > CAP,
    }


def _row(idx, uuid, managed, claim, procs, owners) -> dict:
    proc_owners = sorted({owners.get(p["pid"], "?") for p in procs})
    if claim:
        user, slot = claim
        via = f"condor {slot}"
        if procs:
            via += " pid " + ",".join(p["pid"] for p in procs)
        return {
            "idx": idx,
            "uuid": uuid,
            "state": "claimed",
            "owner": user,
            "owners": proc_owners,
            "via": via,
            "managed": managed,
        }
    if procs:
        via = "direct " + ", ".join(f"pid {p['pid']} ({p['name']})" for p in procs)
        return {
            "idx": idx,
            "uuid": uuid,
            "state": "busy",
            "owner": ",".join(proc_owners),
            "owners": proc_owners,
            "via": via,
            "managed": managed,
        }
    via = "condor" if managed else "direct only"
    return {
        "idx": idx,
        "uuid": uuid,
        "state": "free",
        "owner": "",
        "owners": [],
        "via": via,
        "managed": managed,
    }


def format_audit(a: dict) -> str:
    managed = a["managed"]
    header = f"{a['machine']}  ({len(a['rows'])} GPUs"
    if a.get("local"):
        managed_idx = [str(r["idx"]) for r in a["rows"] if r["managed"]]
        header += f", condor manages idx {','.join(managed_idx) or 'none'}"
    header += ")"
    lines = [header, f"{'idx':<4} {'uuid':<9} {'state':<8} {'owner':<13} via"]
    for r in a["rows"]:
        lines.append(
            f"{str(r['idx']):<4} {r['uuid']:<9} {r['state']:<8} {r['owner']:<13} {r['via']}"
        )
    y = a["you"]
    lines.append(
        f"you hold {y['direct']} direct + {y['condor']} condor + {y['idle_in_queue']} idle-in-queue "
        f"= {y['total']} of {a['cap']}"
    )
    if a["free"]:
        lines.append(
            "free for you: " + ", ".join(f"idx {f['idx']} ({f['via']})" for f in a["free"])
        )
    else:
        lines.append("free for you: none")
    if a["would_exceed"]:
        lines.append(
            f"DO NOT LAUNCH: taking {a['need']} more would put you at {y['total'] + a['need']} of {a['cap']}"
        )
    else:
        lines.append(f"ok to take {a['need']} (would be {y['total'] + a['need']} of {a['cap']})")
    if not managed and a.get("local"):
        lines.append("note: condor manages no GPUs here; only direct runs are possible")
    return "\n".join(lines)
