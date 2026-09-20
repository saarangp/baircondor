"""GPU audit against the per-server cap: who holds each GPU, and how many you could take.

The lab rule is at most CAP GPUs per user per server, counting condor claims AND direct
processes. Neither source alone is enough: condor does not see direct runs (and on some
hosts manages only a subset of the GPUs), while nvidia-smi does not see a condor claim
whose job has not started work yet. This joins the two on GPU UUID.
"""

from __future__ import annotations

import json
import socket
import sys

from .config import get_user
from .history import JOB_STATUS, condor_out

CAP = 3


def run_gpus(args) -> None:
    me = get_user()
    local = socket.gethostname().split(".")[0]
    machine = (getattr(args, "machine", None) or local).split(".")[0]
    is_local = machine.lower() == local.lower()

    slots = parse_slots(condor_out(_slots_cmd(machine)))
    gpus = parse_nvidia_gpus(condor_out(_NVIDIA_GPUS_CMD)) if is_local else []
    apps = parse_nvidia_apps(condor_out(_NVIDIA_APPS_CMD)) if is_local else []
    owners = process_owners([a["pid"] for a in apps]) if apps else {}
    queue = parse_queue(
        condor_out(["condor_q", me, "-af:t", "RequestGpus", "JobStatus", "Requirements"])
    )

    if not slots and not gpus:
        raise ValueError(
            f"no condor slots match '{machine}' and no local nvidia-smi output; "
            "check the machine name with: condor_status -af Machine TotalGpus GPUs_DeviceName"
        )

    report = audit(
        machine, slots, gpus, apps, owners, queue, me, getattr(args, "need", 1), is_local
    )
    print(json.dumps(report, indent=2) if getattr(args, "json", False) else format_audit(report))
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


def process_owners(pids: list[str]) -> dict[str, str]:
    out = condor_out(["ps", "-o", "pid=,user=", "-p", ",".join(pids)])
    return dict(line.split() for line in out.splitlines() if len(line.split()) == 2)


# ── parsers (pure, tested with canned output) ─────────────────────────────────


def _prefix(uuid: str) -> str:
    """'GPU-f47e8274-1cde-...' and 'GPU-f47e8274' both become 'f47e8274'."""
    return uuid.strip().removeprefix("GPU-")[:8]


def parse_slots(text: str) -> list[dict]:
    slots = []
    for line in text.splitlines():
        parts = line.split("\t")
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
        gpus.append({"idx": int(parts[0]), "uuid": _prefix(parts[1]), "name": parts[2]})
    return gpus


def parse_nvidia_apps(text: str) -> list[dict]:
    apps = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        apps.append(
            {"pid": parts[0], "uuid": _prefix(parts[1]), "name": parts[3].rsplit("/", 1)[-1]}
        )
    return apps


def parse_queue(text: str) -> list[dict]:
    """Your jobs from `condor_q USER -af:t RequestGpus JobStatus Requirements`."""
    jobs = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        gpus = int(parts[0]) if parts[0].isdigit() else 0
        req = parts[2] if len(parts) > 2 else ""
        jobs.append(
            {"gpus": gpus, "status": JOB_STATUS.get(int(parts[1]), "?"), "requirements": req}
        )
    return jobs


def _targets(job: dict, machine: str) -> bool:
    """An idle job counts here if its requirements name this machine or pin no machine."""
    req = job["requirements"].lower()
    return machine.lower() in req or "machine" not in req


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
    local: bool = True,
) -> dict:
    managed: set[str] = set()
    claims: dict[str, tuple[str, str]] = {}
    for s in slots:
        if s["type"] != "Dynamic":
            managed.update(s["gpus"])
        if s["user"] and s["type"] != "Partitionable":
            for g in s["gpus"]:
                claims[g] = (s["user"], s["name"])

    if gpus:
        rows = [
            _row(
                g["idx"],
                g["uuid"],
                g["uuid"] in managed,
                claims.get(g["uuid"]),
                [a for a in apps if a["uuid"] == g["uuid"]],
                owners,
            )
            for g in gpus
        ]
    else:
        rows = [_row("-", uuid, True, claims.get(uuid), [], {}) for uuid in sorted(managed)]

    direct = sum(1 for r in rows if r["state"] == "busy" and me in r["owners"])
    condor = sum(1 for r in rows if r["state"] == "claimed" and me in r["owners"])
    idle = sum(j["gpus"] for j in queue if j["status"] == "idle" and _targets(j, machine))
    total = direct + condor + idle
    return {
        "machine": machine,
        "local": local,
        "cap": CAP,
        "rows": rows,
        "you": {"direct": direct, "condor": condor, "idle_in_queue": idle, "total": total},
        "free": [
            {"idx": r["idx"], "uuid": r["uuid"], "via": r["via"]}
            for r in rows
            if r["state"] == "free"
        ],
        "need": need,
        "would_exceed": total + need > CAP,
    }


def _row(idx, uuid, managed, claim, procs, owners) -> dict:
    if claim:
        user, slot = claim
        state, row_owners = "claimed", [user]
        via = f"condor {slot}" + (" pid " + ",".join(p["pid"] for p in procs) if procs else "")
    elif procs:
        state = "busy"
        row_owners = sorted({owners.get(p["pid"], "?") for p in procs})
        via = "direct " + ", ".join(f"pid {p['pid']} ({p['name']})" for p in procs)
    else:
        state, row_owners, via = "free", [], ("condor" if managed else "direct only")
    return {
        "idx": idx,
        "uuid": uuid,
        "state": state,
        "owners": row_owners,
        "via": via,
        "managed": managed,
    }


def format_audit(a: dict) -> str:
    rows = a["rows"]
    header = f"{a['machine']}  ({len(rows)} GPUs"
    if a["local"]:
        header += ", condor manages idx " + (
            ",".join(str(r["idx"]) for r in rows if r["managed"]) or "none"
        )
    lines = [header + ")", f"{'idx':<4} {'uuid':<9} {'state':<8} {'owner':<13} via"]
    for r in rows:
        lines.append(
            f"{str(r['idx']):<4} {r['uuid']:<9} {r['state']:<8} {','.join(r['owners']):<13} {r['via']}"
        )
    y = a["you"]
    lines.append(
        f"you hold {y['direct']} direct + {y['condor']} condor + {y['idle_in_queue']} idle-in-queue "
        f"= {y['total']} of {a['cap']}"
    )
    lines.append(
        "free for you: " + (", ".join(f"idx {f['idx']} ({f['via']})" for f in a["free"]) or "none")
    )
    after = y["total"] + a["need"]
    if a["would_exceed"]:
        lines.append(
            f"DO NOT LAUNCH: taking {a['need']} more would put you at {after} of {a['cap']}"
        )
    else:
        lines.append(f"ok to take {a['need']} (would be {after} of {a['cap']})")
    if a["local"] and not any(r["managed"] for r in rows):
        lines.append("note: condor manages no GPUs here; only direct runs are possible")
    return "\n".join(lines)
