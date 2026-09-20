"""Tests for the GPU audit (parsers on canned output, cap arithmetic)."""

from baircondor.gpus import (
    audit,
    format_audit,
    parse_nvidia_apps,
    parse_nvidia_gpus,
    parse_queue,
    parse_slots,
)

SLOTS_LOCAL = (
    "slot1@HOST.x\tPartitionable\tUnclaimed\tundefined\tGPU-f47e8274,GPU-c543f5d8\n"
    "slot1_1@HOST.x\tDynamic\tClaimed\talice@ad.x\tGPU-f47e8274\n"
)
GPUS_LOCAL = (
    "0, GPU-9e145a8a-b918-3014-0c97-7ed0c332476c, Tesla V100, 4, 32768\n"
    "1, GPU-83a62fe3-8cc0-ebd7-f1ca-1e0a8f57ba3f, Tesla V100, 23826, 32768\n"
    "2, GPU-f47e8274-1cde-0d1b-6327-ec892e2af7c7, Tesla V100, 900, 32768\n"
    "3, GPU-c543f5d8-1946-4735-7c39-d9f41d222b8d, Tesla V100, 4, 32768\n"
)
APPS_LOCAL = (
    "5116, GPU-83a62fe3-8cc0-ebd7-f1ca-1e0a8f57ba3f, 23822, /usr/local/bin/llama-server\n"
    "777, GPU-f47e8274-1cde-0d1b-6327-ec892e2af7c7, 900, python\n"
)
OWNERS = {"5116": "bob", "777": "alice"}
QUEUE = "100\t2\t1\tundefined\n101\t1\t2\tslot1_1@HOST.x\n"


def test_parsers():
    slots = parse_slots(SLOTS_LOCAL)
    assert slots[0]["type"] == "Partitionable" and slots[0]["gpus"] == ["f47e8274", "c543f5d8"]
    assert slots[1]["user"] == "alice" and slots[1]["name"] == "slot1_1"
    gpus = parse_nvidia_gpus(GPUS_LOCAL)
    assert [g["idx"] for g in gpus] == [0, 1, 2, 3] and gpus[2]["uuid"] == "f47e8274"
    apps = parse_nvidia_apps(APPS_LOCAL)
    assert apps[0]["name"] == "llama-server" and apps[1]["uuid"] == "f47e8274"
    q = parse_queue(QUEUE)
    assert q[0] == {"cluster": "100", "gpus": 2, "status": "idle", "host": ""}
    assert q[1]["status"] == "running" and q[1]["host"] == "HOST"
    assert parse_slots("") == [] and parse_nvidia_gpus("garbage") == []


def test_audit_local_attribution_and_cap():
    a = audit(
        "HOST",
        parse_slots(SLOTS_LOCAL),
        parse_nvidia_gpus(GPUS_LOCAL),
        parse_nvidia_apps(APPS_LOCAL),
        OWNERS,
        parse_queue(QUEUE),
        "alice",
        need=1,
    )
    rows = {r["idx"]: r for r in a["rows"]}
    assert rows[0]["state"] == "free" and rows[0]["via"] == "direct only"
    assert rows[1]["state"] == "busy" and rows[1]["owner"] == "bob"
    assert (
        rows[2]["state"] == "claimed"
        and rows[2]["owner"] == "alice"
        and "pid 777" in rows[2]["via"]
    )
    assert rows[3]["state"] == "free" and rows[3]["via"] == "condor"
    # alice: 1 condor claim + 2 GPUs idle in the queue = 3; one more would exceed the cap
    assert a["you"] == {"direct": 0, "condor": 1, "idle_in_queue": 2, "total": 3}
    assert a["would_exceed"] is True
    assert [f["idx"] for f in a["free"]] == [0, 3]
    a["local"] = True
    text = format_audit(a)
    assert "condor manages idx 2,3" in text
    assert "DO NOT LAUNCH" in text


def test_audit_counts_direct_runs_of_the_user():
    a = audit(
        "HOST",
        [],
        parse_nvidia_gpus(GPUS_LOCAL),
        parse_nvidia_apps(APPS_LOCAL),
        {"5116": "bob", "777": "me"},
        [],
        "me",
    )
    assert a["you"]["direct"] == 1 and a["you"]["total"] == 1
    assert a["would_exceed"] is False
    a["local"] = True
    assert "condor manages no GPUs here" in format_audit(a)


def test_audit_remote_uses_condor_claims_only():
    slots = parse_slots(
        "slot1@R.x\tPartitionable\tUnclaimed\tundefined\tGPU-aaaaaaaa,GPU-bbbbbbbb,GPU-cccccccc\n"
        "slot1_1@R.x\tDynamic\tClaimed\tme@ad.x\tGPU-aaaaaaaa\n"
        "slot1_2@R.x\tDynamic\tClaimed\tother@ad.x\tGPU-bbbbbbbb\n"
    )
    a = audit("R", slots, [], [], {}, [], "me", need=2)
    assert [r["state"] for r in a["rows"]] == ["claimed", "claimed", "free"]
    assert a["you"]["condor"] == 1 and a["would_exceed"] is False
    assert "ok to take 2 (would be 3 of 3)" in format_audit(a)
