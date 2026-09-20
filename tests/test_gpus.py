"""Tests for the GPU audit (parsers on canned output, cap arithmetic)."""

from baircondor.gpus import (
    audit,
    format_audit,
    parse_nvidia_apps,
    parse_nvidia_gpus,
    parse_queue,
    parse_slots,
)

SLOTS = (
    "slot1@HOST.x\tPartitionable\tUnclaimed\tundefined\tGPU-f47e8274,GPU-c543f5d8\n"
    "slot1_1@HOST.x\tDynamic\tClaimed\talice@ad.x\tGPU-f47e8274\n"
)
GPUS = (
    "0, GPU-9e145a8a-b918-3014-0c97-7ed0c332476c, Tesla V100, 4, 32768\n"
    "1, GPU-83a62fe3-8cc0-ebd7-f1ca-1e0a8f57ba3f, Tesla V100, 23826, 32768\n"
    "2, GPU-f47e8274-1cde-0d1b-6327-ec892e2af7c7, Tesla V100, 900, 32768\n"
    "3, GPU-c543f5d8-1946-4735-7c39-d9f41d222b8d, Tesla V100, 4, 32768\n"
)
APPS = (
    "5116, GPU-83a62fe3-8cc0-ebd7-f1ca-1e0a8f57ba3f, 23822, /usr/local/bin/llama-server\n"
    "777, GPU-f47e8274-1cde-0d1b-6327-ec892e2af7c7, 900, python\n"
)
OWNERS = {"5116": "bob", "777": "alice"}
QUEUE = (
    '2\t1\tregexp("^HOST", Machine, "i")\n'  # idle, pinned here
    '1\t1\tregexp("^OTHER", Machine, "i")\n'  # idle, pinned elsewhere: not counted here
    "1\t2\tslot\n"  # running: already visible as a claim
)


def _audit(me="alice", need=1, queue=QUEUE):
    return audit(
        "HOST",
        parse_slots(SLOTS),
        parse_nvidia_gpus(GPUS),
        parse_nvidia_apps(APPS),
        OWNERS,
        parse_queue(queue),
        me,
        need,
    )


def test_parsers():
    slots = parse_slots(SLOTS)
    assert slots[0]["type"] == "Partitionable" and slots[0]["gpus"] == ["f47e8274", "c543f5d8"]
    assert slots[1]["user"] == "alice" and slots[1]["name"] == "slot1_1"
    assert [g["idx"] for g in parse_nvidia_gpus(GPUS)] == [0, 1, 2, 3]
    assert parse_nvidia_apps(APPS)[0] == {"pid": "5116", "uuid": "83a62fe3", "name": "llama-server"}
    q = parse_queue(QUEUE)
    assert [(j["gpus"], j["status"]) for j in q] == [(2, "idle"), (1, "idle"), (1, "running")]
    assert parse_slots("") == [] and parse_nvidia_gpus("garbage") == []


def test_audit_attribution_and_cap():
    a = _audit()
    rows = {r["idx"]: r for r in a["rows"]}
    assert rows[0]["state"] == "free" and rows[0]["via"] == "direct only"
    assert rows[1]["state"] == "busy" and rows[1]["owners"] == ["bob"]
    assert (
        rows[2]["state"] == "claimed"
        and rows[2]["owners"] == ["alice"]
        and "pid 777" in rows[2]["via"]
    )
    assert rows[3]["state"] == "free" and rows[3]["via"] == "condor"
    # alice: 1 condor claim + 2 GPUs idle in the queue for this host = 3; one more exceeds the cap
    assert a["you"] == {"direct": 0, "condor": 1, "idle_in_queue": 2, "total": 3}
    assert a["would_exceed"] is True and [f["idx"] for f in a["free"]] == [0, 3]
    text = format_audit(a)
    assert "condor manages idx 2,3" in text and "DO NOT LAUNCH" in text
    # bob's direct run counts for bob
    assert _audit(me="bob", queue="")["you"] == {
        "direct": 1,
        "condor": 0,
        "idle_in_queue": 0,
        "total": 1,
    }


def test_audit_remote_uses_condor_claims_only():
    slots = parse_slots(
        "slot1@R.x\tPartitionable\tUnclaimed\tundefined\tGPU-aaaaaaaa,GPU-bbbbbbbb,GPU-cccccccc\n"
        "slot1_1@R.x\tDynamic\tClaimed\tme@ad.x\tGPU-aaaaaaaa\n"
        "slot1_2@R.x\tDynamic\tClaimed\tother@ad.x\tGPU-bbbbbbbb\n"
    )
    a = audit("R", slots, [], [], {}, [], "me", need=2, local=False)
    assert [r["state"] for r in a["rows"]] == ["claimed", "claimed", "free"]
    assert a["you"]["condor"] == 1 and a["would_exceed"] is False
    assert "ok to take 2 (would be 3 of 3)" in format_audit(a)
