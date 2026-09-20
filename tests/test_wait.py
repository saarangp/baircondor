"""Tests for baircondor wait: hold-aware, blip-proof polling."""

import baircondor.wait as wait_mod
from baircondor.wait import (
    EXIT_HELD,
    EXIT_REMOVED,
    EXIT_TIMEOUT,
    EXIT_UNKNOWN,
    parse_status,
    wait_for_cluster,
)

IDLE = {"status": 1, "exit_code": None, "hold_reason": None}
RUNNING = {"status": 2, "exit_code": None, "hold_reason": None}


def _script(monkeypatch, queue, history):
    """Feed successive condor_q / condor_history answers; sleep is a no-op."""
    q_iter, h_iter = iter(queue), iter(history)
    monkeypatch.setattr(wait_mod, "query_queue", lambda c: next(q_iter))
    monkeypatch.setattr(wait_mod, "query_history", lambda c: next(h_iter))
    monkeypatch.setattr(wait_mod.time, "sleep", lambda s: None)


def test_parse_status():
    assert parse_status("2\tundefined\tundefined\n") == RUNNING
    assert parse_status("5\tundefined\tJob has gone over cgroup memory limit\n")[
        "hold_reason"
    ].startswith("Job has")
    assert parse_status("4\t7\tundefined\n")["exit_code"] == 7
    assert parse_status("") is None and parse_status("undefined\n") is None


def test_exit_codes(monkeypatch):
    # idle, running, one empty read (blip, history not written yet), then history shows completion
    _script(
        monkeypatch,
        [IDLE, RUNNING, None, None],
        [None, {"status": 4, "exit_code": 0, "hold_reason": None}],
    )
    assert wait_for_cluster("1", interval=0) == 0
    _script(monkeypatch, [None], [{"status": 4, "exit_code": 7, "hold_reason": None}])
    assert wait_for_cluster("1", interval=0) == 7
    _script(
        monkeypatch, [{"status": 5, "exit_code": None, "hold_reason": "cgroup memory limit"}], []
    )
    assert wait_for_cluster("1", interval=0) == EXIT_HELD
    _script(monkeypatch, [{"status": 3, "exit_code": None, "hold_reason": None}], [])
    assert wait_for_cluster("1", interval=0) == EXIT_REMOVED
    _script(monkeypatch, [None, None, None], [None, None, None])
    assert wait_for_cluster("1", interval=0) == EXIT_UNKNOWN


def test_timeout(monkeypatch):
    _script(monkeypatch, [RUNNING] * 5, [])
    clock = iter([0, 0, 10, 20, 100, 200])
    monkeypatch.setattr(wait_mod.time, "monotonic", lambda: next(clock))
    assert wait_for_cluster("1", interval=0, timeout=50) == EXIT_TIMEOUT
