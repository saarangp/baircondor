"""Tests for history JSONL append/retrieval and condor status lookup."""

import json
import subprocess
from pathlib import Path

import pytest

from baircondor.history import (
    append_entry,
    condor_out,
    get_entries,
    get_job_status,
    get_last_dirs,
    short_host,
)


@pytest.fixture
def hfile(tmp_path):
    return tmp_path / "history.jsonl"


def test_append_and_read_back(hfile):
    append_entry(Path("/tmp/run1"), "job1", "42", 1, ["python", "a.py"], "alice", hfile)
    append_entry(Path("/tmp/run2"), "job2", None, 0, ["echo"], "alice", hfile)
    append_entry(Path("/tmp/run3"), "job3", "44", 2, ["echo"], "bob", hfile)
    lines = [json.loads(line) for line in hfile.read_text().splitlines()]
    assert [e["jobname"] for e in lines] == ["job1", "job2", "job3"]
    assert lines[1]["cluster_id"] is None
    # newest first, filtered by user, capped by n
    assert [e["jobname"] for e in get_entries(n=5, user="alice", history_file=hfile)] == [
        "job2",
        "job1",
    ]
    assert get_last_dirs(n=1, user="alice", history_file=hfile) == [Path("/tmp/run2")]
    assert get_entries(n=1, user="alice", history_file=hfile)[0]["jobname"] == "job2"


def test_missing_file_gives_nothing(tmp_path):
    assert get_last_dirs(n=3, history_file=tmp_path / "nope.jsonl") == []


def _fake_run(answers):
    """condor_q then condor_history answers, in order."""
    it = iter(answers)

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=next(it), stderr="")

    return run


def test_get_job_status_maps_codes_and_falls_back_to_history(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(["2 slot1@h.x\n"]))
    assert get_job_status("1") == "running"
    monkeypatch.setattr(subprocess, "run", _fake_run(["5 undefined\n"]))
    assert get_job_status("1") == "held"
    monkeypatch.setattr(subprocess, "run", _fake_run(["", "4 slot1@h.x\n"]))
    assert get_job_status("1") == "done"
    monkeypatch.setattr(subprocess, "run", _fake_run(["", ""]))
    assert get_job_status("1") == "?"
    assert get_job_status(None) == "?"


def test_get_job_status_timeout(monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", boom)
    assert get_job_status("1") == "?"


def test_condor_out_and_short_host(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(["hello\n"]))
    assert condor_out(["x"]) == "hello\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert condor_out(["x"]) == ""
    assert short_host("slot1_2@REDLRADADM35839.ad.medctr.ucla.edu") == "REDLRADADM35839"
    assert short_host("undefined") == ""
