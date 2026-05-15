"""Tests for history JSONL append and retrieval."""

import json
import subprocess
from pathlib import Path

import pytest

from baircondor.history import append_entry, get_entries, get_job_status, get_last_dirs


@pytest.fixture
def hfile(tmp_path):
    return tmp_path / "history.jsonl"


# ── append_entry ──────────────────────────────────────────────────────────────


def test_append_entry_creates_file(hfile):
    append_entry(Path("/tmp/run1"), "myjob", "42", 1, ["python", "train.py"], "alice", hfile)
    assert hfile.exists()


def test_append_entry_writes_valid_json(hfile):
    append_entry(Path("/tmp/run1"), "myjob", "42", 1, ["python", "train.py"], "alice", hfile)
    entry = json.loads(hfile.read_text().strip())
    assert entry["jobname"] == "myjob"
    assert entry["cluster_id"] == "42"
    assert entry["gpus"] == 1
    assert entry["run_dir"] == "/tmp/run1"
    assert entry["command"] == ["python", "train.py"]
    assert entry["user"] == "alice"
    assert "timestamp" in entry


def test_append_multiple_entries_one_per_line(hfile):
    for i in range(3):
        append_entry(Path(f"/tmp/run{i}"), "job", str(i), 1, ["echo"], "alice", hfile)
    lines = [l for l in hfile.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


def test_append_none_cluster_id(hfile):
    append_entry(Path("/tmp/run1"), "myjob", None, 1, ["echo"], "alice", hfile)
    entry = json.loads(hfile.read_text().strip())
    assert entry["cluster_id"] is None


# ── get_last_dirs ─────────────────────────────────────────────────────────────


def test_get_last_dirs_returns_most_recent_first(hfile):
    for i in range(5):
        append_entry(Path(f"/tmp/run{i}"), "job", str(i), 1, ["echo"], "alice", hfile)
    dirs = get_last_dirs(n=3, user="alice", history_file=hfile)
    assert dirs == [Path("/tmp/run4"), Path("/tmp/run3"), Path("/tmp/run2")]


def test_get_last_dirs_missing_file(tmp_path):
    dirs = get_last_dirs(n=3, user="alice", history_file=tmp_path / "nope.jsonl")
    assert dirs == []


def test_get_last_dirs_n_larger_than_entries(hfile):
    append_entry(Path("/tmp/run0"), "job", "0", 1, ["echo"], "alice", hfile)
    dirs = get_last_dirs(n=10, user="alice", history_file=hfile)
    assert len(dirs) == 1


# ── get_entries ───────────────────────────────────────────────────────────────


def test_get_entries_filters_by_user(hfile):
    append_entry(Path("/tmp/a"), "job", "1", 1, ["echo"], "alice", hfile)
    append_entry(Path("/tmp/b"), "job", "2", 1, ["echo"], "bob", hfile)
    append_entry(Path("/tmp/c"), "job", "3", 1, ["echo"], "alice", hfile)
    entries = get_entries(n=10, user="alice", history_file=hfile)
    assert len(entries) == 2
    assert all(e["user"] == "alice" for e in entries)


def test_get_entries_empty_file(hfile):
    hfile.write_text("")
    assert get_entries(n=5, user="alice", history_file=hfile) == []


def test_get_entries_respects_n(hfile):
    for i in range(10):
        append_entry(Path(f"/tmp/run{i}"), "job", str(i), 1, ["echo"], "alice", hfile)
    entries = get_entries(n=3, user="alice", history_file=hfile)
    assert len(entries) == 3
    assert entries[0]["cluster_id"] == "9"  # most recent


# ── get_job_status ────────────────────────────────────────────────────────────


def _make_run(stdout):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return run


def test_get_job_status_idle(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _make_run("1\n"))
    assert get_job_status("42") == "idle"


def test_get_job_status_running(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _make_run("2\n"))
    assert get_job_status("42") == "running"


def test_get_job_status_held(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _make_run("5\n"))
    assert get_job_status("42") == "held"


def test_get_job_status_falls_back_to_condor_history(monkeypatch):
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "condor_q":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="4\n", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert get_job_status("42") == "done"
    assert "condor_q" in calls
    assert "condor_history" in calls


def test_get_job_status_unknown_returns_question_mark(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _make_run(""))
    assert get_job_status("42") == "?"


def test_get_job_status_timeout_returns_question_mark(monkeypatch):
    def mock_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 3.0)

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert get_job_status("42") == "?"


def test_get_job_status_none_cluster_id():
    assert get_job_status(None) == "?"
