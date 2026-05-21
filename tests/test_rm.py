"""Tests for baircondor rm subcommand and history --logs flag."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

import baircondor.cli as cli_mod
import baircondor.history as hist_mod
from baircondor.history import append_entry


@pytest.fixture
def hfile(tmp_path):
    return tmp_path / "history.jsonl"


def _make_args_rm(cluster_ids=None, dry_run=False):
    ns = argparse.Namespace()
    ns.cluster_ids = cluster_ids or []
    ns.dry_run = dry_run
    return ns


# ── baircondor rm — explicit cluster IDs ─────────────────────────────────────


def test_rm_calls_condor_rm_with_explicit_ids(monkeypatch):
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job 42.0 marked for removal\n", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    args = _make_args_rm(cluster_ids=["42"])
    cli_mod._cmd_rm(args)

    assert calls == [["condor_rm", "42"]]


def test_rm_multiple_explicit_ids(monkeypatch):
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    args = _make_args_rm(cluster_ids=["10", "11", "12"])
    cli_mod._cmd_rm(args)

    assert calls == [["condor_rm", "10", "11", "12"]]


def test_rm_dry_run_does_not_call_condor_rm(monkeypatch):
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    args = _make_args_rm(cluster_ids=["42"], dry_run=True)
    cli_mod._cmd_rm(args)

    assert calls == []


# ── baircondor rm — fall back to history ─────────────────────────────────────


def test_rm_uses_last_history_entry_when_no_ids(monkeypatch, hfile):
    append_entry(Path("/tmp/run0"), "myjob", "99", 1, ["echo"], "testuser", hfile)

    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job 99.0 marked for removal\n", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(hist_mod, "HISTORY_FILE", hfile)
    monkeypatch.setattr(cli_mod, "get_user", lambda: "testuser")

    args = _make_args_rm()
    cli_mod._cmd_rm(args)

    assert calls == [["condor_rm", "99"]]


def test_rm_no_history_exits(monkeypatch, hfile):
    monkeypatch.setattr(hist_mod, "HISTORY_FILE", hfile)
    monkeypatch.setattr(cli_mod, "get_user", lambda: "testuser")

    args = _make_args_rm()
    with pytest.raises(SystemExit):
        cli_mod._cmd_rm(args)


def test_rm_no_cluster_id_in_entry_exits(monkeypatch, hfile):
    append_entry(Path("/tmp/run0"), "myjob", None, 1, ["echo"], "testuser", hfile)

    monkeypatch.setattr(hist_mod, "HISTORY_FILE", hfile)
    monkeypatch.setattr(cli_mod, "get_user", lambda: "testuser")

    args = _make_args_rm()
    with pytest.raises(SystemExit):
        cli_mod._cmd_rm(args)


def test_rm_dry_run_no_ids_uses_history(monkeypatch, hfile):
    append_entry(Path("/tmp/run0"), "myjob", "77", 1, ["echo"], "testuser", hfile)

    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(hist_mod, "HISTORY_FILE", hfile)
    monkeypatch.setattr(cli_mod, "get_user", lambda: "testuser")

    args = _make_args_rm(dry_run=True)
    cli_mod._cmd_rm(args)

    assert calls == []


# ── baircondor history --logs ─────────────────────────────────────────────────


def test_history_logs_flag_parsed():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="subcommand")
    cli_mod._add_history_parser(sub)

    args = parser.parse_args(["history", "--logs"])
    assert args.logs is True


def test_history_logs_flag_default_false():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="subcommand")
    cli_mod._add_history_parser(sub)

    args = parser.parse_args(["history"])
    assert args.logs is False


def test_history_logs_shows_stdout_stderr_paths(monkeypatch, hfile):
    append_entry(Path("/tmp/myrun"), "myjob", "7", 1, ["echo"], "testuser", hfile)

    printed = []
    monkeypatch.setattr(cli_mod._console, "print", lambda *a, **kw: printed.append(str(a[0]) if a else ""))
    monkeypatch.setattr(hist_mod, "HISTORY_FILE", hfile)
    monkeypatch.setattr(cli_mod, "get_user", lambda: "testuser")
    monkeypatch.setattr(hist_mod, "get_job_status", lambda *a, **kw: "running")

    args = argparse.Namespace(n=5, verbose=False, logs=True)
    cli_mod._cmd_history(args)

    assert any("stdout.txt" in line for line in printed)
    assert any("stderr.txt" in line for line in printed)
    assert any("/tmp/myrun/stdout.txt" in line for line in printed)
    assert any("/tmp/myrun/stderr.txt" in line for line in printed)


def test_history_no_logs_hides_paths(monkeypatch, hfile):
    append_entry(Path("/tmp/myrun"), "myjob", "7", 1, ["echo"], "testuser", hfile)

    printed = []
    monkeypatch.setattr(cli_mod._console, "print", lambda *a, **kw: printed.append(str(a[0]) if a else ""))
    monkeypatch.setattr(hist_mod, "HISTORY_FILE", hfile)
    monkeypatch.setattr(cli_mod, "get_user", lambda: "testuser")
    monkeypatch.setattr(hist_mod, "get_job_status", lambda *a, **kw: "running")

    args = argparse.Namespace(n=5, verbose=False, logs=False)
    cli_mod._cmd_history(args)

    assert not any("stdout.txt" in line for line in printed)
    assert not any("stderr.txt" in line for line in printed)
