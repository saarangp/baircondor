"""Tests for CLI-level behavior."""

import json
import sys
from types import SimpleNamespace

import pytest

from baircondor.cli import _cmd_history, _run_clean


def test_run_clean_converts_valueerror_to_exit():
    def boom(_args):
        raise ValueError("bad thing")

    with pytest.raises(SystemExit) as exc:
        _run_clean(boom, object())
    assert "bad thing" in str(exc.value)


def test_run_clean_passes_through_on_success():
    calls = []
    _run_clean(lambda args: calls.append(args), "sentinel")
    assert calls == ["sentinel"]


# ── history: plain vs TUI dispatch ────────────────────────────────────────────


@pytest.fixture
def history_env(tmp_path, monkeypatch):
    """Point history at a temp file with one entry for the current test user."""
    history_file = tmp_path / "history.jsonl"
    entry = {
        "timestamp": "2026-08-04T12:00:00",
        "user": "testuser",
        "jobname": "myjob",
        "run_dir": str(tmp_path / "run"),
        "cluster_id": "42",
        "gpus": 1,
        "command": ["echo", "hi"],
    }
    history_file.write_text(json.dumps(entry) + "\n")
    monkeypatch.setattr("baircondor.history.HISTORY_FILE", history_file)
    monkeypatch.setattr("baircondor.cli.get_user", lambda: "testuser")
    monkeypatch.setattr("baircondor.history.get_job_status", lambda cluster_id: "running")
    return entry


def _history_args(plain=False, n=3, verbose=False):
    return SimpleNamespace(plain=plain, n=n, verbose=verbose)


def test_history_plain_flag_prints_listing(history_env, capsys):
    _cmd_history(_history_args(plain=True))
    err = capsys.readouterr().err
    assert "myjob" in err
    assert "running" in err


def test_history_non_tty_falls_back_to_plain(history_env, capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    _cmd_history(_history_args(plain=False))
    assert "myjob" in capsys.readouterr().err


def test_history_tty_launches_tui(history_env, monkeypatch):
    launched = {}

    class FakeApp:
        def __init__(self, entries):
            launched["entries"] = entries

        def run(self):
            launched["ran"] = True

    monkeypatch.setattr("baircondor.tui.RunBrowserApp", FakeApp)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    _cmd_history(_history_args(plain=False))
    assert launched["ran"]
    assert launched["entries"][0]["jobname"] == "myjob"
