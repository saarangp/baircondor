"""Tests for CLI-level behavior: error wrapping, history dispatch, profiles listing."""

import json
import sys
from types import SimpleNamespace

import pytest

from baircondor.cli import _cmd_history, _cmd_profiles, _run_clean


def test_run_clean_converts_valueerror_to_exit():
    with pytest.raises(SystemExit, match="bad thing"):
        _run_clean(lambda _: (_ for _ in ()).throw(ValueError("bad thing")), None)
    calls = []
    _run_clean(calls.append, "sentinel")
    assert calls == ["sentinel"]


@pytest.fixture
def history_env(tmp_path, monkeypatch):
    entry = {"timestamp": "2026-08-04T12:00:00", "user": "testuser", "jobname": "myjob"}
    entry.update(run_dir=str(tmp_path / "run"), cluster_id="42", gpus=1, command=["echo", "hi"])
    history_file = tmp_path / "history.jsonl"
    history_file.write_text(
        "\n".join(json.dumps(dict(entry, jobname=f"job{i}")) for i in range(8)) + "\n"
    )
    monkeypatch.setattr("baircondor.history.HISTORY_FILE", history_file)
    monkeypatch.setattr("baircondor.cli.get_user", lambda: "testuser")
    monkeypatch.setattr("baircondor.history.get_job_status", lambda cluster_id: "running")


def _args(plain=False, n=None):
    return SimpleNamespace(plain=plain, n=n, verbose=False)


def test_history_plain_when_flagged_or_piped(history_env, capsys, monkeypatch):
    _cmd_history(_args(plain=True))
    err = capsys.readouterr().err
    assert "job7" in err and "running" in err
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    _cmd_history(_args())
    assert "job7" in capsys.readouterr().err


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
    _cmd_history(_args())
    assert launched["ran"] and len(launched["entries"]) == 5  # default 5
    _cmd_history(_args(n=2))
    assert len(launched["entries"]) == 2


def test_profiles_lists_repo_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    with pytest.raises(SystemExit, match="No .baircondor.yaml"):
        _cmd_profiles(SimpleNamespace(config=str(tmp_path / "none.yaml")))
    (tmp_path / ".baircondor.yaml").write_text("profiles:\n  eval:\n    gpus: 1\n    mem: 64G\n")
    _cmd_profiles(SimpleNamespace(config=str(tmp_path / "none.yaml")))
    out = capsys.readouterr().out
    assert out.startswith(f"# {tmp_path / '.baircondor.yaml'}") and "mem: 64G" in out
