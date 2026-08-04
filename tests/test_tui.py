"""Tests for the interactive history browser."""

import pytest

from baircondor import tui
from baircondor.history import STATUS_COLORS
from baircondor.tui import (
    RunBrowserApp,
    RunDetailScreen,
    entry_command,
    entry_timestamp,
    read_tail,
    status_text,
)


def _entry(run_dir="/tmp/run", **overrides):
    entry = {
        "timestamp": "2026-08-04T12:34:56",
        "user": "me",
        "jobname": "myjob",
        "run_dir": str(run_dir),
        "cluster_id": "123",
        "gpus": 2,
        "command": ["python", "train.py"],
    }
    entry.update(overrides)
    return entry


# ── pure helpers ──────────────────────────────────────────────────────────────


def test_read_tail_missing_file(tmp_path):
    assert read_tail(tmp_path / "nope.txt") is None


def test_read_tail_small_file(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("a\nb\nc\n")
    assert read_tail(p) == "a\nb\nc"


def test_read_tail_caps_lines(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("\n".join(str(i) for i in range(1000)))
    tail = read_tail(p, max_lines=10)
    assert tail == "\n".join(str(i) for i in range(990, 1000))


def test_read_tail_drops_partial_line_when_byte_capped(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("first-line-gets-cut\n" + "\n".join(f"line{i}" for i in range(10)))
    tail = read_tail(p, max_bytes=60)
    assert "first-line-gets-cut" not in tail
    assert tail.endswith("line9")


def test_status_text_styles():
    assert str(status_text("running")) == "● running"
    assert status_text("running").style == STATUS_COLORS["running"]
    assert status_text("unknown-thing").style == "dim"


def test_entry_timestamp():
    assert entry_timestamp(_entry()) == "2026-08-04 12:34"


def test_entry_command_truncates():
    entry = _entry(command=["python"] + ["x" * 50] * 3)
    out = entry_command(entry, max_len=20)
    assert len(out) == 20
    assert out.endswith("...")


# ── app navigation (headless pilot) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_enter_opens_detail_and_esc_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "get_job_status", lambda cluster_id: "running")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stdout.txt").write_text("hello from stdout\n")

    app = RunBrowserApp([_entry(run_dir)])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        assert isinstance(app.screen, RunDetailScreen)

        log = app.screen.query_one("#log-stdout")
        assert any("hello from stdout" in line.text for line in log.lines)

        await pilot.press("escape")
        assert not isinstance(app.screen, RunDetailScreen)


@pytest.mark.asyncio
async def test_detail_missing_file_message(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "get_job_status", lambda cluster_id: "done")
    app = RunBrowserApp([_entry(tmp_path / "gone")])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        log = app.screen.query_one("#log-stdout")
        assert any("not readable" in line.text for line in log.lines)
