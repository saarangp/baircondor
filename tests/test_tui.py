"""Tests for the interactive history browser."""

import pytest

from baircondor import tui
from baircondor.history import STATUS_COLORS, _parse_job_info
from baircondor.tui import (
    RunBrowserApp,
    RunDetailScreen,
    clone_run,
    entry_command,
    entry_timestamp,
    format_meta,
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


def test_parse_job_info_extracts_short_host():
    assert _parse_job_info("2 slot1@redlradadm35840.some.domain") == ("running", "redlradadm35840")
    assert _parse_job_info("1 undefined") == ("idle", "")
    assert _parse_job_info("") == ("?", "")


_META = {
    "jobname": "train",
    "mode": "batch",
    "timestamp": "2026-08-04T12:00:00+00:00",
    "hostname": "redlradadm23589",
    "command": ["python", "train.py"],
    "repo_dir": "/REDLRADADM35839/home/u/eegfm",
    "resources": {"gpus": 2, "cpus": 8, "mem": "48G"},
    "conda": {"env": "eeg2025"},
    "git": {"is_repo": True, "commit": "abc1234def5678", "branch": "main", "dirty": True},
}


def test_format_meta_summarizes_run():
    out = format_meta(_META)
    assert "train  (batch)" in out
    assert "python train.py" in out
    assert "main @ abc1234d, dirty" in out
    assert "gpus=2  cpus=8  mem=48G" in out
    assert "env=eeg2025" in out


def _make_run_dir(tmp_path, name="20260804_120000_abc123", meta_overrides=None):
    import json

    run_dir = tmp_path / "runs" / "train" / name
    run_dir.mkdir(parents=True)
    meta = dict(_META, run_dir=str(run_dir), repo_dir=str(tmp_path))
    meta.update(meta_overrides or {})
    (run_dir / "meta.json").write_text(json.dumps(meta))
    (run_dir / "job.sub").write_text(
        f'output = {run_dir}/stdout.txt\narguments = "{run_dir}/run.sh -- python train.py"\n'
    )
    (run_dir / "run.sh").write_text(f"export BAIRCONDOR_RUN_DIR={run_dir}\n")
    return run_dir


def test_clone_run_rewrites_paths(tmp_path):
    run_dir = _make_run_dir(tmp_path, name="20260804_120000_abc123_smoke")
    new_dir, meta = clone_run(run_dir)

    assert new_dir.parent == run_dir.parent
    assert new_dir != run_dir
    assert new_dir.name.endswith("_smoke")  # tag preserved

    job_sub = (new_dir / "job.sub").read_text()
    assert str(new_dir) in job_sub
    assert str(run_dir) not in job_sub
    run_sh = (new_dir / "run.sh").read_text()
    assert f"BAIRCONDOR_RUN_DIR={new_dir}" in run_sh
    assert (new_dir / "run.sh").stat().st_mode & 0o111  # executable

    import json

    new_meta = json.loads((new_dir / "meta.json").read_text())
    assert new_meta["run_dir"] == str(new_dir)
    assert new_meta["command"] == ["python", "train.py"]
    assert meta["jobname"] == "train"


# ── app navigation (headless pilot) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_enter_opens_detail_and_esc_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "get_job_info", lambda cluster_id: ("running", "redlradadm35840"))
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
    monkeypatch.setattr(tui, "get_job_info", lambda cluster_id: ("done", ""))
    app = RunBrowserApp([_entry(tmp_path / "gone")])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        log = app.screen.query_one("#log-stdout")
        assert any("not readable" in line.text for line in log.lines)


@pytest.mark.asyncio
async def test_detail_info_tab_shows_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "get_job_info", lambda cluster_id: ("done", ""))
    run_dir = _make_run_dir(tmp_path)
    app = RunBrowserApp([_entry(run_dir)])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("4")
        await pilot.pause()
        log = app.screen.query_one("#log-info")
        text = "\n".join(line.text for line in log.lines)
        assert "train  (batch)" in text
        assert "env=eeg2025" in text


@pytest.mark.asyncio
async def test_list_kill_confirms_before_condor_rm(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(tui, "get_job_info", lambda cluster_id: ("running", ""))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="job removed", stderr="")

    monkeypatch.setattr(tui.subprocess, "run", fake_run)

    app = RunBrowserApp([_entry(tmp_path / "run")])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("x")
        assert isinstance(app.screen, tui.Confirm)

        await pilot.press("n")  # decline: nothing runs
        assert not isinstance(app.screen, tui.Confirm)
        assert calls == []

        await pilot.press("k")  # legacy alias still works
        await pilot.press("y")  # confirm: condor_rm runs
        await app.workers.wait_for_complete()
        assert ["condor_rm", "123"] in calls


@pytest.mark.asyncio
async def test_detail_kill_uses_shared_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "get_job_info", lambda cluster_id: ("running", ""))
    app = RunBrowserApp([_entry(tmp_path / "run")])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        assert isinstance(app.screen, RunDetailScreen)
        await pilot.press("x")
        assert isinstance(app.screen, tui.Confirm)
        await pilot.press("n")
        assert isinstance(app.screen, RunDetailScreen)


@pytest.mark.asyncio
async def test_list_resubmit_clones_and_submits(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(tui, "get_job_info", lambda cluster_id: ("done", ""))
    appended = []
    monkeypatch.setattr(tui, "append_entry", lambda *a, **kw: appended.append(a))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="1 job(s) submitted to cluster 99.", stderr="")

    monkeypatch.setattr(tui.subprocess, "run", fake_run)

    run_dir = _make_run_dir(tmp_path)
    app = RunBrowserApp([_entry(run_dir)])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("R")
        assert isinstance(app.screen, tui.Confirm)
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert any(c[0] == "condor_submit" for c in calls)
        assert len(appended) == 1
        assert len(app._entries) == 2  # new run inserted at the top
        assert app._entries[0]["cluster_id"] == "99"
        assert app._entries[0]["run_dir"] != str(run_dir)


@pytest.mark.asyncio
async def test_resubmit_blocked_while_running(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "get_job_info", lambda cluster_id: ("running", ""))
    run_dir = _make_run_dir(tmp_path)
    app = RunBrowserApp([_entry(run_dir)])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()  # let the status worker mark it running
        await pilot.press("R")
        assert not isinstance(app.screen, tui.Confirm)
