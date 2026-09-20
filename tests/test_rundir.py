"""Tests for run dir naming and creation."""

import re

from baircondor.submit import _make_run_dir


def test_run_dir_layout_and_uniqueness(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "alice")
    scratch = tmp_path / "scratch"  # created on demand
    a = _make_run_dir(str(scratch), "condor-runs", "myjob", None, None)
    b = _make_run_dir(str(scratch), "condor-runs", "myjob", "proj", "smoke")
    assert a.parent == scratch / "condor-runs" / "alice" / "myjob"
    assert re.fullmatch(r"\d{8}_\d{6}_[a-z0-9]{6}", a.name)
    assert b.parent == scratch / "condor-runs" / "alice" / "proj" / "myjob"
    assert b.name.endswith("_smoke")
    assert a != _make_run_dir(str(scratch), "condor-runs", "myjob", None, None)
