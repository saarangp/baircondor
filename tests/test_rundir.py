"""Tests for run directory naming and creation."""

from baircondor.submit import _make_run_dir


def test_run_dir_structure(tmp_path):
    run_dir = _make_run_dir(str(tmp_path), "condor-runs", "myjob", None, None)
    # path: tmp_path/condor-runs/<user>/myjob/<timestamp>_<shortid>
    parts = run_dir.relative_to(tmp_path).parts
    assert parts[0] == "condor-runs"
    assert parts[2] == "myjob"
    assert len(parts) == 4


def test_run_dir_timestamp_format(tmp_path):
    run_dir = _make_run_dir(str(tmp_path), "condor-runs", "myjob", None, None)
    dirname = run_dir.name
    ts, shortid = dirname.rsplit("_", 1)
    # timestamp portion: YYYYMMDD_HHMMSS (two underscore-separated parts when split on first _)
    assert len(dirname) > 10
    assert len(shortid) == 6


def test_run_dir_with_project(tmp_path):
    run_dir = _make_run_dir(str(tmp_path), "condor-runs", "myjob", "myproject", None)
    parts = run_dir.relative_to(tmp_path).parts
    assert "myproject" in parts
    assert parts.index("myproject") < parts.index("myjob")


def test_run_dir_scratch_auto_created(tmp_path):
    new_scratch = tmp_path / "brand-new-scratch"
    assert not new_scratch.exists()
    run_dir = _make_run_dir(str(new_scratch), "condor-runs", "myjob", None, None)
    assert new_scratch.exists()
    assert run_dir.is_relative_to(new_scratch)


def test_run_dir_is_unique(tmp_path):
    dirs = {_make_run_dir(str(tmp_path), "condor-runs", "myjob", None, None) for _ in range(20)}
    assert len(dirs) == 20
