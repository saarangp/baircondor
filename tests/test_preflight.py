"""Tests for the preflight command, the --check gate, and the unshared-path guard."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import baircondor.preflight as preflight_mod
from baircondor.preflight import cached_env_warning, check_problems, parse_report, run_preflight
from baircondor.submit import unshared_path_warnings

SAMPLE_REPORT = """\
host: redlradadm35840.example.edu
conda_base: /home/spanchavati/anaconda3
envs_begin
base
eeg2025
eegfm
base
envs_end
repo: /REDLRADADM35839/home/spanchavati/eegfm
repo_exists: yes
git_commit: abc1234
git_branch: main
git_dirty: no
"""
GOOD_REPORT = {
    "conda_base": "/home/u/anaconda3",
    "envs": ["base", "eeg2025"],
    "repo_exists": True,
    "git_commit": "abc1234",
}


def test_parse_report():
    r = parse_report(SAMPLE_REPORT)
    assert r["conda_base"] == "/home/spanchavati/anaconda3"
    assert r["envs"] == ["base", "eeg2025", "eegfm"]  # deduped
    assert r["repo_exists"] is True and r["git_commit"] == "abc1234" and r["git_dirty"] is False
    r = parse_report("host: h\nconda_base: none\nenvs_begin\nenvs_end\nrepo: /x\nrepo_exists: no\n")
    assert r["conda_base"] is None and r["envs"] == [] and r["repo_exists"] is False
    assert parse_report("repo: /x\nrepo_exists: yes\n")["git_commit"] is None  # not a git repo


def _args(tmp_path, **kw):
    base = dict(
        config=str(tmp_path / "no-config.yaml"),
        machine="REDLRADADM35840",
        scratch=str(tmp_path / "scratch"),
    )
    base.update(runs_subdir=None, timeout=300, dry_run=True, conda_env=None)
    return SimpleNamespace(**{**base, **kw})


def test_preflight_dry_run_generates_job(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_mod, "_get_submit_host", lambda: "redlradadm23589")
    run_preflight(_args(tmp_path))
    (run_dir,) = (tmp_path / "scratch").glob("condor-runs/*/.preflight/*")
    job_sub = (run_dir / "job.sub").read_text()
    assert 'requirements = regexp("^REDLRADADM35840", Machine, "i")' in job_sub
    assert "request_cpus = 1" in job_sub and "request_gpus" not in job_sub
    assert f"initialdir = {run_dir}" in job_sub and str(Path.cwd()) in job_sub
    script = (run_dir / "preflight.sh").read_text()
    assert "conda info --base" in script and "git rev-parse --short HEAD" in script


def test_preflight_conda_env_exits_when_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(preflight_mod, "_get_submit_host", lambda: "redlradadm23589")
    monkeypatch.setattr(preflight_mod, "run_check_job", lambda *a, **k: GOOD_REPORT)
    monkeypatch.setattr(preflight_mod, "_local_git_commit", lambda _: "abc1234")
    run_preflight(_args(tmp_path, dry_run=False, conda_env="eeg2025"))
    assert "exists on REDLRADADM35840" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="not found .available: base, eeg2025"):
        run_preflight(_args(tmp_path, dry_run=False, conda_env="typo-env"))
    run_preflight(
        _args(tmp_path, dry_run=False, conda_env="/opt/envs/x")
    )  # path-style: not checked
    assert "not checked" in capsys.readouterr().err


def test_check_problems():
    assert check_problems(GOOD_REPORT, "eeg2025", "abc1234") == []
    (p,) = check_problems(GOOD_REPORT, "typo-env", "abc1234")
    assert "typo-env" in p and "base, eeg2025" in p
    (p,) = check_problems(GOOD_REPORT, "eeg2025", "fff9999")
    assert "abc1234" in p and "fff9999" in p
    bad = {"conda_base": None, "envs": [], "repo_exists": False, "git_commit": None}
    assert len(check_problems(bad, "eeg2025", "abc1234")) == 2  # missing cwd + no conda
    assert check_problems(GOOD_REPORT, "/opt/envs/x", "abc1234") == []
    assert check_problems(GOOD_REPORT, None, "abc1234") == []


def test_cached_env_warning(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(preflight_mod, "HISTORY_FILE", tmp_path / "history.jsonl")
    (tmp_path / "preflight-REDLRADADM35840.json").write_text(
        json.dumps({"timestamp": "2026-08-04T10:00:00", "envs": ["base", "eegfm"]})
    )
    warning = cached_env_warning("REDLRADADM35840", "eeg2025")
    assert (
        "eeg2025" in warning and "2026-08-04T10:00:00" in warning and "submitting anyway" in warning
    )
    assert cached_env_warning("REDLRADADM35840", "eegfm") is None
    assert cached_env_warning("NOSUCHMACHINE", "eeg2025") is None
    assert cached_env_warning("REDLRADADM35840", None) is None


def test_unshared_path_warnings():
    warnings = unshared_path_warnings(
        "REDLRADADM35840",
        "redlradadm23589.example.edu",
        {"cwd": "/raid/u/eegfm", "--scratch": str(Path.home() / "condor-scratch")},
    )
    assert len(warnings) == 2 and "machine-local" in warnings[0] and "/raid/u/eegfm" in warnings[0]
    shared = {"cwd": "/REDLRADADM35839/home/u/eegfm"}
    assert unshared_path_warnings("REDLRADADM35840", "redlradadm23589", shared) == []
    local = {"cwd": "/raid/u/eegfm"}
    assert unshared_path_warnings(None, "redlradadm23589", local) == []
    assert unshared_path_warnings("REDLRADADM23589", "redlradadm23589.example.edu", local) == []
