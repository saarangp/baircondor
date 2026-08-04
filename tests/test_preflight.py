"""Tests for the preflight command and the unshared-path guard."""

from pathlib import Path
from types import SimpleNamespace

from baircondor.preflight import parse_report, run_preflight
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


def test_parse_report_full():
    r = parse_report(SAMPLE_REPORT)
    assert r["host"] == "redlradadm35840.example.edu"
    assert r["conda_base"] == "/home/spanchavati/anaconda3"
    assert r["envs"] == ["base", "eeg2025", "eegfm"]  # deduped
    assert r["repo_exists"] is True
    assert r["git_commit"] == "abc1234"
    assert r["git_branch"] == "main"
    assert r["git_dirty"] is False


def test_parse_report_no_conda_no_repo():
    text = "host: h\nconda_base: none\nenvs_begin\nenvs_end\nrepo: /x\nrepo_exists: no\n"
    r = parse_report(text)
    assert r["conda_base"] is None
    assert r["envs"] == []
    assert r["repo_exists"] is False
    assert r["git_commit"] is None


def test_parse_report_dirty_repo_not_git():
    text = "repo: /x\nrepo_exists: yes\n"
    r = parse_report(text)
    assert r["repo_exists"] is True
    assert r["git_commit"] is None  # path exists but isn't a git repo


# ── run_preflight --dry-run generates the job files ───────────────────────────


def test_preflight_dry_run_generates_job(tmp_path, monkeypatch):
    monkeypatch.setattr("baircondor.preflight._get_submit_host", lambda: "redlradadm23589")
    args = SimpleNamespace(
        config=str(tmp_path / "no-config.yaml"),
        machine="REDLRADADM35840",
        scratch=str(tmp_path / "scratch"),
        runs_subdir=None,
        timeout=300,
        dry_run=True,
    )
    run_preflight(args)

    preflight_dirs = list((tmp_path / "scratch").glob("condor-runs/*/.preflight/*"))
    assert len(preflight_dirs) == 1
    run_dir = preflight_dirs[0]

    job_sub = (run_dir / "job.sub").read_text()
    assert 'requirements = regexp("^REDLRADADM35840", Machine, "i")' in job_sub
    assert "request_cpus = 1" in job_sub
    assert "request_gpus" not in job_sub
    assert f"initialdir = {run_dir}" in job_sub
    assert str(Path.cwd()) in job_sub  # repo path passed as the script argument

    script = (run_dir / "preflight.sh").read_text()
    assert "conda info --base" in script
    assert "envs_begin" in script
    assert "git rev-parse --short HEAD" in script


# ── unshared-path guard ───────────────────────────────────────────────────────


def test_warns_on_machine_local_paths():
    warnings = unshared_path_warnings(
        "REDLRADADM35840",
        "redlradadm23589.example.edu",
        {"cwd": "/raid/spanchavati/eegfm", "--scratch": "/home/spanchavati/condor-scratch"},
    )
    assert len(warnings) == 2
    assert "machine-local" in warnings[0]
    assert "/raid/spanchavati/eegfm" in warnings[0]


def test_warns_on_home_dir_path():
    warnings = unshared_path_warnings(
        "REDLRADADM35840",
        "redlradadm23589",
        {"--scratch": str(Path.home() / "condor-scratch")},
    )
    assert len(warnings) == 1


def test_no_warning_for_shared_paths():
    warnings = unshared_path_warnings(
        "REDLRADADM35840",
        "redlradadm23589",
        {"cwd": "/REDLRADADM35839/home/spanchavati/eegfm"},
    )
    assert warnings == []


def test_no_warning_without_machine_or_same_host():
    paths = {"cwd": "/raid/spanchavati/eegfm"}
    assert unshared_path_warnings(None, "redlradadm23589", paths) == []
    assert unshared_path_warnings("REDLRADADM23589", "redlradadm23589.example.edu", paths) == []
