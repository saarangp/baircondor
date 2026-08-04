"""Tests for the preflight command and the unshared-path guard."""

import json
from pathlib import Path
from types import SimpleNamespace

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


# ── check_problems (submit --check gate) ──────────────────────────────────────

GOOD_REPORT = {
    "conda_base": "/home/u/anaconda3",
    "envs": ["base", "eeg2025"],
    "repo_exists": True,
    "git_commit": "abc1234",
}


def test_check_passes_when_everything_matches():
    assert check_problems(GOOD_REPORT, "eeg2025", "abc1234") == []


def test_check_flags_missing_env():
    problems = check_problems(GOOD_REPORT, "typo-env", "abc1234")
    assert len(problems) == 1
    assert "typo-env" in problems[0]
    assert "base, eeg2025" in problems[0]


def test_check_flags_commit_mismatch():
    problems = check_problems(GOOD_REPORT, "eeg2025", "fff9999")
    assert len(problems) == 1
    assert "abc1234" in problems[0]
    assert "fff9999" in problems[0]


def test_check_flags_missing_repo_and_conda():
    report = {"conda_base": None, "envs": [], "repo_exists": False, "git_commit": None}
    problems = check_problems(report, "eeg2025", "abc1234")
    assert len(problems) == 2  # missing cwd + no conda


def test_check_skips_path_style_env_and_no_env():
    assert check_problems(GOOD_REPORT, "/opt/envs/x", "abc1234") == []
    assert check_problems(GOOD_REPORT, None, "abc1234") == []


# ── cached_env_warning (soft, non-blocking hint) ──────────────────────────────


def _write_cache_file(tmp_path, monkeypatch, envs):
    monkeypatch.setattr(preflight_mod, "HISTORY_FILE", tmp_path / "history.jsonl")
    cache = tmp_path / "preflight-REDLRADADM35840.json"
    cache.write_text(json.dumps({"timestamp": "2026-08-04T10:00:00", "envs": envs}))


def test_cached_warning_when_env_absent(tmp_path, monkeypatch):
    _write_cache_file(tmp_path, monkeypatch, ["base", "eegfm"])
    warning = cached_env_warning("REDLRADADM35840", "eeg2025")
    assert "eeg2025" in warning
    assert "2026-08-04T10:00:00" in warning
    assert "submitting anyway" in warning


def test_no_cached_warning_when_env_present_or_no_cache(tmp_path, monkeypatch):
    _write_cache_file(tmp_path, monkeypatch, ["base", "eeg2025"])
    assert cached_env_warning("REDLRADADM35840", "eeg2025") is None
    assert cached_env_warning("NOSUCHMACHINE", "eeg2025") is None  # no cache file
    assert cached_env_warning("REDLRADADM35840", None) is None


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
