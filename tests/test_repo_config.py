"""Tests for the committed .baircondor.yaml layer and named profiles."""

import importlib
from types import SimpleNamespace

import pytest

from baircondor.api import submit
from baircondor.config import (
    PROFILE_KEYS,
    fill_unset,
    find_repo_config,
    load_config,
    resolve_profile,
)

submit_mod = importlib.import_module("baircondor.submit")

REPO_YAML = """\
defaults:
  scratch: /SHARED/home/${USER}/condor-scratch
profiles:
  eval:
    machine: REDLRADADM35839
    gpus: 1
    cpus: 8
    mem: 64G
    conda_env: laya_env
    conda_base: ~/anaconda3
    sub_lines: ['require_gpus = DeviceUuid != "bad"']
  pretrain:
    machine: REDLRADADM35840
    gpus: 2
    mem: 256G
    conda_env: eeg2025
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".baircondor.yaml").write_text(REPO_YAML)
    (tmp_path / "benchmarking").mkdir()
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def test_find_repo_config_only_inside_a_git_repo(repo, tmp_path):
    assert find_repo_config(repo / "benchmarking") == repo / ".baircondor.yaml"
    # a .baircondor.yaml above the git root does not apply: the search stops at .git
    (tmp_path / ".baircondor.yaml").write_text("profiles: {}\n")
    inner = tmp_path / "other"
    (inner / ".git").mkdir(parents=True)
    assert find_repo_config(inner) is None


def test_load_config_layers_repo_over_personal_and_expands(repo, tmp_path):
    personal = tmp_path / "personal.yaml"
    personal.write_text("defaults:\n  scratch: /raid/alice\n  mem_gpu: 48G\n")
    cfg = load_config(str(personal), repo_dir=repo / "benchmarking")
    assert (
        cfg["defaults"]["scratch"] == "/SHARED/home/alice/condor-scratch"
    )  # repo wins, ${USER} expanded
    assert cfg["defaults"]["mem_gpu"] == "48G"  # personal value kept where the repo is silent
    assert cfg["repo_config"] == str(repo / ".baircondor.yaml")
    assert cfg["profiles"]["eval"]["conda_base"] == str(
        tmp_path / "home" / "anaconda3"
    )  # ~ expanded


def test_resolve_profile_and_fill_unset(repo, tmp_path):
    cfg = load_config(None, repo_dir=repo)
    assert resolve_profile(cfg, None) == {}
    profile = resolve_profile(cfg, "eval")
    args = SimpleNamespace(**{**{k: None for k in PROFILE_KEYS}, "mem": "8G"})
    fill_unset(args, profile)
    assert args.machine == "REDLRADADM35839" and args.gpus == 1
    assert args.mem == "8G"  # already set wins

    with pytest.raises(ValueError, match="available: eval, pretrain"):
        resolve_profile(cfg, "nope")
    (repo / ".baircondor.yaml").write_text("profiles:\n  bad:\n    gpu: 1\n  scalar: laya_env\n")
    cfg = load_config(None, repo_dir=repo)
    with pytest.raises(ValueError, match="unknown keys: gpu"):
        resolve_profile(cfg, "bad")
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_profile(cfg, "scalar")
    (tmp_path / "elsewhere" / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="none was found"):
        resolve_profile(load_config(None, repo_dir=tmp_path / "elsewhere"), "eval")


def test_submit_profile_lands_in_files_once(repo, monkeypatch, tmp_path):
    monkeypatch.chdir(repo)
    run_dir = submit(
        ["echo", "hi"], profile="eval", scratch=str(tmp_path / "scratch"), dry_run=True
    )
    job_sub = (run_dir / "job.sub").read_text()
    assert 'requirements = regexp("^REDLRADADM35839", Machine, "i")' in job_sub
    assert "request_memory = 64G" in job_sub and "request_gpus = 1" in job_sub
    assert job_sub.count('require_gpus = DeviceUuid != "bad"') == 1
    assert 'ENV_NAME="laya_env"' in (run_dir / "run.sh").read_text()
    assert '"profile": "eval"' in (run_dir / "meta.json").read_text()
    # the profile's gpus applies on the API path too, and an explicit sub_line is kept
    run_dir = submit(
        ["echo"],
        profile="pretrain",
        scratch=str(tmp_path / "s2"),
        dry_run=True,
        sub_lines=["+X = 1"],
    )
    job_sub = (run_dir / "job.sub").read_text()
    assert "request_gpus = 2" in job_sub and "+X = 1" in job_sub


def test_after_refuses_when_earlier_job_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submit_mod.subprocess, "check_output", lambda cmd, text: "host\n")
    monkeypatch.setattr("baircondor.wait.wait_for_cluster", lambda cluster, **k: 3)
    with pytest.raises(SystemExit, match="--after 123: that job did not finish cleanly"):
        submit(
            ["echo"],
            gpus=0,
            scratch=str(tmp_path / "s"),
            after="123",
            config=str(tmp_path / "none.yaml"),
        )
    assert next((tmp_path / "s").rglob("job.sub"))  # files were generated before the wait
