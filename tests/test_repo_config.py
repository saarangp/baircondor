"""Tests for the committed .baircondor.yaml layer and named profiles."""

from types import SimpleNamespace

import pytest

from baircondor.api import CondorConfig
from baircondor.config import apply_profile, find_repo_config, load_config

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
    sub = tmp_path / "benchmarking"
    sub.mkdir()
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _args(**kw):
    base = {k: None for k in ("gpus", "cpus", "mem", "disk", "jobname", "scratch", "runs_subdir")}
    base.update({k: None for k in ("project", "tag", "conda_env", "conda_base", "machine")})
    base.update({"pin_submit_host": None, "sub_lines": None})
    base.update(kw)
    return SimpleNamespace(**base)


def test_find_repo_config_walks_up_to_git_root(repo):
    assert find_repo_config(repo / "benchmarking") == repo / ".baircondor.yaml"
    assert find_repo_config(repo) == repo / ".baircondor.yaml"


def test_find_repo_config_stops_at_git_root(tmp_path):
    (tmp_path / ".baircondor.yaml").write_text("profiles: {}\n")
    inner = tmp_path / "other" / "repo"
    inner.mkdir(parents=True)
    (inner / ".git").mkdir()
    assert find_repo_config(inner) is None


def test_load_config_layers_repo_over_personal_and_expands(repo, tmp_path):
    personal = tmp_path / "personal.yaml"
    personal.write_text("defaults:\n  scratch: /raid/alice\n  mem_gpu: 48G\n")
    cfg = load_config(str(personal), repo_dir=repo / "benchmarking")
    assert (
        cfg["defaults"]["scratch"] == "/SHARED/home/alice/condor-scratch"
    )  # repo wins, ${USER} expanded
    assert cfg["defaults"]["mem_gpu"] == "48G"  # personal value kept where repo is silent
    assert cfg["repo_config"] == str(repo / ".baircondor.yaml")
    assert set(cfg["profiles"]) == {"eval", "pretrain"}


def test_apply_profile_fills_unset_and_cli_wins(repo):
    cfg = load_config(None, repo_dir=repo)
    args = _args(mem="8G", sub_lines=["+Cli = 1"])
    profile = apply_profile(cfg, args, "eval")
    assert profile["conda_env"] == "laya_env"
    assert args.machine == "REDLRADADM35839"
    assert args.gpus == 1
    assert args.mem == "8G"  # CLI flag wins
    assert args.conda_base == str(repo / "home" / "anaconda3")  # ~ expanded
    assert args.sub_lines == ['require_gpus = DeviceUuid != "bad"', "+Cli = 1"]


def test_apply_profile_errors(repo, tmp_path):
    cfg = load_config(None, repo_dir=repo)
    with pytest.raises(ValueError, match="available: eval, pretrain"):
        apply_profile(cfg, _args(), "nope")
    (repo / ".baircondor.yaml").write_text("profiles:\n  bad:\n    gpu: 1\n")
    cfg = load_config(None, repo_dir=repo)
    with pytest.raises(ValueError, match="unknown keys: gpu"):
        apply_profile(cfg, _args(), "bad")
    no_repo = tmp_path / "elsewhere"
    no_repo.mkdir()
    (no_repo / ".git").mkdir()
    with pytest.raises(ValueError, match="none was found"):
        apply_profile(load_config(None, repo_dir=no_repo), _args(), "eval")
    assert apply_profile(cfg, _args(), None) == {}


def test_condor_config_from_profile(repo, monkeypatch):
    monkeypatch.chdir(repo)
    cfg = CondorConfig.from_profile("pretrain", mem="300G")
    assert cfg.profile == "pretrain"
    assert cfg.machine == "REDLRADADM35840"
    assert cfg.gpus == 2
    assert cfg.mem == "300G"  # override wins
    assert cfg.conda_env == "eeg2025"


def test_submit_profile_lands_in_files(repo, monkeypatch, tmp_path):
    from baircondor.api import submit

    monkeypatch.chdir(repo)
    run_dir = submit(
        ["echo", "hi"], profile="eval", scratch=str(tmp_path / "scratch"), dry_run=True
    )
    job_sub = (run_dir / "job.sub").read_text()
    assert 'requirements = regexp("^REDLRADADM35839", Machine, "i")' in job_sub
    assert "request_memory = 64G" in job_sub
    assert 'require_gpus = DeviceUuid != "bad"' in job_sub
    assert 'ENV_NAME="laya_env"' in (run_dir / "run.sh").read_text()
    assert '"profile": "eval"' in (run_dir / "meta.json").read_text()
