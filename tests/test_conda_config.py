"""Tests for conda base resolution."""

import subprocess
from types import SimpleNamespace

from baircondor.config import resolve_conda, resolve_pin_submit_host


def _args(conda_env=None, conda_base=None):
    return SimpleNamespace(conda_env=conda_env, conda_base=conda_base)


def test_conda_base_from_cli():
    cfg = {"conda": {"conda_base": "/cfg/base"}}
    out = resolve_conda(cfg, _args(conda_env="myenv", conda_base="/cli/base"))
    assert out["env"] == "myenv"
    assert out["conda_base"] == "/cli/base"


def test_conda_base_from_config():
    cfg = {"conda": {"conda_base": "/cfg/base"}}
    out = resolve_conda(cfg, _args(conda_env="myenv"))
    assert out["conda_base"] == "/cfg/base"


def test_conda_base_autodetect_from_conda_info(monkeypatch):
    cfg = {"conda": {"conda_base": None}}

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="/opt/conda\n")

    monkeypatch.setattr("baircondor.config.subprocess.run", fake_run)
    out = resolve_conda(cfg, _args(conda_env="myenv"))
    assert out["conda_base"] == "/opt/conda"


def test_conda_base_autodetect_from_conda_exe(monkeypatch):
    cfg = {"conda": {"conda_base": None}}

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="")

    monkeypatch.setattr("baircondor.config.subprocess.run", fake_run)
    monkeypatch.setenv("CONDA_EXE", "/home/user/miniconda3/bin/conda")
    out = resolve_conda(cfg, _args(conda_env="myenv"))
    assert out["conda_base"] == "/home/user/miniconda3"


def test_conda_base_autodetect_absent(monkeypatch):
    cfg = {"conda": {"conda_base": None}}

    def fake_run(*args, **kwargs):
        raise OSError

    monkeypatch.setattr("baircondor.config.subprocess.run", fake_run)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    out = resolve_conda(cfg, _args(conda_env="myenv"))
    assert out["conda_base"] is None


def test_pin_submit_host_from_config_default():
    cfg = {"condor": {"pin_submit_host": True}}
    assert resolve_pin_submit_host(cfg, _args()) is True


def test_pin_submit_host_cli_override():
    cfg = {"condor": {"pin_submit_host": True}}
    assert resolve_pin_submit_host(cfg, _args()) is True
    args = _args()
    args.pin_submit_host = False
    assert resolve_pin_submit_host(cfg, args) is False
