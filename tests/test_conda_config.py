"""Tests for conda base resolution."""

import subprocess
from types import SimpleNamespace

import pytest

from baircondor.config import (
    _autodetect_conda_base,
    resolve_conda,
    resolve_machine,
    resolve_pin_submit_host,
    resolve_require_gpus,
)
from baircondor.submit import _validate_conda


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


def test_resolve_conda_no_base_returns_none():
    """Base is resolved at runtime in run.sh, not on the submit host."""
    cfg = {"conda": {"conda_base": None}}
    out = resolve_conda(cfg, _args(conda_env="myenv"))
    assert out["conda_base"] is None


def test_conda_base_bin_conda_normalized():
    cfg = {"conda": {"conda_base": None}}
    out = resolve_conda(cfg, _args(conda_env="myenv", conda_base="/opt/conda/bin/conda"))
    assert out["conda_base"] == "/opt/conda"


def test_conda_base_plain_dir_unchanged():
    cfg = {"conda": {"conda_base": None}}
    out = resolve_conda(cfg, _args(conda_env="myenv", conda_base="/opt/conda"))
    assert out["conda_base"] == "/opt/conda"


# _autodetect_conda_base is no longer called during submission, but the setup wizard
# still uses it to pre-fill a suggestion on the login node.
def test_autodetect_from_conda_info(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="/opt/conda\n")

    monkeypatch.setattr("baircondor.config.subprocess.run", fake_run)
    assert _autodetect_conda_base() == "/opt/conda"


def test_autodetect_from_conda_exe(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="")

    monkeypatch.setattr("baircondor.config.subprocess.run", fake_run)
    monkeypatch.setenv("CONDA_EXE", "/home/user/miniconda3/bin/conda")
    assert _autodetect_conda_base() == "/home/user/miniconda3"


def test_autodetect_absent(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError

    monkeypatch.setattr("baircondor.config.subprocess.run", fake_run)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    assert _autodetect_conda_base() is None


# ── _validate_conda: fail fast on a bad same-host base ────────────────────────


def _write_conda_sh(base):
    activate = base / "etc" / "profile.d"
    activate.mkdir(parents=True)
    (activate / "conda.sh").write_text("")


def test_validate_conda_ok_when_activate_present(tmp_path):
    _write_conda_sh(tmp_path)
    _validate_conda({"env": "train", "conda_base": str(tmp_path)}, None)


def test_validate_conda_exits_when_activate_missing(tmp_path):
    with pytest.raises(SystemExit):
        _validate_conda({"env": "train", "conda_base": str(tmp_path / "nope")}, None)


def test_validate_conda_skips_when_machine_set(tmp_path):
    _validate_conda({"env": "train", "conda_base": str(tmp_path / "nope")}, "OTHER")


def test_validate_conda_skips_when_no_base():
    _validate_conda({"env": "train", "conda_base": None}, None)


def test_pin_submit_host_from_config_default():
    cfg = {"condor": {"pin_submit_host": True}}
    assert resolve_pin_submit_host(cfg, _args()) is True


def test_pin_submit_host_cli_override():
    cfg = {"condor": {"pin_submit_host": True}}
    assert resolve_pin_submit_host(cfg, _args()) is True
    args = _args()
    args.pin_submit_host = False
    assert resolve_pin_submit_host(cfg, args) is False


def test_machine_default_none():
    cfg = {"condor": {"machine": None}}
    args = _args()
    args.machine = None
    assert resolve_machine(cfg, args) is None


def test_machine_from_config_default():
    cfg = {"condor": {"machine": "SOMEHOST"}}
    args = _args()
    args.machine = None
    assert resolve_machine(cfg, args) == "SOMEHOST"


def test_machine_cli_overrides_config():
    cfg = {"condor": {"machine": "SOMEHOST"}}
    args = _args()
    args.machine = "CLIHOST"
    assert resolve_machine(cfg, args) == "CLIHOST"


def test_require_gpus_matches_explicit_machine():
    cfg = {"condor": {"require_gpus": {"REDLRADADM35840": 'UUID != "GPU-bad"'}}}
    assert (
        resolve_require_gpus(cfg, "REDLRADADM35840.ad.medctr.ucla.edu", True, "submit-host")
        == 'UUID != "GPU-bad"'
    )


def test_require_gpus_matches_pinned_submit_host():
    cfg = {"condor": {"require_gpus": {"redlradadm35840": 'UUID != "GPU-bad"'}}}
    assert (
        resolve_require_gpus(cfg, None, True, "redlradadm35840.ad.medctr.ucla.edu")
        == 'UUID != "GPU-bad"'
    )


def test_require_gpus_no_match_returns_none():
    cfg = {"condor": {"require_gpus": {"redlradadm35840": 'UUID != "GPU-bad"'}}}
    assert resolve_require_gpus(cfg, None, True, "otherhost.ad.medctr.ucla.edu") is None


def test_require_gpus_none_when_pinning_disabled_and_no_machine():
    cfg = {"condor": {"require_gpus": {"redlradadm35840": 'UUID != "GPU-bad"'}}}
    assert resolve_require_gpus(cfg, None, False, "redlradadm35840.ad.medctr.ucla.edu") is None


def test_require_gpus_absent_key_returns_none():
    cfg = {"condor": {"require_gpus": {}}}
    assert resolve_require_gpus(cfg, "REDLRADADM35840", True, "submit-host") is None
