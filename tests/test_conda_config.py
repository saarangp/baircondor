"""Tests for config resolution: conda base, machine, host pinning."""

import subprocess
from types import SimpleNamespace

import pytest

from baircondor.config import (
    _autodetect_conda_base,
    resolve_conda,
    resolve_machine,
    resolve_pin_submit_host,
)
from baircondor.submit import _validate_conda


def _args(**kw):
    return SimpleNamespace(
        **{"conda_env": None, "conda_base": None, "machine": None, "pin_submit_host": None, **kw}
    )


def test_resolve_conda_precedence_and_normalization():
    cfg = {"conda": {"conda_base": "/cfg/base"}}
    assert (
        resolve_conda(cfg, _args(conda_env="e", conda_base="/cli/base"))["conda_base"]
        == "/cli/base"
    )
    assert resolve_conda(cfg, _args(conda_env="e"))["conda_base"] == "/cfg/base"
    assert resolve_conda({"conda": {"conda_base": None}}, _args())["conda_base"] is None  # runtime
    out = resolve_conda({"conda": {"conda_base": None}}, _args(conda_base="/opt/conda/bin/conda"))
    assert out["conda_base"] == "/opt/conda"


def test_autodetect_conda_base(monkeypatch):
    monkeypatch.setattr(
        "baircondor.config.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="/opt/conda\n"),
    )
    assert _autodetect_conda_base() == "/opt/conda"
    monkeypatch.setattr(
        "baircondor.config.subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    monkeypatch.setenv("CONDA_EXE", "/home/u/miniconda3/bin/conda")
    assert _autodetect_conda_base() == "/home/u/miniconda3"
    monkeypatch.delenv("CONDA_EXE")
    assert _autodetect_conda_base() is None


def test_validate_conda(tmp_path):
    (tmp_path / "etc" / "profile.d").mkdir(parents=True)
    (tmp_path / "etc" / "profile.d" / "conda.sh").write_text("")
    _validate_conda({"env": "t", "conda_base": str(tmp_path)}, None)
    with pytest.raises(SystemExit):
        _validate_conda({"env": "t", "conda_base": str(tmp_path / "nope")}, None)
    _validate_conda(
        {"env": "t", "conda_base": str(tmp_path / "nope")}, "OTHER"
    )  # checked on the exec host
    _validate_conda({"env": "t", "conda_base": None}, None)


def test_machine_and_pin_resolution():
    cfg = {"condor": {"pin_submit_host": True, "machine": "CFGHOST"}}
    assert resolve_pin_submit_host(cfg, _args()) is True
    assert resolve_pin_submit_host(cfg, _args(pin_submit_host=False)) is False
    assert resolve_machine(cfg, _args()) == "CFGHOST"
    assert resolve_machine(cfg, _args(machine="CLIHOST")) == "CLIHOST"
    assert resolve_machine({"condor": {"machine": None}}, _args()) is None
