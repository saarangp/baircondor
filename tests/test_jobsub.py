"""Tests for job.sub generation and argument escaping."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from baircondor.submit import _condor_escape_arg, _patch_args
from baircondor.templates import validate_sub_line, write_job_sub

submit_mod = importlib.import_module("baircondor.submit")

RES = {"gpus": 1, "cpus": 6, "mem": "24G", "disk": None}


def _sub_text(tmp_path, resources=RES, pin=True, omit_zero=True, machine=None, extra=None):
    write_job_sub(
        tmp_path,
        tmp_path / "repo",
        resources,
        "myjob",
        "submit-host.example.com",
        pin,
        omit_zero,
        machine,
        extra,
    )
    return (tmp_path / "job.sub").read_text()


def test_required_fields(tmp_path):
    text = _sub_text(tmp_path)
    for line in (
        "universe = vanilla",
        f"initialdir = {tmp_path / 'repo'}",
        "executable = /bin/bash",
        "arguments = __ARGS_PLACEHOLDER__",
        "getenv = True",
        f"output = {tmp_path}/stdout.txt",
        f"log    = {tmp_path}/condor.log",
        "request_cpus = 6",
        "request_memory = 24G",
        "request_gpus = 1",
        'requirements = (toLower(Machine) == "submit-host.example.com")',
        '+JobBatchName = "myjob"',
    ):
        assert line in text
    assert "request_disk" not in text


def test_requirements_line(tmp_path):
    assert "requirements =" not in _sub_text(tmp_path, pin=False)
    text = _sub_text(tmp_path, machine="REDLRADADM35840")  # --machine wins over the submit-host pin
    assert 'requirements = regexp("^REDLRADADM35840", Machine, "i")' in text
    assert "toLower(Machine)" not in text
    assert 'regexp("^REDLRADADM35840"' in _sub_text(tmp_path, pin=False, machine="REDLRADADM35840")


def test_gpus_zero_and_disk(tmp_path):
    cpu = {"gpus": 0, "cpus": 4, "mem": "8G", "disk": None}
    assert "request_gpus" not in _sub_text(tmp_path, cpu, omit_zero=True)
    assert "request_gpus = 0" in _sub_text(tmp_path, cpu, omit_zero=False)
    assert "request_disk = 50G" in _sub_text(tmp_path, dict(RES, disk="50G"))


def test_extra_lines_appended_after_batch_name(tmp_path):
    text = _sub_text(tmp_path, extra=['require_gpus = DeviceUuid != "abc"', "  +WantX = True "])
    assert text.index('+JobBatchName = "myjob"') < text.index('require_gpus = DeviceUuid != "abc"')
    assert "\n+WantX = True\n" in text  # stripped
    for bad in ("nonsense", "= x", "a = b\nc = d"):
        with pytest.raises(ValueError):
            validate_sub_line(bad)


def test_condor_escape_arg():
    assert _condor_escape_arg("hello") == "hello"
    assert _condor_escape_arg("/my dir/run.sh") == "'/my dir/run.sh'"
    assert _condor_escape_arg('say "hi"') == """'say ""hi""'"""
    assert _condor_escape_arg("it's") == "'it''s'"
    for bad in ("a\nb", "a\rb"):
        with pytest.raises(ValueError):
            _condor_escape_arg(bad)


def test_patch_args(tmp_path):
    job_sub = tmp_path / "job.sub"
    job_sub.write_text("universe = vanilla\narguments = __ARGS_PLACEHOLDER__\ngetenv = True\n")
    _patch_args(job_sub, Path("/my dir/run.sh"), ["echo", 'say "hi"'])
    text = job_sub.read_text()
    assert "universe = vanilla\n" in text and "getenv = True\n" in text
    assert """arguments = "'/my dir/run.sh' -- echo 'say ""hi""'"\n""" in text


def _run_submit(monkeypatch, tmp_path, hostname, **extra_args):
    monkeypatch.setattr(submit_mod.subprocess, "check_output", lambda cmd, text: hostname)
    monkeypatch.setattr(submit_mod, "_submit", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    base = dict(
        config=str(tmp_path / "no-config.yaml"),
        command=["--", "echo", "hello"],
        gpus=0,
        cpus=None,
        mem=None,
        jobname="job",
        scratch=str(tmp_path / "scratch"),
        runs_subdir=None,
        project=None,
        tag=None,
        conda_env=None,
        conda_base=None,
        dry_run=True,
        pin_submit_host=None,
        machine=None,
    )
    args = SimpleNamespace(**{**base, **extra_args})
    submit_mod.run_submit(args)
    return next((tmp_path / "scratch").glob("**/job.sub")).read_text()


def test_run_submit_pins_to_hostname_f(monkeypatch, tmp_path):
    text = _run_submit(monkeypatch, tmp_path, "REDLRADADM35840.ad.medctr.ucla.edu\n")
    assert 'requirements = (toLower(Machine) == "redlradadm35840.ad.medctr.ucla.edu")' in text
    assert submit_mod._get_submit_host() == "redlradadm35840.ad.medctr.ucla.edu"


def test_run_submit_machine_overrides_submit_host_pin(monkeypatch, tmp_path):
    text = _run_submit(monkeypatch, tmp_path, "OTHERHOST\n", machine="REDLRADADM35840")
    assert 'requirements = regexp("^REDLRADADM35840", Machine, "i")' in text
    assert "toLower(Machine)" not in text
