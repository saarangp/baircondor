"""Tests for the Python API (CondorConfig, submit, interactive)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from baircondor.api import CondorConfig, interactive, submit
from baircondor.config import PROFILE_KEYS


def _run_dir(scratch):
    return next(Path(scratch).rglob("job.sub")).parent


def test_submit_kwargs_and_config_object(tmp_path):
    scratch = str(tmp_path / "s1")
    run_dir = submit(["python", "train.py", "--lr", "1e-4"], gpus=0, scratch=scratch, dry_run=True)
    assert run_dir == _run_dir(scratch)
    assert all((run_dir / f).exists() for f in ("job.sub", "run.sh", "meta.json"))
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["command"] == ["python", "train.py", "--lr", "1e-4"] and meta["mode"] == "batch"

    cfg = CondorConfig(gpus=0, scratch=str(tmp_path / "s2"), jobname="from-model")
    run_dir = submit(["echo"], condor=cfg, jobname="from-kwarg", dry_run=True)  # kwargs win
    assert "from-kwarg" in str(run_dir)


def test_interactive_creates_run_dir(tmp_path):
    scratch = str(tmp_path / "s")
    run_dir = interactive(condor=CondorConfig(gpus=0, scratch=scratch, dry_run=True))
    assert run_dir == _run_dir(scratch)
    assert json.loads((run_dir / "meta.json").read_text())["mode"] == "interactive"


def test_condor_config_model():
    cfg = CondorConfig()
    assert cfg.gpus is None and cfg.dry_run is False  # unset means "resolve from config"
    assert all(getattr(cfg, f) is None for f in ("cpus", "mem", "machine", "profile", "sub_lines"))
    assert CondorConfig(**CondorConfig(gpus=2, mem="32G").model_dump()) == CondorConfig(
        gpus=2, mem="32G"
    )
    with pytest.raises(ValidationError):
        CondorConfig(gps=2)
    # a profile may set exactly the submit fields; keep the two lists in step
    assert PROFILE_KEYS == set(CondorConfig.model_fields) - {"config", "dry_run", "profile"}
