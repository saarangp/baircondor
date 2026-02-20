"""Tests for the Python API (CondorConfig, submit, interactive)."""

import json
from pathlib import Path

import pytest

from baircondor.api import CondorConfig, interactive, submit


@pytest.fixture
def scratch(tmp_path):
    return str(tmp_path / "scratch")


def _find_run_dir(scratch_path):
    """Walk into the scratch dir and return the deepest run directory (contains job.sub)."""
    from pathlib import Path

    for p in Path(scratch_path).rglob("job.sub"):
        return p.parent
    raise FileNotFoundError(f"no run dir found under {scratch_path}")


class TestSubmitKwargs:
    def test_creates_run_dir_files(self, scratch):
        run_dir = submit(["echo", "hello"], gpus=0, scratch=scratch, dry_run=True)
        assert isinstance(run_dir, Path)
        assert run_dir == _find_run_dir(scratch)
        assert (run_dir / "job.sub").exists()
        assert (run_dir / "run.sh").exists()
        assert (run_dir / "meta.json").exists()

    def test_meta_records_command(self, scratch):
        run_dir = submit(
            ["python", "train.py", "--lr", "1e-4"], gpus=0, scratch=scratch, dry_run=True
        )
        assert isinstance(run_dir, Path)
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["command"] == ["python", "train.py", "--lr", "1e-4"]
        assert meta["mode"] == "batch"


class TestSubmitCondorConfig:
    def test_with_config_object(self, scratch):
        cfg = CondorConfig(gpus=0, scratch=scratch, dry_run=True)
        run_dir = submit(["echo", "test"], condor=cfg)
        assert isinstance(run_dir, Path)
        assert run_dir == _find_run_dir(scratch)
        assert (run_dir / "job.sub").exists()
        assert (run_dir / "run.sh").exists()

    def test_kwargs_override_config(self, scratch):
        cfg = CondorConfig(gpus=0, scratch=scratch, jobname="from-model")
        run_dir = submit(["echo", "test"], condor=cfg, jobname="from-kwarg", dry_run=True)
        assert isinstance(run_dir, Path)
        assert "from-kwarg" in str(run_dir)


class TestInteractive:
    def test_creates_run_dir(self, scratch):
        cfg = CondorConfig(gpus=0, scratch=scratch, dry_run=True)
        run_dir = interactive(condor=cfg)
        assert isinstance(run_dir, Path)
        assert run_dir == _find_run_dir(scratch)
        assert (run_dir / "job.sub").exists()
        assert (run_dir / "run.sh").exists()
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["mode"] == "interactive"


class TestCondorConfigDefaults:
    """Verify CondorConfig defaults match the CLI argparse defaults."""

    def test_gpus_default(self):
        assert CondorConfig().gpus == 1

    def test_optional_fields_default_none(self):
        cfg = CondorConfig()
        for field in (
            "cpus",
            "mem",
            "disk",
            "jobname",
            "scratch",
            "runs_subdir",
            "project",
            "tag",
            "conda_env",
            "conda_base",
            "config",
        ):
            assert getattr(cfg, field) is None, f"{field} should default to None"

    def test_dry_run_default_false(self):
        assert CondorConfig().dry_run is False

    def test_model_is_serializable(self):
        cfg = CondorConfig(gpus=2, mem="32G", conda_env="train")
        d = cfg.model_dump()
        assert d["gpus"] == 2
        assert d["mem"] == "32G"
        cfg2 = CondorConfig(**d)
        assert cfg2 == cfg
