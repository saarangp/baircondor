"""Public Python API for baircondor.

Usage::

    from baircondor import CondorConfig, submit

    # Using a CondorConfig model (embeddable in your own pydantic configs)
    cfg = CondorConfig(gpus=2, mem="32G", conda_env="train")
    submit(["python", "train.py", "--lr", "1e-4"], condor=cfg)

    # Or with plain kwargs
    submit(["python", "train.py"], gpus=1, dry_run=True)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

from baircondor.config import PROFILE_KEYS, apply_profile, load_config
from baircondor.submit import run_interactive, run_submit


class CondorConfig(BaseModel):
    """HTCondor resource configuration, embeddable in any pydantic model."""

    model_config = ConfigDict(extra="forbid")

    gpus: int = 1
    cpus: int | None = None
    mem: str | None = None
    disk: str | None = None
    jobname: str | None = None
    scratch: str | None = None
    runs_subdir: str | None = None
    project: str | None = None
    tag: str | None = None
    conda_env: str | None = None
    conda_base: str | None = None
    config: str | None = None
    machine: str | None = None
    profile: str | None = None
    sub_lines: list[str] | None = None
    dry_run: bool = False

    @classmethod
    def from_profile(cls, name: str, repo_dir: Path | None = None, **overrides) -> "CondorConfig":
        """Build a config from a named profile in the repo's .baircondor.yaml.

        Keyword overrides win over the profile, exactly like CLI flags do.
        """
        cfg = load_config(overrides.get("config"), repo_dir=repo_dir)
        ns = SimpleNamespace(**{k: None for k in PROFILE_KEYS})
        apply_profile(cfg, ns, name)
        values = {k: v for k, v in vars(ns).items() if v is not None and k in cls.model_fields}
        values.update(overrides)
        return cls(profile=name, **values)


def _build_namespace(condor: CondorConfig | None, kwargs: dict) -> SimpleNamespace:
    """Merge a CondorConfig (if given) with any kwarg overrides into a SimpleNamespace."""
    base = condor.model_dump() if condor is not None else {}
    base.update(kwargs)
    # Ensure all expected fields exist with defaults
    for field, default in CondorConfig.model_fields.items():
        base.setdefault(field, default.default)
    return SimpleNamespace(**base)


def submit(command: list[str], condor: CondorConfig | None = None, **kwargs) -> Path:
    """Submit a batch job.

    Args:
        command: The command to run (e.g. ``["python", "train.py"]``).
        condor: Optional :class:`CondorConfig` instance.
        **kwargs: Individual overrides (same names as CondorConfig fields).

    Returns:
        Path to the created run directory.
    """
    ns = _build_namespace(condor, kwargs)
    ns.command = command
    return run_submit(ns)


def interactive(condor: CondorConfig | None = None, **kwargs) -> Path:
    """Start an interactive condor session.

    Args:
        condor: Optional :class:`CondorConfig` instance.
        **kwargs: Individual overrides (same names as CondorConfig fields).

    Returns:
        Path to the created run directory.
    """
    ns = _build_namespace(condor, kwargs)
    return run_interactive(ns)
