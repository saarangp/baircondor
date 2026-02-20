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

from types import SimpleNamespace

from pydantic import BaseModel

from baircondor.submit import run_interactive, run_submit


class CondorConfig(BaseModel):
    """HTCondor resource configuration, embeddable in any pydantic model."""

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
    dry_run: bool = False


def _build_namespace(condor: CondorConfig | None, kwargs: dict) -> SimpleNamespace:
    """Merge a CondorConfig (if given) with any kwarg overrides into a SimpleNamespace."""
    base = condor.model_dump() if condor is not None else {}
    base.update(kwargs)
    # Ensure all expected fields exist with defaults
    for field, default in CondorConfig.model_fields.items():
        base.setdefault(field, default.default)
    return SimpleNamespace(**base)


def submit(command: list[str], condor: CondorConfig | None = None, **kwargs) -> None:
    """Submit a batch job.

    Args:
        command: The command to run (e.g. ``["python", "train.py"]``).
        condor: Optional :class:`CondorConfig` instance.
        **kwargs: Individual overrides (same names as CondorConfig fields).
    """
    ns = _build_namespace(condor, kwargs)
    ns.command = command
    run_submit(ns)


def interactive(condor: CondorConfig | None = None, **kwargs) -> None:
    """Start an interactive condor session.

    Args:
        condor: Optional :class:`CondorConfig` instance.
        **kwargs: Individual overrides (same names as CondorConfig fields).
    """
    ns = _build_namespace(condor, kwargs)
    run_interactive(ns)
