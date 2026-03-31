"""Patterns for using baircondor from a pydantic-based experiment repo.

This file is intentionally a reference example, not a built-in queue helper.
Your experiment repo owns config mutation, serialization, and sweep orchestration.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from baircondor import CondorConfig, submit


class DataConfig(BaseModel):
    batch_size: int
    num_workers: int


class OptimizerKwargs(BaseModel):
    lr: float


class OptimizerConfig(BaseModel):
    name: str
    kwargs: OptimizerKwargs


class LoggerConfig(BaseModel):
    group: str


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    optimizer: OptimizerConfig
    logger_config: LoggerConfig
    condor: CondorConfig


def submit_one(config: ExperimentConfig, config_path: Path) -> Path:
    """Pattern: submit one validated config through an existing entrypoint."""
    return submit(
        ["python", "run_pretraining_from_config.py", "--config", str(config_path)],
        condor=config.condor,
    )


def queue_sweep(base_config: ExperimentConfig) -> list[Path]:
    """Pattern: generate variants in caller code and queue them one by one."""
    run_dirs: list[Path] = []

    for lr, batch_size in product([1e-4, 3e-4], [128, 256]):
        variant = base_config.model_copy(deep=True)
        variant.optimizer.kwargs.lr = lr
        variant.data.batch_size = batch_size

        suffix = f"lr{lr:g}-bs{batch_size}"
        variant.condor = variant.condor.model_copy(
            update={
                "jobname": "lejepa-pretrain",
                "project": "eegfm",
                "tag": suffix,
                "dry_run": True,
            }
        )
        variant.logger_config.group = f"{variant.logger_config.group}-{suffix}"

        # Your repo would usually write the variant to a generated config artifact here.
        generated_config_path = Path(f"generated/{suffix}.py")
        run_dir = submit(
            ["python", "run_pretraining_from_config.py", "--config", str(generated_config_path)],
            condor=variant.condor,
        )
        run_dirs.append(run_dir)

    return run_dirs


if __name__ == "__main__":
    base_config = ExperimentConfig(
        data=DataConfig(batch_size=256, num_workers=4),
        optimizer=OptimizerConfig(name="AdamW", kwargs=OptimizerKwargs(lr=1e-4)),
        logger_config=LoggerConfig(group="v2_lejepa_pretraining"),
        condor=CondorConfig(
            gpus=1,
            mem="32G",
            conda_env="train",
            jobname="lejepa-pretrain",
            project="eegfm",
            dry_run=True,
        ),
    )

    single_run_config_path = Path("configs/lejepa_pretraining.py")
    print(f"Single-config pattern run dir: {submit_one(base_config, single_run_config_path)}")

    sweep_run_dirs = queue_sweep(base_config)
    print(f"Queued {len(sweep_run_dirs)} sweep jobs")
