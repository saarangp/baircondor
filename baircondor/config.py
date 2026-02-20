"""Config loading: built-in defaults -> config.yaml -> CLI flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "defaults": {
        "runs_subdir": "condor-runs",
        "cpus_per_gpu": 6,
        "cpus_cpu_only": 4,
        "mem_gpu": "24G",
        "mem_cpu_only": "8G",
    },
    "condor": {
        "omit_request_gpus_when_zero": True,
    },
    "conda": {
        "conda_base": None,
    },
}

_CONFIG_PATH = Path.home() / ".config" / "baircondor" / "config.yaml"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Return merged config: built-in defaults overridden by config file."""
    cfg = _deep_copy(DEFAULTS)

    path = Path(config_path) if config_path else _CONFIG_PATH
    if path.exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, user_cfg)

    return cfg


def resolve_resources(cfg: dict, args) -> dict[str, Any]:
    """Compute final gpus/cpus/mem from config defaults and CLI args."""
    gpus = args.gpus

    if args.cpus is not None:
        cpus = args.cpus
    elif gpus > 0:
        cpus = gpus * cfg["defaults"]["cpus_per_gpu"]
    else:
        cpus = cfg["defaults"]["cpus_cpu_only"]

    if args.mem is not None:
        mem = args.mem
    elif gpus > 0:
        mem = cfg["defaults"]["mem_gpu"]
    else:
        mem = cfg["defaults"]["mem_cpu_only"]

    disk = getattr(args, "disk", None)

    return {"gpus": gpus, "cpus": cpus, "mem": mem, "disk": disk}


def resolve_conda(cfg: dict, args) -> dict[str, str | None]:
    conda_env = getattr(args, "conda_env", None)
    conda_base = getattr(args, "conda_base", None) or cfg["conda"]["conda_base"]
    return {"env": conda_env, "conda_base": conda_base}


# ── helpers ──────────────────────────────────────────────────────────────────


def _deep_copy(d: dict) -> dict:
    import copy

    return copy.deepcopy(d)


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
