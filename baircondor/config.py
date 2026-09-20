"""Config loading: built-in defaults -> personal config.yaml -> repo .baircondor.yaml -> CLI flags."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "defaults": {
        "scratch": "~/condor-scratch",
        "runs_subdir": "condor-runs",
        "cpus_per_gpu": 4,  # effectively like num workers per GPU, but also used to compute CPU-only defaults
        "cpus_cpu_only": 4,
        "mem_gpu": "24G",
        "mem_cpu_only": "8G",
    },
    "condor": {
        "omit_request_gpus_when_zero": True,
        "pin_submit_host": True,
        "machine": None,
    },
    "conda": {
        "conda_base": None,
    },
}

CONFIG_PATH = Path.home() / ".config" / "baircondor" / "config.yaml"
_CONFIG_PATH = CONFIG_PATH

# Committed per-repo config, found by walking up from cwd to the git root. Carries the
# launch recipe (target machine, scratch, conda env, memory) with the code, and named
# profiles for the different job types in one repo (e.g. pretraining vs evaluation).
REPO_CONFIG_NAME = ".baircondor.yaml"

# Keys a profile may set: the submit flags, plus extra job.sub lines.
PROFILE_KEYS = frozenset(
    {
        "gpus",
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
        "machine",
        "pin_submit_host",
        "sub_lines",
    }
)


def get_user() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def load_config(config_path: str | None = None, repo_dir: Path | None = None) -> dict[str, Any]:
    """Return merged config: built-in defaults < personal config < repo .baircondor.yaml.

    ``--config PATH`` replaces the personal config (``~/.config/baircondor/config.yaml``).
    A committed ``.baircondor.yaml`` in the repo (cwd or any parent up to the git root)
    layers on top of that, so a repo's launch recipe wins over personal defaults and CLI
    flags win over both. The repo file's path is recorded under ``cfg["repo_config"]``.
    """
    cfg = _deep_copy(DEFAULTS)
    cfg["profiles"] = {}
    cfg["repo_config"] = None

    path = Path(config_path) if config_path else _CONFIG_PATH
    if path.exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, user_cfg)

    repo_path = find_repo_config(repo_dir)
    if repo_path:
        with open(repo_path) as f:
            repo_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, _expand_strings(repo_cfg))
        cfg["repo_config"] = str(repo_path)

    return cfg


def find_repo_config(start: Path | None = None) -> Path | None:
    """Find .baircondor.yaml in ``start`` (default cwd) or a parent, stopping at the git root."""
    d = (start or Path.cwd()).resolve()
    for p in (d, *d.parents):
        candidate = p / REPO_CONFIG_NAME
        if candidate.is_file():
            return candidate
        if (p / ".git").exists():
            return None
    return None


def apply_profile(cfg: dict, args, name: str | None) -> dict[str, Any]:
    """Fill unset submit args from the named profile in the repo config.

    CLI flags (anything already set on ``args``) win. Returns the profile dict, or {}
    when no profile was requested.
    """
    if not name:
        return {}
    if not cfg.get("repo_config"):
        raise ValueError(
            f"--profile {name} needs a {REPO_CONFIG_NAME} in this repo "
            "(searched from the cwd up to the git root) and none was found"
        )
    profiles = cfg.get("profiles") or {}
    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise ValueError(f"profile '{name}' not in {cfg['repo_config']} (available: {available})")
    profile = profiles[name] or {}
    unknown = sorted(set(profile) - PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"profile '{name}' has unknown keys: {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(PROFILE_KEYS))})"
        )
    for key, value in profile.items():
        if key == "sub_lines":
            existing = list(getattr(args, "sub_lines", None) or [])
            setattr(args, "sub_lines", list(value or []) + existing)
        elif getattr(args, key, None) is None:
            setattr(args, key, value)
    return profile


def resolve_resources(cfg: dict, args) -> dict[str, Any]:
    """Compute final gpus/cpus/mem from config defaults and CLI args."""
    gpus = args.gpus if args.gpus is not None else 1

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
    # An unset base is resolved at runtime inside run.sh on the execute host (hosts
    # differ under --machine), so don't autodetect on the submit host here.
    return {"env": conda_env, "conda_base": _normalize_conda_base(conda_base)}


def _normalize_conda_base(base: str | None) -> str | None:
    """Normalize a base that points at the conda binary (.../bin/conda) to its root."""
    if not base:
        return base
    p = Path(base)
    if p.name == "conda" and p.parent.name == "bin":
        return str(p.parent.parent)
    return base


def resolve_pin_submit_host(cfg: dict, args) -> bool:
    pin_submit_host = getattr(args, "pin_submit_host", None)
    if pin_submit_host is None:
        return bool(cfg["condor"]["pin_submit_host"])
    return pin_submit_host


def resolve_machine(cfg: dict, args) -> str | None:
    """Resolve the explicit --machine target: CLI flag wins over config default."""
    machine = getattr(args, "machine", None)
    if machine is None:
        return cfg["condor"].get("machine")
    return machine


def _autodetect_conda_base() -> str | None:
    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        result = None

    if result and result.returncode == 0:
        base = result.stdout.strip()
        if base:
            return str(Path(base).expanduser())

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_path = Path(conda_exe).expanduser()
        if conda_path.name == "conda":
            return str(conda_path.parent.parent)

    return None


# ── helpers ──────────────────────────────────────────────────────────────────


def _deep_copy(d: dict) -> dict:
    import copy

    return copy.deepcopy(d)


def _expand_strings(obj):
    """Expand ``${USER}``/``$USER`` and a leading ``~`` in every string of a nested config."""
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, dict):
        return {k: _expand_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_strings(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
