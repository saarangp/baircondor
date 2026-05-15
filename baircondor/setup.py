"""First-run setup wizard: detect GPU memory, conda base, prompt for scratch."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

_GPU_MEMORY_TABLE: dict[str, str] = {
    "V100": "48G",
    "Quadro RTX 8000": "48G",
    "H100 NVL": "96G",
    "L40S": "48G",
    "RTX 2080 Ti": "24G",
}

_DEFAULT_MEM = "24G"


def lookup_gpu_memory(model_string: str) -> str:
    for key, mem in _GPU_MEMORY_TABLE.items():
        if key in model_string:
            return mem
    return _DEFAULT_MEM


def detect_gpu_memory() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        return lookup_gpu_memory(first_line)
    except (subprocess.TimeoutExpired, OSError, IndexError):
        return _DEFAULT_MEM


def write_config(data: dict, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def run_wizard(config_path: Path) -> bool:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    from .config import _autodetect_conda_base

    console = Console(stderr=True)
    console.print(
        Panel(
            "[bold]baircondor first-time setup[/bold]\n"
            "Press Enter to accept the detected default, or type a new value.",
            border_style="blue",
        )
    )

    scratch = Prompt.ask("Scratch path", default="~/condor-scratch", console=console)
    conda_base = Prompt.ask(
        "Conda base path", default=_autodetect_conda_base() or "", console=console
    )
    mem_gpu = Prompt.ask("Memory for GPU jobs", default=detect_gpu_memory(), console=console)
    mem_cpu = Prompt.ask("Memory for CPU-only jobs", default="8G", console=console)

    cfg: dict = {
        "defaults": {
            "scratch": scratch,
            "mem_gpu": mem_gpu,
            "mem_cpu_only": mem_cpu,
        },
    }
    if conda_base:
        cfg["conda"] = {"conda_base": conda_base}

    write_config(cfg, config_path)
    console.print(f"[green]Config written to[/green] {config_path}")

    answer = input("Submit now? [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")
