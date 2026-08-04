"""Preflight: run a tiny CPU-only job on a target machine and report what's there.

The lab's exec nodes aren't SSH-able (and 35840 doesn't export its disks), so the
only universal way to ask "what conda envs exist on host X, and how does my cwd
resolve there" is to run a condor job on X and parse its output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from .config import get_user, load_config
from .history import HISTORY_FILE
from .submit import _get_submit_host, _patch_args, _warn_unshared_paths
from .templates import write_job_sub

_console = Console(stderr=True)
_PREFIX = f"[dim]{escape('[baircondor]')}[/dim]"

PREFLIGHT_RESOURCES = {"gpus": 0, "cpus": 1, "mem": "512M", "disk": None}

_PREFLIGHT_SH = """\
#!/usr/bin/env bash
# baircondor preflight: report conda envs and repo state as seen from this host.
set -uo pipefail

if [[ "${1:-}" == "--" ]]; then shift; fi
REPO="${1:-}"

echo "host: $(hostname -f)"

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  for _d in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
    if [[ -f "$_d/etc/profile.d/conda.sh" ]]; then CONDA_BASE="$_d"; break; fi
  done
fi
echo "conda_base: ${CONDA_BASE:-none}"
echo "envs_begin"
if [[ -n "$CONDA_BASE" ]]; then
  echo "base"
  ls -1 "$CONDA_BASE/envs" 2>/dev/null
  ls -1 "$HOME/.conda/envs" 2>/dev/null
fi
echo "envs_end"

echo "repo: $REPO"
if [[ -n "$REPO" && -d "$REPO" ]]; then
  echo "repo_exists: yes"
  cd "$REPO"
  if git rev-parse --git-dir >/dev/null 2>&1; then
    echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)"
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
      echo "git_dirty: yes"
    else
      echo "git_dirty: no"
    fi
  fi
else
  echo "repo_exists: no"
fi
"""


def parse_report(text: str) -> dict:
    """Parse the preflight job's stdout into a dict."""
    report: dict = {"host": None, "conda_base": None, "envs": [], "repo": None}
    report.update({"repo_exists": None, "git_commit": None, "git_branch": None, "git_dirty": None})
    in_envs = False
    for line in text.splitlines():
        line = line.strip()
        if line == "envs_begin":
            in_envs = True
        elif line == "envs_end":
            in_envs = False
        elif in_envs:
            if line and line not in report["envs"]:
                report["envs"].append(line)
        elif ": " in line or line.endswith(":"):
            key, _, value = line.partition(":")
            value = value.strip()
            if key in ("host", "repo", "git_commit", "git_branch"):
                report[key] = value or None
            elif key == "conda_base":
                report[key] = None if value in ("", "none") else value
            elif key in ("repo_exists", "git_dirty"):
                report[key] = value == "yes"
    return report


def run_preflight(args) -> None:
    cfg = load_config(getattr(args, "config", None))
    machine = args.machine
    repo_dir = Path.cwd()
    submit_host = _get_submit_host()
    scratch = args.scratch or cfg["defaults"]["scratch"]
    scratch = str(Path(scratch).expanduser())
    runs_subdir = getattr(args, "runs_subdir", None) or cfg["defaults"]["runs_subdir"]

    _warn_unshared_paths(machine, submit_host, {"cwd": str(repo_dir), "--scratch": scratch})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(scratch) / runs_subdir / get_user() / ".preflight" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    preflight_sh = run_dir / "preflight.sh"
    preflight_sh.write_text(_PREFLIGHT_SH)
    write_job_sub(
        run_dir,
        run_dir,  # initialdir: the run dir itself, so a missing repo can't hold the job
        PREFLIGHT_RESOURCES,
        "preflight",
        submit_host,
        pin_submit_host=False,
        omit_gpus_when_zero=cfg["condor"]["omit_request_gpus_when_zero"],
        machine=machine,
    )
    job_sub = run_dir / "job.sub"
    _patch_args(job_sub, preflight_sh, [str(repo_dir)])

    if getattr(args, "dry_run", False):
        _console.print(f"{_PREFIX} 🧪 {escape('[dry-run]')} would run: condor_submit {job_sub}")
        return

    result = subprocess.run(["condor_submit", str(job_sub)], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, end="", file=sys.stderr)
        sys.exit(result.returncode)
    m = re.search(r"submitted to cluster (\d+)", result.stdout)
    cluster_id = m.group(1) if m else None

    timeout = getattr(args, "timeout", 300)
    _console.print(
        f"{_PREFIX} ⏳ Waiting for preflight job on {escape(machine)} "
        f"(cluster {cluster_id}, timeout {timeout}s)..."
    )
    wait = subprocess.run(
        ["condor_wait", "-wait", str(timeout), str(run_dir / "condor.log")],
        capture_output=True,
        text=True,
    )
    if wait.returncode != 0:
        if cluster_id:
            subprocess.run(["condor_rm", cluster_id], capture_output=True)
        sys.exit(
            f"error: preflight job did not finish within {timeout}s "
            f"({machine} may be busy or unreachable). Job removed; see {run_dir}"
        )

    report = parse_report((run_dir / "stdout.txt").read_text())
    _print_report(report, machine, repo_dir)
    _write_cache(machine, report)


def _print_report(report: dict, machine: str, repo_dir: Path) -> None:
    _console.print(f"[bold]{escape(report.get('host') or machine)}[/bold]")

    base = report.get("conda_base")
    if base:
        _console.print(f"  conda base: {escape(base)}")
        _console.print(f"  conda envs: {escape(', '.join(report['envs']) or '(none)')}")
    else:
        _console.print("  [yellow]no conda installation found[/yellow]")

    _console.print(f"  cwd there : {escape(str(repo_dir))}")
    if not report.get("repo_exists"):
        _console.print(
            f"  [red]✗ this path does not exist on {escape(machine)} — "
            "a job submitted from here would hold. Use a shared /HOSTNAME/... path.[/red]"
        )
        return

    commit = report.get("git_commit")
    if commit is None:
        _console.print("  [green]✓ path exists[/green] (not a git repo)")
        return

    dirty = " (dirty)" if report.get("git_dirty") else ""
    _console.print(
        f"  git there : {escape(commit)} on {escape(report.get('git_branch') or '?')}{dirty}"
    )
    local = _local_git_commit(repo_dir)
    if local and local != commit:
        _console.print(
            f"  [red]✗ {escape(machine)} sees commit {escape(commit)} but your local checkout "
            f"is at {escape(local)} — the job would run DIFFERENT code. This usually means "
            "the path points at a separate machine-local clone.[/red]"
        )
    elif local:
        _console.print("  [green]✓ same commit as your local checkout[/green]")


def _local_git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def cache_file(machine: str) -> Path:
    return HISTORY_FILE.parent / f"preflight-{machine.upper()}.json"


def _write_cache(machine: str, report: dict) -> None:
    path = cache_file(machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "machine": machine,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "conda_base": report.get("conda_base"),
        "envs": report.get("envs", []),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
