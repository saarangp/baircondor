"""Core submission logic: run dir creation, file generation, condor_submit."""

from __future__ import annotations

import os
import random
import socket
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config, resolve_conda, resolve_resources
from .meta import write_meta
from .templates import write_job_sub, write_run_sh


def run_submit(args) -> None:
    cfg = load_config(getattr(args, "config", None))
    resources = resolve_resources(cfg, args)
    conda = resolve_conda(cfg, args)

    # strip leading "--" separator that argparse REMAINDER captures
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.exit("error: a command is required after --")

    repo_dir = Path.cwd()
    submit_host = socket.getfqdn().lower()
    jobname = args.jobname or repo_dir.name
    scratch = args.scratch or cfg["defaults"]["scratch"]
    scratch = str(Path(scratch).expanduser())
    runs_subdir = getattr(args, "runs_subdir", None) or cfg["defaults"]["runs_subdir"]
    run_dir = _make_run_dir(
        scratch,
        runs_subdir,
        jobname,
        getattr(args, "project", None),
        getattr(args, "tag", None),
    )

    _validate_conda(conda)

    run_dir.mkdir(parents=True, exist_ok=False)

    run_sh = write_run_sh(run_dir, repo_dir, jobname, resources, conda)
    write_job_sub(
        run_dir,
        repo_dir,
        resources,
        jobname,
        submit_host,
        cfg["condor"]["omit_request_gpus_when_zero"],
    )
    write_meta(run_dir, repo_dir, jobname, "batch", command, resources, conda)

    job_sub = run_dir / "job.sub"
    # patch job.sub: replace $(args) placeholder with actual arguments
    _patch_args(job_sub, run_sh, command)

    _submit(job_sub, args.dry_run, run_dir, repo_dir)


def run_interactive(args) -> None:
    cfg = load_config(getattr(args, "config", None))
    resources = resolve_resources(cfg, args)
    conda = resolve_conda(cfg, args)

    repo_dir = Path.cwd()
    submit_host = socket.getfqdn().lower()
    jobname = args.jobname or "interactive"
    scratch = args.scratch or cfg["defaults"]["scratch"]
    scratch = str(Path(scratch).expanduser())
    runs_subdir = getattr(args, "runs_subdir", None) or cfg["defaults"]["runs_subdir"]
    run_dir = _make_run_dir(
        scratch,
        runs_subdir,
        jobname,
        getattr(args, "project", None),
        getattr(args, "tag", None),
    )

    _validate_conda(conda)

    run_dir.mkdir(parents=True, exist_ok=False)

    command = ["/bin/bash", "-i"]
    run_sh = write_run_sh(run_dir, repo_dir, jobname, resources, conda)
    write_job_sub(
        run_dir,
        repo_dir,
        resources,
        jobname,
        submit_host,
        cfg["condor"]["omit_request_gpus_when_zero"],
    )
    write_meta(run_dir, repo_dir, jobname, "interactive", command, resources, conda)

    job_sub = run_dir / "job.sub"
    _patch_args(job_sub, run_sh, command)

    _submit_interactive(job_sub, args.dry_run, run_dir)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_run_dir(
    scratch: str, runs_subdir: str, jobname: str, project: str | None, tag: str | None
) -> Path:
    scratch_path = Path(scratch)
    scratch_path.mkdir(parents=True, exist_ok=True)
    if not os.access(scratch_path, os.W_OK):
        sys.exit(f"error: --scratch path is not writable: {scratch}")

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shortid = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    dirname = f"{timestamp}_{shortid}"

    parts = [scratch_path, runs_subdir, user]
    if project:
        parts.append(project)
    parts += [jobname, dirname]
    return Path(*parts)


def _validate_conda(conda: dict) -> None:
    if conda.get("env") and not conda.get("conda_base"):
        sys.exit(
            "error: --conda-env requires a conda base path; auto-detection failed. "
            "Set --conda-base or conda.conda_base in config."
        )


def _condor_escape_arg(arg: str) -> str:
    """Escape one argument for HTCondor new-syntax arguments line.

    Rules: double-quotes are doubled (""), arguments containing spaces/tabs/single-quotes
    are wrapped in single quotes with interior single-quotes doubled ('').
    """
    result = arg.replace('"', '""')
    if " " in result or "\t" in result or "'" in result:
        result = "'" + result.replace("'", "''") + "'"
    return result


def _patch_args(job_sub: Path, run_sh: Path, command: list[str]) -> None:
    """Replace the __ARGS_PLACEHOLDER__ in job.sub with a properly quoted argument string."""
    parts = [str(run_sh), "--"] + command
    inner = " ".join(_condor_escape_arg(p) for p in parts)
    arg_line = f'arguments = "{inner}"'
    text = job_sub.read_text()
    text = text.replace("arguments = __ARGS_PLACEHOLDER__", arg_line)
    job_sub.write_text(text)


def _submit(job_sub: Path, dry_run: bool, run_dir: Path, repo_dir: Path) -> None:
    cmd = ["condor_submit", str(job_sub)]
    print(f"Repo dir : {repo_dir}")
    print(f"Run dir  : {run_dir}")
    print(f"Stdout   : {run_dir}/stdout.txt")
    print(f"Stderr   : {run_dir}/stderr.txt")
    print(f"Log      : {run_dir}/condor.log")
    print(f"Reproduce: condor_submit {job_sub}")

    if dry_run:
        print(f"\n[dry-run] would run: {' '.join(cmd)}")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _submit_interactive(job_sub: Path, dry_run: bool, run_dir: Path) -> None:
    cmd = ["condor_submit", "-interactive", str(job_sub)]
    print(f"Run dir  : {run_dir}")

    if dry_run:
        print(f"\n[dry-run] would run: {' '.join(cmd)}")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\nInteractive session ended. Run dir: {run_dir}")
