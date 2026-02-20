"""Core submission logic: run dir creation, file generation, condor_submit."""

from __future__ import annotations

import os
import random
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
    jobname = args.jobname or repo_dir.name
    runs_subdir = getattr(args, "runs_subdir", None) or cfg["defaults"]["runs_subdir"]
    run_dir = _make_run_dir(
        args.scratch,
        runs_subdir,
        jobname,
        getattr(args, "project", None),
        getattr(args, "tag", None),
    )

    _validate_conda(conda)

    run_dir.mkdir(parents=True, exist_ok=False)

    run_sh = write_run_sh(run_dir, repo_dir, jobname, resources, conda)
    write_job_sub(
        run_dir, repo_dir, resources, jobname, cfg["condor"]["omit_request_gpus_when_zero"]
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
    jobname = args.jobname or "interactive"
    runs_subdir = getattr(args, "runs_subdir", None) or cfg["defaults"]["runs_subdir"]
    run_dir = _make_run_dir(
        args.scratch,
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
        run_dir, repo_dir, resources, jobname, cfg["condor"]["omit_request_gpus_when_zero"]
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
    if not scratch_path.exists():
        sys.exit(f"error: --scratch path does not exist: {scratch}")
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
        sys.exit("error: --conda-env requires --conda-base or conda.conda_base in config")


def _patch_args(job_sub: Path, run_sh: Path, command: list[str]) -> None:
    """Replace the $(args) placeholder in job.sub with the real argument string."""
    import shlex

    arg_str = shlex.join(["--"] + command)
    text = job_sub.read_text()
    text = text.replace(
        f'arguments = "{run_sh}" -- $(args)',
        f'arguments = "{run_sh}" {arg_str}',
    )
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
