"""Generate job.sub and run.sh content."""

from __future__ import annotations

import stat
from pathlib import Path


def write_job_sub(
    run_dir: Path,
    repo_dir: Path,
    resources: dict,
    jobname: str,
    submit_host: str,
    pin_submit_host: bool,
    omit_gpus_when_zero: bool = True,
) -> Path:
    path = run_dir / "job.sub"
    path.write_text(
        _render_job_sub(
            run_dir,
            repo_dir,
            resources,
            jobname,
            submit_host,
            pin_submit_host,
            omit_gpus_when_zero,
        )
    )
    return path


def write_run_sh(run_dir: Path, repo_dir: Path, jobname: str, resources: dict, conda: dict) -> Path:
    path = run_dir / "run.sh"
    path.write_text(_render_run_sh(run_dir, repo_dir, jobname, resources, conda))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ── renderers ────────────────────────────────────────────────────────────────


def _render_job_sub(
    run_dir: Path,
    repo_dir: Path,
    resources: dict,
    jobname: str,
    submit_host: str,
    pin_submit_host: bool,
    omit_gpus_when_zero: bool,
) -> str:
    run_dir / "run.sh"
    lines = [
        "universe = vanilla",
        f"initialdir = {repo_dir}",
        "executable = /bin/bash",
        "arguments = __ARGS_PLACEHOLDER__",
        "getenv = True",
        f"output = {run_dir}/stdout.txt",
        f"error  = {run_dir}/stderr.txt",
        f"log    = {run_dir}/condor.log",
        f"request_cpus = {resources['cpus']}",
        f"request_memory = {resources['mem']}",
    ]

    if pin_submit_host:
        lines.append(f'requirements = (toLower(Machine) == "{submit_host.lower()}")')

    gpus = resources["gpus"]
    if gpus > 0:
        lines.append(f"request_gpus = {gpus}")
    elif not omit_gpus_when_zero:
        lines.append("request_gpus = 0")

    if resources.get("disk"):
        lines.append(f"request_disk = {resources['disk']}")

    lines.append(f'+JobBatchName = "{jobname}"')
    lines.append("")  # trailing newline
    return "\n".join(lines)


def _render_run_sh(
    run_dir: Path, repo_dir: Path, jobname: str, resources: dict, conda: dict
) -> str:
    parts = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"export BAIRCONDOR_RUN_DIR={run_dir}",
        f"export BAIRCONDOR_REPO_DIR={repo_dir}",
        f"export BAIRCONDOR_JOBNAME={jobname}",
        f"export BAIRCONDOR_NUM_GPUS={resources['gpus']}",
        "",
    ]

    if conda.get("env"):
        conda_base = conda.get("conda_base") or ""
        parts += [
            f'source "{conda_base}/etc/profile.d/conda.sh"',
            f'conda activate "{conda["env"]}"',
            "",
        ]

    # skip the literal "--" separator that precedes the user command
    parts += [
        '# drop the "--" separator before the user command',
        'if [[ "${1:-}" == "--" ]]; then shift; fi',
        "",
        'exec "$@"',
    ]
    return "\n".join(parts) + "\n"
