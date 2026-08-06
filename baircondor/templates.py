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
    machine: str | None = None,
    require_gpus: str | None = None,
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
            machine,
            require_gpus,
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
    machine: str | None,
    require_gpus: str | None = None,
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

    # An explicit --machine target wins over the default submit-host pin (and over
    # --no-pin-submit-host). The prefix-anchored, case-insensitive regexp tolerates
    # short-name vs FQDN (e.g. "^redlradadm35840" matches "redlradadm35840.ad...").
    if machine:
        lines.append(f'requirements = regexp("^{machine}", Machine, "i")')
    elif pin_submit_host:
        lines.append(f'requirements = (toLower(Machine) == "{submit_host.lower()}")')

    gpus = resources["gpus"]
    if gpus > 0:
        lines.append(f"request_gpus = {gpus}")
        if require_gpus:
            lines.append(f"require_gpus = {require_gpus}")
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
        # Resolve the conda base on the EXECUTE host at runtime, not on the submit host:
        # under --machine the two hosts have different conda installs. Honor an explicit
        # base if given, else `conda info --base`, else probe the usual $HOME locations.
        conda_base = conda.get("conda_base") or ""
        parts += [
            f'CONDA_BASE="{conda_base}"',
            'if [[ -z "$CONDA_BASE" ]]; then',
            "  if command -v conda >/dev/null 2>&1; then",
            '    CONDA_BASE="$(conda info --base)"',
            "  else",
            '    for _d in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do',
            '      if [[ -f "$_d/etc/profile.d/conda.sh" ]]; then CONDA_BASE="$_d"; break; fi',
            "    done",
            "  fi",
            "fi",
            'if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then',
            '  echo "baircondor: could not find a conda installation on $(hostname);'
            ' set --conda-base" >&2',
            "  exit 1",
            "fi",
            f'ENV_NAME="{conda["env"]}"',
            # fail fast with the available envs instead of conda's generic activate error
            # (path-style envs with "/" are activated as-is and skip the existence check)
            'if [[ "$ENV_NAME" != */* && "$ENV_NAME" != base'
            ' && ! -d "$CONDA_BASE/envs/$ENV_NAME" && ! -d "$HOME/.conda/envs/$ENV_NAME" ]]; then',
            "  echo \"baircondor: conda env '$ENV_NAME' not found on $(hostname).\" >&2",
            '  echo "baircondor: available envs: base'
            " $(ls -1 \"$CONDA_BASE/envs\" 2>/dev/null | tr '\\n' ' ')\" >&2",
            "  exit 1",
            "fi",
            'source "$CONDA_BASE/etc/profile.d/conda.sh"',
            'conda activate "$ENV_NAME"',
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
