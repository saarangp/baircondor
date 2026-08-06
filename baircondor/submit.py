"""Core submission logic: run dir creation, file generation, condor_submit."""

from __future__ import annotations

import os
import random
import re
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from .config import (
    get_user,
    load_config,
    resolve_conda,
    resolve_machine,
    resolve_pin_submit_host,
    resolve_require_gpus,
    resolve_resources,
)
from .history import append_entry
from .meta import write_meta
from .templates import write_job_sub, write_run_sh

_console = Console(stderr=True)
_PREFIX = f"[dim]{escape('[baircondor]')}[/dim]"


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        _console.print(f"{_PREFIX} {msg}")


def _get_submit_host() -> str:
    """Return the submit host exactly as reported by the local host configuration."""
    return subprocess.check_output(["hostname", "-f"], text=True).strip().lower()


def run_submit(args) -> Path:
    cfg = load_config(getattr(args, "config", None))
    resources = resolve_resources(cfg, args)
    conda = resolve_conda(cfg, args)
    pin_submit_host = resolve_pin_submit_host(cfg, args)
    machine = resolve_machine(cfg, args)

    # strip leading "--" separator that argparse REMAINDER captures
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.exit("error: a command is required after --")

    repo_dir = Path.cwd()
    submit_host = _get_submit_host()
    user = get_user()
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

    _validate_conda(conda, machine)
    _warn_unshared_paths(machine, submit_host, {"cwd": str(repo_dir), "--scratch": scratch})

    quiet = getattr(args, "quiet", False)

    if getattr(args, "check", False):
        if not machine:
            sys.exit("error: --check requires --machine (or a condor.machine config default)")
        if args.dry_run:
            _log(f"🧪 [dry-run] would run a preflight check on {machine} first", quiet)
        else:
            from .preflight import run_live_check

            run_live_check(
                machine, scratch, runs_subdir, repo_dir, submit_host, cfg, conda.get("env")
            )
    elif machine:
        from .preflight import cached_env_warning

        warning = cached_env_warning(machine, conda.get("env"))
        if warning:
            _console.print(f"[yellow]⚠ {escape(warning)}[/yellow]")

    run_dir.mkdir(parents=True, exist_ok=False)
    _log(f"📁 Created run dir: {run_dir}", quiet)

    require_gpus = resolve_require_gpus(cfg, machine, pin_submit_host, submit_host)

    run_sh = write_run_sh(run_dir, repo_dir, jobname, resources, conda)
    _log("📝 Generated run.sh", quiet)
    write_job_sub(
        run_dir,
        repo_dir,
        resources,
        jobname,
        submit_host,
        pin_submit_host,
        cfg["condor"]["omit_request_gpus_when_zero"],
        machine=machine,
        require_gpus=require_gpus,
    )
    _log("📝 Generated job.sub", quiet)
    write_meta(run_dir, repo_dir, jobname, "batch", command, resources, conda)
    _log("📝 Generated meta.json", quiet)

    job_sub = run_dir / "job.sub"
    # patch job.sub: replace $(args) placeholder with actual arguments
    _patch_args(job_sub, run_sh, command)

    _submit(
        job_sub,
        args.dry_run,
        run_dir,
        repo_dir,
        quiet,
        jobname=jobname,
        gpus=resources["gpus"],
        command=command,
        user=user,
    )

    return run_dir


def run_interactive(args) -> Path:
    cfg = load_config(getattr(args, "config", None))
    resources = resolve_resources(cfg, args)
    conda = resolve_conda(cfg, args)
    pin_submit_host = resolve_pin_submit_host(cfg, args)
    machine = resolve_machine(cfg, args)

    repo_dir = Path.cwd()
    submit_host = _get_submit_host()
    user = get_user()
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

    _validate_conda(conda, machine)
    _warn_unshared_paths(machine, submit_host, {"cwd": str(repo_dir), "--scratch": scratch})

    quiet = getattr(args, "quiet", False)
    run_dir.mkdir(parents=True, exist_ok=False)
    _log(f"📁 Created run dir: {run_dir}", quiet)

    require_gpus = resolve_require_gpus(cfg, machine, pin_submit_host, submit_host)

    command = ["/bin/bash", "-i"]
    run_sh = write_run_sh(run_dir, repo_dir, jobname, resources, conda)
    _log("📝 Generated run.sh", quiet)
    write_job_sub(
        run_dir,
        repo_dir,
        resources,
        jobname,
        submit_host,
        pin_submit_host,
        cfg["condor"]["omit_request_gpus_when_zero"],
        machine=machine,
        require_gpus=require_gpus,
    )
    _log("📝 Generated job.sub", quiet)
    write_meta(run_dir, repo_dir, jobname, "interactive", command, resources, conda)
    _log("📝 Generated meta.json", quiet)

    job_sub = run_dir / "job.sub"
    _patch_args(job_sub, run_sh, command)

    _submit_interactive(
        job_sub,
        args.dry_run,
        run_dir,
        quiet,
        jobname=jobname,
        gpus=resources["gpus"],
        user=user,
    )

    return run_dir


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_run_dir(
    scratch: str, runs_subdir: str, jobname: str, project: str | None, tag: str | None
) -> Path:
    scratch_path = Path(scratch)
    scratch_path.mkdir(parents=True, exist_ok=True)
    if not os.access(scratch_path, os.W_OK):
        sys.exit(f"error: --scratch path is not writable: {scratch}")

    user = get_user()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shortid = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    dirname = f"{timestamp}_{shortid}"
    if tag:
        dirname = f"{dirname}_{tag}"

    parts = [scratch_path, runs_subdir, user]
    if project:
        parts.append(project)
    parts += [jobname, dirname]
    return Path(*parts)


def unshared_path_warnings(
    machine: str | None, submit_host: str, paths: dict[str, str]
) -> list[str]:
    """Flag machine-local paths (/home, /raid, $HOME) for cross-machine runs.

    Under --machine the job resolves every path on the target host, so a /home or
    /raid path silently points at the target's own disk — a stale clone there runs
    instead of your code. Shared paths look like /HOSTNAME/{home,raid}/...
    """
    if not machine or submit_host.lower().startswith(machine.lower()):
        return []
    local_prefixes = ("/home/", "/raid/", str(Path.home()) + "/")
    warnings = []
    for label, path in paths.items():
        p = str(path)
        if p.startswith(local_prefixes) or p in ("/home", "/raid", str(Path.home())):
            top = "/" + p.lstrip("/").split("/", 1)[0]
            warnings.append(
                f"{label} '{p}' is machine-local; on {machine} it resolves to that host's "
                f"own {top}, so the job may run a stale copy or hold. "
                f"Use a shared /HOSTNAME/... path."
            )
    return warnings


def _warn_unshared_paths(machine: str | None, submit_host: str, paths: dict[str, str]) -> None:
    for w in unshared_path_warnings(machine, submit_host, paths):
        _console.print(f"[yellow]⚠ {escape(w)}[/yellow]")


def _validate_conda(conda: dict, machine: str | None) -> None:
    base = conda.get("conda_base")
    if not base:
        return  # no explicit base: run.sh resolves one at runtime on the execute host
    if machine:
        return  # base is interpreted on a different host; the submit host can't check it
    activate = Path(base).expanduser() / "etc" / "profile.d" / "conda.sh"
    if not activate.is_file():
        sys.exit(
            f"error: conda base '{base}' has no etc/profile.d/conda.sh (expected {activate}). "
            "Check --conda-base."
        )


def _condor_escape_arg(arg: str) -> str:
    """Escape one argument for HTCondor new-syntax arguments line.

    Rules: double-quotes are doubled (""), arguments containing spaces/tabs/single-quotes
    are wrapped in single quotes with interior single-quotes doubled ('').
    """
    if "\n" in arg or "\r" in arg:
        raise ValueError(
            "command arguments cannot contain newlines: HTCondor's arguments line "
            "cannot span multiple lines. Remove the embedded newline."
        )
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


def _submit(
    job_sub: Path,
    dry_run: bool,
    run_dir: Path,
    repo_dir: Path,
    quiet: bool = False,
    jobname: str = "",
    gpus: int = 0,
    command: list[str] | None = None,
    user: str = "",
) -> None:
    cmd = ["condor_submit", str(job_sub)]
    _log(f"🗂️  Repo dir : {repo_dir}", quiet)
    _log(f"📂 Run dir  : {run_dir}", quiet)
    _log(f"📄 Stdout   : {run_dir}/stdout.txt", quiet)
    _log(f"📄 Stderr   : {run_dir}/stderr.txt", quiet)
    _log(f"📋 Log      : {run_dir}/condor.log", quiet)
    _log(f"🔁 Reproduce: condor_submit {job_sub}", quiet)

    if dry_run:
        _log(f"🧪 [dry-run] would run: {' '.join(cmd)}", quiet)
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        _log(f"❌ condor_submit failed (exit {result.returncode})", quiet=False)
        sys.exit(result.returncode)

    m = re.search(r"submitted to cluster (\d+)", result.stdout)
    cluster_id = m.group(1) if m else None
    if cluster_id:
        _log(f"🚀 Submitted — cluster {cluster_id}", quiet)
    _log("✅ Done.", quiet)

    append_entry(run_dir, jobname, cluster_id, gpus, command or [], user)


def _submit_interactive(
    job_sub: Path,
    dry_run: bool,
    run_dir: Path,
    quiet: bool = False,
    jobname: str = "",
    gpus: int = 0,
    user: str = "",
) -> None:
    cmd = ["condor_submit", "-interactive", str(job_sub)]
    _log(f"📂 Run dir  : {run_dir}", quiet)

    if dry_run:
        _log(f"🧪 [dry-run] would run: {' '.join(cmd)}", quiet)
        return

    append_entry(run_dir, jobname, None, gpus, ["/bin/bash", "-i"], user)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    _log(f"✅ Interactive session ended. Run dir: {run_dir}", quiet)
