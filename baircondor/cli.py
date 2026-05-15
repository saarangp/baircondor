"""baircondor CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .config import CONFIG_PATH
from .submit import run_interactive, run_submit

_console = Console(stderr=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="baircondor",
        description="HTCondor job submission helper for BAIR lab GPU servers.",
    )
    parser.add_argument("--config", metavar="PATH", help="Path to config YAML file.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    _add_submit_parser(sub)
    _add_interactive_parser(sub)
    _add_history_parser(sub)
    _add_last_parser(sub)
    sub.add_parser("config", help="Print the config file path.")
    sub.add_parser("setup", help="Re-run the setup wizard.")

    args = parser.parse_args()

    if args.subcommand == "submit":
        _maybe_run_wizard(args)
        run_submit(args)
    elif args.subcommand == "interactive":
        _maybe_run_wizard(args)
        run_interactive(args)
    elif args.subcommand == "history":
        _cmd_history(args)
    elif args.subcommand == "last":
        _cmd_last(args)
    elif args.subcommand == "config":
        print(CONFIG_PATH)
    elif args.subcommand == "setup":
        _cmd_setup()


# ── setup wizard ──────────────────────────────────────────────────────────────


def _maybe_run_wizard(args) -> None:
    if getattr(args, "dry_run", False):
        return
    config_path = Path(getattr(args, "config", None) or CONFIG_PATH)
    if config_path.exists():
        return
    if not sys.stdin.isatty():
        return
    from .setup import run_wizard

    _console.print("[yellow]No config file found.[/yellow] Running first-time setup...\n")
    proceed = run_wizard(config_path)
    if not proceed:
        sys.exit(0)


def _cmd_setup() -> None:
    from .setup import run_wizard

    if CONFIG_PATH.exists():
        answer = input(f"Config already exists at {CONFIG_PATH}. Overwrite? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            sys.exit(0)
    run_wizard(CONFIG_PATH)


# ── history / last ────────────────────────────────────────────────────────────


def _cmd_history(args) -> None:
    import os

    from rich.text import Text

    from .history import HISTORY_FILE, get_entries, get_job_status

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    cap = 50
    entries = get_entries(n=cap + 1, user=user, history_file=HISTORY_FILE)

    if not entries:
        _console.print("[dim]No submissions yet.[/dim]")
        return

    overflow = len(entries) > cap
    entries = entries[:cap]
    shown = min(args.n, len(entries))
    display = entries[:shown]

    for entry in display:
        ts = entry.get("timestamp", "")[:16].replace("T", " ")
        jobname = entry.get("jobname", "?")
        cluster_id = entry.get("cluster_id")
        run_dir = entry.get("run_dir", "")
        gpus = entry.get("gpus", 0)
        command = entry.get("command", [])

        status = get_job_status(cluster_id)
        status_style = _status_style(status)

        summary = Text()
        summary.append(f"[{ts}]  ", style="dim")
        summary.append(jobname, style="bold")
        summary.append("  ")
        summary.append(f"● {status}", style=status_style)
        _console.print(summary)
        _console.print(f"  {run_dir}", style="dim cyan")

        if args.verbose:
            cmd_str = " ".join(command)
            if len(cmd_str) > 60:
                cmd_str = cmd_str[:57] + "..."
            _console.print(f"  gpus={gpus}  cmd: {cmd_str}", style="dim")

        _console.print()

    if overflow or shown < len(entries):
        total = len(entries)
        _console.print(f"[dim]Showing {shown} of {total}+ entries. Use -n N to see more.[/dim]")


def _cmd_last(args) -> None:
    import os

    from .history import HISTORY_FILE, get_last_dirs

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    dirs = get_last_dirs(n=args.n, user=user, history_file=HISTORY_FILE)
    if not dirs:
        print("No submissions yet.", file=sys.stderr)
        return
    for d in dirs:
        print(d)


def _status_style(status: str) -> str:
    return {
        "idle": "yellow",
        "running": "green",
        "done": "dim green",
        "failed": "red",
        "held": "red",
        "removed": "dim red",
    }.get(status, "dim")


# ── subcommand parsers ────────────────────────────────────────────────────────


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scratch",
        default=None,
        metavar="PATH",
        help="Root directory for run dirs (default: ~/condor-scratch). "
        "Use fast local storage like /raid/$USER for GPU servers.",
    )
    p.add_argument(
        "--jobname",
        metavar="NAME",
        help="Job name used in the run dir path and condor's JobBatchName "
        "(default: current directory name, or 'interactive').",
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        metavar="N",
        help="Number of GPUs to request. Use 0 for CPU-only jobs. Default: 1.",
    )
    p.add_argument(
        "--cpus",
        type=int,
        metavar="N",
        help="Number of CPUs. Default: 4 per GPU, or 4 for CPU-only jobs. "
        "Override in config with cpus_per_gpu / cpus_cpu_only.",
    )
    p.add_argument(
        "--mem",
        metavar="MEM",
        help="Memory request, passed verbatim to condor (e.g. 32G, 12000MB). "
        "Default: 24G for GPU jobs, 8G for CPU-only.",
    )
    p.add_argument(
        "--disk",
        metavar="DISK",
        help="Disk request, passed verbatim to condor (e.g. 10G). Omitted by default.",
    )
    p.add_argument(
        "--tag",
        metavar="TAG",
        help="String appended to the run dir name. "
        "Example: --tag smoke-test creates .../20260219_161635_abc123_smoke-test/",
    )
    p.add_argument(
        "--project",
        metavar="NAME",
        help="Grouping folder inserted into the run dir path. "
        "Example: --project eegfm creates .../condor-runs/$USER/eegfm/<jobname>/...",
    )
    p.add_argument(
        "--runs-subdir",
        default=None,
        metavar="NAME",
        help="Subdirectory under scratch for all runs (default: condor-runs).",
    )
    p.add_argument(
        "--conda-env",
        metavar="ENVNAME",
        help="Conda environment to activate before running your command.",
    )
    p.add_argument(
        "--conda-base",
        metavar="PATH",
        help="Path to conda installation (e.g. /raid/$USER/miniconda3). "
        "Auto-detected if omitted.",
    )
    p.add_argument(
        "--pin-submit-host",
        dest="pin_submit_host",
        action="store_true",
        default=None,
        help="Pin job to the server you submitted from (default: on).",
    )
    p.add_argument(
        "--no-pin-submit-host",
        dest="pin_submit_host",
        action="store_false",
        default=None,
        help="Allow condor to schedule the job on any eligible host.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate run dir and files but do not call condor_submit. "
        "Useful for checking job.sub before submitting.",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress informational output.",
    )


def _add_submit_parser(sub) -> None:
    p = sub.add_parser("submit", help="Submit a non-interactive batch job.")
    _common_args(p)
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        metavar="-- COMMAND...",
        help="Command to run (after --).",
    )

    def _validate(args):
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            p.error("a command is required after --")

    p.set_defaults(_validate=_validate)


def _add_interactive_parser(sub) -> None:
    p = sub.add_parser("interactive", help="Start an interactive condor shell.")
    _common_args(p)


def _add_history_parser(sub) -> None:
    p = sub.add_parser("history", help="Show recent job submissions.")
    p.add_argument(
        "-n",
        type=int,
        default=3,
        metavar="N",
        help="Number of entries to show (default: 3).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show GPUs and command in addition to the default fields.",
    )


def _add_last_parser(sub) -> None:
    p = sub.add_parser("last", help="Print the path(s) of the most recent run dir(s).")
    p.add_argument(
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="Number of paths to print (default: 1).",
    )


if __name__ == "__main__":
    main()
