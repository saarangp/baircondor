"""baircondor CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console

from .config import CONFIG_PATH, get_user
from .submit import run_interactive, run_submit

_console = Console(stderr=True, soft_wrap=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="baircondor",
        description="HTCondor job submission helper for BAIR lab GPU servers.",
    )
    parser.add_argument("--config", metavar="PATH", help="Path to config YAML file.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    _add_submit_parser(sub)
    _add_interactive_parser(sub)
    _add_preflight_parser(sub)
    _add_history_parser(sub)
    _add_last_parser(sub)
    _add_gpus_parser(sub)
    _add_wait_parser(sub)
    sub.add_parser("profiles", help="List the profiles in this repo's .baircondor.yaml.")
    sub.add_parser("config", help="Print the config file path.")
    sub.add_parser("setup", help="Re-run the setup wizard.")
    _add_install_skill_parser(sub)

    args = parser.parse_args()

    if args.subcommand == "submit":
        _maybe_run_wizard(args)
        _run_clean(run_submit, args)
    elif args.subcommand == "interactive":
        _maybe_run_wizard(args)
        _run_clean(run_interactive, args)
    elif args.subcommand == "preflight":
        from .preflight import run_preflight

        _run_clean(run_preflight, args)
    elif args.subcommand == "history":
        _cmd_history(args)
    elif args.subcommand == "last":
        _cmd_last(args)
    elif args.subcommand == "gpus":
        from .gpus import run_gpus

        _run_clean(run_gpus, args)
    elif args.subcommand == "wait":
        from .wait import run_wait

        _run_clean(run_wait, args)
    elif args.subcommand == "profiles":
        _run_clean(_cmd_profiles, args)
    elif args.subcommand == "install-skill":
        from .skill import run_install_skill

        _run_clean(run_install_skill, args)
    elif args.subcommand == "config":
        print(CONFIG_PATH)
    elif args.subcommand == "setup":
        _cmd_setup()


def _run_clean(handler, args) -> None:
    """Run a submission handler, turning input errors into a clean message (no traceback)."""
    try:
        handler(args)
    except ValueError as e:
        sys.exit(f"error: {e}")


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
    from .history import HISTORY_FILE, get_entries

    cap = 50
    entries = get_entries(n=cap + 1, user=get_user(), history_file=HISTORY_FILE)

    if not entries:
        _console.print("[dim]No submissions yet.[/dim]")
        return

    if not args.plain and sys.stdout.isatty() and sys.stderr.isatty():
        from .tui import RunBrowserApp

        RunBrowserApp(entries[: args.n or 5]).run()
        return

    _print_history(entries, cap, args)


def _print_history(entries: list[dict], cap: int, args) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from rich.text import Text

    from .history import get_job_status

    has_more = len(entries) > cap
    entries = entries[:cap]
    display = entries[: args.n or 3]

    with ThreadPoolExecutor(max_workers=len(display)) as ex:
        statuses = list(ex.map(lambda e: get_job_status(e.get("cluster_id")), display))

    for entry, status in zip(display, statuses):
        ts = entry.get("timestamp", "")[:16].replace("T", " ")
        jobname = entry.get("jobname", "?")
        run_dir = entry.get("run_dir", "")
        gpus = entry.get("gpus", 0)
        command = entry.get("command", [])

        summary = Text()
        summary.append(f"[{ts}]  ", style="dim")
        summary.append(jobname, style="bold")
        summary.append("  ")
        summary.append(f"● {status}", style=_status_style(status))
        _console.print(summary)
        _console.print(f"  {run_dir}", style="dim cyan")

        if args.verbose:
            cmd_str = " ".join(command)
            if len(cmd_str) > 60:
                cmd_str = cmd_str[:57] + "..."
            _console.print(f"  gpus={gpus}  cmd: {cmd_str}", style="dim")

        _console.print()

    if has_more or len(display) < len(entries):
        total = f"{cap}+" if has_more else str(len(entries))
        _console.print(f"[dim]Showing {len(display)} of {total}. Use -n N to see more.[/dim]")


def _cmd_last(args) -> None:
    from .history import HISTORY_FILE, get_last_dirs

    dirs = get_last_dirs(n=args.n, user=get_user(), history_file=HISTORY_FILE)
    if not dirs:
        print("No submissions yet.", file=sys.stderr)
        return
    for d in dirs:
        print(d)


def _cmd_profiles(args) -> None:
    from .config import REPO_CONFIG_NAME, load_config

    cfg = load_config(getattr(args, "config", None))
    if not cfg.get("repo_config"):
        sys.exit(f"No {REPO_CONFIG_NAME} found from {Path.cwd()} up to the git root.")
    print(f"# {cfg['repo_config']}")
    profiles = cfg.get("profiles") or {}
    if not profiles:
        print("(no profiles defined)")
        return
    print(yaml.safe_dump(profiles, default_flow_style=False, sort_keys=False), end="")


def _status_style(status: str) -> str:
    from .history import STATUS_COLORS

    return STATUS_COLORS.get(status, "dim")


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
        default=None,
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
        help="Path to conda installation (e.g. /raid/$USER/miniconda3). If omitted, it is "
        "detected at runtime on the execute host (conda on PATH, else ~/anaconda3, "
        "~/miniconda3, ~/miniforge3).",
    )
    p.add_argument(
        "--machine",
        metavar="NAME",
        default=None,
        help="Pin the job to a specific execute host by name (e.g. REDLRADADM35840), "
        "regardless of where you submit from. Matches machines whose name starts with "
        "NAME, case-insensitively. Takes priority over --pin-submit-host / "
        "--no-pin-submit-host (a config condor.machine default likewise wins over "
        "--no-pin-submit-host). Paths (cwd, --scratch, --conda-base) are resolved on "
        "that host; cwd and --scratch must be shared-filesystem paths.",
    )
    p.add_argument(
        "--pin-submit-host",
        dest="pin_submit_host",
        action="store_true",
        default=None,
        help="Pin job to the server you submitted from (default: on). "
        "Ignored when --machine is given.",
    )
    p.add_argument(
        "--no-pin-submit-host",
        dest="pin_submit_host",
        action="store_false",
        default=None,
        help="Allow condor to schedule the job on any eligible host.",
    )
    p.add_argument(
        "--profile",
        metavar="NAME",
        help="Fill unset flags from this profile in the repo's .baircondor.yaml "
        "(found from the cwd up to the git root). Flags you pass explicitly still win. "
        "See `baircondor profiles`.",
    )
    p.add_argument(
        "--sub-line",
        dest="sub_lines",
        action="append",
        metavar="'KEY = VALUE'",
        help="Extra line appended verbatim to job.sub; repeatable. Example: "
        "--sub-line 'require_gpus = DeviceUuid != \"39392cfc-...\"' to avoid a faulty GPU.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate run dir and files but do not call condor_submit. "
        "The only option that never submits.",
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
        "--check",
        action="store_true",
        help="Run a preflight job on the --machine target first, then SUBMIT if it passes. "
        "Aborts only if the conda env is missing there, the cwd doesn't resolve, or its git "
        "commit differs from your local checkout. This is a gate, not a dry run; use "
        "--dry-run to generate files without submitting.",
    )
    p.add_argument(
        "--after",
        metavar="CLUSTER",
        help="Wait (on this host) until condor cluster CLUSTER finishes with exit 0, then "
        "submit. If it is held, removed, or fails, nothing is submitted. Blocks the shell; "
        "run under nohup for long chains.",
    )
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


def _add_preflight_parser(sub) -> None:
    p = sub.add_parser(
        "preflight",
        help="Run a tiny CPU-only job on a machine to report its conda envs "
        "and how your cwd resolves there (git commit, existence).",
    )
    p.add_argument(
        "--machine",
        required=True,
        metavar="NAME",
        help="Machine to inspect (e.g. REDLRADADM35840). Same matching as submit --machine.",
    )
    p.add_argument(
        "--scratch",
        default=None,
        metavar="PATH",
        help="Root for the preflight's run dir; must be a shared /HOSTNAME/... path "
        "for cross-machine checks (same rule as submit).",
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
        help="Also require this conda env to exist on the machine; exit non-zero with the "
        "env list if it does not.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SECS",
        help="Max seconds to wait for the preflight job to finish (default: 300).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the preflight job files but do not submit.",
    )


def _add_history_parser(sub) -> None:
    p = sub.add_parser(
        "history",
        help="Browse recent job submissions (interactive TUI in a terminal).",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="Print the plain listing instead of the interactive browser "
        "(automatic when output is piped).",
    )
    p.add_argument(
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Number of entries to show (default: 5 interactive, 3 plain).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show GPUs and command in addition to the default fields.",
    )


def _add_gpus_parser(sub) -> None:
    p = sub.add_parser(
        "gpus",
        help="Audit GPU usage against the 3-per-server cap: who holds each GPU, via "
        "condor or a direct process, and how many you could still take.",
    )
    p.add_argument(
        "--machine",
        metavar="NAME",
        help="Machine to audit (default: this host). Remote machines show condor claims "
        "only, which is the full picture there since they are not SSH-able.",
    )
    p.add_argument(
        "--need",
        type=int,
        default=1,
        metavar="N",
        help="GPUs you intend to take; exit 1 if that would exceed the cap (default: 1).",
    )
    p.add_argument("--json", action="store_true", help="Print the audit as JSON.")


def _add_wait_parser(sub) -> None:
    p = sub.add_parser(
        "wait",
        help="Block until a condor job leaves the queue. Exit 0 on success, the job's exit "
        "code on failure, 3 if held (prints HoldReason), 4 if removed, 2 if unknown.",
    )
    p.add_argument(
        "cluster",
        nargs="?",
        default="last",
        metavar="CLUSTER",
        help="Cluster id, or 'last' for your most recent submission (default).",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=30,
        metavar="SECS",
        help="Poll interval (default: 30).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SECS",
        help="Give up after this many seconds with exit 5 (default: wait forever).",
    )


def _add_install_skill_parser(sub) -> None:
    p = sub.add_parser(
        "install-skill",
        help="Symlink the bundled agent skill into ~/.claude/skills and ~/.codex/skills.",
    )
    p.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        metavar="PATH",
        help="Skill directory to install into (repeatable). Default: both agents' dirs.",
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
