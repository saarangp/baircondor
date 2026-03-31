"""baircondor CLI entrypoint."""

from __future__ import annotations

import argparse

from .submit import run_interactive, run_submit


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="baircondor",
        description="HTCondor job submission helper for BAIR lab GPU servers.",
    )
    parser.add_argument("--config", metavar="PATH", help="Path to config YAML file.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    _add_submit_parser(sub)
    _add_interactive_parser(sub)

    args = parser.parse_args()

    if args.subcommand == "submit":
        run_submit(args)
    elif args.subcommand == "interactive":
        run_interactive(args)


# ── subcommand parsers ────────────────────────────────────────────────────────


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scratch",
        default=None,
        metavar="PATH",
        help="Scratch directory for run dirs (default: ~/condor-scratch).",
    )
    p.add_argument(
        "--jobname", metavar="NAME", help="Job name (default: repo dir name or 'interactive')."
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        metavar="N",
        help="Number of GPUs to request (0 for CPU-only). Default: 1.",
    )
    p.add_argument(
        "--cpus",
        type=int,
        metavar="N",
        help="Number of CPUs (default: gpus * cpus_per_gpu or cpus_cpu_only).",
    )
    p.add_argument(
        "--mem", metavar="MEM", help="Memory request, passed verbatim (e.g. 32G, 12000MB)."
    )
    p.add_argument("--disk", metavar="DISK", help="Disk request, passed verbatim.")
    p.add_argument("--tag", metavar="TAG", help="Optional string appended to run dir name.")
    p.add_argument("--project", metavar="NAME", help="Optional grouping folder inside runs_subdir.")
    p.add_argument(
        "--runs-subdir",
        default=None,
        metavar="NAME",
        help="Subdirectory under scratch for all runs (default: condor-runs).",
    )
    p.add_argument("--conda-env", metavar="ENVNAME", help="Conda environment to activate.")
    p.add_argument("--conda-base", metavar="PATH", help="Conda installation base path.")
    p.add_argument(
        "--pin-submit-host",
        dest="pin_submit_host",
        action="store_true",
        default=None,
        help="Pin job to the submit host.",
    )
    p.add_argument(
        "--no-pin-submit-host",
        dest="pin_submit_host",
        action="store_false",
        default=None,
        help="Allow scheduling on any eligible host.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Generate files but do not call condor_submit."
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
        # strip leading "--" if present
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            p.error("a command is required after --")

    p.set_defaults(_validate=_validate)


def _add_interactive_parser(sub) -> None:
    p = sub.add_parser("interactive", help="Start an interactive condor shell.")
    _common_args(p)


if __name__ == "__main__":
    main()
