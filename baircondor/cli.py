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
