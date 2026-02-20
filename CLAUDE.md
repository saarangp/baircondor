# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`baircondor` is a Python CLI tool that wraps `condor_submit` to standardize HTCondor job submission for ~20 lab users across multiple GPU servers. Each server has its own schedd; users SSH into their target server and submit there.

Two subcommands:
- `baircondor submit` — non-interactive batch job submission
- `baircondor interactive` — interactive shell allocation via `condor_submit -interactive`

## Architecture

```
baircondor/
  cli.py        — argparse entrypoint; dispatches to submit/interactive handlers
  submit.py     — core logic: run dir creation, file generation, condor_submit invocation
  config.py     — YAML config loading, built-in defaults, CLI flag override merging
  templates.py  — generates job.sub and run.sh content
  meta.py       — generates meta.json (resources, git info, timestamps)
tests/
  test_rundir.py    — run dir naming and creation
  test_jobsub.py    — job.sub field validation
  test_meta.py      — meta.json key validation
pyproject.toml
spec.md         — full v1 specification (source of truth for behavior)
```

## Key Design Decisions

**Run directory layout:**
```
{scratch}/{runs_subdir}/{user}/{jobname}/{YYYYMMDD_HHMMSS}_{shortid}/
  job.sub       condor submit description
  run.sh        wrapper script (executable); actual condor executable
  meta.json     reproducibility metadata
  stdout.txt    condor stdout
  stderr.txt    condor stderr
  condor.log    condor event log
```

**job.sub:** `initialdir` is set to `repo_dir` (cwd at invocation), so relative paths work like interactive runs. `executable = /bin/bash`, `arguments = <abs_path_run.sh> -- <command...>`. `request_gpus` is omitted when `--gpus 0` (controlled by `condor.omit_request_gpus_when_zero`, default true).

**run.sh:** Sets `set -euo pipefail`, exports `BAIRCONDOR_*` env vars, optionally activates conda, then `exec "$@"`.

**Config search order:** `--config PATH` > `~/.config/baircondor/config.yaml` > built-in defaults. CLI flags always win.

**Memory defaults (v1):** No per-GPU scaling — single string defaults (`mem_gpu = "24G"`, `mem_cpu_only = "8G"`). Memory strings are passed through verbatim.

**CPU defaults:** `cpus = gpus * cpus_per_gpu` (default 6 per GPU) or `cpus_cpu_only` (default 4) when `--gpus 0`.

**No v1 features:** No `transfer_input_files`, no `requirements` expressions, no Docker, no retry logic, no sweep/array abstraction.

## Dependencies

- `pyyaml` — config file parsing
- `pytest` — tests (dev only)
- Python 3.11+

## Dev Commands

```bash
pip install -e ".[dev]"    # install with dev deps
pytest tests/              # run all tests
pytest tests/test_jobsub.py -v  # run specific test file

# smoke test (dry run, no condor needed)
baircondor submit --scratch /tmp --gpus 0 --dry-run -- echo hello
```
