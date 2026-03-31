# baircondor

Kinda like submitit but for condor (and also way simpler)

A small CLI wrapper around `condor_submit` that standardizes HTCondor job submission across UCLA BAIR lab GPU servers.

Each server runs its own HTCondor schedd. SSH into the server you want and submit there.
baircondor pins the job to that exact submit host so run-dir paths stay local and predictable.

## Installation

```bash
pip install -e ".[dev]"   # with test deps
pip install -e .           # without
```

## Quick start

**Batch job (GPU):**
```bash
baircondor submit --gpus 2 -- python pretraining.py --config config.py
```

**Batch job (CPU-only):**
```bash
baircondor submit --gpus 0 -- python eval.py --checkpoint ckpt.pt
```

**Interactive shell (with GPU):**
```bash
baircondor interactive --gpus 1 --mem 32G
```

**Dry run (generate files, don't submit):**
```bash
baircondor submit --gpus 0 --dry-run -- echo hello
```

**Tagged run dir (helpful for smoke tests / sweeps):**
```bash
baircondor submit --gpus 1 --tag smoke-test -- python examples/gpu_test.py
```

**Allow scheduling on any eligible host:**
```bash
baircondor submit --no-pin-submit-host --gpus 1 -- python examples/gpu_test.py
```

## What it does

For every submission, baircondor creates a timestamped run directory under your scratch path (`~/condor-scratch` by default, auto-created if missing):

```
~/condor-scratch/condor-runs/$USER/<jobname>/20260219_161635_abc123/
  job.sub       condor submit description
  run.sh        wrapper script executed by condor
  meta.json     git commit, resources, timestamp
  stdout.txt    your job's stdout
  stderr.txt    your job's stderr
  condor.log    condor event log
```

`initialdir` in `job.sub` is set to your current working directory (the repo), so relative paths in your scripts behave exactly like they do interactively.
By default, `requirements` in `job.sub` is set to the submit host reported by `hostname -f` (`toLower(Machine) == "<submit-host>"`), so jobs run on the same server where you submitted. Use `--no-pin-submit-host` to disable host pinning.

## CLI reference

All options work for both `submit` and `interactive`:

| Flag | Default | Description |
|---|---|---|
| `--scratch PATH` | `~/condor-scratch` | Scratch directory for run dirs |
| `--gpus N` | `1` | GPUs to request; `0` = CPU-only |
| `--cpus N` | `gpus × 4` or `4` | CPUs to request |
| `--mem MEM` | `24G` or `8G` | Memory, passed verbatim (e.g. `48G`, `12000MB`) |
| `--disk DISK` | *(omitted)* | Disk request, passed verbatim |
| `--jobname NAME` | repo dir name | Label for the job and run dir |
| `--project NAME` | *(omitted)* | Extra grouping folder inside `condor-runs/` |
| `--tag TAG` | *(omitted)* | Appended to run dir name |
| `--conda-env ENV` | *(omitted)* | Conda env to activate before running |
| `--conda-base PATH` | from config or auto-detect | Path to conda installation |
| `--pin-submit-host` | from config (`true`) | Pin job to submit host |
| `--no-pin-submit-host` | *(off)* | Disable host pinning |
| `--dry-run` | `false` | Generate files only; don't call condor |
| `--config PATH` | `~/.config/baircondor/config.yaml` | Config file override |

## Config file

Create `~/.config/baircondor/config.yaml` to set lab-wide or personal defaults:

```yaml
defaults:
  scratch: /raid/myuser  # override: use fast local storage instead of ~/condor-scratch
  runs_subdir: condor-runs
  cpus_per_gpu: 8        # override: 8 CPUs per GPU instead of default 4
  mem_gpu: "48G"         # override: 48G default for GPU jobs

conda:
  conda_base: /raid/saarang/miniconda3

condor:
  omit_request_gpus_when_zero: true
  pin_submit_host: true
```

CLI flags always override the config file.

If you want host pinning off by default, set:

```yaml
condor:
  pin_submit_host: false
```

## Environment variables available in your job

| Variable | Value |
|---|---|
| `BAIRCONDOR_RUN_DIR` | absolute path to this run's directory |
| `BAIRCONDOR_REPO_DIR` | your repo directory (cwd at submission) |
| `BAIRCONDOR_JOBNAME` | the job name |
| `BAIRCONDOR_NUM_GPUS` | number of GPUs requested |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Pre-commit hooks (autoflake → isort → black) run automatically on commit after:
```bash
pip install pre-commit
pre-commit install
```


## Python API

The Python API is a low-level submission primitive for experiment repos. `baircondor`
handles job submission and resource metadata; your training repo still owns config
validation, config mutation, sweep generation, and queue orchestration.

The public API stays intentionally small:
- `CondorConfig`
- `submit(command, condor=..., **kwargs)`
- `interactive(condor=..., **kwargs)`

This means `run_many`-style helpers are not built into baircondor in this PR.
The intended pattern is that caller code generates validated config variants and
calls `submit(...)` repeatedly.

### Pattern 1: embed condor settings in a validated config

If your launcher already validates experiment configs with pydantic, keep
submission metadata next to the rest of the experiment config:

```python
from pydantic import BaseModel

from baircondor import CondorConfig


class ExperimentConfig(BaseModel):
    data: dict
    model: dict
    trainer_config: dict
    logger_config: dict
    condor: CondorConfig


config = ExperimentConfig(
    data={"batch_size": 256, "num_workers": 4},
    model={"name": "EEGLEJEPA"},
    trainer_config={"devices": [0]},
    logger_config={"group": "v2_lejepa_pretraining"},
    condor=CondorConfig(
        gpus=1,
        mem="32G",
        conda_env="train",
        jobname="lejepa-pretrain",
        project="eegfm",
    ),
)
```

Since `CondorConfig` is a pydantic model, unknown `condor` fields like `gps: 2`
raise immediately rather than being ignored silently.

### Pattern 2: self-submit one validated config

The common pattern is: load config, validate it, and optionally re-invoke your
existing training entrypoint through `submit(...)`.

```python
import argparse
from pathlib import Path

from baircondor import submit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--condor", action="store_true")
    args = parser.parse_args()

    config = load_and_validate_config(args.config)  # returns your pydantic config object

    if args.condor:
        run_dir = submit(
            ["python", "run_pretraining.py", str(args.config)],
            condor=config.condor,
            tag=config.logger_config.group,
        )
        print(f"Submitted. Run dir: {run_dir}")
        return

    train(config)
```

### Pattern 3: caller-owned sweep loops

For sweeps, keep the loop in your experiment repo. Start from one validated base
config, clone or update the fields you care about, derive per-run metadata such
as `jobname` and `tag`, and call `submit(...)` for each variant.

```python
from copy import deepcopy
from itertools import product

from baircondor import submit


run_dirs = []
for lr, batch_size in product([1e-4, 3e-4], [128, 256]):
    variant = deepcopy(base_config)
    variant.optimizer.kwargs.lr = lr
    variant.data.batch_size = batch_size

    suffix = f"lr{lr:g}-bs{batch_size}"
    run_dir = submit(
        ["python", "run_pretraining_from_config.py", "--config", f"generated/{suffix}.py"],
        condor=variant.condor,
        jobname="lejepa-pretrain",
        project="eegfm",
        tag=suffix,
    )
    run_dirs.append(run_dir)
```

In practice, experiment repos usually materialize each variant into a generated
config artifact before calling `submit(...)`. baircondor does not do that
serialization for you.

Both `submit()` and `interactive()` return a `pathlib.Path` to the created run
directory, so caller code can track queued runs immediately.

See `examples/python_api_patterns.py` for a concrete reference example.


## Status

**What works today:**
- `submit` and `interactive` subcommands with full run-dir generation (`job.sub`, `run.sh`, `meta.json`)
- Python API: `CondorConfig` pydantic model + `submit()` / `interactive()` functions
- Config cascade (built-in defaults → `~/.config/baircondor/config.yaml` → CLI flags)
- `--dry-run` mode (no condor needed)
- Conda env activation
- Git metadata capture in `meta.json`

**What needs testing on real clusters:**
- End-to-end `condor_submit` on each lab server (GPU jobs, CPU-only jobs, interactive sessions)
- Stress testing: concurrent submissions, large scratch dirs, edge-case job names

**Not yet implemented:**
- Docker universe support (container jobs, local registry, bind mounts)
