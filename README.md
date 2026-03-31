# baircondor

Kinda like submitit but for condor (and also way simpler)

A small CLI wrapper around `condor_submit` that standardizes HTCondor job submission across UCLA BAIR lab GPU servers.

Each server runs its own HTCondor schedd. SSH into the server you want and submit there.
By default, baircondor pins the job to that exact host so run-dir paths stay local and predictable.

## Installation

```bash
pip install -e .            # minimal
pip install -e ".[dev]"     # with pytest + flake8
```

## First-time setup

baircondor works out of the box with no config file. The defaults that matter:

| Setting | Default | You may want to change if… |
|---|---|---|
| `scratch` | `~/condor-scratch` | You want runs on fast local storage (e.g. `/raid/$USER`) |
| `conda_base` | auto-detected | Auto-detection fails or you use a non-standard conda path |
| `pin_submit_host` | `true` | You want condor to schedule across multiple hosts |

To set lab-wide or personal defaults, create `~/.config/baircondor/config.yaml`:

```yaml
defaults:
  scratch: /raid/myuser          # fast local storage
  cpus_per_gpu: 8                # 8 CPUs per GPU (e.g. for dataloading workers)
  mem_gpu: "48G"                 # more memory for large models

conda:
  conda_base: /raid/myuser/miniconda3

condor:
  pin_submit_host: true          # default; set false to allow cross-host scheduling
```

CLI flags always override the config file. See `examples/config.yaml` for every available option.

## Quick start

**GPU batch job:**
```bash
baircondor submit --gpus 2 -- python pretraining.py --config config.py
```

**CPU-only batch job:**
```bash
baircondor submit --gpus 0 -- python eval.py --checkpoint ckpt.pt
```

**Interactive shell with a GPU:**
```bash
baircondor interactive --gpus 1 --mem 32G
```

**Dry run** (generate files, don't submit — no condor needed):
```bash
baircondor submit --gpus 1 --dry-run -- python train.py --lr 1e-4
```

<details>
<summary>Sample dry-run output</summary>

```
Repo dir : /home/myuser/my-project
Run dir  : /raid/myuser/condor-runs/myuser/my-project/20260219_161635_abc123
Stdout   : /raid/myuser/condor-runs/myuser/my-project/20260219_161635_abc123/stdout.txt
Stderr   : /raid/myuser/condor-runs/myuser/my-project/20260219_161635_abc123/stderr.txt
Log      : /raid/myuser/condor-runs/myuser/my-project/20260219_161635_abc123/condor.log
Reproduce: condor_submit /raid/myuser/condor-runs/myuser/my-project/20260219_161635_abc123/job.sub

[dry-run] would run: condor_submit .../job.sub
```

</details>

**Tagged run** (appends a label to the run dir name):
```bash
baircondor submit --gpus 1 --tag smoke-test -- python examples/gpu_test.py
# creates .../20260219_161635_abc123_smoke-test/
```

**Project grouping** (adds a folder level for organizing related experiments):
```bash
baircondor submit --gpus 1 --project eegfm --jobname pretrain -- python train.py
# creates .../condor-runs/$USER/eegfm/pretrain/20260219_161635_abc123/
```

**Disable host pinning** (let condor schedule on any eligible host):
```bash
baircondor submit --no-pin-submit-host --gpus 1 -- python examples/gpu_test.py
```

## What it does

For every submission, baircondor creates a timestamped run directory:

```
<scratch>/<runs_subdir>/$USER/[<project>/]<jobname>/<YYYYMMDD_HHMMSS>_<shortid>[_<tag>]/
  job.sub         HTCondor submit description
  run.sh          wrapper script executed by condor
  meta.json       git commit, resources, timestamp
  stdout.txt      job stdout (written by condor)
  stderr.txt      job stderr (written by condor)
  condor.log      condor event log
```

With the default config this looks like:
```
~/condor-scratch/condor-runs/myuser/my-project/20260219_161635_abc123/
```

**Key behaviors:**
- `initialdir` in `job.sub` is set to your current working directory (the repo), so relative paths in your scripts work exactly like they do interactively.
- `requirements` pins jobs to the submit host by default. Use `--no-pin-submit-host` or set `condor.pin_submit_host: false` in your config to let condor schedule across hosts.
- `run.sh` sets `set -euo pipefail`, exports `BAIRCONDOR_*` env vars, optionally activates conda, then `exec`s your command.

## Environment variables available in your job

Your command runs inside `run.sh`, which exports these before `exec`ing your command:

| Variable | Value |
|---|---|
| `BAIRCONDOR_RUN_DIR` | Absolute path to this run's directory |
| `BAIRCONDOR_REPO_DIR` | Your repo directory (cwd at submission time) |
| `BAIRCONDOR_JOBNAME` | The job name |
| `BAIRCONDOR_NUM_GPUS` | Number of GPUs requested |

Use `BAIRCONDOR_RUN_DIR` to save outputs (checkpoints, results) alongside the job metadata:

```python
import os
from pathlib import Path

run_dir = Path(os.environ.get("BAIRCONDOR_RUN_DIR", "."))
torch.save(model.state_dict(), run_dir / "checkpoint.pt")
```

<details>
<summary><h2>CLI reference</h2></summary>

All options work for both `submit` and `interactive`:

| Flag | Default | Description |
|---|---|---|
| `--scratch PATH` | `~/condor-scratch` | Root directory for run dirs |
| `--gpus N` | `1` | GPUs to request; `0` = CPU-only |
| `--cpus N` | `4 per GPU` or `4` | CPUs to request |
| `--mem MEM` | `24G` / `8G` (CPU-only) | Memory, passed verbatim (e.g. `48G`, `12000MB`) |
| `--disk DISK` | *(omitted)* | Disk request, passed verbatim |
| `--jobname NAME` | current dir name | Label for the job and run dir path |
| `--project NAME` | *(omitted)* | Grouping folder: `.../condor-runs/$USER/<project>/<jobname>/...` |
| `--tag TAG` | *(omitted)* | Appended to run dir: `..._<tag>/` |
| `--runs-subdir NAME` | `condor-runs` | Subdirectory under scratch |
| `--conda-env ENV` | *(omitted)* | Conda env to activate before running |
| `--conda-base PATH` | auto-detected | Path to conda installation |
| `--pin-submit-host` | `true` (from config) | Pin job to this server |
| `--no-pin-submit-host` | | Let condor schedule on any eligible host |
| `--dry-run` | `false` | Generate files only; don't call `condor_submit` |
| `--config PATH` | `~/.config/baircondor/config.yaml` | Config file override |

Run `baircondor submit --help` or `baircondor interactive --help` for full details.

</details>

<details>
<summary><h2>Config file reference</h2></summary>

Create `~/.config/baircondor/config.yaml` to set personal or lab-wide defaults.
The full set of options with their built-in defaults:

```yaml
defaults:
  scratch: ~/condor-scratch      # root dir for all run directories
  runs_subdir: condor-runs       # subdirectory under scratch
  cpus_per_gpu: 4                # CPUs per GPU (used to compute default --cpus)
  cpus_cpu_only: 4               # CPUs when --gpus 0
  mem_gpu: "24G"                 # default memory for GPU jobs
  mem_cpu_only: "8G"             # default memory for CPU-only jobs

condor:
  omit_request_gpus_when_zero: true   # don't emit request_gpus when --gpus 0
  pin_submit_host: true               # pin jobs to the server you submitted from

conda:
  conda_base: null               # path to conda install; auto-detected if omitted
```

CLI flags always override the config file. The config file overrides built-in defaults.

</details>

<details>
<summary><h2>Python API</h2></summary>

The Python API lets you submit jobs programmatically from experiment code.
baircondor handles job submission and resource metadata; your training repo owns
config validation, config mutation, sweep generation, and queue orchestration.

The public API:
- `CondorConfig` — a pydantic model for HTCondor resource settings (embeddable in your own configs)
- `submit(command, condor=..., **kwargs)` — submit a batch job, returns `Path` to run dir
- `interactive(condor=..., **kwargs)` — start an interactive session, returns `Path` to run dir

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

Since `CondorConfig` uses `extra="forbid"`, unknown fields like `gps=2`
raise immediately rather than being silently ignored.

<details>
<summary>Pattern 2: self-submit one validated config</summary>

Load config, validate it, and re-invoke your training entrypoint through `submit(...)`:

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

</details>

<details>
<summary>Pattern 3: caller-owned sweep loops</summary>

For sweeps, keep the loop in your experiment repo. Start from one validated base
config, derive per-run metadata, and call `submit(...)` for each variant:

```python
from itertools import product

from baircondor import submit


run_dirs = []
for lr, batch_size in product([1e-4, 3e-4], [128, 256]):
    variant = base_config.model_copy(deep=True)
    variant.optimizer.kwargs.lr = lr
    variant.data.batch_size = batch_size

    suffix = f"lr{lr:g}-bs{batch_size}"
    variant.condor = variant.condor.model_copy(
        update={"jobname": "lejepa-pretrain", "project": "eegfm", "tag": suffix}
    )

    # Your repo would usually write the variant to a generated config file here.
    generated_config_path = f"generated/{suffix}.yaml"
    run_dir = submit(
        ["python", "run_pretraining.py", "--config", generated_config_path],
        condor=variant.condor,
    )
    run_dirs.append(run_dir)
```

</details>

Both `submit()` and `interactive()` return a `pathlib.Path` to the created run
directory, so caller code can track queued runs immediately.

See `examples/python_api_patterns.py` for a runnable reference example.

</details>

<details>
<summary><h2>Checking job status</h2></summary>

After submitting, use standard HTCondor commands to check on your jobs:

```bash
# List your queued/running jobs
condor_q

# Detailed info for a specific job
condor_q -l <job_id>

# Check completed jobs
condor_history -limit 10

# Watch a running job's output in real-time
tail -f /raid/myuser/condor-runs/myuser/my-project/20260219_161635_abc123/stdout.txt
```

**If a job fails:**
1. Check `stderr.txt` in the run directory for your script's error output
2. Check `condor.log` for condor-level issues (eviction, memory limits, host problems)
3. Check `job.sub` to verify the resources and command look correct
4. Re-run locally with the same command to see if the error reproduces:
   ```bash
   cd /path/to/your/repo
   bash /path/to/run-dir/run.sh -- python train.py --lr 1e-4
   ```

</details>

<details>
<summary><h2>Development</h2></summary>

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Pre-commit hooks (autoflake → isort → black) run automatically on commit after:
```bash
pip install pre-commit
pre-commit install
```

</details>

## Status

**What works today:**
- `submit` and `interactive` subcommands with full run-dir generation (`job.sub`, `run.sh`, `meta.json`)
- Python API: `CondorConfig` pydantic model + `submit()` / `interactive()` functions
- Config cascade (built-in defaults → `~/.config/baircondor/config.yaml` → CLI flags)
- `--dry-run` mode (no condor needed)
- Conda env activation
- Git metadata capture in `meta.json`
- Host pinning toggle (`--pin-submit-host` / `--no-pin-submit-host`)

**What needs testing on real clusters:**
- End-to-end `condor_submit` on each lab server (GPU jobs, CPU-only jobs, interactive sessions)

**Not yet implemented:**
- Docker universe support (container jobs, local registry, bind mounts)
