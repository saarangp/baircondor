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
`requirements` in `job.sub` is set to the submit host (`toLower(Machine) == "<submit-host>"`), so jobs run on the same server where you submitted.

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
```

CLI flags always override the config file.

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

You can also submit jobs programmatically — useful for embedding condor config in your own pydantic training configs:

```python
from baircondor import CondorConfig, submit

# Using a CondorConfig model (embed in your own pydantic configs)
cfg = CondorConfig(gpus=2, mem="32G", conda_env="train")
run_dir = submit(["python", "train.py", "--lr", "1e-4"], condor=cfg)
print(f"Submitted. Run dir: {run_dir}")

# Or with plain kwargs
run_dir = submit(["python", "train.py"], gpus=1, dry_run=True)

# Interactive session
from baircondor import interactive
run_dir = interactive(condor=CondorConfig(gpus=1, mem="32G"))
```

Both `submit()` and `interactive()` return a `pathlib.Path` to the created run directory, so you can log it, save it to a config, or reference outputs after the job completes.

`CondorConfig` fields mirror the CLI flags (same names, same defaults). Kwargs override the model, matching the "CLI flags always win" convention.

### Example: `--condor` flag in a training script

A common pattern is to keep condor settings in your training config YAML, then add a `--condor` flag to your runner script. When the flag is set, the script submits *itself* to condor (without the flag) and exits. Otherwise it trains locally as usual.

**Config (e.g. `configs/pretrain.yaml`):**
```yaml
model:
  hidden_dim: 256
  num_layers: 4

training:
  lr: 1e-4
  epochs: 100

condor:
  gpus: 2
  mem: "32G"
  conda_env: "train"
```

**Runner script (e.g. `run_pretraining.py`):**
```python
import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--condor", action="store_true", help="Submit to condor instead of running locally")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    if args.condor:
        from baircondor import CondorConfig, submit

        condor_cfg = CondorConfig(**config["condor"])
        # Re-invoke this script without --condor, so it trains normally inside the job
        run_dir = submit(["python", __file__, str(args.config)], condor=condor_cfg)
        print(f"Submitted. Run dir: {run_dir}")
        return

    # Normal training — runs locally or inside the condor job
    train(config)
```

```bash
# Submit to condor
python run_pretraining.py configs/pretrain.yaml --condor

# Run locally (e.g. for debugging)
python run_pretraining.py configs/pretrain.yaml
```

Since `CondorConfig` is a pydantic model, it validates the condor section of your YAML at load time — typos like `gps: 2` or `mem: 32` (missing the `G`) will raise immediately rather than failing silently on the cluster.


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
