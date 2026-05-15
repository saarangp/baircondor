# baircondor

Kinda like submitit but for condor (and also way simpler)

A small CLI wrapper around `condor_submit` for BAIR lab GPU servers.

## Installation

```bash
pip install -e .
```

## Usage

```bash
baircondor setup                              # first-time setup (auto-runs on first submit too)
baircondor submit --gpus N -- your command   # submit a GPU batch job
baircondor interactive --gpus 1              # interactive shell with a GPU
baircondor history                           # recent submissions
baircondor last                              # path to most recent run dir (shell-composable)
```

That's it for most use cases. Everything else is optional.

---

<details>
<summary><b>First-time setup</b></summary>

Run `baircondor setup` (or just submit — it auto-triggers if no config exists).

The wizard detects your conda base and GPU type, then asks you to confirm:

```
Scratch path [~/condor-scratch]: /raid/myuser
Conda base path [/raid/myuser/miniconda3]:
Memory for GPU jobs [48G]:
Memory for CPU-only jobs [8G]:
```

This writes `~/.config/baircondor/config.yaml`. Edit it later, or re-run `baircondor setup` to redo it.

To edit the config directly:
```bash
$EDITOR $(baircondor config)
```

</details>

<details>
<summary><b>Finding your runs</b></summary>

After submitting, use `baircondor history` to see recent jobs with live status:

```
[2026-05-15 14:23]  myproject  ● running
  /raid/myuser/condor-runs/myuser/myproject/20260515_142301_abc123

[2026-05-14 09:11]  eval-run  ● done
  /raid/myuser/condor-runs/myuser/eval-run/20260514_091145_xyz789
```

Options: `-n N` (show N entries, default 3), `-v` (also show GPUs and command).

For shell use, `baircondor last` prints just the path:

```bash
tail -f $(baircondor last)/stderr.txt
ls $(baircondor last -n 3)
```

</details>

<details>
<summary><b>Common submit patterns</b></summary>

**GPU batch job:**
```bash
baircondor submit --gpus 2 -- python pretraining.py --config config.py
```

**CPU-only batch job:**
```bash
baircondor submit --gpus 0 -- python eval.py --checkpoint ckpt.pt
```

**Interactive shell:**
```bash
baircondor interactive --gpus 1 --mem 32G
```

**Dry run** (check job.sub without submitting):
```bash
baircondor submit --gpus 1 --dry-run -- python train.py
```

**Tagged run** (adds a label to the run dir name):
```bash
baircondor submit --gpus 1 --tag smoke-test -- python examples/gpu_test.py
```

**Project grouping** (extra folder level):
```bash
baircondor submit --gpus 1 --project eegfm --jobname pretrain -- python train.py
```

</details>

<details>
<summary><b>Run directory layout</b></summary>

Every submission creates a timestamped directory:

```
<scratch>/<runs_subdir>/$USER/[<project>/]<jobname>/<YYYYMMDD_HHMMSS>_<shortid>[_<tag>]/
  job.sub         HTCondor submit description
  run.sh          wrapper script executed by condor
  meta.json       git commit, resources, timestamp
  stdout.txt      job stdout
  stderr.txt      job stderr
  condor.log      condor event log
```

`initialdir` in `job.sub` is set to your cwd at submission time, so relative paths work exactly as they do interactively.

Environment variables available inside your job:

| Variable | Value |
|---|---|
| `BAIRCONDOR_RUN_DIR` | Absolute path to the run directory |
| `BAIRCONDOR_REPO_DIR` | Your repo directory (cwd at submission) |
| `BAIRCONDOR_JOBNAME` | The job name |
| `BAIRCONDOR_NUM_GPUS` | Number of GPUs requested |

```python
run_dir = Path(os.environ.get("BAIRCONDOR_RUN_DIR", "."))
torch.save(model.state_dict(), run_dir / "checkpoint.pt")
```

</details>

<details>
<summary><b>CLI reference</b></summary>

All flags work for both `submit` and `interactive`:

| Flag | Default | Description |
|---|---|---|
| `--scratch PATH` | `~/condor-scratch` | Root directory for run dirs |
| `--gpus N` | `1` | GPUs to request; `0` = CPU-only |
| `--cpus N` | `4 per GPU` or `4` | CPUs to request |
| `--mem MEM` | `24G` / `8G` (CPU-only) | Memory, passed verbatim (e.g. `48G`, `12000MB`) |
| `--disk DISK` | *(omitted)* | Disk request, passed verbatim |
| `--jobname NAME` | current dir name | Label for the job and run dir path |
| `--project NAME` | *(omitted)* | Grouping folder in the run dir path |
| `--tag TAG` | *(omitted)* | Appended to run dir: `..._<tag>/` |
| `--runs-subdir NAME` | `condor-runs` | Subdirectory under scratch |
| `--conda-env ENV` | *(omitted)* | Conda env to activate before running |
| `--conda-base PATH` | auto-detected | Path to conda installation |
| `--pin-submit-host` | `true` | Pin job to this server |
| `--no-pin-submit-host` | | Let condor schedule on any eligible host |
| `--dry-run` | `false` | Generate files only; don't submit |
| `--config PATH` | `~/.config/baircondor/config.yaml` | Config file override |

</details>

<details>
<summary><b>Config file reference</b></summary>

`~/.config/baircondor/config.yaml` — full options with defaults:

```yaml
defaults:
  scratch: ~/condor-scratch
  runs_subdir: condor-runs
  cpus_per_gpu: 4
  cpus_cpu_only: 4
  mem_gpu: "24G"
  mem_cpu_only: "8G"

condor:
  omit_request_gpus_when_zero: true
  pin_submit_host: true

conda:
  conda_base: null    # auto-detected if omitted
```

CLI flags always override the config file.

</details>

<details>
<summary><b>Python API</b></summary>

```python
from baircondor import CondorConfig, submit

cfg = CondorConfig(gpus=2, mem="32G", conda_env="train", project="eegfm")
run_dir = submit(["python", "train.py", "--lr", "1e-4"], condor=cfg)
```

`CondorConfig` is a pydantic model — embed it in your own experiment configs:

```python
class ExperimentConfig(BaseModel):
    model: dict
    condor: CondorConfig

config = ExperimentConfig(
    model={"name": "EEGLEJEPA"},
    condor=CondorConfig(gpus=1, mem="32G", conda_env="train"),
)
run_dir = submit(["python", "train.py"], condor=config.condor)
```

See `examples/python_api_patterns.py` for sweep and self-submit patterns.

</details>

<details>
<summary><b>Debugging failed jobs</b></summary>

```bash
tail -f $(baircondor last)/stderr.txt   # watch stderr live
cat $(baircondor last)/condor.log       # condor-level events
cat $(baircondor last)/job.sub          # verify resources and command
```

To reproduce locally:
```bash
bash $(baircondor last)/run.sh -- python train.py --lr 1e-4
```

</details>

<details>
<summary><b>Development</b></summary>

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
