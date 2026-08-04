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
baircondor preflight --machine NAME          # check conda envs + repo state on another machine
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

After submitting, `baircondor history` opens an interactive browser of your recent jobs:
arrow keys to move, **enter** to open a run (live-tailing stdout, with tabs for stderr and
condor.log; `1`/`2`/`3` switch tabs), **esc**/**left** to go back, **k** to kill the job,
**y** to copy the run dir path, **r** to refresh statuses, **q** to quit.

When output is piped (or with `--plain`) it prints a plain listing instead:

```
[2026-05-15 14:23]  myproject  ● running
  /raid/myuser/condor-runs/myuser/myproject/20260515_142301_abc123

[2026-05-14 09:11]  eval-run  ● done
  /raid/myuser/condor-runs/myuser/eval-run/20260514_091145_xyz789
```

Plain-mode options: `-n N` (show N entries, default 3), `-v` (also show GPUs and command).

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

**Target a specific machine** (the host that holds your data/cache, regardless of where you submit from):
```bash
baircondor submit --machine REDLRADADM35839 --gpus 1 -- python train.py
```

> **Cross-machine runs need a shared filesystem.** The job's run dir and cwd must resolve to the
> same absolute path on both the submit and target hosts. On the shared NFS namespace, pass an
> explicit shared path for `--scratch` (e.g. `/REDLRADADM35839/home/$USER/condor-scratch`) and
> `cd` into one before submitting — the default `~/condor-scratch` and node-local `/raid` are not
> visible on other hosts and the job will hold with a "No such file … stdout.txt" error. Hosts
> that don't export their filesystem (e.g. PHI-restricted nodes) aren't reachable this way and
> would need condor file transfer, which baircondor doesn't do yet.

**What resolves where.** The job runs entirely on the target machine, so every path is
interpreted from its point of view:

| Thing | Where it lives / resolves |
|---|---|
| Your command, GPUs, data reads | Execute on the `--machine` host |
| Repo cwd (`initialdir`) and `--scratch` | Captured at submit time; must be the same absolute path on both hosts (shared NFS, e.g. `/REDLRADADM35839/home/$USER/...`) |
| `--conda-base` (or auto-detection) | Resolved on the execute host — point it at that machine's conda install |

Submitting with a machine-local cwd or `--scratch` (`/home/...`, `/raid/...`, `~`) while
`--machine` targets a different host prints a warning, since those paths resolve to the
*target's* own disk and can silently run a stale clone.

**Checking a machine before you submit.** The exec nodes aren't SSH-able, so
`baircondor preflight` runs a tiny CPU-only condor job on the target and reports back:

```bash
cd /REDLRADADM35839/home/$USER/myrepo
baircondor preflight --machine REDLRADADM35840 --scratch /REDLRADADM35839/home/$USER/condor-scratch
```

It prints the machine's conda base and env list, whether your cwd exists there, and the
git commit it sees at that path (with a loud warning if it differs from your local
checkout). Results are cached under `~/.local/share/baircondor/`. Jobs that reach a
missing conda env anyway fail immediately with the list of envs that do exist on that
host, visible in the stderr tab of `baircondor history`.

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
| `--machine NAME` | *(omitted)* | Pin to a specific host by name; wins over submit-host pinning |
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
  machine: null    # pin to a specific host by name; overrides pin_submit_host

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
