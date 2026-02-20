# baircondor v1 Spec (Lean Condor Submit Helper)

## Context / Assumptions

- Lab has **multiple GPU servers** (e.g., 6). Each server currently has its **own schedd** (one schedd per server, for now).
- Users typically **SSH into the target GPU server** and submit jobs there. Jobs execute on that same server.
- Shared filesystem `/radraid` exists but is **slow**; servers also have **fast local storage** (varies by server, e.g. `/raid/$USER` or `/home/$USER`).
- Condor supports **vanilla** universe, **`request_gpus`** (vanilla), and **`condor_submit -interactive`**.
- No preemption; no automatic sweeps (for now).
- Must support both **GPU** and **CPU-only** jobs. CPU-only is represented as `--gpus 0`.
- Memory strings should be passed through **verbatim** (e.g. `64G`, `12000MB`).

---

## Goals (v1)

- Standardize Condor submission UX across ~20+ users.
- Make each submission self-contained and debuggable by generating a per-run directory containing:
  - Condor submit description
  - execution wrapper script
  - Condor logs
  - metadata
- Keep launcher/framework agnostic (Lightning, torchrun, accelerate, plain python, etc.).
- Use **repo directory as working directory** (so relative paths behave like interactive runs).

---

## Non-goals (v1)

- No cross-server scheduling / “submit anywhere; Condor picks host”.
- No sweep/array abstraction.
- No automatic `torchrun` / Lightning integration.
- No scratch autodetect.
- No repo shipping / `transfer_input_files`.
- No retry / auto-resubmit logic beyond Condor defaults.
- Docker support is optional (can be added later); conda activation is supported.

---

## CLI Commands

### 1) `baircondor submit`

Submits a non-interactive job.

**Usage:**
```bash
baircondor submit [options] -- <command...>
```

**Key behavior:**
- Determine `repo_dir = os.getcwd()` at invocation time.
- Create a unique `run_dir` under `--scratch`.
- Generate `job.sub`, `run.sh`, `meta.json` in `run_dir`.
- Submit with `condor_submit <run_dir>/job.sub`.
- Print job id and key paths.

#### Required options
- `--scratch PATH`  
  Example: `/raid/$USER` or `/home/$USER`.

#### Common options
- `--jobname NAME` (default: derived from repo directory name)
- `--gpus N` (default: 1; allow 0)
- `--cpus N` (default computed; see Defaults)
- `--mem MEM` (default computed; see Defaults; pass through verbatim)
- `--disk DISK` (optional; pass through verbatim if used)
- `--tag TAG` (optional string appended to run dir name)
- `--project NAME` (optional grouping folder)
- `--runs-subdir NAME` (default: `condor-runs`)
- `--dry-run` (generate files and print intended `condor_submit` command; do not submit)

#### Conda options
- `--conda-env ENVNAME` (optional)
- `--conda-base PATH` (optional; otherwise from config)

---

### 2) `baircondor interactive`

Submits an interactive allocation using `condor_submit -interactive` and drops into a shell under that allocation.

**Usage:**
```bash
baircondor interactive [options]
```

**Key behavior:**
- Same run_dir + generated artifacts as `submit`.
- Instead of running a provided command, it runs an interactive shell:
  - `/bin/bash -i`
- Uses `condor_submit -interactive <run_dir>/job.sub`.

#### Required options
- `--scratch PATH`

#### Common options
Same as `submit`:
- `--jobname` (default: `interactive`)
- `--gpus` (default: 1; allow 0)
- `--cpus`, `--mem`, `--disk`
- `--tag`, `--project`, `--runs-subdir`
- `--dry-run`

#### Conda options
Same as `submit`:
- `--conda-env`, `--conda-base`

---

## Directory Layout

### Run directory path template

```
{scratch}/{runs_subdir}/{user}/{jobname}/{YYYYMMDD_HHMMSS}_{shortid}/
```

Where:
- `user` is `$USER` or equivalent from system/user info.
- `shortid` is a random short identifier (e.g. 6–10 chars) to prevent collisions.

### Run directory contents

- `job.sub` — generated submit description
- `run.sh` — generated wrapper used as the actual execution entrypoint
- `meta.json` — metadata for reproducibility/debugging
- `stdout.txt` — Condor stdout
- `stderr.txt` — Condor stderr
- `condor.log` — Condor event log

---

## Execution Model

### Working directory must be the repo directory

- `initialdir` in `job.sub` is set to `repo_dir`.
- This ensures relative paths in user code behave like interactive runs.

### Wrapper script: `run.sh`

`run.sh` responsibilities:
- `set -euo pipefail`
- Export standardized environment variables:

  - `baircondor_RUN_DIR=<run_dir>`
  - `baircondor_REPO_DIR=<repo_dir>`
  - `baircondor_JOBNAME=<jobname>`
  - `baircondor_NUM_GPUS=<gpus>`

- If `--conda-env` is provided:
  - `source <conda_base>/etc/profile.d/conda.sh`
  - `conda activate <env>`

- `exec "$@"` to run the provided command.

`run.sh` must be written to `run_dir` and marked executable (`chmod +x`).

### Interactive shell behavior

`baircondor interactive` uses the same wrapper and executes:
- `/bin/bash -i`

So:
- conda activation (if configured) occurs before the interactive shell starts
- baircondor_* variables are available in the shell

---

## Condor Submit File: `job.sub`

### Common required fields (batch + interactive)

- `universe = vanilla`
- `initialdir = <repo_dir>`
- `executable = /bin/bash`
- `getenv = True`
- `output = <run_dir>/stdout.txt`
- `error  = <run_dir>/stderr.txt`
- `log    = <run_dir>/condor.log`
- `request_cpus = <cpus>`
- `request_memory = <mem>`

### GPU request behavior

- If `gpus > 0`:
  - include: `request_gpus = <gpus>`
- If `gpus == 0`:
  - by default **omit** the `request_gpus` line (configurable)

Config key:
- `condor.omit_request_gpus_when_zero` (default: `true`)

### Batch mode arguments

For `baircondor submit`:
- `arguments = <abs_path_to_run.sh> -- <user_command_and_args...>`

Note: the wrapper should ignore the literal `--` token, or the generator should omit it—implementation choice. The goal is to preserve argument boundaries reliably.

### Interactive mode arguments

For `baircondor interactive`:
- `arguments = <abs_path_to_run.sh> -- /bin/bash -i`

### Recommended nice-to-haves (include if supported)

- `+JobBatchName = "<jobname>"`

### Explicitly *not used* in v1

- No `transfer_input_files`
- No `should_transfer_files`
- No `requirements` / machine selection logic (since one schedd per server and user submits on the server they want)

---

## Defaults & Configuration

### Config file (recommended)

Support config file with this search order:
1. `--config PATH` (optional)
2. `~/.config/baircondor/config.toml` (preferred) or `config.yaml`
3. built-in defaults

CLI flags override config.

### Default values

- `defaults.runs_subdir = "condor-runs"`
- `defaults.cpus_per_gpu = 6`
- `defaults.cpus_cpu_only = 4` (or reuse `cpus_per_gpu`; configurable)
- `defaults.mem_per_gpu = "24G"`
- `defaults.mem_cpu_only = "8G"` (configurable)
- `condor.omit_request_gpus_when_zero = true`
- `conda.conda_base = <required if conda-env used>` (no autodetect required; can be set via flag or config)

### Default computation rules

If user does not pass `--cpus`:
- If `gpus > 0`: `cpus = gpus * defaults.cpus_per_gpu`
- If `gpus == 0`: `cpus = defaults.cpus_cpu_only`

If user does not pass `--mem`:
- If `gpus > 0`: `mem = gpus * defaults.mem_per_gpu` (string multiplication is not possible; implement as config-driven string or compute only if mem is numeric+unit; simplest v1: require mem default be explicit and compute by repeating? Better: set a single default mem for gpu jobs, not per-gpu. Implementation choice—see note below.)
- If `gpus == 0`: `mem = defaults.mem_cpu_only`

**Implementation note (memory):**
Because memory strings are pass-through and may include units, v1 should avoid unit parsing unless desired.
Simplest robust behavior:
- Provide `defaults.mem_gpu = "24G"` and `defaults.mem_cpu_only = "8G"` (no per-gpu multiplication).
- Still allow user override via `--mem`.

(If you do want per-gpu mem scaling later, implement a small parser.)

---

## Metadata: `meta.json`

Write `meta.json` containing at least:

- `user`
- `hostname`
- `timestamp` (ISO 8601)
- `repo_dir`
- `run_dir`
- `jobname`
- `mode` (`"batch"` or `"interactive"`)
- `command` (array of strings; for interactive, store `["/bin/bash", "-i"]`)
- `resources`:
  - `gpus`, `cpus`, `mem`, `disk` (if set)
- `conda`:
  - `env` (if set)
  - `conda_base` (resolved path if used)

Optional (best-effort; do not fail if unavailable):
- `git`:
  - `is_repo` boolean
  - `commit` SHA
  - `branch` (if available)
  - `dirty` boolean

---

## UX Output Requirements

After successful submission, print:

- Job submitted: `<cluster>.<proc>` (if available from `condor_submit` output)
- Repo dir: `<repo_dir>`
- Run dir: `<run_dir>`
- Stdout: `<run_dir>/stdout.txt`
- Stderr: `<run_dir>/stderr.txt`
- Condor log: `<run_dir>/condor.log`
- Reproduce: `condor_submit <run_dir>/job.sub`

For interactive:
- After the interactive session ends, print:
  - `Interactive session ended. Run dir: <run_dir>`

---

## Error Handling

- If `--scratch` does not exist or is not writable: exit non-zero with clear message.
- If `--conda-env` is provided but `conda_base` cannot be resolved: exit non-zero with clear message.
- If `condor_submit` fails: print stderr/stdout and exit non-zero.
- `--dry-run` should never call Condor; it only generates and prints what would happen.

---

## Implementation Notes (keep it lean)

- Suggested language: Python.
- Suggested CLI parsing: `argparse` (no dependencies) or `typer` (if allowed).
- Do not deeply parse classads. Just generate files and invoke:
  - `condor_submit` (batch)
  - `condor_submit -interactive` (interactive)

- Preserve user command arguments exactly:
  - store as JSON array in meta
  - avoid shell-escaping bugs by passing args through wrapper where possible

- `run.sh` should be simple, robust bash.

---

## Minimal Test Plan

### Unit tests
- Run dir naming and creation.
- Generated `job.sub` contains required fields.
- `request_gpus` omitted when `--gpus 0` (default behavior).
- `meta.json` contains expected keys.

### Integration (optional)
- `--dry-run` creates all files and prints expected output.

### Manual acceptance checklist
- Batch CPU-only:
  - `baircondor submit --scratch /home/$USER --gpus 0 --cpus 4 --mem 8G -- python -c "print('hi')"`
- Batch GPU:
  - `baircondor submit --scratch /raid/$USER --gpus 1 --cpus 8 --mem 32G -- python -c "import torch; print(torch.cuda.is_available())"`
- Interactive GPU:
  - `baircondor interactive --scratch /raid/$USER --gpus 1 --cpus 8 --mem 32G`
  - verify `nvidia-smi` works
- Interactive with conda:
  - `baircondor interactive --scratch /raid/$USER --gpus 0 --conda-env myenv --cpus 4 --mem 8G`
  - verify `which python` and imports

---

## Open Questions (optional for v1; can defer)

1. Should disk requests be included (`request_disk`)?
2. Should we include `+JobBatchName` always?
3. Should we implement Docker mode in v1.1 (local registry, bind mounts)?