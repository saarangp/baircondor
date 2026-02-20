# baircondor — Progress Tracker

## Done

- [x] Wrote `spec.md` (full v1 specification)
- [x] Created `CLAUDE.md` (project guidance for Claude Code)
- [x] Confirmed implementation choices: argparse, YAML config, Python 3.11+
- [x] `pyproject.toml` — package metadata, entry points, dependencies
- [x] `baircondor/__init__.py`
- [x] `baircondor/config.py` — YAML config loading, built-in defaults, CLI flag merging
- [x] `baircondor/templates.py` — `job.sub` and `run.sh` generation
- [x] `baircondor/meta.py` — `meta.json` generation (git info, resources, timestamps)
- [x] `baircondor/submit.py` — run dir creation, file writing, `condor_submit` invocation
- [x] `baircondor/cli.py` — `baircondor submit` and `baircondor interactive` subcommands
- [x] `tests/test_rundir.py` — run dir naming and creation (5 tests)
- [x] `tests/test_jobsub.py` — `job.sub` required fields, GPU omission behavior (6 tests)
- [x] `tests/test_meta.py` — `meta.json` required keys (5 tests)
- [x] Dry-run smoke test passes: `baircondor submit --scratch /tmp --gpus 0 --dry-run -- echo hello`

## In Progress

- python api

## TODO

### Acceptance (manual, requires lab server with condor) ✓
- [x] Batch CPU-only: `baircondor submit --scratch /home/$USER --gpus 0 --cpus 4 --mem 8G -- python -c "print('hi')"`
- [x] Batch GPU: `baircondor submit --scratch /raid/$USER --gpus 1 --cpus 8 --mem 32G -- python -c "import torch; print(torch.cuda.is_available())"`
- [x] Interactive GPU: `baircondor interactive --scratch /raid/$USER --gpus 1 --cpus 8 --mem 32G`; verify `nvidia-smi`
- [x] Interactive with conda: `baircondor interactive --scratch /raid/$USER --gpus 0 --conda-env myenv --cpus 4 --mem 8G`; verify `which python`

### Nice-to-haves / v1.1
- [x] `~/.config/baircondor/config.yaml` example file / template (`examples/config.yaml`)
- [ ] Shell completion (argcomplete or manual)
- [ ] Docker support
- [ ] Python API / pydantic integration: expose a `submit()` function and a `CondorConfig` pydantic model so jobs can be queued programmatically from Python configs (e.g. embed `condor: CondorConfig` in a project's pydantic training config and call `baircondor.submit(command=[...], condor=cfg.condor)`). Key refactor: decouple `resolve_resources` from argparse `Namespace` so it accepts plain kwargs.
