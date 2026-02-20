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

_(nothing active)_

## TODO

### Acceptance (manual, requires lab server with condor)
- [ ] Batch CPU-only: `baircondor submit --scratch /home/$USER --gpus 0 --cpus 4 --mem 8G -- python -c "print('hi')"`
- [ ] Batch GPU: `baircondor submit --scratch /raid/$USER --gpus 1 --cpus 8 --mem 32G -- python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Interactive GPU: `baircondor interactive --scratch /raid/$USER --gpus 1 --cpus 8 --mem 32G`; verify `nvidia-smi`
- [ ] Interactive with conda: `baircondor interactive --scratch /raid/$USER --gpus 0 --conda-env myenv --cpus 4 --mem 8G`; verify `which python`

### Nice-to-haves / v1.1
- [ ] `~/.config/baircondor/config.yaml` example file / template
- [ ] Shell completion (argcomplete or manual)
- [ ] Docker support
