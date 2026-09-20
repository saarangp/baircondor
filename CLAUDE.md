# CLAUDE.md

@AGENTS.md

This file is for working on the baircondor code itself. `AGENTS.md` (imported above) is
for using baircondor to run experiments; it applies here too.

## Project overview

`baircondor` wraps `condor_submit` for the lab's GPU servers: a run dir per submission
(`job.sub`, `run.sh`, `meta.json`, logs), a personal config plus a committed per-repo
config with named profiles, machine pinning, preflight checks, a GPU cap audit, a
hold-aware `wait`, a history browser, and a Python API.

```
baircondor/
  cli.py        argparse entrypoint; one subparser per subcommand
  submit.py     run dir creation, file generation, condor_submit; --check, --after
  config.py     defaults < ~/.config/baircondor/config.yaml < .baircondor.yaml; profiles
  templates.py  job.sub and run.sh renderers (extra_lines for --sub-line)
  meta.py       meta.json (resources, conda, profile, git)
  preflight.py  CPU-only report job on a target machine; --check gate; env cache
  gpus.py       GPU audit: nvidia-smi + ps joined with condor slot claims
  wait.py       block on a cluster id; held/removed/blip handling
  history.py    ~/.local/share/baircondor/history.jsonl; condor_q status lookup
  tui.py        textual browser behind `history` (TTY only)
  setup.py      first-run wizard
  skill.py      installs skills/baircondor/SKILL.md for Claude Code and Codex
  api.py        CondorConfig (pydantic), submit(), interactive(), from_profile()
tests/          one file per module; parsers are tested on canned command output
```

## Design decisions

- `initialdir` is the cwd at submit time; `executable = /bin/bash`,
  `arguments = <run.sh> -- <command>`. `run.sh` exports `BAIRCONDOR_*`, activates conda
  (base resolved on the execute host), then `exec "$@"`.
- Precedence: CLI flags > `.baircondor.yaml` profile/defaults > personal config (or
  `--config PATH`) > built-in defaults. `${USER}` and `~` expand in the repo file.
- `--check` submits if the checks pass. `--dry-run` is the only no-submit option.
- `wait` treats an empty `condor_q` as an exit only after a `condor_history` record;
  held jobs return exit 3 and are never released or resubmitted by the tool.
- No `transfer_input_files`, no Docker, no retries, no DAGs.

## Dev commands

```bash
pip install -e ".[dev]"
pytest tests/ -v
baircondor submit --gpus 0 --dry-run -- echo hello     # smoke test, no condor needed
pre-commit install                                      # autoflake, isort, black
```

Keep changes small and direct; no speculative guards or wrappers. Run the tests after
every change and add one for each new behavior. Write docs in plain English.
