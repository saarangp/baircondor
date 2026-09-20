# Changelog

## 2026-09-20

### For agents and labmates
- `AGENTS.md`: the rules for running experiments here through Claude Code or Codex,
  learned from the swm and Laya2 campaigns. `examples/AGENTS.template.md` is the cluster
  section to paste into an experiment repo. `CLAUDE.md` now imports `AGENTS.md`.
- `baircondor install-skill` symlinks a bundled skill into `~/.claude/skills` and
  `~/.codex/skills`.

### Repo config and profiles
- A committed `.baircondor.yaml` (cwd up to the git root) layers over the personal
  config and defines named `profiles`. `baircondor submit --profile NAME` fills unset
  flags from it; explicit flags still win. `baircondor profiles` lists them. `${USER}`
  and `~` expand. The profile is recorded in `meta.json`. The Python API takes
  `profile="NAME"` the same way.

### GPU cap audit
- `baircondor gpus [--machine NAME] [--need N] [--json]`: per-GPU owner and source
  (direct process or condor slot), your total against the 3-per-server cap including
  idle jobs in the queue, and exit 1 when a launch would exceed it.

### Waiting and chaining
- `baircondor wait [CLUSTER|last]`: blocks until the job leaves the queue. Exit 3 on a
  hold with `HoldReason` printed (never releases or resubmits), 4 removed, 2 unknown,
  5 timeout; a single empty `condor_q` is not treated as an exit.
- `submit --after CLUSTER`: wait for that job to succeed on the submit host, then submit.

### Smaller
- `--sub-line 'key = value'` (repeatable, also `sub_lines` in a profile) appends verbatim
  lines to `job.sub`, e.g. `require_gpus = DeviceUuid != "..."` to avoid a faulty GPU.
- `preflight --conda-env ENV` exits non-zero when the env is missing on the target.
- Output no longer wraps long paths when piped or logged.
- `--check` help and README now say it submits when the checks pass; `--dry-run` is the
  only no-submit option.

## 2026-08-04

### Interactive history browser
- `baircondor history` now opens a terminal UI showing your last 5 runs
  (`-n N` for more) with live status and the machine each job landed on.
- Press **enter** on a run to live-tail its stdout, with tabs for stderr,
  condor.log, and an info tab (resources, conda env, git commit, condor
  requirements).
- **x** cancels the selected job (with a y/n confirm), from the list or the
  detail view.
- **R** resubmits a finished or failed run: same command, resources, and
  machine pinning, in a fresh run dir.
- **y** copies the run dir path, **r** refreshes statuses, **q** quits.
- Piped output and `--plain` keep the old scriptable listing; `baircondor
  last` is unchanged.

### Cross-machine safety
- Jobs that land on a machine without the requested conda env now fail
  immediately with the list of envs that do exist there, instead of dying
  with a cryptic activation error.
- New `baircondor preflight --machine NAME`: runs a tiny CPU-only job on a
  target machine and reports its conda envs, whether your cwd exists there,
  and the git commit it sees (with a warning if it differs from your local
  checkout).
- New `baircondor submit --check`: runs that preflight automatically and only
  submits if the env exists, the cwd resolves, and the git commit matches.
- Submitting with `--machine` to a different host now warns when your cwd or
  `--scratch` is a machine-local path (`/home`, `/raid`, `~`), since those
  resolve to the target's own disk and can silently run a stale clone.
- Preflight results are cached; a later submit naming an env missing from the
  cached list gets a soft "as of <time>" warning (never blocks).

### Machine pinning
- `--machine NAME` pins a job to a specific execute host (added earlier, now
  documented properly): the README explains what resolves where, and the
  shared-path requirements for cross-machine runs.
- Conda base is now resolved at runtime on the execute host, so cross-machine
  jobs pick up the target machine's conda install automatically.

## Earlier

- `--machine` flag to pin jobs to a named execute host.
- `baircondor history` / `baircondor last`, first-run setup wizard, pretty
  stderr output with `--quiet`.
- Python API: `CondorConfig` pydantic model and `submit()` for sweeps and
  self-submitting scripts.
- `--no-pin-submit-host` to let condor schedule on any eligible host.
- Core: `submit` and `interactive` subcommands, run dir layout
  (job.sub / run.sh / meta.json / logs), YAML config with CLI overrides.
