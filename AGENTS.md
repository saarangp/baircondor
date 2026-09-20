# Working with baircondor as an agent

Rules for Claude Code, Codex, and similar tools running experiments on the lab's HTCondor
GPU servers. `README.md` documents the commands; this file says how to use them safely.
`baircondor install-skill` installs the same rules as a skill for both tools.

## The one submit path

`baircondor` is the only way to submit condor jobs. Never write a `job.sub` by hand or
call `condor_submit` directly. If you need a submit-description line baircondor does
not generate, pass it with `--sub-line 'key = value'`.

## Machines

- **Submit host**: the machine you are on (`hostname`). Condor commands work here. It
  may also have GPUs you can use directly.
- **Exec nodes**: reachable only through a condor job (`--machine NAME`). Not SSH-able.
  List them and their GPUs with `condor_status -af Machine TotalGpus GPUs_DeviceName`.
  Do not hardcode an inventory in your notes; it changes.
- **Disks are per machine.** `$HOME` and `/raid` are different disks on every host. Some
  machines export theirs to the others as `/MACHINE/home/USER` and `/MACHINE/raid/USER`.
  A job's cwd and `--scratch` must be on an exported path that the target machine also
  mounts, or the job holds with "No such file ... stdout.txt". As of 2026-09-20:

  | Machine | Exports |
  |---|---|
  | REDLRADADM14958, 14959, 23589 | home and raid |
  | REDLRADADM35838, 35839 | home only |
  | REDLRADADM35840 | nothing; its own disks are visible nowhere else |

- Condor may manage only some of a host's GPUs. On REDLRADADM23589, condor manages 4 of
  the 8 (indices 4 to 7); the others are direct-only. `baircondor gpus` shows the split.

## Ask before spending compute

A run that uses GPU time needs the user's go-ahead unless they already gave it. Questions
about the cluster (which machine holds the data, which conda env a job needs, what a
policy does) are for the user, not for experiments.

## The GPU budget: 3 per user per server

Condor claims and direct processes both count. Before any launch, run the audit and paste
its table into your message:

```
baircondor gpus --need N              # this host
baircondor gpus --machine NAME        # an exec node
```

It joins `nvidia-smi` (with process owners) and condor's slot claims, adds your idle jobs
that can land on that machine, and exits 1 with `DO NOT LAUNCH` if taking N more would
exceed the cap. Your own direct runs count too.

## Two ways to run

- **Direct** on the submit host, for short work: pin the GPUs the audit listed as free.
  Trainers take every visible GPU otherwise.

  ```
  CUDA_VISIBLE_DEVICES=2 python train.py --config configs/x.py
  ```

- **Condor**, for exec nodes, long jobs, and queued work. Prefer the repo's recipe:

  ```
  baircondor profiles                              # what the repo defines
  baircondor submit --profile eval -- bash benchmarking/phase_x.sh
  ```

  Profiles live in a committed `.baircondor.yaml` (see README, "Repo config and
  profiles"). Flags you pass explicitly override the profile. Without a profile, the
  cross-machine launch line needs all of: `--machine`, a shared `--scratch`, a shared cwd,
  `--conda-env`, `--conda-base` as it is on the execute host, and a `--mem` taken from the
  repo's previous runs (the defaults are too low for training; a memory-limit hold looks
  like "cgroup memory limit" in `HoldReason`).

## What submits and what does not

- `--dry-run` generates the run dir and files and never submits. It is the only such flag.
- `--check` runs a preflight job on the target and then submits if the checks pass. It is
  a gate, not a dry run.
- `baircondor preflight --machine NAME --conda-env ENV` inspects a machine with a tiny
  CPU-only job and exits non-zero if the env is missing there. Use it to validate.

## Agent-safe commands

- `baircondor history --plain -n 5 -v` and `baircondor last`. The TUI and
  `baircondor interactive` need a terminal; do not call them.
- `baircondor wait [CLUSTER|last]` blocks until the job leaves the queue: exit 0 on
  success, the job's exit code on failure, 3 if held (it prints `HoldReason`), 4 if
  removed. One empty `condor_q` answer is not an exit; it waits for a `condor_history`
  record.
- `baircondor submit --after CLUSTER ...` waits for that job to succeed, then submits.
  It blocks your shell, so run long chains under `nohup`.
- `tail -f $(baircondor last)/stderr.txt` to watch a job.

## Held and failed jobs

A held job never leaves the queue on its own. When `wait` reports one, stop: print the
`HoldReason`, read `$(baircondor last)/stderr.txt`, and ask the user. Do not release or
resubmit on your own; holds have several causes and repeating the launch is rarely the fix.

Debug a failed job from `$(baircondor last)/stderr.txt`, then `condor.log`, then `job.sub`
to confirm what was requested. Do that before any resubmit.

## Running several jobs in order under the cap

`--after` chains one job on another. For a serial queue that keeps you under the cap,
wait until your running jobs clear before each submit:

```
while condor_q $USER -af RequestGpus JobStatus | grep -q "^2 [12]$"; do sleep 300; done
baircondor submit --profile pretrain -- python run.py --config configs/a.py
sleep 600   # let it register before the next check
```

## Putting this in your own repo

Copy `examples/AGENTS.template.md` into your experiment repo's `AGENTS.md` or `CLAUDE.md`
and fill the placeholders. Commit a `.baircondor.yaml` with one profile per job type so
launches come from the file, not from memory.
