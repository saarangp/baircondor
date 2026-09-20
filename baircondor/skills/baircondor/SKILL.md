---
name: baircondor
description: Launch, watch, and debug GPU jobs on the lab's HTCondor servers with baircondor. Load before any GPU run on the cluster, whether through condor or a direct process, and when a job is held, failed, or missing.
---

# baircondor

`baircondor` is the only way to submit condor jobs here. Never write a `job.sub` or call
`condor_submit` yourself.

## Before any GPU run

1. **Ask first.** A run that costs GPU time needs the user's go-ahead unless they already
   gave it. Questions about the cluster (which machine, which env, where data lives) are
   questions for the user, not experiments.
2. **Audit the cap.** The rule is 3 GPUs per user per server, condor and direct combined:

   ```
   baircondor gpus --need N            # this host
   baircondor gpus --machine NAME      # an exec node
   ```

   Paste its table in your message. If it says `DO NOT LAUNCH`, do not launch.
3. **Pin direct runs.** `CUDA_VISIBLE_DEVICES=<free idx>` on every direct command, using
   only indices the audit listed as free. Trainers take every visible GPU otherwise.

## Launching

- Prefer the repo's recipe: `baircondor submit --profile NAME -- <command>`. See
  `baircondor profiles`. Flags you pass override the profile.
- Exec nodes need a shared cwd and `--scratch` (a `/MACHINE/home/USER/...` path) and the
  execute host's `--conda-base`.
- `--dry-run` is the only flag that does not submit. `--check` runs checks and then
  submits if they pass.
- Chain: `baircondor submit --after CLUSTER ...` waits for that job to succeed first.

## After launching

- `baircondor wait [CLUSTER|last]` blocks until the job leaves the queue. Exit 3 means
  held: it prints `HoldReason`. Do not release or resubmit; report it to the user.
- `baircondor history --plain -n 5 -v` and `baircondor last` are the agent-safe views.
  The TUI and `baircondor interactive` need a terminal.
- Debug from `$(baircondor last)/stderr.txt`, then `condor.log`, then `job.sub`, before
  any resubmit.
