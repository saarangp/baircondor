# Lab cluster (paste into your experiment repo's AGENTS.md or CLAUDE.md)

Rules for running this repo's jobs on the lab's condor servers. General baircondor rules
live in the baircondor repo's `AGENTS.md`; read that once per session.

## Which machine is this session on?

Run `hostname` before proposing any command.

- **<LAPTOP or other machine>**: no GPU, no data. Prepare code and hand over one
  copy-pasteable line per launch. Never launch from here.
- **<SUBMIT HOST>** (`REDLRADADM____`): condor commands work; GPUs usable directly with
  `CUDA_VISIBLE_DEVICES` pinned. Repo checkout: `<shared path, e.g. /REDLRADADM35839/home/$USER/<repo>>`.

## Where the data is

| Data | Machine | Path |
|---|---|---|
| <pretraining shards> | REDLRADADM____ | <path as seen from that machine> |
| <eval cache> | REDLRADADM____ | <path> |

Jobs that read this data must run on that machine (`--machine`).

## Launch recipes

Committed in `.baircondor.yaml`; do not retype them. `baircondor profiles` lists them.

```yaml
defaults:
  scratch: /REDLRADADM____/home/${USER}/condor-scratch    # exported path the target mounts
profiles:
  pretrain:
    machine: REDLRADADM____
    gpus: 2
    cpus: 8
    mem: 256G                 # from previous runs; the default is far too low
    conda_env: <pretraining env>
    conda_base: /home/${USER}/anaconda3   # as it is on the execute host
  eval:
    machine: REDLRADADM____
    gpus: 1
    cpus: 8
    mem: 64G
    conda_env: <eval env>     # a different env than pretraining; mixing them fails fast
    conda_base: /home/${USER}/anaconda3
```

```
baircondor gpus --machine REDLRADADM____ --need 2     # paste the table, then:
baircondor submit --profile pretrain -- python run_pretraining.py --config configs/<cfg>.py
baircondor submit --profile eval --after <CLUSTER> -- bash benchmarking/phase_<x>.sh
```

## Before any launch

1. Get the user's go-ahead for anything that spends GPU time.
2. `baircondor gpus` on the target; 3 GPUs per server per user, condor plus direct.
