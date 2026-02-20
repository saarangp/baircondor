# Examples

## `gpu_test.py`

GPU smoke test that verifies the full baircondor pipeline: submission, scheduling, GPU allocation, and output.

The script checks CUDA availability, collects device info, runs a 4096x4096 matmul benchmark, and saves `result.json` to the run directory.

**Requires:** PyTorch with CUDA support.

### Usage

```bash
# Standalone (local GPU, no condor)
python examples/gpu_test.py

# Via baircondor
baircondor submit --gpus 1 -- python examples/gpu_test.py

# Disable submit-host pinning (allow scheduling on any eligible host)
baircondor submit --no-pin-submit-host --gpus 1 -- python examples/gpu_test.py

# With a conda env
baircondor submit --gpus 1 --conda-env myenv -- python examples/gpu_test.py
# (Uses --conda-base from config, or auto-detects from your shell conda setup)

# Dry run (no GPU needed, just checks command wiring)
baircondor submit --gpus 1 --dry-run -- python examples/gpu_test.py
```

### Output

When run inside a baircondor job, results are saved to `$BAIRCONDOR_RUN_DIR/result.json`. When run standalone, `result.json` is written to the current directory.

```
=== baircondor GPU smoke test ===
Device:       NVIDIA A100-SXM4-80GB
CUDA:         12.1
PyTorch:      2.1.0
GPU count:    1
Matrix size:  4096x4096
Matmul time:  0.0842s
Result saved: /tmp/runs/user/gpu-test/20260219_150000_abc123/result.json
PASS
```
