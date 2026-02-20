#!/usr/bin/env python3
"""GPU smoke test for baircondor.

Verifies the full pipeline: submission -> scheduling -> GPU allocation -> output.

Usage:
    # Standalone (local GPU, no condor)
    python examples/gpu_test.py

    # Via baircondor (real submission)
    baircondor submit --scratch /tmp --gpus 1 -- python examples/gpu_test.py

    # With conda env
    baircondor submit --scratch /tmp --gpus 1 --conda-env myenv -- python examples/gpu_test.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def check_cuda():
    if not torch.cuda.is_available():
        print("FAIL: No CUDA device visible to PyTorch.", file=sys.stderr)
        print("Check that the node has a GPU and drivers are loaded.", file=sys.stderr)
        sys.exit(1)


def collect_device_info():
    return {
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "gpu_count": torch.cuda.device_count(),
    }


def run_matmul_benchmark(size=4096):
    device = torch.device("cuda:0")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    # Synchronize before timing to flush any pending ops
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b  # noqa: F841
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed


def build_result(device_info, elapsed, matrix_size):
    return {
        **device_info,
        "matrix_size": matrix_size,
        "matmul_seconds": round(elapsed, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_result(result):
    run_dir = os.environ.get("BAIRCONDOR_RUN_DIR")
    if run_dir:
        out_path = Path(run_dir) / "result.json"
    else:
        out_path = Path("result.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return out_path


def print_summary(result, out_path):
    print("=== baircondor GPU smoke test ===")
    print(f"Device:       {result['device_name']}")
    print(f"CUDA:         {result['cuda_version']}")
    print(f"PyTorch:      {result['pytorch_version']}")
    print(f"GPU count:    {result['gpu_count']}")
    print(f"Matrix size:  {result['matrix_size']}x{result['matrix_size']}")
    print(f"Matmul time:  {result['matmul_seconds']:.4f}s")
    print(f"Result saved: {out_path}")
    print("PASS")


def main():
    check_cuda()
    device_info = collect_device_info()

    matrix_size = 4096
    elapsed = run_matmul_benchmark(matrix_size)

    result = build_result(device_info, elapsed, matrix_size)
    out_path = save_result(result)
    print_summary(result, out_path)


if __name__ == "__main__":
    main()
