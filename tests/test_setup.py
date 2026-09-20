"""Tests for the setup wizard: GPU memory table and config writing."""

import subprocess

import yaml

from baircondor.setup import detect_gpu_memory, lookup_gpu_memory, write_config


def test_lookup_gpu_memory_table():
    assert lookup_gpu_memory("Tesla V100-SXM2-32GB") == "48G"
    assert lookup_gpu_memory("NVIDIA H100 NVL") == "96G"
    assert lookup_gpu_memory("NVIDIA GeForce RTX 2080 Ti") == "24G"
    assert lookup_gpu_memory("NVIDIA A100-SXM4-80GB") == "24G"  # unknown -> default
    assert lookup_gpu_memory("") == "24G"


def test_detect_gpu_memory_uses_first_gpu(monkeypatch):
    def mock_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            cmd, 0, stdout="NVIDIA H100 NVL\nTesla V100\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert detect_gpu_memory() == "96G"


def test_write_config_creates_parents_and_valid_yaml(tmp_path):
    cfg_path = tmp_path / "a" / "b" / "config.yaml"
    write_config({"defaults": {"scratch": "/tmp/scratch"}}, cfg_path)
    assert yaml.safe_load(cfg_path.read_text())["defaults"]["scratch"] == "/tmp/scratch"
