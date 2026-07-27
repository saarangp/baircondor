"""Tests for the setup wizard: GPU memory table and config writing."""

import subprocess

import yaml

from baircondor.setup import detect_gpu_memory, lookup_gpu_memory, write_config

# ── lookup_gpu_memory ─────────────────────────────────────────────────────────


def test_lookup_v100():
    assert lookup_gpu_memory("Tesla V100-SXM2-32GB") == "48G"


def test_lookup_quadro_rtx_8000():
    assert lookup_gpu_memory("Quadro RTX 8000") == "48G"


def test_lookup_h100_nvl():
    assert lookup_gpu_memory("NVIDIA H100 NVL") == "96G"


def test_lookup_l40s():
    assert lookup_gpu_memory("NVIDIA L40S") == "48G"


def test_lookup_rtx_2080ti():
    assert lookup_gpu_memory("NVIDIA GeForce RTX 2080 Ti") == "24G"


def test_lookup_unknown_gpu_returns_default():
    assert lookup_gpu_memory("NVIDIA A100-SXM4-80GB") == "24G"
    assert lookup_gpu_memory("") == "24G"


# ── detect_gpu_memory ─────────────────────────────────────────────────────────


def test_detect_gpu_memory_calls_nvidia_smi(monkeypatch):
    def mock_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(cmd, 0, stdout="Tesla V100-SXM2-32GB\n", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert detect_gpu_memory() == "48G"


def test_detect_gpu_memory_uses_first_gpu(monkeypatch):
    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="NVIDIA H100 NVL\nNVIDIA H100 NVL\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert detect_gpu_memory() == "96G"


def test_detect_gpu_memory_unknown_model(monkeypatch):
    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="NVIDIA A100-SXM4-80GB\n", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert detect_gpu_memory() == "24G"


# ── write_config ──────────────────────────────────────────────────────────────


def test_write_config_creates_file(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    write_config({"defaults": {"scratch": "/tmp/scratch"}}, cfg_path)
    assert cfg_path.exists()


def test_write_config_creates_parent_dirs(tmp_path):
    cfg_path = tmp_path / "a" / "b" / "config.yaml"
    write_config({"defaults": {"scratch": "/tmp/scratch"}}, cfg_path)
    assert cfg_path.exists()


def test_write_config_content_is_valid_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    data = {
        "defaults": {"scratch": "/raid/user", "mem_gpu": "48G", "mem_cpu_only": "8G"},
        "conda": {"conda_base": "/raid/user/miniconda3"},
    }
    write_config(data, cfg_path)
    loaded = yaml.safe_load(cfg_path.read_text())
    assert loaded == data
