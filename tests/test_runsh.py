"""Tests for run.sh generation (conda base resolved at runtime on the exec host)."""

import pytest

from baircondor.templates import write_run_sh


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path


@pytest.fixture
def repo_dir(tmp_path):
    return tmp_path / "repo"


def _run_sh_text(run_dir, repo_dir, conda, gpus=1):
    resources = {"gpus": gpus, "cpus": 4, "mem": "24G", "disk": None}
    write_run_sh(run_dir, repo_dir, "myjob", resources, conda)
    return (run_dir / "run.sh").read_text()


def test_no_conda_env_has_no_conda_logic(run_dir, repo_dir):
    text = _run_sh_text(run_dir, repo_dir, {})
    assert "conda activate" not in text
    assert "conda info --base" not in text


def test_explicit_base_used_verbatim(run_dir, repo_dir):
    text = _run_sh_text(run_dir, repo_dir, {"env": "train", "conda_base": "/opt/conda"})
    # the base is seeded non-empty, so the runtime fallback is skipped on the exec host
    assert 'CONDA_BASE="/opt/conda"' in text
    assert 'source "$CONDA_BASE/etc/profile.d/conda.sh"' in text
    assert 'ENV_NAME="train"' in text
    assert 'conda activate "$ENV_NAME"' in text


def test_no_base_resolves_at_runtime(run_dir, repo_dir):
    text = _run_sh_text(run_dir, repo_dir, {"env": "train", "conda_base": None})
    # empty seed, then runtime fallbacks
    assert 'CONDA_BASE=""' in text
    assert "conda info --base" in text
    assert "$HOME/anaconda3" in text
    assert "$HOME/miniconda3" in text
    assert "$HOME/miniforge3" in text
    assert 'ENV_NAME="train"' in text
    assert 'conda activate "$ENV_NAME"' in text


def test_missing_env_fails_fast_with_available_list(run_dir, repo_dir):
    text = _run_sh_text(run_dir, repo_dir, {"env": "train", "conda_base": None})
    # the env-existence check runs before conda activate and lists what IS there
    assert '"$CONDA_BASE/envs/$ENV_NAME"' in text
    assert '"$HOME/.conda/envs/$ENV_NAME"' in text
    assert "not found on $(hostname)" in text
    assert "available envs: base" in text
    assert text.index("not found on") < text.index("conda activate")


def test_path_style_env_skips_existence_check(run_dir, repo_dir):
    # envs given as a path (containing "/") are activated as-is
    text = _run_sh_text(run_dir, repo_dir, {"env": "/opt/envs/train", "conda_base": None})
    assert '"$ENV_NAME" != */*' in text


def test_no_base_errors_when_conda_missing(run_dir, repo_dir):
    text = _run_sh_text(run_dir, repo_dir, {"env": "train", "conda_base": None})
    # a clear failure path instead of a silent bad activation
    assert "exit 1" in text
