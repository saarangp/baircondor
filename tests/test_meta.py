"""Tests for meta.json generation."""

import json

import pytest

from baircondor.meta import write_meta


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run"


@pytest.fixture
def repo_dir(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    return d


def _load_meta(run_dir, repo_dir, **kwargs):
    run_dir.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        jobname="myjob",
        mode="batch",
        command=["python", "train.py"],
        resources={"gpus": 1, "cpus": 6, "mem": "24G"},
        conda={},
    )
    defaults.update(kwargs)
    write_meta(run_dir, repo_dir, **defaults)
    return json.loads((run_dir / "meta.json").read_text())


def test_required_keys_present(run_dir, repo_dir):
    meta = _load_meta(run_dir, repo_dir)
    for key in (
        "user",
        "hostname",
        "timestamp",
        "repo_dir",
        "run_dir",
        "jobname",
        "mode",
        "command",
        "resources",
        "conda",
        "git",
    ):
        assert key in meta, f"missing key: {key}"


def test_resources_keys(run_dir, repo_dir):
    meta = _load_meta(run_dir, repo_dir)
    assert "gpus" in meta["resources"]
    assert "cpus" in meta["resources"]
    assert "mem" in meta["resources"]


def test_git_key_present(run_dir, repo_dir):
    meta = _load_meta(run_dir, repo_dir)
    assert "is_repo" in meta["git"]


def test_mode_values(run_dir, repo_dir):
    meta_batch = _load_meta(run_dir, repo_dir, mode="batch")
    assert meta_batch["mode"] == "batch"

    (run_dir / "meta.json").unlink()
    meta_interactive = _load_meta(
        run_dir, repo_dir, mode="interactive", command=["/bin/bash", "-i"]
    )
    assert meta_interactive["mode"] == "interactive"


def test_command_stored_as_list(run_dir, repo_dir):
    meta = _load_meta(run_dir, repo_dir, command=["python", "train.py", "--lr", "1e-3"])
    assert meta["command"] == ["python", "train.py", "--lr", "1e-3"]
