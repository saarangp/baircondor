"""Tests for meta.json generation."""

import json

from baircondor.meta import write_meta


def test_meta_contents(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resources = {"gpus": 1, "cpus": 4, "mem": "24G", "disk": None}
    write_meta(
        run_dir,
        tmp_path,
        "job",
        "batch",
        ["python", "t.py"],
        resources,
        {"env": "e"},
        profile="eval",
    )
    meta = json.loads((run_dir / "meta.json").read_text())
    assert {
        "user",
        "hostname",
        "timestamp",
        "repo_dir",
        "run_dir",
        "jobname",
        "mode",
        "command",
        "git",
    } <= set(meta)
    assert meta["mode"] == "batch" and meta["command"] == ["python", "t.py"]
    assert meta["resources"] == {"gpus": 1, "cpus": 4, "mem": "24G"}  # None dropped
    assert meta["conda"] == {"env": "e"} and meta["profile"] == "eval"
    assert meta["git"]["is_repo"] is False  # tmp_path is not a git repo
