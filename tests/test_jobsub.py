"""Tests for job.sub generation."""

import pytest

from baircondor.templates import write_job_sub


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path


@pytest.fixture
def repo_dir(tmp_path):
    return tmp_path / "repo"


def _sub_text(run_dir, repo_dir, resources, jobname="myjob", omit_zero=True):
    write_job_sub(run_dir, repo_dir, resources, jobname, omit_zero)
    return (run_dir / "job.sub").read_text()


def test_required_fields_present(run_dir, repo_dir):
    resources = {"gpus": 1, "cpus": 6, "mem": "24G", "disk": None}
    text = _sub_text(run_dir, repo_dir, resources)
    assert "universe = vanilla" in text
    assert f"initialdir = {repo_dir}" in text
    assert "executable = /bin/bash" in text
    assert "getenv = True" in text
    assert f"output = {run_dir}/stdout.txt" in text
    assert f"error  = {run_dir}/stderr.txt" in text
    assert f"log    = {run_dir}/condor.log" in text
    assert "request_cpus = 6" in text
    assert "request_memory = 24G" in text
    assert "request_gpus = 1" in text


def test_gpu_omitted_when_zero_and_flag_true(run_dir, repo_dir):
    resources = {"gpus": 0, "cpus": 4, "mem": "8G", "disk": None}
    text = _sub_text(run_dir, repo_dir, resources, omit_zero=True)
    assert "request_gpus" not in text


def test_gpu_included_when_zero_and_flag_false(run_dir, repo_dir):
    resources = {"gpus": 0, "cpus": 4, "mem": "8G", "disk": None}
    text = _sub_text(run_dir, repo_dir, resources, omit_zero=False)
    assert "request_gpus = 0" in text


def test_disk_included_when_set(run_dir, repo_dir):
    resources = {"gpus": 1, "cpus": 6, "mem": "24G", "disk": "50G"}
    text = _sub_text(run_dir, repo_dir, resources)
    assert "request_disk = 50G" in text


def test_disk_omitted_when_not_set(run_dir, repo_dir):
    resources = {"gpus": 1, "cpus": 6, "mem": "24G", "disk": None}
    text = _sub_text(run_dir, repo_dir, resources)
    assert "request_disk" not in text


def test_job_batch_name(run_dir, repo_dir):
    resources = {"gpus": 1, "cpus": 6, "mem": "24G", "disk": None}
    text = _sub_text(run_dir, repo_dir, resources, jobname="testjob")
    assert '+JobBatchName = "testjob"' in text
