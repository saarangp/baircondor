"""Tests for job.sub generation."""

from pathlib import Path

import pytest

from baircondor.submit import _condor_escape_arg, _patch_args
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


def test_arguments_placeholder_present(run_dir, repo_dir):
    """Template should emit the placeholder that _patch_args will replace."""
    resources = {"gpus": 1, "cpus": 6, "mem": "24G", "disk": None}
    text = _sub_text(run_dir, repo_dir, resources)
    assert "arguments = __ARGS_PLACEHOLDER__" in text


# --- HTCondor argument escaping tests ---


class TestCondorEscapeArg:
    def test_simple_arg(self):
        assert _condor_escape_arg("hello") == "hello"

    def test_arg_with_space(self):
        assert _condor_escape_arg("/my dir/run.sh") == "'/my dir/run.sh'"

    def test_arg_with_tab(self):
        assert _condor_escape_arg("a\tb") == "'a\tb'"

    def test_arg_with_double_quote(self):
        assert _condor_escape_arg('say "hi"') == """'say ""hi""'"""

    def test_arg_with_single_quote(self):
        assert _condor_escape_arg("it's") == "'it''s'"

    def test_double_dash(self):
        assert _condor_escape_arg("--") == "--"

    def test_path_no_spaces(self):
        assert _condor_escape_arg("/usr/local/bin/python") == "/usr/local/bin/python"


class TestPatchArgs:
    def test_simple_command(self, tmp_path):
        job_sub = tmp_path / "job.sub"
        job_sub.write_text("arguments = __ARGS_PLACEHOLDER__\n")
        run_sh = Path("/home/user/runs/run.sh")
        _patch_args(job_sub, run_sh, ["python", "train.py"])
        text = job_sub.read_text()
        assert text == 'arguments = "/home/user/runs/run.sh -- python train.py"\n'

    def test_path_with_spaces(self, tmp_path):
        job_sub = tmp_path / "job.sub"
        job_sub.write_text("arguments = __ARGS_PLACEHOLDER__\n")
        run_sh = Path("/my dir/runs/run.sh")
        _patch_args(job_sub, run_sh, ["echo", "hello"])
        text = job_sub.read_text()
        assert text == """arguments = "'/my dir/runs/run.sh' -- echo hello"\n"""

    def test_args_with_double_quotes(self, tmp_path):
        job_sub = tmp_path / "job.sub"
        job_sub.write_text("arguments = __ARGS_PLACEHOLDER__\n")
        run_sh = Path("/home/user/run.sh")
        _patch_args(job_sub, run_sh, ["echo", 'say "hi"'])
        text = job_sub.read_text()
        assert 'arguments = "/home/user/run.sh -- echo \'say ""hi""\'"\n' == text

    def test_preserves_surrounding_content(self, tmp_path):
        job_sub = tmp_path / "job.sub"
        job_sub.write_text("universe = vanilla\narguments = __ARGS_PLACEHOLDER__\ngetenv = True\n")
        run_sh = Path("/home/user/run.sh")
        _patch_args(job_sub, run_sh, ["python", "test.py"])
        text = job_sub.read_text()
        assert "universe = vanilla\n" in text
        assert "getenv = True\n" in text
        assert 'arguments = "/home/user/run.sh -- python test.py"' in text
