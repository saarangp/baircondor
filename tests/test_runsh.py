"""Tests for run.sh generation (conda base resolved at runtime on the exec host)."""

from baircondor.templates import write_run_sh


def _run_sh_text(run_dir, conda):
    write_run_sh(run_dir, run_dir / "repo", "myjob", {"gpus": 1, "cpus": 4, "mem": "24G"}, conda)
    return (run_dir / "run.sh").read_text()


def test_no_conda_env_has_no_conda_logic(tmp_path):
    text = _run_sh_text(tmp_path, {})
    assert "conda activate" not in text
    assert 'exec "$@"' in text


def test_conda_env_activation(tmp_path):
    text = _run_sh_text(tmp_path, {"env": "train", "conda_base": "/opt/conda"})
    assert 'CONDA_BASE="/opt/conda"' in text and 'ENV_NAME="train"' in text
    # missing env fails fast with the list of envs, before activation
    assert text.index("not found on") < text.index('conda activate "$ENV_NAME"')
    assert "available envs: base" in text
    # no explicit base: resolved at runtime on the execute host
    text = _run_sh_text(tmp_path, {"env": "train", "conda_base": None})
    assert 'CONDA_BASE=""' in text and "conda info --base" in text and "$HOME/miniconda3" in text
    assert '"$ENV_NAME" != */*' in text  # path-style envs skip the existence check
