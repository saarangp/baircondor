"""Tests for the bundled agent skill and its installer."""

import pytest

from baircondor.skill import SKILL_SRC, install_skill


def test_bundled_skill_has_frontmatter():
    text = (SKILL_SRC / "SKILL.md").read_text()
    assert text.startswith("---\nname: baircondor\n")
    assert "baircondor gpus" in text and "baircondor wait" in text


def test_install_skill_symlinks_and_is_idempotent(tmp_path):
    dirs = [tmp_path / "claude" / "skills", tmp_path / "codex" / "skills"]
    links = install_skill(dirs)
    assert [link.name for link in links] == ["baircondor", "baircondor"]
    assert all(link.is_symlink() and (link / "SKILL.md").is_file() for link in links)
    assert install_skill(dirs) == links  # second run keeps the same links


def test_install_skill_refuses_to_clobber(tmp_path):
    d = tmp_path / "skills"
    (d / "baircondor").mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        install_skill([d])
