"""Tests for the bundled agent skill and its installer."""

import pytest

from baircondor.skill import SKILL_SRC, install_skill


def test_install_skill(tmp_path):
    assert (SKILL_SRC / "SKILL.md").read_text().startswith("---\nname: baircondor\n")
    dirs = [tmp_path / "claude" / "skills", tmp_path / "codex" / "skills"]
    links = install_skill(dirs)
    assert all(link.is_symlink() and (link / "SKILL.md").is_file() for link in links)
    assert install_skill(dirs) == links  # idempotent
    (tmp_path / "taken" / "baircondor").mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        install_skill([tmp_path / "taken"])
