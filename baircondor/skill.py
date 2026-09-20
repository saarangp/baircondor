"""Install the bundled agent skill for Claude Code and Codex."""

from __future__ import annotations

from pathlib import Path

SKILL_NAME = "baircondor"
SKILL_SRC = Path(__file__).parent / "skills" / SKILL_NAME
DEFAULT_DIRS = (Path.home() / ".claude" / "skills", Path.home() / ".codex" / "skills")


def run_install_skill(args) -> None:
    dirs = [Path(d).expanduser() for d in (getattr(args, "dirs", None) or DEFAULT_DIRS)]
    for target in install_skill(dirs):
        print(target)


def install_skill(dirs: list[Path], src: Path = SKILL_SRC) -> list[Path]:
    """Symlink the skill directory into each skills dir; returns the links made or kept."""
    if not (src / "SKILL.md").is_file():
        raise ValueError(f"bundled skill not found at {src}")
    links = []
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        link = d / SKILL_NAME
        if link.is_symlink() and link.resolve() == src.resolve():
            links.append(link)
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"{link} already exists and is not a link to {src}; remove it first")
        link.symlink_to(src.resolve(), target_is_directory=True)
        links.append(link)
    return links
