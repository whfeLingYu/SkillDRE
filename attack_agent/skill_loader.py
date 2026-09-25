"""Load original skill packages from SkillsBench tasks.

A SkillsBench skill lives at:

    tasks/<task-id>/environment/skills/<skill-name>/
        SKILL.md
        scripts/*.py
        ...

The task-level instruction.md sits at `tasks/<task-id>/instruction.md`. We
walk upward from the skill directory to discover it.

The whole skill directory is loaded as a flat path -> content map, so the
prompt can show — and the agent can edit — any file in the skill tree, not
just `SKILL.md` and `scripts/`.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import SkillPackage
from .utils import read_text


# Text-like extensions we surface to the agent. Anything else (images,
# binary data, .stl, etc.) is preserved on disk but not loaded into the
# prompt-side `files` map.
TEXT_SUFFIXES = {
    ".md", ".py", ".sh", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".txt", ".csv", ".html", ".css",
    ".sql", ".rst",
}
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}
MAX_FILE_BYTES = 200_000   # avoid choking the prompt on huge files


def discover_skill_dirs(root: Path) -> list[Path]:
    """Return every directory under `root` that holds a SKILL.md file."""
    return sorted({p.parent for p in root.rglob("SKILL.md") if p.is_file()})


def _find_task_dir(skill_dir: Path) -> Path | None:
    """Walk upward to find the enclosing `tasks/<task-id>` directory."""
    for parent in [skill_dir, *skill_dir.parents]:
        if parent.parent.name == "tasks":
            return parent
        if (parent / "instruction.md").exists() and (parent / "task.toml").exists():
            return parent
    return None


def _find_instruction_md(skill_dir: Path) -> Path | None:
    task_dir = _find_task_dir(skill_dir)
    if task_dir is not None:
        candidate = task_dir / "instruction.md"
        if candidate.exists():
            return candidate
    for parent in [skill_dir, *skill_dir.parents]:
        candidate = parent / "instruction.md"
        if candidate.exists():
            return candidate
    return None


def _collect_files(skill_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(skill_dir).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        rel = path.relative_to(skill_dir).as_posix()
        files[rel] = read_text(path)
    return files


def load_skill_package(skill_dir: Path) -> SkillPackage:
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.exists():
        raise FileNotFoundError(f"SKILL.md not found under {skill_dir}")

    instruction_path = _find_instruction_md(skill_dir)
    return SkillPackage(
        name=skill_dir.name,
        skill_dir=skill_dir.resolve(),
        files=_collect_files(skill_dir),
        instruction_path=instruction_path.resolve() if instruction_path else None,
        instruction_md=read_text(instruction_path) if instruction_path else "",
        task_dir=_find_task_dir(skill_dir),
    )


def load_skills_from_root(root: Path, max_skills: int | None = None) -> list[SkillPackage]:
    dirs = discover_skill_dirs(root)
    if max_skills is not None:
        dirs = dirs[:max_skills]
    return [load_skill_package(d) for d in dirs]


def render_skill_package_for_prompt(
    pkg: SkillPackage,
    max_file_chars: int | None = 4000,
    *,
    include_instruction: bool = True,
) -> str:
    """Compose a compact textual view of the original skill, suitable for an LLM prompt."""
    parts: list[str] = []
    parts.append(f"## Original skill: `{pkg.name}/`")

    if include_instruction and pkg.instruction_md:
        parts.append(f"### Task instruction.md (this is what the downstream "
                     f"agent will be asked to do)\n\n{pkg.instruction_md.strip()}")

    if not pkg.files:
        parts.append("(skill directory is empty)")
        return "\n\n".join(parts)

    parts.append("### Skill directory tree\n```\n" + _render_tree(pkg.files) + "\n```")

    parts.append("### File contents")
    for rel, content in pkg.files.items():
        lang = _fence_lang(rel)
        if max_file_chars is None or len(content) <= max_file_chars:
            snippet = content
        else:
            snippet = content[:max_file_chars] + "\n... [truncated]"
        parts.append(f"#### `{rel}`\n```{lang}\n{snippet}\n```")
    return "\n\n".join(parts)


def _render_tree(files: dict[str, str]) -> str:
    return "\n".join(sorted(files.keys()))


def _fence_lang(rel: str) -> str:
    ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
    return {
        "py": "python", "sh": "bash", "js": "javascript", "ts": "typescript",
        "md": "markdown", "json": "json", "yaml": "yaml", "yml": "yaml",
        "toml": "toml", "html": "html", "css": "css", "sql": "sql",
    }.get(ext, "")
