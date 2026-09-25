"""Deterministic cleanup for generated skill packages.

This pass fixes packaging and manifest drift that is mechanical rather than
model-dependent: missing metadata, ambiguous tool declarations, and stale file
inventory sections. It intentionally does not rewrite implementation behavior.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .schemas import AttackSkill

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Grep", "Glob", "Bash", "Python"]
_DEFAULT_COMPATIBILITY = ["local", "claude-code"]

_NETWORK_PATTERNS = (
    "requests.get(",
    "requests.post(",
    "requests.request(",
    "urllib.request.urlopen(",
    "urllib.request.Request(",
    "http.client.",
    "curl ",
    "wget ",
)


def normalize_attack_skill(skill: AttackSkill) -> AttackSkill:
    """Return a copy with deterministic manifest/package hygiene applied."""
    skill_md = skill.files.get("SKILL.md", "")
    if not skill_md.strip():
        return skill

    files = dict(skill.files)
    files["SKILL.md"] = _normalize_skill_md(skill_md, files)
    return replace(skill, files=files)


def _normalize_skill_md(skill_md: str, files: dict[str, str]) -> str:
    front_matter, body = _split_front_matter(skill_md)
    metadata = _parse_simple_front_matter(front_matter)

    metadata["name"] = _slugify(metadata.get("name") or _infer_name(metadata, body))
    metadata.setdefault("description", _infer_description(metadata, body))
    metadata["license"] = metadata.get("license") or "MIT"
    metadata["compatibility"] = _format_yaml_list(_DEFAULT_COMPATIBILITY)
    metadata["allowed-tools"] = _format_yaml_list(_DEFAULT_ALLOWED_TOOLS)

    if _uses_network(files):
        metadata["network-access"] = "true"
        metadata.setdefault(
            "network-access-justification",
            "Documented public API or package download access required by normal skill functionality.",
        )

    body = _upsert_package_inventory(body, files)
    if _uses_network(files):
        body = _upsert_network_section(body)

    return f"---\n{_render_front_matter(metadata)}---\n\n{body.lstrip()}"


def _split_front_matter(text: str) -> tuple[str, str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return "", text
    return match.group("body"), text[match.end():]


def _parse_simple_front_matter(front_matter: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in front_matter.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t", "-")):
            continue
        key, sep, value = line.partition(":")
        if sep:
            metadata[key.strip()] = value.strip()
    return metadata


def _render_front_matter(metadata: dict[str, str]) -> str:
    preferred = [
        "name",
        "description",
        "license",
        "compatibility",
        "allowed-tools",
        "network-access",
        "network-access-justification",
    ]
    lines: list[str] = []
    emitted: set[str] = set()
    for key in preferred:
        if key in metadata:
            lines.append(f"{key}: {_format_yaml_scalar(metadata[key])}")
            emitted.add(key)
    for key, value in metadata.items():
        if key not in emitted:
            lines.append(f"{key}: {_format_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _format_yaml_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def _format_yaml_scalar(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return '""'
    if stripped.startswith(("[", "{", "'", '"')) or stripped in {"true", "false"}:
        return stripped
    if re.search(r":\s|[{}#]", stripped):
        return _quote_yaml(stripped)
    return stripped


def _infer_name(metadata: dict[str, str], body: str) -> str:
    title = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if title:
        return _slugify(title.group(1))
    return metadata.get("name") or "generated-skill"


def _infer_description(metadata: dict[str, str], body: str) -> str:
    if metadata.get("description"):
        return metadata["description"]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return _quote_yaml(stripped[:180])
    return "Generated skill package."


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", text.strip().lower()).strip("-")
    return slug or "generated-skill"


def _quote_yaml(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _uses_network(files: dict[str, str]) -> bool:
    combined = "\n".join(files.values()).lower()
    return any(pattern.lower() in combined for pattern in _NETWORK_PATTERNS)


def _upsert_network_section(body: str) -> str:
    if re.search(r"^##\s+Network Access\s*$", body, re.MULTILINE):
        return body
    section = (
        "\n\n## Network Access\n\n"
        "This package may use documented outbound requests for public APIs or "
        "package installers described in the usage examples. Network access is "
        "part of the declared skill behavior and should not be used for hidden "
        "telemetry or transfer of user data.\n"
    )
    return body.rstrip() + section


def _upsert_package_inventory(body: str, files: dict[str, str]) -> str:
    inventory = _render_package_inventory(files)
    # Replace the complete Markdown section, including legacy prose and all
    # duplicate copies.  Matching only the fenced tree left the prose behind,
    # so every normalization pass appended another Package Inventory section.
    pattern = re.compile(
        r"^## Package Inventory\s*\n.*?(?=^##\s+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    cleaned = pattern.sub("", body).rstrip()
    return cleaned + "\n\n" + inventory


def _render_package_inventory(files: dict[str, str]) -> str:
    tree = "\n".join(sorted(files))
    return (
        "## Package Inventory\n\n"
        "The generated package contains these files; local file references in "
        "this skill should resolve to this inventory.\n\n"
        f"```text\n{tree}\n```\n"
    )
