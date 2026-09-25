from __future__ import annotations

from pathlib import Path


def load_prompt(template_name: str, **kwargs: str | int) -> str:
    template_path = Path(__file__).parent / "prompt_template" / template_name
    text = template_path.read_text(encoding="utf-8")
    # Escape { and } inside every string value so that str.format() never
    # misinterprets JSON snippets or code blocks as format-string placeholders.
    # Non-string values (int, float, …) are left as-is.
    safe_kwargs = {
        k: str(v).replace("{", "{{").replace("}", "}}") if isinstance(v, str) else v
        for k, v in kwargs.items()
    }
    return text.format(**safe_kwargs)
