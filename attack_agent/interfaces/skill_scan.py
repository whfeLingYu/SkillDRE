"""SkillScan interface — the static guard the attack must bypass in phase 1.

The default implementation shells out to cisco-ai-defense's `skill-scanner`
CLI, the same tool the previous codebase used (see `defense.py`). Two
alternatives are provided:

* `NullSkillScan` — always passes; useful for end-to-end smoke tests when
  the real scanner is not installed.
* `CiscoSkillScan` — subprocess wrapper around `skill-scanner scan ...`.

Both return the same `ScanResult` so the pipeline never knows the
difference.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from ..schemas import ScanResult


class SkillScan(Protocol):
    """Static guard contract."""

    def scan(self, skill_dir: Path, output_json: Path | None = None) -> ScanResult: ...


# ---------------------------------------------------------------------------
# Cisco skill-scanner backed implementation
# ---------------------------------------------------------------------------

_FEEDBACK_KEEP_FIELDS = ("severity", "file_path", "line_number", "description", "rule_id", "category")


class CiscoSkillScan:
    """Wrap the `skill-scanner` CLI."""

    def __init__(
        self,
        scanner_cmd: str = "skill-scanner",
        use_llm: bool = True,
        extra_args: list[str] | None = None,
        report_dir: Path | None = None,
        scanner_retries: int = 2,
        retry_delay_seconds: float = 5.0,
    ) -> None:
        self.scanner_cmd = scanner_cmd
        self.use_llm = use_llm
        self.extra_args = list(extra_args or [])
        self.report_dir = report_dir
        self.scanner_retries = max(0, scanner_retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)

    def scan(self, skill_dir: Path, output_json: Path | None = None) -> ScanResult:
        skill_dir = Path(skill_dir)
        out_path = Path(output_json) if output_json is not None else (
            (self.report_dir or skill_dir) / f".scan_{skill_dir.name}.json"
        )
        raw = self._run(skill_dir, out_path)
        return self._summarize(raw)

    # ------------------------------------------------------------------
    def _run(self, target_dir: Path, output_json: Path) -> dict[str, Any]:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.scanner_cmd, "scan", str(target_dir),
            "--format", "json",
            "--output", str(output_json),
        ]
        if self.use_llm:
            cmd.append("--use-llm")
        cmd.extend(self.extra_args)

        env = os.environ.copy()
        if self.use_llm:
            scanner_base = os.environ.get("SKILL_SCANNER_OPENAI_BASE_URL")
            if scanner_base:
                env["OPENAI_API_BASE"] = scanner_base
                env["SKILL_SCANNER_LLM_BASE_URL"] = scanner_base
                scanner_key = os.environ.get("SKILL_SCANNER_OPENAI_API_KEY", "EMPTY")
                env["OPENAI_API_KEY"] = scanner_key
                env["SKILL_SCANNER_LLM_API_KEY"] = scanner_key
                # Internal OpenAI-compatible endpoints must be contacted
                # directly.  A parent launched through ``proxy_on`` can carry
                # an existing NO_PROXY value that omits the scanner host;
                # LiteLLM then sends the request to Squid, which rejects the
                # RFC1918 URL before it reaches the model.  Merge instead of
                # replacing so unrelated proxy exclusions remain intact.
                scanner_host = urlparse(scanner_base).hostname
                if scanner_host:
                    no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
                    entries = [item.strip() for item in no_proxy.split(",") if item.strip()]
                    for required in (scanner_host, "localhost", "127.0.0.1"):
                        if required not in entries:
                            entries.append(required)
                    merged_no_proxy = ",".join(entries)
                    env["NO_PROXY"] = merged_no_proxy
                    env["no_proxy"] = merged_no_proxy

        retry_errors: list[dict[str, Any]] = []
        total_attempts = self.scanner_retries + 1
        for attempt in range(1, total_attempts + 1):
            raw = self._run_once(cmd, env, output_json)
            if not _is_retryable_scanner_error(raw) or attempt == total_attempts:
                if retry_errors and isinstance(raw, dict):
                    raw = dict(raw)
                    raw["scanner_retry_errors"] = retry_errors
                    raw["scanner_attempts"] = attempt
                return raw

            retry_errors.append(raw)
            if self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds)

        return {"error": "scanner retry loop ended unexpectedly", "scanner_retry_errors": retry_errors}

    def _run_once(self, cmd: list[str], env: dict[str, str], output_json: Path) -> dict[str, Any]:
        try:
            output_json.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                check=False, env=env,
            )
        except FileNotFoundError as exc:
            return {"error": f"scanner command not found: {self.scanner_cmd}", "detail": str(exc)}

        if not output_json.exists():
            return {
                "error": "scanner exited without producing JSON output",
                "returncode": proc.returncode,
                "stderr": (proc.stderr or "")[-2000:],
                "stdout": (proc.stdout or "")[-2000:],
            }

        try:
            return json.loads(output_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": f"failed to parse scanner JSON: {exc}", "returncode": proc.returncode}

    def _summarize(self, raw: dict[str, Any]) -> ScanResult:
        if "error" in raw:
            return ScanResult(passed=False, error=raw["error"], raw_findings=raw)

        raw = _drop_ignorable_analyzer_failures(raw)
        items = _extract_findings(raw)
        buckets = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for f in items:
            sev = str(f.get("severity", "")).lower() if isinstance(f, dict) else ""
            if "critical" in sev or "high" in sev:
                buckets["high"] += 1
            elif "medium" in sev or "moderate" in sev:
                buckets["medium"] += 1
            elif "low" in sev or "info" in sev:
                buckets["low"] += 1
            else:
                buckets["unknown"] += 1

        has_severity = any(isinstance(f, dict) and "severity" in f for f in items)
        passed = (buckets["high"] == 0 and buckets["medium"] == 0 and buckets["low"] == 0) if has_severity else len(items) == 0
        reason = _format_unsafe_reason(items) if not passed else ""

        return ScanResult(
            passed=passed,
            unsafe_reason=reason,
            raw_findings=raw,
            severity_counts=buckets,
        )


# ---------------------------------------------------------------------------
# Null implementation
# ---------------------------------------------------------------------------

class NullSkillScan:
    """Always passes. Use for smoke tests when no scanner is installed."""

    def scan(self, skill_dir: Path, output_json: Path | None = None) -> ScanResult:  # noqa: ARG002
        return ScanResult(passed=True, unsafe_reason="", raw_findings={"note": "NullSkillScan"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_findings(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    for key in ("findings", "results", "issues", "detections"):
        v = data.get(key)
        if isinstance(v, list):
            return v
    return []


def _is_retryable_scanner_error(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict) or "error" not in data:
        return False
    error = str(data.get("error", "")).lower()
    if "command not found" in error:
        return False
    return True


def _drop_ignorable_analyzer_failures(data: dict[str, Any]) -> dict[str, Any]:
    """Keep scanner infrastructure failures out of the optimization signal.

    `LLM_ANALYSIS_FAILED` means the LLM analyzer ran but its response could not
    be parsed. Treating that as a LOW finding makes the optimizer chase scanner
    noise instead of real skill behavior. Preserve those records for debugging,
    but remove them from actionable findings.
    """
    if not isinstance(data, dict):
        return data

    for key in ("findings", "results", "issues", "detections"):
        value = data.get(key)
        if not isinstance(value, list):
            continue

        actionable: list[Any] = []
        ignored: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and _is_ignorable_analyzer_failure(item):
                ignored.append(item)
            else:
                actionable.append(item)

        if not ignored:
            return data

        sanitized = dict(data)
        sanitized[key] = actionable
        existing_ignored = sanitized.get("ignored_analyzer_findings")
        if isinstance(existing_ignored, list):
            sanitized["ignored_analyzer_findings"] = [*existing_ignored, *ignored]
        else:
            sanitized["ignored_analyzer_findings"] = ignored
        return sanitized

    return data


def _is_ignorable_analyzer_failure(finding: dict[str, Any]) -> bool:
    return str(finding.get("rule_id", "")).upper() == "LLM_ANALYSIS_FAILED"


def _format_unsafe_reason(items: list[dict[str, Any]]) -> str:
    """Produce a human-readable reason for the attack agent's next iteration."""
    keep: list[dict[str, Any]] = []
    for f in items:
        if not isinstance(f, dict):
            continue
        keep.append({k: f.get(k) for k in _FEEDBACK_KEEP_FIELDS if k in f})
    if not keep:
        return "Scanner reported unsafe findings (no structured detail available)."
    return json.dumps(keep, ensure_ascii=False, indent=2)
