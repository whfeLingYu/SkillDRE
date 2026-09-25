"""Skill-Sonar interface — phase-2 runtime defense.

The real Skill-Sonar is not part of this repo yet. We define the contract
plus a `StubSkillSonar` that returns "no alarm" so the pipeline can run end
to end. Replace with a concrete implementation that monitors the running
agent process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..schemas import SonarResult


class SkillSonar(Protocol):
    """Runtime monitoring contract.

    `start(run_dir)` is called before the bench harness launches; `stop()`
    returns the collected observations. Implementations may run an async
    monitor in between; the stub does nothing.
    """

    def start(self, run_dir: Path, attack_target_hint: str = "") -> None: ...
    def stop(self) -> SonarResult: ...


class StubSkillSonar:
    """No-op Skill-Sonar. Always returns 'no alarm'."""

    def __init__(self) -> None:
        self._started: bool = False
        self._run_dir: Path | None = None
        self._hint: str = ""

    def start(self, run_dir: Path, attack_target_hint: str = "") -> None:
        self._started = True
        self._run_dir = Path(run_dir)
        self._hint = attack_target_hint

    def stop(self) -> SonarResult:
        if not self._started:
            return SonarResult(alarmed=False, alarm_reason="", events=[],
                               error="StubSkillSonar.stop() called before start()")
        self._started = False
        return SonarResult(
            alarmed=False,
            alarm_reason="",
            events=[{"note": "StubSkillSonar: no runtime monitoring performed.",
                     "run_dir": str(self._run_dir)}],
            runtime_audit={
                "protocol_status": "compliant",
                "runtime_protocol_valid": True,
                "interruptions": [],
                "summary": {"interruption_count": 0},
            },
        )
