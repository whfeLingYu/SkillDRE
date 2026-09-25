"""Skillbench runner interface — phase-2 dynamic harness.

The full bench harness (Daytona / Dockerized `bench eval create ...`) is
heavy. We define the contract here and ship a `StubBenchRunner` that
performs a minimal local invocation (no container) so the pipeline can run
end to end. A real implementation should shell out to:

    cd skillsbench
    uv run bench eval create -t tasks/<task-id> -a claude-agent-acp -s <skills-dir>

and parse the resulting reward / artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..schemas import AttackSkill, AttackTarget, BenchRunResult, SkillPackage


class BenchRunner(Protocol):
    def run(
        self,
        source_skill: SkillPackage,
        attack_skill: AttackSkill,
        attack_target: AttackTarget,
        run_dir: Path,
    ) -> BenchRunResult: ...


class StubBenchRunner:
    """No-op runner. Records inputs and returns task_reward=0.

    Replace with a real harness call when wiring phase 2.
    """

    def run(
        self,
        source_skill: SkillPackage,
        attack_skill: AttackSkill,
        attack_target: AttackTarget,
        run_dir: Path,
    ) -> BenchRunResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "stub_bench_manifest.txt"
        manifest_path.write_text(
            "StubBenchRunner — no real execution performed.\n"
            f"source_skill: {source_skill.name}\n"
            f"task_dir: {source_skill.task_dir}\n"
            f"attack_skill_files: {sorted(attack_skill.files.keys())}\n"
            f"attack_target.objective: {attack_target.objective}\n",
            encoding="utf-8",
        )
        return BenchRunResult(
            completed=True,
            task_reward=0.0,
            stdout="StubBenchRunner did not execute the bench harness.",
            stderr="",
            artifacts={"manifest": str(manifest_path)},
        )
