"""Pluggable interfaces for the attack-agent pipeline.

Each module here defines a thin protocol class plus a default implementation
that the pipeline talks to. Swap any of them out (e.g. real Skill-Sonar, a
populated strategy library, a Daytona-based bench runner) without touching
`pipeline.py`.
"""

from .skill_scan import SkillScan, CiscoSkillScan, NullSkillScan, ScanResult
from .skill_sonar import SkillSonar, StubSkillSonar
from .strategy_library import StrategyLibrary, InMemoryStrategyLibrary
from .bench_runner import BenchRunner, StubBenchRunner

__all__ = [
    "SkillScan",
    "CiscoSkillScan",
    "NullSkillScan",
    "ScanResult",
    "SkillSonar",
    "StubSkillSonar",
    "StrategyLibrary",
    "InMemoryStrategyLibrary",
    "BenchRunner",
    "StubBenchRunner",
]
