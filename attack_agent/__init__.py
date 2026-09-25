"""Auto Skill-Attack Agent — automated red-team test generation."""

from .pipeline import run_pipeline, run_pipeline_over_skills
from .skill_loader import load_skill_package, load_skills_from_root

__all__ = [
    "__version__",
    "run_pipeline",
    "run_pipeline_over_skills",
    "load_skill_package",
    "load_skills_from_root",
]
__version__ = "0.2.0"
