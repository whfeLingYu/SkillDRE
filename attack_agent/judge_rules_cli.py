"""Generate one reviewed judge rule for every skill-level attack target.

Exposed as the ``judge-rules`` subcommand of ``python -m attack_agent.main``,
and also runnable standalone via ``python -m attack_agent.judge_rules_cli``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .judge_rule_builder import generate_judge_rules_batch
from .llm_client import default_model

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGET_ROOT = REPO_ROOT / "skillsbench" / "target_results" / "target-results-final"
DEFAULT_GENERATOR_MODEL = ""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-results-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL or default_model())
    parser.add_argument(
        "--review-models",
        default="",
        help="Optional comma-separated reviewer models; empty disables LLM review",
    )
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument(
        "--target-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited target IDs to process; blank lines and # comments are ignored",
    )
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reset-review-history",
        action="store_true",
        help="Use the cached spec and feedback but do not anchor candidate scoring to old votes",
    )
    parser.add_argument(
        "--review-cached",
        action="store_true",
        help="Review accepted cached rules and regenerate only reviewer-rejected rules",
    )


def run(args: argparse.Namespace) -> dict:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env.judge_rules")
        load_dotenv()
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = args.target_results_root.parent / f"judge-rules-{stamp}"
    reviewers = [value.strip() for value in args.review_models.split(",") if value.strip()]
    target_ids = None
    if args.target_ids_file is not None:
        target_ids = [
            line.strip()
            for line in args.target_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not target_ids:
            raise SystemExit(f"target ID file is empty: {args.target_ids_file}")
    return generate_judge_rules_batch(
        target_results_root=args.target_results_root,
        output_root=output,
        generator_model_id=args.generator_model,
        reviewer_model_ids=reviewers,
        max_iterations=args.max_iterations,
        workers=args.workers,
        max_targets=args.max_targets,
        target_ids=target_ids,
        save_debug=args.save_debug,
        force=args.force,
        review_cached=args.review_cached,
        reset_review_history=args.reset_review_history,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
