"""Attack-target pre-stage with multi-model voting.

This module separates target construction from adversarial skill generation.
The accepted target is then treated as immutable by later phase-1 refinement.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .generator import _extract_json
from .llm_client import chat
from .prompt_loader import load_prompt
from .schemas import (
    AttackIntent,
    AttackTarget,
    IntentBuildResult,
    IntentBuildRound,
    IntentVote,
    SkillPackage,
    TargetBuildResult,
    TargetBuildRound,
    TargetVote,
)
from .skill_loader import render_skill_package_for_prompt

TARGET_CRITERIA = (
    "malicious",
    "evaluable",
    "executable",
    "task_compatible",
    "stable_template",
    "intent_aligned",
)

INTENT_CRITERIA = (
    "malicious",
    "task_relevant",
    "broad_enough",
    "not_over_specific",
    "sandbox_safe",
)


def build_attack_target(
    source_skill: SkillPackage,
    *,
    generator_model_id: str,
    voter_model_ids: Iterable[str],
    attack_intent: AttackIntent | None = None,
    initial_target: AttackTarget | None = None,
    initial_vote_feedback: str = "",
    max_iterations: int = 3,
    outer_feedback: str = "",
    debug_dir: Optional[Path] = None,
    temperature: float = 0.7,
) -> TargetBuildResult:
    """Generate and refine one attack target until every voter approves it."""
    voter_models = _dedupe_models(voter_model_ids) or [generator_model_id]
    vote_attempts = max(1, _env_int("ATTACK_AGENT_TARGET_VOTE_ATTEMPTS", 2))
    generate_attempts = max(1, _env_int("ATTACK_AGENT_TARGET_GENERATE_ATTEMPTS", 2))
    rounds: list[TargetBuildRound] = []
    previous_target: AttackTarget | None = initial_target
    vote_feedback = initial_vote_feedback

    for round_idx in range(1, max(1, max_iterations) + 1):
        prompt = load_prompt(
            "generate_attack_target.txt",
            source_skill_view=render_skill_package_for_prompt(source_skill),
            attack_intent_block=_render_attack_intent(attack_intent),
            outer_feedback_block=_render_outer_feedback(outer_feedback),
            previous_target_block=_render_previous_target(previous_target),
            vote_feedback_block=vote_feedback or "(no prior target-vote feedback)",
        )
        target: AttackTarget | None = None
        last_generate_exc: Exception | None = None
        for generate_attempt in range(1, generate_attempts + 1):
            raw = chat(model_id=generator_model_id, user_prompt=prompt, temperature=temperature)
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                suffix = "" if generate_attempt == 1 else f"_retry_{generate_attempt}"
                (debug_dir / f"target_round_{round_idx}_generate{suffix}_prompt.txt").write_text(prompt, encoding="utf-8")
                (debug_dir / f"target_round_{round_idx}_generate{suffix}_response.txt").write_text(raw or "<empty>", encoding="utf-8")
            try:
                target = _coerce_attack_target(_extract_json(raw))
                break
            except Exception as exc:
                last_generate_exc = exc
        if target is None:
            raise ValueError(f"Attack target generation failed after {generate_attempts} attempt(s): {last_generate_exc}")
        votes = _judge_attack_target_many(
            source_skill,
            target,
            attack_intent=attack_intent,
            model_ids=voter_models,
            debug_dir=debug_dir,
            debug_tag_prefix=f"target_round_{round_idx}",
            attempts=vote_attempts,
        )
        accepted = all(v.passed for v in votes)
        build_round = TargetBuildRound(index=round_idx, target=target, votes=votes, accepted=accepted)
        rounds.append(build_round)

        if accepted:
            return TargetBuildResult(target=target, accepted=True, rounds=rounds)

        previous_target = target
        vote_feedback = _render_vote_feedback(votes)

    return TargetBuildResult(target=rounds[-1].target, accepted=False, rounds=rounds)


def build_attack_intent(
    skills: list[SkillPackage],
    *,
    model_id: str,
    voter_model_ids: Iterable[str],
    max_iterations: int = 3,
    debug_dir: Optional[Path] = None,
    temperature: float = 0.7,
) -> IntentBuildResult:
    if not skills:
        raise ValueError("build_attack_intent requires at least one skill")
    voter_models = _dedupe_models(voter_model_ids) or [model_id]
    vote_attempts = max(1, _env_int("ATTACK_AGENT_TARGET_VOTE_ATTEMPTS", 2))
    generate_attempts = max(1, _env_int("ATTACK_AGENT_TARGET_GENERATE_ATTEMPTS", 2))
    rounds: list[IntentBuildRound] = []
    previous_intent: AttackIntent | None = None
    vote_feedback = ""

    for round_idx in range(1, max(1, max_iterations) + 1):
        prompt = load_prompt(
            "generate_attack_intent.txt",
            task_context=_render_task_context(skills),
            previous_intent_block=_render_attack_intent(previous_intent),
            vote_feedback_block=vote_feedback or "(no prior intent-vote feedback)",
        )
        intent: AttackIntent | None = None
        last_generate_exc: Exception | None = None
        for generate_attempt in range(1, generate_attempts + 1):
            raw = chat(model_id=model_id, user_prompt=prompt, temperature=temperature)
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                suffix = "" if generate_attempt == 1 else f"_retry_{generate_attempt}"
                (debug_dir / f"intent_round_{round_idx}{suffix}_prompt.txt").write_text(prompt, encoding="utf-8")
                (debug_dir / f"intent_round_{round_idx}{suffix}_response.txt").write_text(raw or "<empty>", encoding="utf-8")
            try:
                intent = _coerce_attack_intent(_extract_json(raw))
                break
            except Exception as exc:
                last_generate_exc = exc
        if intent is None:
            raise ValueError(f"Attack intent generation failed after {generate_attempts} attempt(s): {last_generate_exc}")
        votes = _judge_attack_intent_many(
            skills,
            intent,
            model_ids=voter_models,
            debug_dir=debug_dir,
            debug_tag_prefix=f"intent_round_{round_idx}",
            attempts=vote_attempts,
        )
        accepted = all(v.passed for v in votes)
        build_round = IntentBuildRound(index=round_idx, intent=intent, votes=votes, accepted=accepted)
        rounds.append(build_round)
        if accepted:
            return IntentBuildResult(intent=intent, accepted=True, rounds=rounds)

        previous_intent = intent
        vote_feedback = _render_vote_feedback(votes, criteria=INTENT_CRITERIA)

    return IntentBuildResult(intent=rounds[-1].intent, accepted=False, rounds=rounds)


def judge_attack_intent(
    skills: list[SkillPackage],
    intent: AttackIntent,
    *,
    model_id: str,
    debug_dir: Optional[Path] = None,
    debug_tag: str = "intent_vote",
) -> IntentVote:
    prompt = load_prompt(
        "judge_attack_intent.txt",
        task_context=_render_task_context(skills),
        attack_intent_json=json.dumps(asdict(intent), ensure_ascii=False, indent=2),
    )
    raw = chat(model_id=model_id, user_prompt=prompt, temperature=0.7)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{debug_tag}_prompt.txt").write_text(prompt, encoding="utf-8")
        (debug_dir / f"{debug_tag}_response.txt").write_text(raw or "<empty>", encoding="utf-8")

    data = _extract_json(raw)
    criteria = {name: bool(data.get(name, False)) for name in INTENT_CRITERIA}
    suggestions_raw = data.get("suggestions", [])
    suggestions = [str(x) for x in suggestions_raw if x] if isinstance(suggestions_raw, list) else []
    return IntentVote(
        model_id=model_id,
        passed=all(criteria.values()),
        explanation=str(data.get("explanation", "")).strip(),
        suggestions=suggestions,
        **criteria,
    )


def judge_attack_target(
    source_skill: SkillPackage,
    target: AttackTarget,
    *,
    attack_intent: AttackIntent | None = None,
    model_id: str,
    debug_dir: Optional[Path] = None,
    debug_tag: str = "target_vote",
) -> TargetVote:
    prompt = load_prompt(
        "judge_attack_target.txt",
        source_skill_view=render_skill_package_for_prompt(source_skill),
        attack_intent_json=_render_attack_intent(attack_intent),
        attack_target_json=json.dumps(asdict(target), ensure_ascii=False, indent=2),
    )
    raw = chat(model_id=model_id, user_prompt=prompt, temperature=0.7)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{debug_tag}_prompt.txt").write_text(prompt, encoding="utf-8")
        (debug_dir / f"{debug_tag}_response.txt").write_text(raw or "<empty>", encoding="utf-8")

    data = _extract_json(raw)
    criteria = {name: bool(data.get(name, False)) for name in TARGET_CRITERIA}
    passed = all(criteria.values())
    suggestions_raw = data.get("suggestions", [])
    suggestions = [str(x) for x in suggestions_raw if x] if isinstance(suggestions_raw, list) else []
    return TargetVote(
        model_id=model_id,
        passed=passed,
        explanation=str(data.get("explanation", "")).strip(),
        suggestions=suggestions,
        **criteria,
    )


def _safe_judge_attack_target(
    source_skill: SkillPackage,
    target: AttackTarget,
    *,
    attack_intent: AttackIntent | None,
    model_id: str,
    debug_dir: Optional[Path],
    debug_tag: str,
    attempts: int,
) -> TargetVote:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            tag = debug_tag if attempt == 1 else f"{debug_tag}_retry_{attempt}"
            return judge_attack_target(
                source_skill,
                target,
                attack_intent=attack_intent,
                model_id=model_id,
                debug_dir=debug_dir,
                debug_tag=tag,
            )
        except Exception as exc:
            last_exc = exc
    return TargetVote(
        model_id=model_id,
        passed=False,
        explanation=f"Target vote failed after {attempts} attempt(s): {last_exc}",
        suggestions=["Return valid JSON and make the target satisfy every review criterion."],
    )


def _judge_attack_target_many(
    source_skill: SkillPackage,
    target: AttackTarget,
    *,
    attack_intent: AttackIntent | None,
    model_ids: list[str],
    debug_dir: Optional[Path],
    debug_tag_prefix: str,
    attempts: int,
) -> list[TargetVote]:
    votes_by_idx: dict[int, TargetVote] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(model_ids))) as executor:
        futures = {
            executor.submit(
                _safe_judge_attack_target,
                source_skill,
                target,
                attack_intent=attack_intent,
                model_id=model_id,
                debug_dir=debug_dir,
                debug_tag=f"{debug_tag_prefix}_vote_{idx}",
                attempts=attempts,
            ): idx
            for idx, model_id in enumerate(model_ids, start=1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            votes_by_idx[idx] = future.result()
    return [votes_by_idx[idx] for idx in sorted(votes_by_idx)]


def _safe_judge_attack_intent(
    skills: list[SkillPackage],
    intent: AttackIntent,
    *,
    model_id: str,
    debug_dir: Optional[Path],
    debug_tag: str,
    attempts: int,
) -> IntentVote:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            tag = debug_tag if attempt == 1 else f"{debug_tag}_retry_{attempt}"
            return judge_attack_intent(
                skills,
                intent,
                model_id=model_id,
                debug_dir=debug_dir,
                debug_tag=tag,
            )
        except Exception as exc:
            last_exc = exc
    return IntentVote(
        model_id=model_id,
        passed=False,
        explanation=f"Intent vote failed after {attempts} attempt(s): {last_exc}",
        suggestions=["Return valid JSON and make the intent satisfy every review criterion."],
    )


def _judge_attack_intent_many(
    skills: list[SkillPackage],
    intent: AttackIntent,
    *,
    model_ids: list[str],
    debug_dir: Optional[Path],
    debug_tag_prefix: str,
    attempts: int,
) -> list[IntentVote]:
    votes_by_idx: dict[int, IntentVote] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(model_ids))) as executor:
        futures = {
            executor.submit(
                _safe_judge_attack_intent,
                skills,
                intent,
                model_id=model_id,
                debug_dir=debug_dir,
                debug_tag=f"{debug_tag_prefix}_vote_{idx}",
                attempts=attempts,
            ): idx
            for idx, model_id in enumerate(model_ids, start=1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            votes_by_idx[idx] = future.result()
    return [votes_by_idx[idx] for idx in sorted(votes_by_idx)]


def serialize_target_build_result(result: TargetBuildResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "target": asdict(result.target),
        "rounds": [
            {
                "round": item.index,
                "accepted": item.accepted,
                "target": asdict(item.target),
                "votes": [asdict(vote) for vote in item.votes],
            }
            for item in result.rounds
        ],
    }


def serialize_attack_intent(intent: AttackIntent) -> dict[str, Any]:
    return asdict(intent)


def serialize_intent_build_result(result: IntentBuildResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "intent": asdict(result.intent),
        "rounds": [
            {
                "round": item.index,
                "accepted": item.accepted,
                "intent": asdict(item.intent),
                "votes": [asdict(vote) for vote in item.votes],
            }
            for item in result.rounds
        ],
    }


def _coerce_attack_intent(data: dict[str, Any]) -> AttackIntent:
    if not isinstance(data, dict):
        raise ValueError(f"Intent response must be an object, got {type(data).__name__}")
    intent_raw = data.get("attack_intent", data)
    if not isinstance(intent_raw, dict):
        raise ValueError("`attack_intent` must be an object")
    intent = AttackIntent(
        objective=str(intent_raw.get("objective", "")).strip(),
        malicious_side_effect=str(intent_raw.get("malicious_side_effect", "")).strip(),
        constraints=[str(x) for x in intent_raw.get("constraints", []) if x],
        success_theme=str(intent_raw.get("success_theme", "")).strip(),
    )
    if not intent.objective:
        raise ValueError("`attack_intent.objective` is required")
    if not intent.malicious_side_effect:
        raise ValueError("`attack_intent.malicious_side_effect` is required")
    return intent


def _coerce_attack_target(data: dict[str, Any]) -> AttackTarget:
    if not isinstance(data, dict):
        raise ValueError(f"Target response must be an object, got {type(data).__name__}")
    target_raw = data.get("attack_target", data)
    if not isinstance(target_raw, dict):
        raise ValueError("`attack_target` must be an object")
    target = AttackTarget(
        objective=str(target_raw.get("objective", "")).strip(),
        success_signals=[str(x) for x in target_raw.get("success_signals", []) if x],
        artifacts=[str(x) for x in target_raw.get("artifacts", []) if x],
    )
    if not target.objective:
        raise ValueError("`attack_target.objective` is required")
    if not target.success_signals and not target.artifacts:
        raise ValueError("attack target must include success_signals or artifacts")
    return target


def _dedupe_models(models: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for model in models:
        model = (model or "").strip()
        if model and model not in seen:
            seen[model] = None
    return list(seen)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _render_outer_feedback(text: str) -> str:
    text = (text or "").strip()
    return text if text else "(no prior phase-2 feedback — first outer iteration)"


def _render_previous_target(target: AttackTarget | None) -> str:
    if target is None:
        return "(no previous target)"
    return json.dumps(asdict(target), ensure_ascii=False, indent=2)


def _render_attack_intent(intent: AttackIntent | None) -> str:
    if intent is None:
        return "(no task-level attack intent provided; design a skill-specific target directly)"
    return json.dumps(asdict(intent), ensure_ascii=False, indent=2)


def _render_task_context(skills: list[SkillPackage]) -> str:
    first = skills[0]
    task_name = first.task_dir.name if first.task_dir is not None else first.name
    parts = [f"## Task: `{task_name}`"]
    if first.instruction_md:
        parts.append(f"### Task instruction.md\n\n{first.instruction_md.strip()[:6000]}")
    parts.append("### Skills under this task")
    for skill in skills:
        skill_md = (skill.skill_md or "").strip()
        snippet = skill_md[:1600] + ("\n... [truncated]" if len(skill_md) > 1600 else "")
        tree = "\n".join(sorted(skill.files.keys()))
        parts.append(
            f"#### `{skill.name}`\n"
            f"Tree:\n```\n{tree}\n```\n"
            f"SKILL.md excerpt:\n```markdown\n{snippet}\n```"
        )
    return "\n\n".join(parts)


def _render_vote_feedback(votes: list, *, criteria: Iterable[str] = TARGET_CRITERIA) -> str:
    blocks = []
    for vote in votes:
        failed = [name for name in criteria if not getattr(vote, name)]
        status = "PASS" if vote.passed else f"FAIL ({', '.join(failed) or 'passed=false'})"
        suggestions = "; ".join(vote.suggestions) if vote.suggestions else "(no concrete suggestions)"
        blocks.append(
            f"Model: {vote.model_id}\n"
            f"Status: {status}\n"
            f"Explanation: {vote.explanation or '(none)'}\n"
            f"Suggestions: {suggestions}"
        )
    return "\n\n".join(blocks)
