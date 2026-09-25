#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Auto Skill-Attack Agent — driver script.
#
# Layout the script expects under TASKS_ROOT (you point this at YOUR
# skillsbench clone — it does NOT have to live inside this repo):
#
#   <TASKS_ROOT>/<task-id>/
#       instruction.md
#       task.toml
#       environment/skills/<skill-name>/SKILL.md
#       environment/skills/<skill-name>/scripts/...
#       ...
#
# Four run modes (pick with the first CLI arg; defaults to targets):
#
#   single   run ONE skill   (set TASK and SKILL, or SKILL_DIR for a direct path)
#   task     run every skill under ONE task (set TASK)
#   all      run every skill under TASKS_ROOT (cap with MAX_SKILLS)
#   targets  only build task-level attack_intent + skill-level attack_target
#
# Examples (defaults: TASKS_ROOT=skillsbench/tasks, MODE=targets):
#
#   # build attack intents/targets for every skill under skillsbench/tasks
#   ./run.sh
#
#   # quick smoke test
#   MAX_SKILLS=1 ./run.sh
#
#   # tune target-generation concurrency
#   TARGET_WORKERS=8 ./run.sh
#
#   # explicit single skill
#   ./run.sh single grid-dispatch-operator economic-dispatch
#
#   # external skillsbench checkout:
#   # edit TASKS_ROOT below, then run the mode you need.
#
#   # loop until SkillScan passes — no upper bound
#   PHASE1_ITERS=0 ENABLE_SKILL_SCAN=1 ./run.sh
#
#   # batch / alternate modes
#   ./run.sh task grid-dispatch-operator
#   MAX_SKILLS=10 ./run.sh all
#   ./run.sh targets
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- repo layout ------------------------------------------------------------
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ---- target-only defaults --------------------------------------------------
# These are intentionally direct script parameters, not environment fallbacks.
# Edit them here when you want to change the full target-generation run.
if [[ -d "${REPO_ROOT}/skillsbench/tasks" ]]; then
    TASKS_ROOT="${REPO_ROOT}/skillsbench/tasks"
    SKILLS_ROOTS=("${REPO_ROOT}/skillsbench/tasks")
    if [[ -d "${REPO_ROOT}/skillsbench/tasks_excluded" ]]; then
        SKILLS_ROOTS+=("${REPO_ROOT}/skillsbench/tasks_excluded")
    fi
else
    TASKS_ROOT="${REPO_ROOT}/examples/tasks"
    SKILLS_ROOTS=("${REPO_ROOT}/examples/tasks")
fi
MODE="${1:-targets}"
if [[ "${MODE}" == "targets" || "${MODE}" == "repair-targets" ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/skillsbench/target_results"
else
    OUTPUT_ROOT="${REPO_ROOT}/outputs"
fi

# Optional single-skill/task parameters for non-target modes.
TASK="${2:-citation-check}"
SKILL="${3:-citation-management}"
SKILL_DIR=""

# ---- model / LLM ------------------------------------------------------------
# Load .env as defaults: only fill keys that are NOT already in the
# environment, so vars passed on the command line (FOO=bar ./run.sh ...)
# always win over .env.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        # strip comments + blanks
        [[ -z "${_line}" || "${_line}" =~ ^[[:space:]]*# ]] && continue
        # split KEY=VALUE (only on first =)
        _key="${_line%%=*}"
        _val="${_line#*=}"
        # trim leading/trailing whitespace on key
        _key="${_key#"${_key%%[![:space:]]*}"}"
        _key="${_key%"${_key##*[![:space:]]}"}"
        [[ -z "${_key}" ]] && continue
        # strip optional surrounding quotes on value
        _val="${_val%\"}"; _val="${_val#\"}"
        _val="${_val%\'}"; _val="${_val#\'}"
        # only set if currently empty/unset
        if [[ -z "${!_key:-}" ]]; then
            export "${_key}=${_val}"
        fi
    done < "${REPO_ROOT}/.env"
    unset _line _key _val
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export NO_PROXY="${NO_PROXY:-}"
export no_proxy="${no_proxy:-${NO_PROXY}}"
MODEL="${MODEL:-${OPENAI_MODEL:-deepseek-v4-pro}}"
export OPENAI_MODEL="${OPENAI_MODEL:-${MODEL}}"
# Attack-target pre-stage. TARGET_MODELS is a comma-separated list of OpenAI
# model ids used as voters. Use `model@base_url` when voters live on different
# OpenAI-compatible endpoints. Empty means "use MODEL as the only voter".
# Example:
#   TARGET_MODELS=model-a,model-b TARGET_ITERS=5 ./run.sh
DEFAULT_TARGET_MODELS=""
TARGET_MODELS="${TARGET_MODELS:-${DEFAULT_TARGET_MODELS}}"
TARGET_ITERS="${TARGET_ITERS:-30}"
TARGET_WORKERS="${TARGET_WORKERS:-8}"
export ATTACK_AGENT_TARGET_VOTE_ATTEMPTS="${ATTACK_AGENT_TARGET_VOTE_ATTEMPTS:-2}"
export ATTACK_AGENT_TARGET_GENERATE_ATTEMPTS="${ATTACK_AGENT_TARGET_GENERATE_ATTEMPTS:-2}"
THINKING="${THINKING:-1}"                    # 1 enables model thinking via llm_client extra_body
export ATTACK_AGENT_ENABLE_THINKING="${THINKING}"

# For optimization modes (`single`, `task`, `all`) you can point at a prebuilt
# target output and skip one-time target generation.
TARGET_RESULTS_ROOT="${TARGET_RESULTS_ROOT:-}"   # e.g. skillsbench/target_results/target-results-...
ATTACK_TARGET_FILE="${ATTACK_TARGET_FILE:-}"     # single-skill override
REPAIR_TARGET_RESULTS_ROOT="${REPAIR_TARGET_RESULTS_ROOT:-}"

# ---- pipeline knobs ---------------------------------------------------------
# 0 = UNLIMITED (loop until the condition is met):
#   PHASE1_ITERS=0  → keep refining until SkillScan passes (or LLM crashes)
#   OUTER_ITERS=0   → keep going until phase-2 verdict succeeds
PHASE1_ITERS="${PHASE1_ITERS:-30}"
# Soft-pass threshold for low-severity findings.
# When high=0 medium=0 low<=N, stop refining and accept the skill.
# 0 = require all-zero (hard pass only). Set e.g. PHASE1_LOW_THRESHOLD=2
# to accept skills that only have ≤2 low-severity findings remaining.
PHASE1_LOW_THRESHOLD="${PHASE1_LOW_THRESHOLD:-0}"
CANDIDATES_PER_ROUND="${CANDIDATES_PER_ROUND:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
JSON_REPAIR_ATTEMPTS="${JSON_REPAIR_ATTEMPTS:-3}"
OUTER_ITERS="${OUTER_ITERS:-1}"
ENABLE_PHASE2="${ENABLE_PHASE2:-0}"      # 1 to turn on bench + sonar + judge
JUDGE="${JUDGE:-heuristic}"              # heuristic | llm

# ---- SkillScan --------------------------------------------------------------
ENABLE_SKILL_SCAN="${ENABLE_SKILL_SCAN:-1}"   # 1 = use real skill-scanner CLI
SCANNER_CMD="${SCANNER_CMD:-skill-scanner}"
NO_SCANNER_LLM="${NO_SCANNER_LLM:-0}"         # 1 disables --use-llm on the scanner
SCANNER_LLM_MAX_TOKENS="${SCANNER_LLM_MAX_TOKENS:-16384}"
SCANNER_RETRIES="${SCANNER_RETRIES:-2}"       # retries scanner infrastructure failures
SCANNER_RETRY_DELAY="${SCANNER_RETRY_DELAY:-5}"

# ---- Strategy library (default off — we want cold-start measurements first)
ENABLE_DEFAULT_SEEDS="${ENABLE_DEFAULT_SEEDS:-0}"
STRATEGY_LIB="${STRATEGY_LIB:-}"

# ---- batch -----------------------------------------------------------------
MAX_SKILLS="${MAX_SKILLS:-}"             # cap for MODE=all/task ; empty = no cap
SKIP_SKILLS="${SKIP_SKILLS:-0}"          # skip first N discovered skills in MODE=all/task
NUM_WORKERS="${NUM_WORKERS:-1}"          # parallel workers for optimization batch mode
BATCH_ROOT="${BATCH_ROOT:-}"             # existing batch output dir to append/continue into

# ---- debug -----------------------------------------------------------------
SAVE_DEBUG="${SAVE_DEBUG:-0}"            # 1 to persist per-round prompt/response under _debug/

# ---- mode / target ---------------------------------------------------------
# Defaults are defined near the top of this script as direct edit points.

# ---- python interpreter -----------------------------------------------------
PYTHON="${PYTHON:-python3}"

# Non-target modes are optimization stages. They must reuse prebuilt attack
# targets instead of generating new ones inside the optimization loop.
if [[ "${MODE}" != "targets" && -z "${TARGET_RESULTS_ROOT}" && -z "${ATTACK_TARGET_FILE}" ]]; then
    latest_target_results="$("${PYTHON}" - "${REPO_ROOT}/skillsbench/target_results" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
chosen = ""
chosen_count = -1
for candidate in sorted(root.glob("target-results-*")):
    summary_path = candidate / "batch_summary.json"
    if not summary_path.exists() or not any(candidate.rglob("attack_target.json")):
        continue
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(summary, list):
        continue
    if any(item.get("error") for item in summary if isinstance(item, dict)):
        continue
    if len(summary) >= chosen_count:
        chosen = str(candidate)
        chosen_count = len(summary)
print(chosen)
PY
)"
    if [[ -n "${latest_target_results}" ]]; then
        TARGET_RESULTS_ROOT="${latest_target_results}"
    else
        echo "[error] no prebuilt attack targets found for MODE=${MODE}." >&2
        echo "[hint ] run 'bash run.sh' first, or set TARGET_RESULTS_ROOT / ATTACK_TARGET_FILE in run.sh." >&2
        exit 2
    fi
fi
if [[ "${MODE}" == "repair-targets" && -z "${REPAIR_TARGET_RESULTS_ROOT}" ]]; then
    REPAIR_TARGET_RESULTS_ROOT="${TARGET_RESULTS_ROOT}"
fi

# -----------------------------------------------------------------------------
# Build the CLI flags shared by every invocation.
# -----------------------------------------------------------------------------
declare -a COMMON_FLAGS=(
    --output         "${OUTPUT_ROOT}"
    --model          "${MODEL}"
    --target-models  "${TARGET_MODELS}"
    --target-iterations "${TARGET_ITERS}"
    --phase1-iterations     "${PHASE1_ITERS}"
    --phase1-low-threshold  "${PHASE1_LOW_THRESHOLD}"
    --candidates-per-round  "${CANDIDATES_PER_ROUND}"
    --max-restarts          "${MAX_RESTARTS}"
    --json-repair-attempts  "${JSON_REPAIR_ATTEMPTS}"
    --outer-iterations      "${OUTER_ITERS}"
    --judge          "${JUDGE}"
)

if [[ "${ENABLE_PHASE2}" == "1" ]]; then
    COMMON_FLAGS+=(--enable-phase2)
fi

if [[ "${ENABLE_SKILL_SCAN}" == "1" ]]; then
    COMMON_FLAGS+=(--enable-skill-scan --scanner-cmd "${SCANNER_CMD}")
    COMMON_FLAGS+=(--scanner-retries "${SCANNER_RETRIES}")
    COMMON_FLAGS+=(--scanner-retry-delay "${SCANNER_RETRY_DELAY}")
    if [[ "${NO_SCANNER_LLM}" == "1" ]]; then
        COMMON_FLAGS+=(--no-scanner-llm)
    elif [[ -n "${SCANNER_LLM_MAX_TOKENS}" && "${SCANNER_LLM_MAX_TOKENS}" != "0" ]]; then
        COMMON_FLAGS+=(--scanner-llm-max-tokens "${SCANNER_LLM_MAX_TOKENS}")
    fi
fi

if [[ -n "${STRATEGY_LIB}" ]]; then
    COMMON_FLAGS+=(--strategy-lib "${STRATEGY_LIB}")
elif [[ "${ENABLE_DEFAULT_SEEDS}" == "1" ]]; then
    COMMON_FLAGS+=(--enable-default-seeds)
fi

if [[ "${SAVE_DEBUG}" == "1" ]]; then
    COMMON_FLAGS+=(--save-debug)
fi
if [[ -n "${TARGET_RESULTS_ROOT}" ]]; then
    COMMON_FLAGS+=(--target-results-root "${TARGET_RESULTS_ROOT}")
fi
if [[ -n "${ATTACK_TARGET_FILE}" ]]; then
    COMMON_FLAGS+=(--attack-target-file "${ATTACK_TARGET_FILE}")
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
banner() {
    echo
    echo "================================================================="
    echo "  $*"
    echo "================================================================="
}

require_tasks_root() {
    if [[ "${#SKILLS_ROOTS[@]}" -eq 0 ]]; then
        echo "[error] SKILLS_ROOTS is empty." >&2
        echo "[hint ] point it at your skillsbench tasks/ dir, e.g.:" >&2
        echo "          TASKS_ROOT=/path/to/skillsbench/tasks ./run.sh ..." >&2
        return 1
    fi
    local root
    for root in "${SKILLS_ROOTS[@]}"; do
        if [[ ! -d "${root}" ]]; then
            echo "[error] skills root does not exist: ${root}" >&2
            return 1
        fi
    done
}

resolve_skill_dir() {
    # $1=task  $2=skill -> echo full skill dir
    local task="$1" skill="$2"
    if [[ -z "${task}" || -z "${skill}" ]]; then
        echo "[error] MODE=single requires TASK and SKILL (or SKILL_DIR)." >&2
        echo "[hint ] e.g. ./run.sh single <task-id> <skill-name>" >&2
        return 1
    fi
    require_tasks_root || return 1
    local root dir
    for root in "${SKILLS_ROOTS[@]}"; do
        dir="${root}/${task}/environment/skills/${skill}"
        if [[ -f "${dir}/SKILL.md" ]]; then
            echo "${dir}"
            return 0
        fi
    done
    echo "[error] SKILL.md not found for task='${task}' skill='${skill}' under configured roots." >&2
    for root in "${SKILLS_ROOTS[@]}"; do
        echo "[hint ] checked: ${root}/${task}/environment/skills/${skill}" >&2
    done
    echo "[hint ] available skills under task '${task}':" >&2
    for root in "${SKILLS_ROOTS[@]}"; do
        if [[ -d "${root}/${task}/environment/skills" ]]; then
            for s in "${root}/${task}/environment/skills"/*/SKILL.md; do
                [[ -f "${s}" ]] && echo "          $(basename "$(dirname "${s}")")" >&2
            done
        fi
    done
    return 1
}

validate_skill_dir() {
    # $1 = absolute skill dir -> validate & echo
    local dir="$1"
    if [[ ! -f "${dir}/SKILL.md" ]]; then
        echo "[error] SKILL_DIR has no SKILL.md: ${dir}" >&2
        return 1
    fi
    echo "${dir}"
}

resolve_task_skills_root() {
    # $1=task -> root containing all SKILL.md of that task
    local task="$1"
    if [[ -z "${task}" ]]; then
        echo "[error] MODE=task requires TASK." >&2
        return 1
    fi
    require_tasks_root || return 1
    local root task_root matches=()
    for root in "${SKILLS_ROOTS[@]}"; do
        task_root="${root}/${task}/environment/skills"
        if [[ -d "${task_root}" ]]; then
            matches+=("${task_root}")
        fi
    done
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "[error] no skills dir for task '${task}' under configured roots." >&2
        return 1
    fi
    printf '%s\n' "${matches[@]}"
}

run_redteam_one() {
    local skill_dir="$1"
    banner "redteam: ${skill_dir##${REPO_ROOT}/}"
    "${PYTHON}" -m attack_agent.main redteam \
        --skill-dir "${skill_dir}" \
        "${COMMON_FLAGS[@]}"
}

run_batch() {
    local label="$1"
    shift
    local roots=("$@")
    banner "batch: ${label}"
    local flags=("${COMMON_FLAGS[@]}")
    if [[ -n "${MAX_SKILLS}" ]]; then
        flags+=(--max-skills "${MAX_SKILLS}")
    fi
    if [[ "${SKIP_SKILLS}" != "0" ]]; then
        flags+=(--skip-skills "${SKIP_SKILLS}")
    fi
    if [[ "${NUM_WORKERS}" != "1" ]]; then
        flags+=(--num-workers "${NUM_WORKERS}")
    fi
    if [[ -n "${BATCH_ROOT}" ]]; then
        flags+=(--batch-root "${BATCH_ROOT}")
    fi
    local root
    for root in "${roots[@]}"; do
        flags+=(--skills-root "${root}")
    done
    "${PYTHON}" -m attack_agent.main batch \
        "${flags[@]}"
}

run_targets() {
    local label="$1"
    shift
    local roots=("$@")
    banner "target batch: ${label}"
    local flags=(
        --output "${OUTPUT_ROOT}"
        --model "${MODEL}"
        --target-models "${TARGET_MODELS}"
        --target-iterations "${TARGET_ITERS}"
        --target-workers "${TARGET_WORKERS}"
    )
    local root
    for root in "${roots[@]}"; do
        flags+=(--skills-root "${root}")
    done
    if [[ -n "${MAX_SKILLS}" ]]; then
        flags+=(--max-skills "${MAX_SKILLS}")
    fi
    if [[ "${SAVE_DEBUG}" == "1" ]]; then
        flags+=(--save-debug)
    fi
    "${PYTHON}" -m attack_agent.main targets "${flags[@]}"
}

run_repair_targets() {
    local label="$1"
    shift
    local roots=("$@")
    if [[ -z "${REPAIR_TARGET_RESULTS_ROOT}" ]]; then
        echo "[error] repair-targets requires REPAIR_TARGET_RESULTS_ROOT." >&2
        echo "[hint ] e.g. REPAIR_TARGET_RESULTS_ROOT=${REPO_ROOT}/skillsbench/target_results/target-results-... bash run.sh repair-targets" >&2
        return 1
    fi
    banner "repair target batch: ${label}"
    local flags=(
        --output "${OUTPUT_ROOT}"
        --model "${MODEL}"
        --target-models "${TARGET_MODELS}"
        --target-iterations "${TARGET_ITERS}"
        --target-workers "${TARGET_WORKERS}"
        --repair-target-results-root "${REPAIR_TARGET_RESULTS_ROOT}"
    )
    local root
    for root in "${roots[@]}"; do
        flags+=(--skills-root "${root}")
    done
    if [[ "${SAVE_DEBUG}" == "1" ]]; then
        flags+=(--save-debug)
    fi
    if [[ -n "${MAX_SKILLS}" ]]; then
        flags+=(--max-skills "${MAX_SKILLS}")
    fi
    "${PYTHON}" -m attack_agent.main targets "${flags[@]}"
}

print_config() {
    cat <<EOF
[config]
  REPO_ROOT            = ${REPO_ROOT}
  TASKS_ROOT           = ${TASKS_ROOT:-<unset>}
  SKILLS_ROOTS         = ${SKILLS_ROOTS[*]:-<unset>}
  OUTPUT_ROOT          = ${OUTPUT_ROOT}
  MODE                 = ${MODE}
  TASK                 = ${TASK:-<unset>}
  SKILL                = ${SKILL:-<unset>}
  SKILL_DIR (override) = ${SKILL_DIR:-<unset>}
  MODEL                = ${MODEL}
  TARGET_MODELS        = ${TARGET_MODELS:-<MODEL>}
  TARGET_ITERS         = ${TARGET_ITERS}
  TARGET_WORKERS       = ${TARGET_WORKERS}
  TARGET_VOTE_ATTEMPTS = ${ATTACK_AGENT_TARGET_VOTE_ATTEMPTS}
  TARGET_GEN_ATTEMPTS  = ${ATTACK_AGENT_TARGET_GENERATE_ATTEMPTS}
  TARGET_RESULTS_ROOT  = ${TARGET_RESULTS_ROOT:-<none>}
  ATTACK_TARGET_FILE   = ${ATTACK_TARGET_FILE:-<none>}
  REPAIR_TARGET_ROOT   = ${REPAIR_TARGET_RESULTS_ROOT:-<none>}
  OPENAI_BASE_URL      = ${OPENAI_BASE_URL}
  OPENAI_API_BASE      = ${OPENAI_API_BASE}
  NO_PROXY             = ${NO_PROXY}
  PHASE1_ITERS         = ${PHASE1_ITERS}
  PHASE1_LOW_THRESHOLD = ${PHASE1_LOW_THRESHOLD}
  CANDIDATES_PER_ROUND = ${CANDIDATES_PER_ROUND}
  MAX_RESTARTS         = ${MAX_RESTARTS}
  JSON_REPAIR_ATTEMPTS = ${JSON_REPAIR_ATTEMPTS}
  OUTER_ITERS          = ${OUTER_ITERS}
  ENABLE_PHASE2        = ${ENABLE_PHASE2}
  JUDGE                = ${JUDGE}
  ENABLE_SKILL_SCAN    = ${ENABLE_SKILL_SCAN}
  SCANNER_CMD          = ${SCANNER_CMD}
  NO_SCANNER_LLM       = ${NO_SCANNER_LLM}
  SCANNER_LLM_MAX_TOKENS = ${SCANNER_LLM_MAX_TOKENS}
  ENABLE_DEFAULT_SEEDS = ${ENABLE_DEFAULT_SEEDS}
  STRATEGY_LIB         = ${STRATEGY_LIB:-<none>}
  MAX_SKILLS           = ${MAX_SKILLS:-<none>}
  SKIP_SKILLS          = ${SKIP_SKILLS}
  NUM_WORKERS          = ${NUM_WORKERS}
  BATCH_ROOT           = ${BATCH_ROOT:-<none>}
  SAVE_DEBUG           = ${SAVE_DEBUG}
  PYTHON               = ${PYTHON}
EOF
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
print_config

case "${MODE}" in
    single)
        if [[ -n "${SKILL_DIR}" ]]; then
            TARGET_DIR="$(validate_skill_dir "${SKILL_DIR}")"
        else
            TARGET_DIR="$(resolve_skill_dir "${TASK}" "${SKILL}")"
        fi
        run_redteam_one "${TARGET_DIR}"
        ;;

    task)
        mapfile -t TASK_SKILLS_ROOTS < <(resolve_task_skills_root "${TASK}")
        run_batch "all skills in task ${TASK}" "${TASK_SKILLS_ROOTS[@]}"
        ;;

    all)
        require_tasks_root
        run_batch "every SKILL.md under configured roots" "${SKILLS_ROOTS[@]}"
        ;;

    targets)
        require_tasks_root
        run_targets "attack intents and targets under configured roots" "${SKILLS_ROOTS[@]}"
        ;;

    repair-targets)
        require_tasks_root
        run_repair_targets "repair failed attack targets under configured roots" "${SKILLS_ROOTS[@]}"
        ;;

    *)
        echo "[error] unknown MODE='${MODE}' (expected: single | task | all | targets | repair-targets)" >&2
        exit 2
        ;;
esac
