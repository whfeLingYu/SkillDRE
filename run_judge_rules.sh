#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Edit these values directly before a full batch run.
GENERATOR_API_BASE_URL=""
REVIEW_API_BASE_URL=""
REVIEW_API_KEY=""
MODEL_NAME="deepseek-v4-pro"
QWEN_API_BASE_URL=""
TARGET_RESULTS_ROOT="${REPO_ROOT}/skillsbench/target_results/target-results-final"
OUTPUT_ROOT="${REPO_ROOT}/skillsbench/target_results/judge-rules-final_dsv4pro"
GENERATOR_MODEL="${MODEL_NAME}${GENERATOR_API_BASE_URL:+@${GENERATOR_API_BASE_URL}}"

# Two-stage workflow:
#   0 = generate/repair only; do not call a reviewer.
#   1 = review cached rules, then repair only reviewer-rejected rules.
ENABLE_REVIEW=1
REVIEW_MODEL_SPECS="${REVIEW_MODEL_SPECS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-12}"
WORKERS=12          # Up to 12 targets; at most 36 simultaneous reviewer requests.
MAX_TARGETS=""       # Set to 1 for a smoke run; empty means all 260.
TARGET_IDS_FILE="${TARGET_IDS_FILE:-}" # Optional newline-delimited target IDs for a selective run.
SAVE_DEBUG=0
FORCE=0             # 0 = audit cache and repair only invalid rules; 1 = regenerate all.
RESET_REVIEW_HISTORY="${RESET_REVIEW_HISTORY:-0}"
PYTHON="${PYTHON:-python3}"

if [[ "${ENABLE_REVIEW}" == "1" ]]; then
    REVIEW_MODELS="${REVIEW_MODEL_SPECS}"
    REVIEW_CACHED=1
else
    REVIEW_MODELS=""
    REVIEW_CACHED=0
fi

# Always use the values above, regardless of inherited shell variables.
export OPENAI_API_KEY="${REVIEW_API_KEY}"
export OPENAI_BASE_URL="${REVIEW_API_BASE_URL}"
export OPENAI_API_BASE="${REVIEW_API_BASE_URL}"
export OPENAI_MODEL="${MODEL_NAME}"
export ATTACK_AGENT_ENABLE_THINKING=0
export ATTACK_AGENT_NO_AUTH_HOSTS="${ATTACK_AGENT_NO_AUTH_HOSTS:-}"

# urllib honors exact hosts more consistently than CIDR entries in NO_PROXY.
export NO_PROXY="${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"

# API resilience. Retry count excludes the first request. Model requests have
# no client-side deadline.
export ATTACK_AGENT_FORCE_HTTP_CLIENT="${ATTACK_AGENT_FORCE_HTTP_CLIENT:-1}"
export ATTACK_AGENT_LLM_NUM_RETRIES="${ATTACK_AGENT_LLM_NUM_RETRIES:-8}"
export ATTACK_AGENT_LLM_RETRY_BASE_DELAY="${ATTACK_AGENT_LLM_RETRY_BASE_DELAY:-2}"
export ATTACK_AGENT_LLM_RETRY_MAX_DELAY="${ATTACK_AGENT_LLM_RETRY_MAX_DELAY:-60}"
export ATTACK_AGENT_REVIEW_ATTEMPTS="${ATTACK_AGENT_REVIEW_ATTEMPTS:-3}"
export ATTACK_AGENT_JUDGE_RULE_STRATEGY="${ATTACK_AGENT_JUDGE_RULE_STRATEGY:-1}"
export ATTACK_AGENT_KIMI_REVIEW_CONCURRENCY="${ATTACK_AGENT_KIMI_REVIEW_CONCURRENCY:-2}"
export ATTACK_AGENT_KIMI_REVIEW_RETRIES="${ATTACK_AGENT_KIMI_REVIEW_RETRIES:-1}"
export ATTACK_AGENT_LLM_STALL_WARNING_SECONDS="${ATTACK_AGENT_LLM_STALL_WARNING_SECONDS:-300}"

declare -a ARGS=(
    --target-results-root "${TARGET_RESULTS_ROOT}"
    --output "${OUTPUT_ROOT}"
    --generator-model "${GENERATOR_MODEL}"
    --review-models "${REVIEW_MODELS}"
    --max-iterations "${MAX_ITERATIONS}"
    --workers "${WORKERS}"
)
if [[ -n "${MAX_TARGETS}" ]]; then
    ARGS+=(--max-targets "${MAX_TARGETS}")
fi
if [[ -n "${TARGET_IDS_FILE}" ]]; then
    ARGS+=(--target-ids-file "${TARGET_IDS_FILE}")
fi
if [[ "${SAVE_DEBUG}" == "1" ]]; then
    ARGS+=(--save-debug)
fi
if [[ "${FORCE}" == "1" ]]; then
    ARGS+=(--force)
fi
if [[ "${REVIEW_CACHED}" == "1" ]]; then
    ARGS+=(--review-cached)
fi
if [[ "${RESET_REVIEW_HISTORY}" == "1" ]]; then
    ARGS+=(--reset-review-history)
fi

echo "[judge-rules] target root : ${TARGET_RESULTS_ROOT}"
echo "[judge-rules] output      : ${OUTPUT_ROOT}"
echo "[judge-rules] generate API: ${GENERATOR_API_BASE_URL} (no auth)"
echo "[judge-rules] review API  : ${REVIEW_API_BASE_URL}"
echo "[judge-rules] generator   : ${GENERATOR_MODEL}"
echo "[judge-rules] review      : ${ENABLE_REVIEW}"
echo "[judge-rules] reviewers   : ${REVIEW_MODELS}"
echo "[judge-rules] vote policy : unanimous consent of all configured reviewers"
echo "[judge-rules] review cache: ${REVIEW_CACHED}"
echo "[judge-rules] reset votes : ${RESET_REVIEW_HISTORY}"
echo "[judge-rules] target filter: ${TARGET_IDS_FILE:-<all>}"
echo "[judge-rules] workers     : ${WORKERS}"
echo "[judge-rules] iterations  : ${MAX_ITERATIONS}"
echo "[judge-rules] HTTP client : ${ATTACK_AGENT_FORCE_HTTP_CLIENT}"
echo "[judge-rules] LLM retries : ${ATTACK_AGENT_LLM_NUM_RETRIES}"
echo "[judge-rules] retry delay : ${ATTACK_AGENT_LLM_RETRY_BASE_DELAY}-${ATTACK_AGENT_LLM_RETRY_MAX_DELAY}s"
echo "[judge-rules] review tries: ${ATTACK_AGENT_REVIEW_ATTEMPTS}"
echo "[judge-rules] repair strategy: ${ATTACK_AGENT_JUDGE_RULE_STRATEGY}"
echo "[judge-rules] Kimi concurrency: ${ATTACK_AGENT_KIMI_REVIEW_CONCURRENCY}"
echo "[judge-rules] Kimi retries: ${ATTACK_AGENT_KIMI_REVIEW_RETRIES}"
echo "[judge-rules] request timeout: disabled"
echo "[judge-rules] stall warn  : ${ATTACK_AGENT_LLM_STALL_WARNING_SECONDS}s (warning only)"

cd "${REPO_ROOT}"
"${PYTHON}" -u -m attack_agent.main judge-rules "${ARGS[@]}"
