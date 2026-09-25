#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Edit these values when evaluating another Phase-2 batch.
PHASE2_ROOT="${PHASE2_ROOT:-${REPO_ROOT}/outputs/phase2_all_20260816-132358}"
RULES_ROOT="${RULES_ROOT:-${REPO_ROOT}/skillsbench/target_results/judge-rules-final_dsv4pro}"
OUTPUT_DIR="${OUTPUT_DIR:-${PHASE2_ROOT}/asr_evaluation}"
SELECTION="${SELECTION:-best}"
REFERENCE_ROOT="${REFERENCE_ROOT:-}"
PYTHON="${PYTHON:-python3}"

args=(
  --phase2-root "$PHASE2_ROOT"
  --rules-root "$RULES_ROOT"
  --output-dir "$OUTPUT_DIR"
  --selection "$SELECTION"
)
if [[ -n "$REFERENCE_ROOT" ]]; then
  args+=(--reference-root "$REFERENCE_ROOT")
fi

echo "[asr] phase2 root : $PHASE2_ROOT"
echo "[asr] rules root  : $RULES_ROOT"
echo "[asr] output      : $OUTPUT_DIR"
echo "[asr] selection   : $SELECTION"

cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"
"$PYTHON" -u -m attack_agent.main asr-eval "${args[@]}"
