#!/usr/bin/env bash
# run_comparison.sh — reproducible skill-A/B + version-B-evolution for ONE solver/model.
#
# This is the single, committed entry point for every model run (7B / 14B / GLM). It
# replaces ad-hoc host scripts so each result is reproducible from git: same script,
# different env, one journal + CSV set per run under results/<TAG>/.
#
# Three blocks (select via BLOCKS):
#   1  operator A/B      — OPS_LIMIT elementwise tasks × {no_skill, v0}
#   2  diagnosis A/B     — all perf_diag tasks × {no_skill, v0}
#   3  version-B evolve  — FAMILY train tasks, B_GENS generations, git-tagged per gen
#
# Usage examples:
#   SOLVER=qwen MODEL_DIR=/root/autodl-tmp/models/qwen25-coder-14b NT_QUANT=4bit \
#     TAG=qwen14b bash evaluation/run_comparison.sh
#   SOLVER=glm GLM_MODEL=glm-5.2 GLM_AGENT=1 \
#     GLM_API_KEY=xxx TAG=glm52 bash evaluation/run_comparison.sh
#   SOLVER=qwen MODEL_DIR=... TAG=qwen7b BLOCKS="1 2" bash evaluation/run_comparison.sh
#
# Env:
#   SOLVER    qwen|glm|fake                 (required)
#   TAG       results label                 (required)
#   SKILL_ROOT skill package root           (default: dir above this script)
#   MODEL_DIR local model dir               (qwen)
#   NT_QUANT  4bit|""                        (qwen; 4-bit bitsandbytes)
#   GLM_MODEL/GLM_API_KEY/GLM_AGENT          (glm)
#   FAMILY    evolution family              (default elementwise)
#   OPS_LIMIT operator task count           (default 6)
#   B_GENS    evolution generations         (default 3)
#   BLOCKS    space-separated block ids     (default "1 2 3")
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="${SKILL_ROOT:-$(cd "$HERE/.." && pwd)}"
: "${SOLVER:?set SOLVER=qwen|glm|fake}"
: "${TAG:?set TAG=<label>}"
FAMILY="${FAMILY:-elementwise}"
OPS_LIMIT="${OPS_LIMIT:-6}"
B_GENS="${B_GENS:-3}"
BLOCKS="${BLOCKS:-1 2 3}"

RES="$HERE/results/$TAG"
mkdir -p "$RES"
J="$RES/journal.jsonl"
rm -f "$J"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# solver-specific args threaded to run_matrix / moo_loop
SOLVER_ARGS=(--solver "$SOLVER")
[[ -n "${MODEL_DIR:-}" ]] && SOLVER_ARGS+=(--model-dir "$MODEL_DIR")
[[ "$SOLVER" == "qwen" && -n "${MODEL_DIR:-}" ]] || true

echo "== run_comparison TAG=$TAG SOLVER=$SOLVER model=${GLM_MODEL:-${MODEL_DIR:-?}} blocks=[$BLOCKS] =="
echo "== results -> $RES =="

RUNNER="$HERE/proxy_runner/run_matrix.py"
MOO="$HERE/evolve_moo/moo_loop.py"

if [[ " $BLOCKS " == *" 1 "* ]]; then
  echo "===== [1] operator A/B: $OPS_LIMIT elementwise × {no_skill,v0} ====="
  python "$RUNNER" --skill-root "$SKILL_ROOT" --split all --limit "$OPS_LIMIT" \
    --modes no_skill v0 "${SOLVER_ARGS[@]}" --run-id baseline \
    --journal "$J" --csv "$RES/ab_operators.csv" --out-dir "$RES/ep_ops"
fi

if [[ " $BLOCKS " == *" 2 "* ]]; then
  echo "===== [2] diagnosis A/B: perf_diag × {no_skill,v0} ====="
  python "$RUNNER" --skill-root "$SKILL_ROOT" --split all --family perf_diag \
    --modes no_skill v0 "${SOLVER_ARGS[@]}" --run-id baseline_diag \
    --journal "$J" --csv "$RES/ab_diagnosis.csv" --out-dir "$RES/ep_diag"
fi

if [[ " $BLOCKS " == *" 3 "* ]]; then
  echo "===== [3] version-B evolution: $FAMILY × $B_GENS gen ====="
  # isolated git-tracked copy so v0 stays pristine and each gen is a reviewable tag
  BCOPY="/tmp/nt-skill-$TAG"
  rm -rf "$BCOPY"; cp -r "$SKILL_ROOT" "$BCOPY"
  find "$BCOPY" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  printf '__pycache__/\n*.pyc\nevaluation/results/episodes/\n' > "$BCOPY/.gitignore"
  rm -rf "$BCOPY/.git" "$BCOPY/evaluation/results/episodes"
  ( cd "$BCOPY" && git init -q && git add -A && git commit -qm v0 && git tag "skill-$TAG-gen0" )
  git config --global --add safe.directory "$BCOPY" 2>/dev/null || true
  python "$MOO" --skill-root "$BCOPY" --journal "$J" --generations "$B_GENS" \
    "${SOLVER_ARGS[@]}" --family "$FAMILY" --mode b --out-dir "$RES/evolve"
  ( cd "$BCOPY" && echo "evolution tags: $(git tag | tr '\n' ' ')" )
fi

echo "===== [summary] TAG=$TAG ====="
python "$HERE/journal/curves.py" summary "$J" || true
echo "===== DONE TAG=$TAG (artifacts in $RES) ====="
