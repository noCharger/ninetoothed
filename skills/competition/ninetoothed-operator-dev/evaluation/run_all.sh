#!/usr/bin/env bash
# run_all.sh — full self-evolution pipeline on a GPU host.
#
# Runs: baseline (learning-before) -> version B -> version A -> comparison (learning-after).
# Expects: torch + ninetoothed + CUDA + `claude` CLI available; run from the skill root's
# evaluation/ dir. Pass --fake to exercise the wiring with no GPU/claude (stub solver).
#
# Usage:
#   bash evaluation/run_all.sh                 # real run on GPU host
#   bash evaluation/run_all.sh --fake          # offline wiring check
#   B_GENS=6 A_GENS=4 bash evaluation/run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$HERE/.." && pwd)"
RESULTS="$HERE/results"
JOURNAL="$RESULTS/journal.jsonl"
FAKE=""
[[ "${1:-}" == "--fake" ]] && FAKE="--fake --no-git"
B_GENS="${B_GENS:-5}"
A_GENS="${A_GENS:-4}"

mkdir -p "$RESULTS/baseline"
echo "== [0/4] baseline: 24 tasks x {no_skill, v0} =="
python "$HERE/proxy_runner/run_matrix.py" --skill-root "$SKILL_ROOT" \
    --split all --modes no_skill v0 --run-id baseline \
    --journal "$JOURNAL" --csv "$RESULTS/baseline/matrix.csv" ${FAKE:+--fake}

echo "== [1/4] version B: light-MOO self-evolution, $B_GENS generations =="
python "$HERE/evolve_moo/moo_loop.py" --skill-root "$SKILL_ROOT" \
    --journal "$JOURNAL" --generations "$B_GENS" $FAKE

echo "== [2/4] version A: controller + memory, $A_GENS generations =="
python "$HERE/evolve_full/loop.py" --skill-root "$SKILL_ROOT" \
    --journal "$JOURNAL" --generations "$A_GENS" $FAKE

echo "== [3/4] learning-after: comparison + curves =="
python "$HERE/journal/curves.py" summary "$JOURNAL"
python "$HERE/journal/curves.py" compare "$JOURNAL" --modes no_skill v0 a b || true
python "$HERE/journal/curves.py" curve "$JOURNAL" --run b --csv "$RESULTS/b_curve.csv" || true
python "$HERE/journal/curves.py" curve "$JOURNAL" --run a --csv "$RESULTS/a_curve.csv" || true

echo "== done. journal: $JOURNAL =="
