#!/usr/bin/env bash
#
# Validation -> selection -> test, in that order.
#
# MATH-500 is the VALIDATION set: it chooses which round's verifier policy wins.
# ProcessBench is the TEST set: it is run on that one policy and its number is
# what gets reported.
#
# The order matters and the separation is the point. Self-play reward cannot do
# the selecting (r = +0.275 against held-out F1, n = 10, not significant;
# picking on it returns a policy worse than the untuned cold start), and
# selecting on ProcessBench would make the reported ProcessBench figure
# meaningless.
#
#   scripts/eval_policies_processbench.py stays as a separate ANALYSIS script:
#   80 items across every round, for understanding the trajectory. Do not use
#   it to choose a policy.
#
# Usage:
#   ./scripts/final_evaluation.sh <verify-prefix> <n-rounds> [model]
#   ./scripts/final_evaluation.sh hverify 10 claude-haiku-4-5
#
set -uo pipefail
cd "$(dirname "$0")/.."

PREFIX=${1:?usage: final_evaluation.sh <verify-prefix> <n-rounds> [model]}
ROUNDS=${2:?usage: final_evaluation.sh <verify-prefix> <n-rounds> [model]}
MODEL=${3:-claude-haiku-4-5}

VAL_ROOT=${VAL_ROOT:-data/validation_math500/${PREFIX}}
TEST_ROOT=${TEST_ROOT:-data/final_test/${PREFIX}}
PERTURBER=${PERTURBER:-${PREFIX%verify}perturb-v1}   # pinned; see below
VAL_N=${VAL_N:-40}
VAL_K=${VAL_K:-3}
CLEAN_FRAC=${CLEAN_FRAC:-0.25}
BACKEND=${BACKEND:-api}
CONCURRENCY=${CONCURRENCY:-6}
VERSIONS=$(seq 1 "$ROUNDS" | tr '\n' ' ')

echo "prefix=$PREFIX rounds=$ROUNDS model=$MODEL backend=$BACKEND"
echo

# --- 1. build the validation set once, with a PINNED perturber --------------
# Pinned so that difficulty does not drift between versions. Rebuilding it
# invalidates comparisons with anything already scored, hence no --force here.
echo "── 1/4  validation set (MATH-500, perturber=$PERTURBER)"
python scripts/validate_math500.py --root "$VAL_ROOT" --backend "$BACKEND" \
    --model "$MODEL" --concurrency "$CONCURRENCY" \
    prepare --n "$VAL_N" --k "$VAL_K" --clean-frac "$CLEAN_FRAC" \
    --perturber "$PERTURBER" || echo "   (already built — reusing)"

# --- 2. score every verifier version on it ----------------------------------
echo
echo "── 2/4  scoring ${PREFIX}-v1..v${ROUNDS} on validation"
python scripts/validate_math500.py --root "$VAL_ROOT" --backend "$BACKEND" \
    --model "$MODEL" --concurrency "$CONCURRENCY" \
    run --prefix "$PREFIX" --versions $VERSIONS || exit 2

# --- 3. select ---------------------------------------------------------------
echo
echo "── 3/4  selecting the winner"
python scripts/validate_math500.py --root "$VAL_ROOT" \
    select --prefix "$PREFIX" || exit 2
SELECTED=$(python -c "import json;print(json.load(open('$VAL_ROOT/selected.json'))['selected'])")
VER=${SELECTED##*-v}
echo "   selected: $SELECTED"

# --- 4. the reported result: full ProcessBench, winner only ------------------
echo
echo "── 4/4  FULL ProcessBench on $SELECTED (3,400 items — this is the reported number)"
python scripts/processbench_eval.py --root "$TEST_ROOT" prepare --round "$VER" \
    --verify-skill "$SELECTED" --full --seed 0 2>/dev/null \
  || echo "   (already prepared — reusing)"
python scripts/eval_policies_processbench.py --prefix "$PREFIX" --versions "$VER" \
    --model "$MODEL" --backend "$BACKEND" --concurrency "$CONCURRENCY" \
    --root "$TEST_ROOT" --skills-root .claude/skills || exit 2

echo
echo "──────────────────────────────────────────────────────────────"
echo "validation (selection) : $VAL_ROOT/selected.json"
echo "test (reported result) : $TEST_ROOT"
echo "analysis (all rounds)  : scripts/eval_policies_processbench.py, 80-item sample"
