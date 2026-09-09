#!/usr/bin/env bash
#
# Four self-play experiments, varying who writes the policy and who plays.
#
# The 10-round Haiku run (HAIKU_FAILURE.md) found no improvement on held-out
# data and traced it to the *policy updater*, not the players: Haiku fitted
# per-round noise at n≈2 per error type, blacklisted half the taxonomy, and
# never deleted a rule, while its verification was fine (0/36 missed, 1/44 false
# alarms). These runs test that diagnosis directly.
#
#   A  strong updater + weak players   <- the discriminating experiment
#   B  weak updater   + weak players   <- the published baseline, re-run
#   C, D  the same contrast on other model families
#
# If A improves on held-out ProcessBench and B does not, the updater was the
# bottleneck and the architecture is sound.
#
# Usage:
#   ./run_experiments.sh                 # all four
#   ./run_experiments.sh A C             # only those
#   DRY_RUN=1 ./run_experiments.sh       # render every prompt, call nothing
#   ROUNDS=3 EPISODES=4 ./run_experiments.sh A
#
set -uo pipefail
cd "$(dirname "$0")"

# --- knobs ------------------------------------------------------------------
ROUNDS=${ROUNDS:-10}
EPISODES=${EPISODES:-8}
K=${K:-3}
SEED=${SEED:-42}
PER_TOPIC=${PER_TOPIC:-300}
START=${START:-cold}
PB_PER_SUBSET=${PB_PER_SUBSET:-20}     # ProcessBench items per subset, per policy
PB_BACKEND=${PB_BACKEND:-api}          # api | batch | claude-code
DRY_RUN=${DRY_RUN:-0}
EVAL=${EVAL:-1}                        # 0 to skip the held-out evaluation

# --- model ids --------------------------------------------------------------
# Override any of these from the environment if an id is wrong for your account.
# The GPT and DeepSeek ids below postdate this script's author's knowledge; they
# are taken from the request verbatim and NOT verified against a live API.
A_PLAYERS=${A_PLAYERS:-claude-haiku-4-5}
A_UPDATER=${A_UPDATER:-claude-opus-5}

B_PLAYERS=${B_PLAYERS:-claude-haiku-4-5}
B_UPDATER=${B_UPDATER:-claude-haiku-4-5}

C_PLAYERS=${C_PLAYERS:-gpt-5.6-terra}
C_UPDATER=${C_UPDATER:-gpt-5.6-sol}

D_PLAYERS=${D_PLAYERS:-deepseek-v4-flash}
D_UPDATER=${D_UPDATER:-deepseek-v4-pro}

LOGS=logs/experiments
mkdir -p "$LOGS"

# --- helpers ----------------------------------------------------------------
have_key() { [ -n "${!1:-}" ]; }

# Which credential a model needs, from its id.
key_for() {
  case "$1" in
    claude-*)  echo ANTHROPIC_API_KEY ;;
    gpt-*|o1*|o3*|o4*|chatgpt*) echo OPENAI_API_KEY ;;
    deepseek*) echo DEEPSEEK_API_KEY ;;
    *)         echo "" ;;
  esac
}

# Claude models can go through the local CLI, which needs no API key; everything
# else must use the direct API.
backend_for() { case "$1" in claude-*) echo claude-code ;; *) echo api ;; esac; }

run_experiment() {   # name players updater
  local name=$1 players=$2 updater=$3
  local slug tag backend pkey ukey root log
  slug=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  tag="exp_${slug}"
  backend=$(backend_for "$players")
  # A run mixing a CLI player with an API updater is not expressible in one
  # process, so a mixed pair forces the API backend for both.
  if [ "$(backend_for "$updater")" != "$backend" ]; then backend=api; fi

  root="data/skill_rollouts/${tag}"
  log="${LOGS}/${tag}.log"

  echo "──────────────────────────────────────────────────────────────"
  echo "Experiment ${name}: players=${players}  updater=${updater}"
  echo "  backend=${backend}  root=${root}"

  if [ "$backend" = "api" ]; then
    for m in "$players" "$updater"; do
      k=$(key_for "$m")
      if [ -z "$k" ]; then
        echo "  SKIP — cannot infer a provider for '${m}'; set the id explicitly."; return 1
      fi
      if ! have_key "$k"; then
        echo "  SKIP — \$${k} is not set (needed for ${m})."; return 1
      fi
    done
  fi

  local dry=(); [ "$DRY_RUN" = "1" ] && dry=(--dry-run)
  # Each experiment gets its own policy namespace so versions never collide.
  python scripts/run_pipeline.py \
      --rounds "$ROUNDS" --episodes "$EPISODES" --k "$K" --seed "$SEED" \
      --start "$START" --freeze none --source hendrycks --per-topic "$PER_TOPIC" \
      --model "$players" --updater-model "$updater" \
      --backend "$backend" \
      --perturb-prefix "${tag}_perturb" --verify-prefix "${tag}_verify" \
      --root "$root" --retry-format 1 --resume \
      "${dry[@]}" >"$log" 2>&1
  local rc=$?

  if [ $rc -ne 0 ]; then
    echo "  pipeline exited $rc — see $log"
    tail -3 "$log" | sed 's/^/    /'
    return $rc
  fi
  echo "  self-play done → $root"
  [ -f "$root/summary_table.md" ] && sed 's/^/    /' "$root/summary_table.md"

  # --- held-out evaluation of every verifier policy the run produced --------
  if [ "$EVAL" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    local versions
    versions=$(seq 1 "$ROUNDS" | tr '\n' ' ')
    echo "  evaluating ${tag}_verify v1..v${ROUNDS} on ProcessBench (${PB_BACKEND})"
    python scripts/eval_policies_processbench.py \
        --prefix "${tag}_verify" --versions $versions \
        --model "$players" --per-subset "$PB_PER_SUBSET" --seed 0 \
        --backend "$PB_BACKEND" --concurrency 6 \
        --root "data/policy_evals/${tag}" >>"$log" 2>&1 \
      && echo "  eval done → data/policy_evals/${tag}" \
      || { echo "  eval incomplete (rerun this script to resume) — see $log"; }
  fi
}

# --- dispatch ---------------------------------------------------------------
usage() {
  sed -n '3,$p' "$0" | sed -n '/^#/!q;p' | sed 's/^#\{1,\} \{0,1\}//'
  cat <<'USAGE'

Experiments
  A  claude-opus-5      updater + claude-haiku-4-5   players
  B  claude-haiku-4-5   updater + claude-haiku-4-5   players
  C  gpt-5.6-sol        updater + gpt-5.6-terra      players
  D  deepseek-v4-pro    updater + deepseek-v4-flash  players

Environment overrides (defaults in brackets)
  ROUNDS [10]          self-play rounds per experiment
  EPISODES [8]         episodes per round
  K [3]                errors injected per episode
  SEED [42]            problem sampling seed
  PER_TOPIC [300]      problems drawn from the dataset
  START [cold]         cold | warm  initial policies
  EVAL [1]             0 skips the held-out ProcessBench evaluation
  PB_PER_SUBSET [20]   ProcessBench items per subset, per policy
  PB_BACKEND [api]     api | batch | claude-code
  DRY_RUN [0]          1 renders every prompt and calls nothing
  A_PLAYERS/A_UPDATER  ... D_PLAYERS/D_UPDATER  override any model id

Credentials
  Claude experiments run through `claude -p` and need no key. OpenAI and
  DeepSeek need OPENAI_API_KEY / DEEPSEEK_API_KEY; an experiment whose key is
  missing is skipped with a message, the others still run.
  If you keep keys in configs/.secrets (gitignored):
      set -a; . configs/.secrets; set +a

Resuming
  Re-running the same command resumes: completed rounds and answered evaluation
  items are skipped, so an interrupted run never re-pays for finished work.
USAGE
}

case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
esac

declare -a WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=(A B C D)

echo "rounds=$ROUNDS episodes=$EPISODES k=$K start=$START dry_run=$DRY_RUN eval=$EVAL"
declare -a FAILED=()
RAN=0
for e in "${WANT[@]}"; do
  case "$(echo "$e" | tr "[:lower:]" "[:upper:]")" in
    A) RAN=$((RAN+1)); run_experiment A "$A_PLAYERS" "$A_UPDATER" || FAILED+=(A) ;;
    B) RAN=$((RAN+1)); run_experiment B "$B_PLAYERS" "$B_UPDATER" || FAILED+=(B) ;;
    C) RAN=$((RAN+1)); run_experiment C "$C_PLAYERS" "$C_UPDATER" || FAILED+=(C) ;;
    D) RAN=$((RAN+1)); run_experiment D "$D_PLAYERS" "$D_UPDATER" || FAILED+=(D) ;;
    *) echo "unknown experiment '${e}' (choose from A B C D)"; FAILED+=("$e") ;;
  esac
done

echo "──────────────────────────────────────────────────────────────"
if [ "$RAN" -eq 0 ]; then
  echo "nothing ran — check the experiment names"
elif [ ${#FAILED[@]} -eq 0 ]; then
  echo "all $RAN requested experiment(s) completed"
else
  echo "did not complete: ${FAILED[*]}  (logs in ${LOGS}/)"
fi
echo
echo "Compare held-out F1 across experiments:"
echo "  for d in data/policy_evals/exp_*; do echo \"\$d\"; \\"
echo "    python -c \"import json,glob;[print(' ',f.split('/')[-3], json.load(open(f))['overall']['processbench_f1']) for f in sorted(glob.glob(\\\"\$d/*/round_*/eval_summary.json\\\"))]\"; done"
