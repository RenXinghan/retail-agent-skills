#!/usr/bin/env bash
# Experiment 2: failure-only mining and validation gating (retail, skills_index mode).
# Libraries are mined from the "mine" part of the train runs, gated on the "val"
# part, and only then evaluated on the test split.
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep macOS awake for the whole run; idle sleep pauses it for hours.
if [[ $(uname) == Darwin && -z ${CAFFEINATED:-} ]]; then
  CAFFEINATED=1 exec caffeinate -i "$0" "$@"
fi

RAS=.venv/bin/ras
TRAIN=results/raw/main-train-vanilla/results.json
VAL_TRIALS=${VAL_TRIALS:-2}
TEST_TRIALS=${TEST_TRIALS:-3}

# 1. Two libraries from the mining part: the original miner and the failure-only miner.
$RAS mine --results "$TRAIN" --part mine --max-skills 12 --out skills/exp2-all.yaml
$RAS mine --results "$TRAIN" --part mine --failures-only --max-skills 6 --out skills/exp2-fail.yaml

# 2. Run each on the validation part and gate it against the plain agent's runs there.
for lib in all fail; do
  $RAS run --mode skills_index --skills "skills/exp2-$lib.yaml" --split train --part val \
    --trials "$VAL_TRIALS" --name "exp2-val-$lib"
  $RAS gate --skills "skills/exp2-$lib.yaml" --val "results/raw/exp2-val-$lib/results.json" \
    --baseline "$TRAIN" --out "skills/exp2-$lib-gated.yaml" || echo "gate kept nothing for $lib"
done

# 3. Evaluate every distinct library on the test split.
for lib in all fail all-gated fail-gated; do
  file="skills/exp2-$lib.yaml"
  [ -f "$file" ] || { echo "skip $lib: no library"; continue; }
  if [[ $lib == *-gated ]] && cmp -s "$file" "skills/exp2-${lib%-gated}.yaml"; then
    echo "skip $lib: gate kept every skill"; continue
  fi
  $RAS run --mode skills_index --skills "$file" --split test --trials "$TEST_TRIALS" \
    --name "exp2-test-$lib"
done

$RAS report results/raw/main-test-vanilla/results.json results/raw/exp2-test-*/results.json \
  | tee results/exp2-test.md
