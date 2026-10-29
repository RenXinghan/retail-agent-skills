#!/usr/bin/env bash
# Full protocol. Memory is built from the train split only; the test split is
# evaluated once per configuration. Override sizes for a cheap pilot, e.g.
#   NUM_TASKS=5 TRAIN_TRIALS=1 TEST_TRIALS=1 ./scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep macOS awake for the whole run; idle sleep pauses it for hours.
if [[ $(uname) == Darwin && -z ${CAFFEINATED:-} ]]; then
  CAFFEINATED=1 exec caffeinate -i "$0" "$@"
fi

TRAIN_TRIALS=${TRAIN_TRIALS:-2}
TEST_TRIALS=${TEST_TRIALS:-3}
TOP_K=${TOP_K:-3}
LIMIT=()
[ -n "${NUM_TASKS:-}" ] && LIMIT=(--num-tasks "$NUM_TASKS")
TAG=${TAG:-main}
RAS=.venv/bin/ras

# 1. Collect train trajectories with the plain agent.
$RAS run --mode vanilla --split train --trials "$TRAIN_TRIALS" ${LIMIT[@]+"${LIMIT[@]}"} --name "$TAG-train-vanilla"
TRAIN=results/raw/$TAG-train-vanilla/results.json

# 2. Distill skills from them.
$RAS mine --results "$TRAIN" --out "skills/$TAG.yaml"

# 3. Evaluate every memory configuration on the held-out test split.
$RAS run --mode vanilla --split test --trials "$TEST_TRIALS" ${LIMIT[@]+"${LIMIT[@]}"} --name "$TAG-test-vanilla"
$RAS run --mode trajectories --trajectories "$TRAIN" --top-k "$TOP_K" --split test \
  --trials "$TEST_TRIALS" ${LIMIT[@]+"${LIMIT[@]}"} --name "$TAG-test-trajectories"
$RAS run --mode skills_inject --skills "skills/$TAG.yaml" --top-k "$TOP_K" --split test \
  --trials "$TEST_TRIALS" ${LIMIT[@]+"${LIMIT[@]}"} --name "$TAG-test-skills_inject"
$RAS run --mode skills_index --skills "skills/$TAG.yaml" --split test \
  --trials "$TEST_TRIALS" ${LIMIT[@]+"${LIMIT[@]}"} --name "$TAG-test-skills_index"

$RAS report results/raw/"$TAG"-test-*/results.json | tee "results/$TAG-test.md"
