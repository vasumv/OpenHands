#!/usr/bin/env bash
set -eo pipefail

# Fix for Nix environment - numpy needs libstdc++
export LD_LIBRARY_PATH=/nix/store/dj06r96j515npcqi9d8af1d1c60bx2vn-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
EVAL_LIMIT=$3
NUM_WORKERS=$4

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi

checkout_eval_branch

# Only 'CodeActAgent' is supported for Fray currently
AGENT="CodeActAgent"
MAX_ITER=30  # Reasonable default for concurrency bug fixing

get_openhands_version

echo "AGENT: $AGENT"
echo "MAX_ITER: $MAX_ITER"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"

export PYTHONPATH=$(pwd)

COMMAND="poetry run python ./evaluation/benchmarks/fray/run_infer.py \
    --llm-config $MODEL_CONFIG \
    --agent-cls $AGENT \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS
"

if [ -n "$EVAL_LIMIT" ]; then
  echo "EVAL_LIMIT: $EVAL_LIMIT"
  COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

# Run the command
eval $COMMAND
