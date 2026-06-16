#!/usr/bin/env bash
# run_history_experiments.sh — history-aware consensus encoder comparison.
#
# Our novel contribution beyond the COLA paper: the consensus builder is fed a
# WINDOW of past observations through a temporal encoder, instead of a single
# frame. This script compares the encoder ladder on the consensus label:
#   identity     -> window=1, single-frame  == the paper's vanilla COLA
#   gru          -> recurrent history encoder
#   transformer  -> multi-head self-attention history encoder
# Headline question: does attention (transformer) beat vanilla (identity)?
#
# 4-agent simple_spread (fully observable): the fastest scenario where COLA still
# produces a non-degenerate consensus label (3-agent collapses, 6-agent is slow),
# so encoder differences are measurable on a stable task. cb_variant is fixed to
# 'cola' (this ablation varies the ENCODER, not the label content).
#
# Logs to WandB project cola-history. COMPLETED skips finished runs on re-run.
# Usage: bash run_history_experiments.sh   (inside tmux)

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON="${PYTHON:-$(command -v python)}"
MAX_JOBS=6                 # window=5 runs use ~3.5 GB each; 6 fits ~32 GB
MAX_STEPS=800000           # ~plateau for simple_spread; 1M if time allows
N_AGENTS=4
SCENARIO="simple_spread"
ENCODERS=(identity gru transformer)
SEEDS=(42 43 44 45 46)
HISTORY_WINDOW=5           # used for gru/transformer; identity forces window=1
DEVICE="cpu"               # transformer can be re-benchmarked on cuda if slow
BUFFER_CAPACITY=150000

COMPLETED=()               # "<encoder>_<seed>" entries to skip on re-run

LOGDIR="logs/history_experiments"
mkdir -p "$LOGDIR"

_jobs=()
_wait_for_slot() {
    while [ "${#_jobs[@]}" -ge "$MAX_JOBS" ]; do
        local new_jobs=()
        for pid in "${_jobs[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then new_jobs+=("$pid"); fi
        done
        _jobs=("${new_jobs[@]+"${new_jobs[@]}"}")
        if [ "${#_jobs[@]}" -ge "$MAX_JOBS" ]; then sleep 5; fi
    done
    return 0
}
_launch() {
    local encoder="$1" seed="$2"
    local window="$HISTORY_WINDOW"
    [ "$encoder" = "identity" ] && window=1
    local run_name="hist_${encoder}_seed${seed}"
    local logfile="${LOGDIR}/${run_name}.log"
    echo "[$(date '+%H:%M:%S')] START  ${run_name}  (window=${window}, log: ${logfile})"
    PYTHONPATH=src "$PYTHON" -u train_cola.py \
        --scenario        "$SCENARIO" \
        --n_agents        "$N_AGENTS" \
        --use_history_path \
        --history_encoder "$encoder" \
        --history_window  "$window" \
        --cb_variant      cola \
        --device          "$DEVICE" \
        --max_steps       "$MAX_STEPS" \
        --buffer_capacity "$BUFFER_CAPACITY" \
        --batch_size      1024 \
        --seed            "$seed" \
        --eval_interval   20000 \
        --eval_episodes   10 \
        --lr_emb          1e-3 \
        --use_wandb \
        --wandb_project   "cola-history" \
        --wandb_run_name  "$run_name" \
        --wandb_group     "hist_${encoder}" \
        --wandb_tags      "history,${encoder},maddpg,${SCENARIO}_n${N_AGENTS}" \
        --save_model_path "models/hist_${run_name}.pth" \
        > "$logfile" 2>&1 &
    local pid=$!
    _jobs+=("$pid")
    echo "[$(date '+%H:%M:%S')] PID    ${pid}  -> ${run_name}"
}

TOTAL=$(( ${#ENCODERS[@]} * ${#SEEDS[@]} ))
echo "================================================================"
echo "  History encoder comparison: ${ENCODERS[*]}  x  ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "  Scenario: ${SCENARIO} (n_agents=${N_AGENTS})  |  steps: ${MAX_STEPS}  |  device: ${DEVICE}"
echo "  Max parallel: $MAX_JOBS  |  WandB: cola-history  |  Logs: $LOGDIR/"
echo "================================================================"

for ENCODER in "${ENCODERS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        skip=false
        for done in "${COMPLETED[@]+"${COMPLETED[@]}"}"; do
            if [ "$done" = "${ENCODER}_${SEED}" ]; then skip=true; break; fi
        done
        if [ "$skip" = true ]; then
            echo "[$(date '+%H:%M:%S')] SKIP   ${ENCODER}_seed${SEED} (already completed)"
            continue
        fi
        _wait_for_slot
        _launch "$ENCODER" "$SEED"
    done
done

echo ""
echo "All jobs launched — waiting for remaining ${#_jobs[@]} to finish..."
wait
echo ""
echo "================================================================"
echo "  History encoder comparison complete."
echo "  WandB project: cola-history  |  Logs: $LOGDIR/"
echo "================================================================"
