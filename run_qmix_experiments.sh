#!/usr/bin/env bash
# run_qmix_experiments.sh — cross-paradigm reproduction: COLA-QMIX vs vanilla.
#
# Tests the "universal plug-in" claim on a VALUE-DECOMPOSITION method (QMIX), which
# is off-policy and discrete-action: the consensus embedding is appended to each
# per-agent utility network's input while the monotonic mixing network is left
# unchanged. Compares two label conditions, identical otherwise:
#   vanilla -> --no_cola   (NullConsensusBuilder, constant embedding == baseline QMIX)
#   cola    -> learned DINO consensus label (full method)
#
# Keep N_AGENTS / SCENARIO identical to the MADDPG headline ablation so the
# cola_qmix runs can be plotted next to cola_maddpg / cola_mappo (cross-paradigm panel).
#
# Logs to WandB project cola-qmix. COMPLETED skips finished runs on re-run.
# Usage: bash run_qmix_experiments.sh   (inside tmux)
#
# The QMIX loop now does PERIODIC greedy eval (--eval_interval), logging
# eval/eval_episode_return, eval/eval_distinct_classes and eval/eval_consensus_entropy
# as curves — parity with the MADDPG / MAPPO / history-aware paths for cross-paradigm plots.
#
# NOTE: train_qmix.py only supports simple_spread / simple_tag (no --obs_mask /
# --disable_comm) and has no --cb_variant, so QMIX is cola vs no_cola only, and
# the Cooperative Pantomime task stays MADDPG/MAPPO-only.

set -euo pipefail

PYTHON="${PYTHON:-$(command -v python)}"
MAX_JOBS=2                 # off-policy + replay buffer; 2 fits a free 4-core box comfortably
MAX_STEPS=1000000          # QMIX on discretised MPE; trim if it plateaus early
N_AGENTS=6                 # MATCH the MADDPG headline ablation (cross-paradigm)
SCENARIO="simple_spread"
VARIANTS=(vanilla cola)
SEEDS=(42 43 44)
LOG_INTERVAL=10000         # train-curve cadence
DEVICE="cpu"               # tiny feed-forward net: CPU is fine
EVAL_EPISODES=20           # episodes per periodic eval
EVAL_INTERVAL=20000        # env steps between evals (match MADDPG/MAPPO/history paths)
BUFFER_CAPACITY=250000     # 6-agent buffer is heavy; 250k matches the MADDPG runs

COMPLETED=()               # "<variant>_<seed>" entries to skip on re-run

LOGDIR="logs/qmix_experiments"
mkdir -p "$LOGDIR" models

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
    local variant="$1" seed="$2"
    # vanilla == baseline QMIX (no consensus); cola == learned label.
    local cola_flag=""
    [ "$variant" = "vanilla" ] && cola_flag="--no_cola"
    local run_name="${variant}_qmix_seed${seed}"
    local logfile="${LOGDIR}/${run_name}.log"
    echo "[$(date '+%H:%M:%S')] START  ${run_name}  (variant=${variant}, log: ${logfile})"
    PYTHONPATH=src "$PYTHON" -u train_qmix.py \
        --scenario        "$SCENARIO" \
        --n_agents        "$N_AGENTS" \
        $cola_flag \
        --device          "$DEVICE" \
        --max_steps       "$MAX_STEPS" \
        --buffer_capacity "$BUFFER_CAPACITY" \
        --log_interval    "$LOG_INTERVAL" \
        --eval_episodes   "$EVAL_EPISODES" \
        --eval_interval   "$EVAL_INTERVAL" \
        --seed            "$seed" \
        --use_wandb \
        --wandb_project   "cola-qmix" \
        --wandb_run_name  "$run_name" \
        --wandb_group     "qmix_${variant}" \
        --wandb_tags      "${variant},qmix,${SCENARIO}_n${N_AGENTS}" \
        --save_model_path "models/${run_name}.pth" \
        > "$logfile" 2>&1 &
    local pid=$!
    _jobs+=("$pid")
    echo "[$(date '+%H:%M:%S')] PID    ${pid}  -> ${run_name}"
}

TOTAL=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
echo "================================================================"
echo "  QMIX cross-paradigm: ${VARIANTS[*]}  x  ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "  Scenario: ${SCENARIO} (n_agents=${N_AGENTS})  |  steps: ${MAX_STEPS}  |  device: ${DEVICE}"
echo "  Max parallel: $MAX_JOBS  |  WandB: cola-qmix  |  Logs: $LOGDIR/"
echo "================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        skip=false
        for done in "${COMPLETED[@]+"${COMPLETED[@]}"}"; do
            if [ "$done" = "${VARIANT}_${SEED}" ]; then skip=true; break; fi
        done
        if [ "$skip" = true ]; then
            echo "[$(date '+%H:%M:%S')] SKIP   ${VARIANT}_seed${SEED} (already completed)"
            continue
        fi
        _wait_for_slot
        _launch "$VARIANT" "$SEED"
    done
done

echo ""
echo "All jobs launched — waiting for remaining ${#_jobs[@]} to finish..."
wait
echo ""
echo "================================================================"
echo "  QMIX cross-paradigm comparison complete."
echo "  WandB project: cola-qmix  |  Logs: $LOGDIR/"
echo "================================================================"
