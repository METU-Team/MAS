#!/usr/bin/env bash
# run_mappo_experiments.sh — cross-paradigm reproduction: COLA-MAPPO vs vanilla.
#
# Tests the "universal plug-in" claim on an ON-POLICY actor-critic (MAPPO), which
# has NO replay buffer: the consensus builder is trained from freshly collected
# trajectories. Compares two label conditions, identical otherwise:
#   vanilla -> --no_cola   (NullConsensusBuilder, constant embedding == baseline MAPPO)
#   cola    -> learned DINO consensus label (full method)
#
# Keep N_AGENTS / SCENARIO identical to the MADDPG headline ablation so the
# cola_mappo runs can be plotted next to cola_maddpg (cross-paradigm panel).
#
# Logs to WandB project cola-mappo. COMPLETED skips finished runs on re-run.
# Usage: bash run_mappo_experiments.sh   (inside tmux)
#
# The MAPPO loop now does PERIODIC greedy eval (--eval_interval), logging
# eval/eval_episode_return, eval/eval_distinct_classes and eval/eval_consensus_entropy
# as curves — parity with the MADDPG / history-aware paths for cross-paradigm plots.

set -euo pipefail

PYTHON="${PYTHON:-$(command -v python)}"
MAX_JOBS=2                 # on-policy, CPU; 2 fits a free 4-core box comfortably
MAX_STEPS=2000000          # MAPPO is sample-inefficient; trim if it plateaus early
N_AGENTS=6                 # MATCH the MADDPG headline ablation (cross-paradigm)
SCENARIO="simple_spread"
VARIANTS=(vanilla cola)
SEEDS=(42 43 44)
LOG_INTERVAL=10000         # train-curve cadence (rounds up to n_rollout_steps)
DEVICE="cpu"               # tiny net + on-policy: CPU is fine
EVAL_EPISODES=20           # episodes per periodic eval
EVAL_INTERVAL=20000        # env steps between evals (match MADDPG/history paths)

COMPLETED=()               # "<variant>_<seed>" entries to skip on re-run

LOGDIR="logs/mappo_experiments"
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
    # vanilla == baseline MAPPO (no consensus); cola == learned label.
    local cola_flag=""
    [ "$variant" = "vanilla" ] && cola_flag="--no_cola"
    local run_name="${variant}_mappo_seed${seed}"
    local logfile="${LOGDIR}/${run_name}.log"
    echo "[$(date '+%H:%M:%S')] START  ${run_name}  (variant=${variant}, log: ${logfile})"
    PYTHONPATH=src "$PYTHON" -u train_mappo.py \
        --scenario        "$SCENARIO" \
        --n_agents        "$N_AGENTS" \
        $cola_flag \
        --device          "$DEVICE" \
        --max_steps       "$MAX_STEPS" \
        --log_interval    "$LOG_INTERVAL" \
        --eval_episodes   "$EVAL_EPISODES" \
        --eval_interval   "$EVAL_INTERVAL" \
        --seed            "$seed" \
        --use_wandb \
        --wandb_project   "cola-mappo" \
        --wandb_run_name  "$run_name" \
        --wandb_group     "mappo_${variant}" \
        --wandb_tags      "${variant},mappo,${SCENARIO}_n${N_AGENTS}" \
        --save_model_path "models/${run_name}.pth" \
        > "$logfile" 2>&1 &
    local pid=$!
    _jobs+=("$pid")
    echo "[$(date '+%H:%M:%S')] PID    ${pid}  -> ${run_name}"
}

TOTAL=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
echo "================================================================"
echo "  MAPPO cross-paradigm: ${VARIANTS[*]}  x  ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "  Scenario: ${SCENARIO} (n_agents=${N_AGENTS})  |  steps: ${MAX_STEPS}  |  device: ${DEVICE}"
echo "  Max parallel: $MAX_JOBS  |  WandB: cola-mappo  |  Logs: $LOGDIR/"
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
echo "  MAPPO cross-paradigm comparison complete."
echo "  WandB project: cola-mappo  |  Logs: $LOGDIR/"
echo "================================================================"
