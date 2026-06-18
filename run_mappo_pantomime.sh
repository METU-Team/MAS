#!/usr/bin/env bash
# run_mappo_pantomime.sh — Cooperative Pantomime on MAPPO: COLA vs vanilla.
#
# Pantomime = simple_reference (2 agents, 3 landmarks; each agent sees the OTHER
# agent's target colour but NOT its own) with --disable_comm (message channel
# silenced), so agents must infer their goal from the partner's BEHAVIOUR alone.
# This is the most paper-faithful genuinely-partially-observable MPE task, i.e.
# the setting where COLA's consensus signal is supposed to help.
#
#   vanilla -> --no_cola  (NullConsensusBuilder, constant embedding == baseline)
#   cola    -> learned DINO consensus label
#
# Logs to WandB project cola-mappo-pantomime (separate from the simple_spread
# cross-paradigm runs). Periodic eval is on (eval_episode_return + distinct +
# entropy). Usage: bash run_mappo_pantomime.sh   (inside tmux)
#
# NOTE: Pantomime is 2-agent -> collapse-prone + high variance (see EXPERIMENTS_LOG
# #3, MADDPG). Run several seeds; do not over-read a single seed.

set -euo pipefail

# Pantomime is tiny (2 agents); pack single-thread runs onto the free 4-core box.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON="${PYTHON:-$(command -v python)}"
MAX_JOBS=4                 # 2 agents -> very light; 4 fit a free 4-core box
MAX_STEPS=1000000          # 2-agent reference; trim if it plateaus earlier
SCENARIO="simple_reference"
VARIANTS=(vanilla cola)
SEEDS=(42 43 44)
LOG_INTERVAL=10000
DEVICE="cpu"
EVAL_EPISODES=20
EVAL_INTERVAL=20000

COMPLETED=()               # "<variant>_<seed>" entries to skip on re-run

LOGDIR="logs/mappo_pantomime"
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
    local cola_flag=""
    [ "$variant" = "vanilla" ] && cola_flag="--no_cola"
    local run_name="${variant}_pantomime_seed${seed}"
    local logfile="${LOGDIR}/${run_name}.log"
    echo "[$(date '+%H:%M:%S')] START  ${run_name}  (variant=${variant}, log: ${logfile})"
    PYTHONPATH=src "$PYTHON" -u train_mappo.py \
        --scenario        "$SCENARIO" \
        --disable_comm \
        $cola_flag \
        --device          "$DEVICE" \
        --max_steps       "$MAX_STEPS" \
        --log_interval    "$LOG_INTERVAL" \
        --eval_episodes   "$EVAL_EPISODES" \
        --eval_interval   "$EVAL_INTERVAL" \
        --seed            "$seed" \
        --use_wandb \
        --wandb_project   "cola-mappo-pantomime" \
        --wandb_run_name  "$run_name" \
        --wandb_group     "mappo_${variant}" \
        --wandb_tags      "${variant},mappo,pantomime,simple_reference" \
        --save_model_path "models/${run_name}.pth" \
        > "$logfile" 2>&1 &
    local pid=$!
    _jobs+=("$pid")
    echo "[$(date '+%H:%M:%S')] PID    ${pid}  -> ${run_name}"
}

TOTAL=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
echo "================================================================"
echo "  MAPPO Pantomime: ${VARIANTS[*]}  x  ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "  Scenario: ${SCENARIO} + disable_comm (2 agents)  |  steps: ${MAX_STEPS}"
echo "  Max parallel: $MAX_JOBS  |  WandB: cola-mappo-pantomime  |  Logs: $LOGDIR/"
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
echo "  MAPPO Pantomime comparison complete."
echo "  WandB project: cola-mappo-pantomime  |  Logs: $LOGDIR/"
echo "================================================================"
