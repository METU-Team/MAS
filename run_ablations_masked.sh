#!/usr/bin/env bash
# run_ablations_masked.sh — partial-observability control ablation, 6-agent
# simple_spread, 3 jobs.
#
# Identical setup to run_ablations_parallel.sh BUT with --obs_mask others: each
# agent sees its own velocity/position and the landmarks, but NOT the other
# agents' relative positions. This is the regime where COLA's learned, view-
# invariant consensus label is theoretically motivated — agents can no longer
# read neighbours directly and must coordinate through the consensus signal.
# Expectation: the cola - shuffled gap should widen vs the fully-observable run.
#
# Runs the 3-variant control ladder (cola / shuffled / no_cola) over 3 seeds.
# random is dropped: with per-agent actor/critic nets its fixed per-agent label
# is absorbed by the bias, so it is ~equivalent to no_cola.
#
# Logs to a SEPARATE WandB project so it never mixes with the fully-observable
# ablation. Already-completed runs are listed in COMPLETED and skipped, so the
# script is safe to re-run after an interruption.
#
# Usage: bash run_ablations_masked.sh
#   SIGHUP-safe launch:  setsid bash run_ablations_masked.sh &  (or use tmux)

set -euo pipefail

PYTHON="/home/ahmetysnocak/miniconda3/envs/mas_env/bin/python"
MAX_JOBS=3
MAX_STEPS=1000000
N_AGENTS=6
OBS_MASK=others
PROJECT="cola-6agent-masked"
SCENARIOS=(simple_spread)
VARIANTS=(cola shuffled no_cola)
SEEDS=(42 43 44)

# Already-finished runs ("<scenario>_<variant>_<seed>") — fresh start -> empty.
COMPLETED=()

LOGDIR="logs/ablations_6agent_masked"
mkdir -p "$LOGDIR"

# Job pool helpers
_jobs=()
_wait_for_slot() {
    while [ "${#_jobs[@]}" -ge "$MAX_JOBS" ]; do
        local new_jobs=()
        for pid in "${_jobs[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_jobs+=("$pid")
            fi
        done
        _jobs=("${new_jobs[@]+"${new_jobs[@]}"}")
        if [ "${#_jobs[@]}" -ge "$MAX_JOBS" ]; then sleep 5; fi
    done
    return 0   # never let a false test leak out and trip `set -e`
}
_launch() {
    local scenario="$1" variant="$2" seed="$3"
    local run_name="${scenario}_${variant}_seed${seed}_masked"
    local logfile="${LOGDIR}/${run_name}.log"
    echo "[$(date '+%H:%M:%S')] START  ${run_name}  (log: ${logfile})"
    PYTHONPATH=src "$PYTHON" train_cola.py \
        --scenario     "$scenario" \
        --n_agents     "$N_AGENTS" \
        --obs_mask     "$OBS_MASK" \
        --max_steps    "$MAX_STEPS" \
        --seed         "$seed" \
        --cb_variant   "$variant" \
        --eval_interval 20000 \
        --lr_emb       1e-3 \
        --use_wandb \
        --wandb_project  "$PROJECT" \
        --wandb_run_name "$run_name" \
        --wandb_group    "${scenario}_${variant}" \
        --wandb_tags     "control_ablation,maddpg,${variant},${scenario},obs_mask_${OBS_MASK},partial_obs" \
        --save_model_path "models/ablation6a_masked_${run_name}.pth" \
        > "$logfile" 2>&1 &
    local pid=$!
    _jobs+=("$pid")
    echo "[$(date '+%H:%M:%S')] PID    ${pid}  -> ${run_name}"
}

TOTAL=$(( ${#SCENARIOS[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]} ))
echo "================================================================"
echo "  Partial-obs control ablation (obs_mask=${OBS_MASK})"
echo "  ${#SCENARIOS[@]} scenarios x ${#VARIANTS[@]} variants x ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "  Scenarios: ${SCENARIOS[*]}  |  N_AGENTS: ${N_AGENTS}"
echo "  Max parallel: $MAX_JOBS  |  Skipping completed: ${COMPLETED[*]:-(none)}"
echo "  WandB project: $PROJECT  |  Logs: $LOGDIR/"
echo "================================================================"

for SCENARIO in "${SCENARIOS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Skip already-completed runs
            skip=false
            for done in "${COMPLETED[@]+"${COMPLETED[@]}"}"; do
                if [ "$done" = "${SCENARIO}_${VARIANT}_${SEED}" ]; then skip=true; break; fi
            done
            if [ "$skip" = true ]; then
                echo "[$(date '+%H:%M:%S')] SKIP   ${SCENARIO}_${VARIANT}_seed${SEED} (already completed)"
                continue
            fi
            _wait_for_slot
            _launch "$SCENARIO" "$VARIANT" "$SEED"
        done
    done
done

# Wait for all remaining jobs
echo ""
echo "All jobs launched — waiting for remaining ${#_jobs[@]} to finish..."
wait
echo ""
echo "================================================================"
echo "  All masked-ablation runs complete."
echo "  WandB project: $PROJECT"
echo "  Logs: $LOGDIR/"
echo "================================================================"
