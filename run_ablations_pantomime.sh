#!/usr/bin/env bash
# run_ablations_pantomime.sh — Cooperative Pantomime control ablation.
#
# Paper-faithful showcase task for COLA (ConsensusLearning.pdf, Appendix A.1):
# simple_reference with the communication channel DISABLED. Two agents, three
# landmarks; each agent sees the OTHER agent's target but not its own, and there
# is no message channel — so each agent must infer its hidden goal from the
# partner's movement, and signal the partner's goal through its own movement.
# Genuine view diversity + a functional need to agree => the regime where the
# view-invariant consensus label should actually help (unlike uniform masking,
# which impoverished the input and re-collapsed the label).
#
# 3-variant control ladder (cola / shuffled / no_cola) over 3 seeds. random is
# dropped (its fixed per-agent label is absorbed by per-agent net biases).
#
# Logs to a SEPARATE WandB project (cola-pantomime). COMPLETED skips finished
# runs on re-run. n_agents is fixed at 2 by the env.
#
# Usage: bash run_ablations_pantomime.sh    (inside tmux for disconnect safety)

set -euo pipefail

# Pin per-process math threads so many parallel single-thread env loops pack
# cleanly onto the available cores instead of oversubscribing them.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON="${PYTHON:-$(command -v python)}"
MAX_JOBS=9
MAX_STEPS=1000000
PROJECT="cola-pantomime"
SCENARIO="simple_reference"
VARIANTS=(cola shuffled no_cola)
SEEDS=(42 43 44)

# Already-finished runs ("<variant>_<seed>") — fresh start -> empty.
COMPLETED=()

LOGDIR="logs/ablations_pantomime"
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
    return 0
}
_launch() {
    local variant="$1" seed="$2"
    local run_name="pantomime_${variant}_seed${seed}"
    local logfile="${LOGDIR}/${run_name}.log"
    echo "[$(date '+%H:%M:%S')] START  ${run_name}  (log: ${logfile})"
    PYTHONPATH=src "$PYTHON" -u train_cola.py \
        --scenario     "$SCENARIO" \
        --disable_comm \
        --device       cpu \
        --max_steps    "$MAX_STEPS" \
        --seed         "$seed" \
        --cb_variant   "$variant" \
        --eval_interval 20000 \
        --lr_emb       1e-3 \
        --use_wandb \
        --wandb_project  "$PROJECT" \
        --wandb_run_name "$run_name" \
        --wandb_group    "pantomime_${variant}" \
        --wandb_tags     "control_ablation,maddpg,${variant},pantomime,no_comm" \
        --save_model_path "models/ablation_pant_${run_name}.pth" \
        > "$logfile" 2>&1 &
    local pid=$!
    _jobs+=("$pid")
    echo "[$(date '+%H:%M:%S')] PID    ${pid}  -> ${run_name}"
}

TOTAL=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
echo "================================================================"
echo "  Cooperative Pantomime control ablation (simple_reference, no comm)"
echo "  ${#VARIANTS[@]} variants x ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "  Max parallel: $MAX_JOBS  |  Skipping completed: ${COMPLETED[*]:-(none)}"
echo "  WandB project: $PROJECT  |  Logs: $LOGDIR/"
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
echo "  All Pantomime ablation runs complete."
echo "  WandB project: $PROJECT"
echo "  Logs: $LOGDIR/"
echo "================================================================"
