#!/usr/bin/env bash
# run_experiments.sh — Launch all 6 comparison runs (1.5 M steps each).
#
# Usage:
#   bash run_experiments.sh [wandb_project] [wandb_entity]
#
# Runs in order: MADDPG, COLA-MADDPG, MAPPO, COLA-MAPPO, QMIX, COLA-QMIX.
# All logs go to WandB project <wandb_project> (default: cola-marl).
# Checkpoints saved to models/ with timestamped names.
#
# Requirements: PYTHONPATH=src must resolve cola_framework.
# Set USE_WANDB=1 to enable WandB logging.

set -euo pipefail

PYTHON="${PYTHON:-/home/okan_saglam/masProje/grf_venv39/bin/python}"
WANDB_PROJECT="${1:-cola-marl}"
WANDB_ENTITY="${2:-}"
MAX_STEPS=1500000
SCENARIO="simple_spread"
N_AGENTS=3
SEED=42

USE_WANDB="${USE_WANDB:-0}"
WANDB_FLAG=""
ENTITY_FLAG=""
if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAG="--use_wandb"
    [ -n "$WANDB_ENTITY" ] && ENTITY_FLAG="--wandb_entity $WANDB_ENTITY"
fi

echo "============================================================"
echo "  COLA MARL — Comparison Experiments"
echo "  Steps: $MAX_STEPS | Scenario: $SCENARIO | WandB: $USE_WANDB"
echo "============================================================"

# ── 1. Vanilla MADDPG (baseline, no COLA) ─────────────────────────────────
echo ""
echo "[1/6] Vanilla MADDPG (baseline)"
PYTHONPATH=src $PYTHON train_cola.py \
  --scenario "$SCENARIO" \
  --n_agents "$N_AGENTS" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --no_cola \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "maddpg_${SCENARIO}" \
  --wandb_group "maddpg_${SCENARIO}" \
  --wandb_tags "baseline,maddpg,no_cola" \
  $WANDB_FLAG $ENTITY_FLAG \
  --save_model_path "models/maddpg_baseline.pth"

# ── 2. COLA-MADDPG ──────────────────────────────────────────────────────────
echo ""
echo "[2/6] COLA-MADDPG"
PYTHONPATH=src $PYTHON train_cola.py \
  --scenario "$SCENARIO" \
  --n_agents "$N_AGENTS" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "cola_maddpg_${SCENARIO}" \
  --wandb_group "maddpg_${SCENARIO}" \
  --wandb_tags "cola,maddpg" \
  $WANDB_FLAG $ENTITY_FLAG \
  --save_model_path "models/cola_maddpg.pth"

# ── 3. Vanilla MAPPO (baseline, no COLA) ──────────────────────────────────
echo ""
echo "[3/6] Vanilla MAPPO (baseline)"
PYTHONPATH=src $PYTHON train_mappo.py \
  --scenario "$SCENARIO" \
  --n_agents "$N_AGENTS" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --no_cola \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "mappo_${SCENARIO}" \
  --wandb_group "mappo_${SCENARIO}" \
  --wandb_tags "baseline,mappo,no_cola" \
  $WANDB_FLAG $ENTITY_FLAG \
  --save_model_path "models/mappo_baseline.pth"

# ── 4. COLA-MAPPO ────────────────────────────────────────────────────────────
echo ""
echo "[4/6] COLA-MAPPO"
PYTHONPATH=src $PYTHON train_mappo.py \
  --scenario "$SCENARIO" \
  --n_agents "$N_AGENTS" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "cola_mappo_${SCENARIO}" \
  --wandb_group "mappo_${SCENARIO}" \
  --wandb_tags "cola,mappo" \
  $WANDB_FLAG $ENTITY_FLAG \
  --save_model_path "models/cola_mappo.pth"

# ── 5. Vanilla QMIX (baseline, no COLA, discrete actions) ─────────────────
echo ""
echo "[5/6] Vanilla QMIX (baseline)"
PYTHONPATH=src $PYTHON train_qmix.py \
  --scenario "$SCENARIO" \
  --n_agents "$N_AGENTS" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --no_cola \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "qmix_${SCENARIO}" \
  --wandb_group "qmix_${SCENARIO}" \
  --wandb_tags "baseline,qmix,no_cola" \
  $WANDB_FLAG $ENTITY_FLAG \
  --save_model_path "models/qmix_baseline.pth"

# ── 6. COLA-QMIX ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] COLA-QMIX"
PYTHONPATH=src $PYTHON train_qmix.py \
  --scenario "$SCENARIO" \
  --n_agents "$N_AGENTS" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "cola_qmix_${SCENARIO}" \
  --wandb_group "qmix_${SCENARIO}" \
  --wandb_tags "cola,qmix" \
  $WANDB_FLAG $ENTITY_FLAG \
  --save_model_path "models/cola_qmix.pth"

echo ""
echo "============================================================"
echo "  All 6 runs complete."
echo "  WandB project: $WANDB_PROJECT"
echo "  Checkpoints:   models/"
echo ""
echo "  To generate videos after training:"
echo "    PYTHONPATH=src $PYTHON watch_cola.py  --no_video  # MADDPG"
echo "    PYTHONPATH=src $PYTHON watch_mappo.py --no_video  # MAPPO"
echo "    PYTHONPATH=src $PYTHON watch_qmix.py  --no_video  # QMIX"
echo "============================================================"
