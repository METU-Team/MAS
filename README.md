# COLA MARL Project

This project implements a modular COLA (Consensus Learning) framework on top of MADDPG for cooperative multi-agent reinforcement learning.

The codebase is designed so each major part of the training system lives in a separate module. This makes it easier to replace components, test them independently, and build additional tooling such as model watchers and video pipelines.

## What Is Implemented Right Now

The current implementation covers the full pipeline from environment interaction to training, evaluation, experiment logging, artifact saving, and post-training watching.

Current capabilities:

- Environment wrapper for PettingZoo MPE parallel environments.
- Replay buffer with random batch sampling.
- Consensus Builder (student-teacher style with EMA and centering).
- Consensus embedding module.
- Per-agent actor module.
- Centralized critic module.
- MADDPG update logic with COLA integration.
- Full training loop orchestrator.
- Evaluation loop for trained policies.
- WandB logger integration using API key file.
- Timestamped model checkpoint saving to avoid overwriting previous runs.
- Metadata sidecar JSON generation for each checkpoint.
- Watcher that loads checkpoints (latest by default), runs rollouts, and records mp4 videos.
- Step-by-step module gate tests for every major stage.

## Project Structure

Key files and folders:

- Main training entrypoint: [train_cola.py](train_cola.py)
- Main watcher entrypoint: [watch_cola.py](watch_cola.py)
- Framework package root: [src/cola_framework](src/cola_framework)
- Unit and gate tests: [tests](tests)
- Saved model artifacts: [models](models)
- Generated rollout videos: [eval_videos](eval_videos)
- WandB API key file path used by default: [src/apiKey.txt](src/apiKey.txt)

Framework modules are split by responsibility:

- Environments: [src/cola_framework/envs](src/cola_framework/envs)
- Buffers: [src/cola_framework/buffers](src/cola_framework/buffers)
- Consensus: [src/cola_framework/consensus](src/cola_framework/consensus)
- Embedding: [src/cola_framework/embedding](src/cola_framework/embedding)
- Policies (Actors): [src/cola_framework/policies](src/cola_framework/policies)
- Critics: [src/cola_framework/critics](src/cola_framework/critics)
- Trainers (Update Logic): [src/cola_framework/trainers](src/cola_framework/trainers)
- Training Loops: [src/cola_framework/loops](src/cola_framework/loops)
- Evaluation: [src/cola_framework/evaluation](src/cola_framework/evaluation)
- Monitoring: [src/cola_framework/monitoring](src/cola_framework/monitoring)
- Watchers: [src/cola_framework/watchers](src/cola_framework/watchers)
- Interface contracts: [src/cola_framework/interfaces](src/cola_framework/interfaces)

## End-to-End Data Flow

Training flow:

1. Environment produces observations.
2. Consensus Builder predicts per-agent consensus labels.
3. Embedding module transforms labels into dense vectors.
4. Actors consume local obs + embedding and output actions.
5. Environment steps and transitions are pushed into Replay Buffer.
6. Updater samples batches and applies:
   - Consensus loss update
   - Critic updates
   - Actor updates
   - Target network soft updates
7. Loop logs metrics and runs periodic tracking.
8. Final evaluation runs without exploration noise.
9. Checkpoint and metadata are saved.

Watcher flow:

1. Load checkpoint (latest by default from models folder).
2. Rebuild model modules from checkpoint configuration.
3. Run inference episodes.
4. Optionally record mp4 videos.
5. Return summary metrics and video file paths.

## Setup

## 1) Python Environment

The project has been developed with Python 3.9.

If you prefer creating your own environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Install Dependencies

```bash
pip install -r requirements.txt
```

## 3) Set Module Path

Commands in this README use:

```bash
PYTHONPATH=src
```

## 4) WandB API Key

For WandB online logging, place your key in:

- [src/apiKey.txt](src/apiKey.txt)

The training logger reads this file by default.

## Training Usage

Basic training example:

```bash
PYTHONPATH=src python train_cola.py \
  --scenario simple_spread \
  --n_agents 3 \
  --max_steps 50000 \
  --use_wandb
```

Quick smoke run:

```bash
PYTHONPATH=src python train_cola.py \
  --max_steps 20 \
  --warmup_steps 4 \
  --train_freq 2 \
  --batch_size 4 \
  --log_interval 10 \
  --eval_episodes 2 \
  --wandb_mode disabled
```

Useful training options:

- Scenario: `--scenario simple_spread|simple_tag`
- Device: `--device auto|cpu|cuda`
- WandB: `--use_wandb --wandb_project --wandb_run_name --wandb_entity`
- Output base path: `--save_model_path models/final_cola_model.pth`

## Saved Artifacts

Each training run now saves unique timestamped files to avoid overwrite:

- Checkpoint: `models/final_cola_model_<scenario>_YYYYMMDD_HHMMSS.pth`
- Metadata: `models/final_cola_model_<scenario>_YYYYMMDD_HHMMSS.json`

The metadata JSON is useful for automation tooling (watchers, render pipelines, transfer jobs), because it stores dimensions, scenario, and run summary.

## Watcher Usage

Default behavior: use the newest `.pth` file in the models folder.

Basic watcher run:

```bash
PYTHONPATH=src python watch_cola.py --episodes 1 --use_virtual_display
```

Use a specific checkpoint:

```bash
PYTHONPATH=src python watch_cola.py \
  --checkpoint_path models/final_cola_model_simple_spread_20260421_131523.pth \
  --episodes 1 \
  --use_virtual_display
```

No video mode (fast check):

```bash
PYTHONPATH=src python watch_cola.py --episodes 1 --no_video
```

Record video options:

- Output folder: `--output_dir eval_videos`
- FPS: `--fps 20`
- Prefix: `--video_prefix watch`
- Headless rendering support: `--use_virtual_display`

## Running Tests

Run all current gate tests:

```bash
PYTHONPATH=src python tests/test_step1_env_wrapper.py
PYTHONPATH=src python tests/test_step2_replay_buffer.py
PYTHONPATH=src python tests/test_step3_consensus_builder.py
PYTHONPATH=src python tests/test_step4_consensus_embedding.py
PYTHONPATH=src python tests/test_step5_actor.py
PYTHONPATH=src python tests/test_step6_critic.py
PYTHONPATH=src python tests/test_step7_update_logic.py
PYTHONPATH=src python tests/test_step8_training_loop.py
PYTHONPATH=src python tests/test_step9_logging_and_evaluation.py
PYTHONPATH=src python tests/test_watcher_checkpoint_selection.py
```

## Current Implementation Notes

- The actor outputs actions in `[-1, 1]`; environment wrapper maps these values to the environment action space before stepping.
- Consensus agreement is tracked during training updates.
- Training loop logs periodic metrics and supports external logger injection.
- Watcher loads checkpoints and can generate mp4 outputs.
