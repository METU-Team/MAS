"""Entrypoint for COLA + MAPPO training on MPE.

Run from repository root, for example:
  PYTHONPATH=src python train_mappo.py \\
    --scenario simple_spread --n_agents 3 --max_steps 1000000

Use --no_cola for the vanilla MAPPO baseline (no consensus signal).
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Optional

import torch

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.buffers.rollout_buffer import RolloutBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.critics.value_function import CentralizedValueFunction
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.evaluation.mappo_policy_evaluator import MAPPOPolicyEvaluator
from cola_framework.loops.mappo_training_loop import MAPPOTrainingConfig, MAPPOTrainingLoop
from cola_framework.monitoring.wandb_logger import WandbLogger
from cola_framework.policies.gaussian_actor import GaussianActor
from cola_framework.trainers.mappo_updater import MAPPOUpdater


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train COLA + MAPPO on MPE.")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread",
                        choices=["simple_spread", "simple_tag"])
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--max_cycles", type=int, default=100)

    # Core model
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)

    # Baseline: disable COLA consensus signal
    parser.add_argument("--no_cola", action="store_true",
                        help="Run vanilla MAPPO without COLA consensus signal.")

    # Rollout / PPO
    parser.add_argument("--n_rollout_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_minibatches", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--c_value", type=float, default=0.5)
    parser.add_argument("--c_entropy", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--log_std_init", type=float, default=-0.5)

    # GAE
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # Optimization
    parser.add_argument("--lr_cb", type=float, default=3e-4)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_value", type=float, default=1e-3)

    # Training loop
    parser.add_argument("--max_steps", type=int, default=2_000_000)
    parser.add_argument("--log_interval", type=int, default=10_000)
    parser.add_argument("--reward_window", type=int, default=1_000)

    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=20)

    # Logging / WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cola-marl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--api_key_path", type=str, default="src/apiKey.txt")

    # Runtime and outputs
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_model_path", type=str,
                        default="models/final_mappo_model.pth")

    return parser


def _maybe_create_wandb_logger(args, config_payload: dict) -> Optional[WandbLogger]:
    if not args.use_wandb:
        return None
    key_path = args.api_key_path
    if not os.path.isabs(key_path):
        key_path = os.path.join(REPO_ROOT, key_path)
    return WandbLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=config_payload,
        mode=args.wandb_mode,
        api_key_path=key_path,
    )


def _build_timestamped_save_path(base_save_path: str, scenario: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(base_save_path)
    stem, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".pth"
    unique_name = "{}_{}_{}{}".format(stem, scenario, timestamp, ext)
    return os.path.join(os.path.dirname(base_save_path), unique_name)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    env = MPEWrapper(
        scenario=args.scenario,
        n_agents=args.n_agents,
        max_cycles=args.max_cycles,
        device=device,
    )

    rollout_buffer = RolloutBuffer(
        n_steps=args.n_rollout_steps,
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        state_dim=env.state_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device,
    )

    if args.no_cola:
        consensus_builder = NullConsensusBuilder(k=args.k).to(device)
    else:
        consensus_builder = ConsensusBuilder(
            obs_dim=env.obs_dim,
            k=args.k,
            hidden_dim=args.hidden_dim,
        ).to(device)

    embedding_layer = ConsensusEmbedding(k=args.k, emb_dim=args.emb_dim).to(device)

    actors = [
        GaussianActor(
            obs_dim=env.obs_dim,
            emb_dim=args.emb_dim,
            action_dim=env.action_dim,
            hidden_dim=args.hidden_dim,
            log_std_init=args.log_std_init,
        ).to(device)
        for _ in range(env.n_agents)
    ]

    value_fns = [
        CentralizedValueFunction(
            state_dim=env.state_dim,
            n_agents=env.n_agents,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        for _ in range(env.n_agents)
    ]

    opt_cb = torch.optim.Adam(consensus_builder.student.parameters(), lr=args.lr_cb)
    opt_actors = [torch.optim.Adam(a.parameters(), lr=args.lr_actor) for a in actors]
    opt_values = [torch.optim.Adam(v.parameters(), lr=args.lr_value) for v in value_fns]

    updater = MAPPOUpdater(
        consensus_builder=consensus_builder,
        embedding_layer=embedding_layer,
        actors=actors,
        value_fns=value_fns,
        opt_cb=opt_cb,
        opt_actors=opt_actors,
        opt_values=opt_values,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        clip_eps=args.clip_eps,
        c_value=args.c_value,
        c_entropy=args.c_entropy,
        max_grad_norm=args.max_grad_norm,
    )

    loop_config = MAPPOTrainingConfig(
        max_steps=args.max_steps,
        n_rollout_steps=args.n_rollout_steps,
        log_interval=args.log_interval,
        reward_window=args.reward_window,
    )

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    wandb_logger = _maybe_create_wandb_logger(args, run_config)

    def _on_log(record: dict) -> None:
        print(
            "[train] step={step} mean_reward={mean_reward:.4f} "
            "updates={update_count}".format(**record)
        )
        if wandb_logger is not None:
            wandb_logger.log_metrics(record, step=int(record["step"]))

    training_loop = MAPPOTrainingLoop(
        env=env,
        rollout_buffer=rollout_buffer,
        consensus_builder=consensus_builder,
        embedding_layer=embedding_layer,
        actors=actors,
        value_fns=value_fns,
        updater=updater,
        config=loop_config,
        on_log=_on_log,
    )

    train_result = training_loop.run()

    evaluator = MAPPOPolicyEvaluator(
        env=env,
        consensus_builder=consensus_builder,
        embedding_layer=embedding_layer,
        actors=actors,
    )
    eval_metrics = evaluator.evaluate(n_episodes=args.eval_episodes)

    if wandb_logger is not None:
        wandb_logger.log_metrics(eval_metrics, step=int(train_result["total_steps"]))
        wandb_logger.finish(
            {
                "total_steps": train_result["total_steps"],
                "update_count": train_result["update_count"],
                "mean_episode_return_eval": eval_metrics["mean_episode_return"],
                "consensus_agreement_eval": eval_metrics["consensus_agreement_eval"],
            }
        )

    save_path = args.save_model_path
    if not os.path.isabs(save_path):
        save_path = os.path.join(REPO_ROOT, save_path)
    save_path = _build_timestamped_save_path(save_path, args.scenario)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "args": vars(args),
        "train_result": train_result,
        "eval_metrics": eval_metrics,
        "consensus_builder": consensus_builder.state_dict(),
        "embedding_layer": embedding_layer.state_dict(),
        "actors": [a.state_dict() for a in actors],
        "value_fns": [v.state_dict() for v in value_fns],
    }
    torch.save(checkpoint, save_path)

    stem = os.path.splitext(save_path)[0]
    summary = {
        "train_result": train_result,
        "eval_metrics": eval_metrics,
        "model_path": save_path,
        "device": device,
    }

    full_summary_path = stem + "_full_summary.json"
    with open(full_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logs_path = stem + "_logs.jsonl"
    with open(logs_path, "w", encoding="utf-8") as f:
        for record in train_result.get("logs", []):
            f.write(json.dumps(record) + "\n")

    metadata_path = stem + ".json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": save_path,
                "algorithm": "COLA-MAPPO" if not args.no_cola else "MAPPO",
                "scenario": args.scenario,
                "n_agents": env.n_agents,
                "obs_dim": env.obs_dim,
                "action_dim": env.action_dim,
                "state_dim": env.state_dim,
                "device": device,
                "train_result": {
                    "total_steps": train_result["total_steps"],
                    "update_count": train_result["update_count"],
                },
                "eval_metrics": eval_metrics,
            },
            f,
            indent=2,
        )

    summary["metadata_path"] = metadata_path
    print("Training completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
