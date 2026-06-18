"""Entrypoint for COLA + QMIX training on discrete-action MPE.

Run from repository root, for example:
  PYTHONPATH=src python train_qmix.py \\
    --scenario simple_spread --n_agents 3 --max_steps 1000000

Use --no_cola for the vanilla QMIX baseline (no consensus signal).
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

from cola_framework.buffers.replay_buffer import ReplayBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.critics.mixing_network import MixingNetwork
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.discrete_mpe_wrapper import DiscreteMPEWrapper
from cola_framework.evaluation.qmix_policy_evaluator import QMIXPolicyEvaluator
from cola_framework.loops.qmix_training_loop import QMIXTrainingConfig, QMIXTrainingLoop
from cola_framework.monitoring.wandb_logger import WandbLogger
from cola_framework.policies.q_network import QNetwork
from cola_framework.trainers.qmix_updater import QMIXUpdater


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train COLA + QMIX on discrete MPE.")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread",
                        choices=["simple_spread", "simple_tag"])
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--max_cycles", type=int, default=100)

    # Core model
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--mix_hidden_dim", type=int, default=32)

    # Baseline: disable COLA consensus signal
    parser.add_argument("--no_cola", action="store_true",
                        help="Run vanilla QMIX without COLA consensus signal.")

    # Replay and training
    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=1_024)
    parser.add_argument("--warmup_steps", type=int, default=1_024)
    parser.add_argument("--train_freq", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=2_000_000)

    # Epsilon-greedy exploration
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=500_000)

    # Optimization
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau_polyak", type=float, default=0.005)
    parser.add_argument("--lr_cb", type=float, default=3e-4)
    parser.add_argument("--lr_qmix", type=float, default=5e-4)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10_000)
    parser.add_argument("--reward_window", type=int, default=1_000)

    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=20_000,
                        help="Env steps between periodic greedy evals (0 disables).")

    # WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cola-marl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated WandB tags.")
    parser.add_argument("--api_key_path", type=str, default="src/apiKey.txt")

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_model_path", type=str,
                        default="models/final_qmix_model.pth")

    return parser


def _maybe_create_wandb_logger(args, config_payload: dict) -> Optional[WandbLogger]:
    if not args.use_wandb:
        return None
    key_path = args.api_key_path
    if not os.path.isabs(key_path):
        key_path = os.path.join(REPO_ROOT, key_path)
    tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
    group = args.wandb_group or "qmix_{}".format(args.scenario)
    return WandbLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=config_payload,
        tags=tags,
        group=group,
        mode=args.wandb_mode,
        api_key_path=key_path,
    )


def _build_timestamped_save_path(base_save_path: str, scenario: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem, ext = os.path.splitext(os.path.basename(base_save_path))
    if not ext:
        ext = ".pth"
    unique_name = "{}_{}_{}{}".format(stem, scenario, timestamp, ext)
    return os.path.join(os.path.dirname(base_save_path), unique_name)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    env = DiscreteMPEWrapper(
        scenario=args.scenario,
        n_agents=args.n_agents,
        max_cycles=args.max_cycles,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,   # = 1 (index storage)
        state_dim=env.state_dim,
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

    q_networks = [
        QNetwork(env.obs_dim, args.emb_dim, env.n_actions, args.hidden_dim).to(device)
        for _ in range(env.n_agents)
    ]
    target_q_networks = [copy.deepcopy(qn).to(device) for qn in q_networks]

    mixing_network = MixingNetwork(
        n_agents=env.n_agents,
        state_dim=env.state_dim,
        mix_hidden_dim=args.mix_hidden_dim,
    ).to(device)
    target_mixing_network = copy.deepcopy(mixing_network).to(device)

    opt_cb = torch.optim.Adam(consensus_builder.student.parameters(), lr=args.lr_cb)

    # All QMIX parameters share a single optimizer (matching original paper).
    qmix_params = (
        [p for qn in q_networks for p in qn.parameters()]
        + list(mixing_network.parameters())
    )
    opt_qmix = torch.optim.Adam(qmix_params, lr=args.lr_qmix)

    updater = QMIXUpdater(
        consensus_builder=consensus_builder,
        embedding_layer=embedding_layer,
        q_networks=q_networks,
        target_q_networks=target_q_networks,
        mixing_network=mixing_network,
        target_mixing_network=target_mixing_network,
        opt_cb=opt_cb,
        opt_qmix=opt_qmix,
        gamma=args.gamma,
        tau_polyak=args.tau_polyak,
        max_grad_norm=args.max_grad_norm,
    )

    loop_config = QMIXTrainingConfig(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        train_freq=args.train_freq,
        batch_size=args.batch_size,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        eps_decay_steps=args.eps_decay_steps,
        log_interval=args.log_interval,
        reward_window=args.reward_window,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    wandb_logger = _maybe_create_wandb_logger(args, run_config)

    def _on_log(record: dict) -> None:
        print(
            "[train] step={step:>8d}  ep_return={episode_return:>8.3f}"
            "  episodes={episode_count:>5d}  eps={epsilon:.4f}".format(**record)
        )
        if wandb_logger is not None:
            step = int(record["step"])
            wandb_record = {"train/" + k: v for k, v in record.items() if k != "step"}
            wandb_logger.log_metrics(wandb_record, step=step)

    evaluator = QMIXPolicyEvaluator(
        env=env,
        consensus_builder=consensus_builder,
        embedding_layer=embedding_layer,
        q_networks=q_networks,
    )

    def _on_eval(record: dict, step: int) -> None:
        print(
            "[eval ] step={:>8d}  eval_return={:>8.3f}  distinct={:>4.1f}"
            "  entropy={:>5.3f}".format(
                step,
                float(record.get("eval_episode_return", record.get("mean_episode_return", 0.0))),
                float(record.get("eval_distinct_classes", 0.0)),
                float(record.get("eval_consensus_entropy", 0.0)),
            )
        )
        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {"eval/" + k: v for k, v in record.items()}, step=int(step)
            )

    training_loop = QMIXTrainingLoop(
        env=env,
        replay_buffer=replay_buffer,
        consensus_builder=consensus_builder,
        embedding_layer=embedding_layer,
        q_networks=q_networks,
        updater=updater,
        config=loop_config,
        on_log=_on_log,
        evaluator=evaluator,
        on_eval=_on_eval,
    )

    train_result = training_loop.run()

    # Final summary eval (reuses the same evaluator instance built above).
    eval_metrics = evaluator.evaluate(n_episodes=args.eval_episodes)

    if wandb_logger is not None:
        eval_wandb = {"eval/" + k: v for k, v in eval_metrics.items()}
        wandb_logger.log_metrics(eval_wandb, step=int(train_result["total_steps"]))
        wandb_logger.finish(
            {
                "total_steps": train_result["total_steps"],
                "train_steps": train_result["train_steps"],
                "eval/episode_return": eval_metrics["mean_episode_return"],
                "eval/consensus_agreement": eval_metrics["consensus_agreement_eval"],
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
        "q_networks": [qn.state_dict() for qn in q_networks],
        "mixing_network": mixing_network.state_dict(),
    }
    torch.save(checkpoint, save_path)

    stem = os.path.splitext(save_path)[0]
    summary = {
        "train_result": train_result,
        "eval_metrics": eval_metrics,
        "model_path": save_path,
        "device": device,
    }

    with open(stem + "_full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(stem + "_logs.jsonl", "w") as f:
        for record in train_result.get("logs", []):
            f.write(json.dumps(record) + "\n")
    with open(stem + ".json", "w") as f:
        json.dump(
            {
                "checkpoint_path": save_path,
                "algorithm": "COLA-QMIX" if not args.no_cola else "QMIX",
                "scenario": args.scenario,
                "n_agents": env.n_agents,
                "n_actions": env.n_actions,
                "obs_dim": env.obs_dim,
                "state_dim": env.state_dim,
                "device": device,
                "train_result": {
                    "total_steps": train_result["total_steps"],
                    "train_steps": train_result["train_steps"],
                },
                "eval_metrics": eval_metrics,
            },
            f,
            indent=2,
        )

    print("Training completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
