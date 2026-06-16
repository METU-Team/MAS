"""Single project entrypoint for COLA + MADDPG training and evaluation.

Run from repository root, for example:
  PYTHONPATH=src /home/okan_saglam/masProje/grf_venv39/bin/python train_cola.py \
    --scenario simple_spread --n_agents 3 --max_steps 50000 --use_wandb
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
from cola_framework.buffers.sequence_buffer import SequenceReplayBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.history_aware_builder import HistoryAwareConsensusBuilder
from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.consensus.random_label_builder import RandomLabelConsensusBuilder
from cola_framework.consensus.shuffled_label_builder import ShuffledLabelConsensusBuilder
from cola_framework.critics.centralized_critic import Critic
from cola_framework.encoders.gru_encoder import GRUHistoryEncoder
from cola_framework.encoders.identity_encoder import IdentityHistoryEncoder
from cola_framework.encoders.transformer_encoder import TransformerHistoryEncoder
from cola_framework.encoders.window_encoder import WindowConcatEncoder
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.evaluation.history_aware_policy_evaluator import HistoryAwarePolicyEvaluator
from cola_framework.evaluation.policy_evaluator import PolicyEvaluator
from cola_framework.loops.cola_training_loop import COLATrainingConfig, COLATrainingLoop
from cola_framework.loops.history_aware_training_loop import HistoryAwareCOLATrainingLoop
from cola_framework.monitoring.wandb_logger import WandbLogger
from cola_framework.policies.actor import Actor
from cola_framework.trainers.history_aware_maddpg_updater import HistoryAwareMADDPGUpdater
from cola_framework.trainers.maddpg_updater import MADDPGUpdater
from cola_framework.utils.window_manager import ObservationWindowManager


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train COLA + MADDPG on MPE.")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", choices=["simple_spread", "simple_tag", "simple_reference"])
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument(
        "--disable_comm",
        action="store_true",
        help="Cooperative Pantomime: silence the communication channel in "
        "simple_reference so agents must infer hidden targets from behaviour.",
    )
    parser.add_argument(
        "--obs_mask",
        type=str,
        default="none",
        choices=["none", "others", "comm", "others_comm"],
        help="Partial-observability mask for simple_spread: hide other-agent "
        "relative positions and/or the communication channel.",
    )
    parser.add_argument("--max_cycles", type=int, default=100)

    # Core model
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)

    # Baseline: disable COLA (use NullConsensusBuilder) for fair comparison
    parser.add_argument("--no_cola", action="store_true", help="Run vanilla MADDPG without COLA consensus signal.")

    # Control-ablation: swap only the *content* of the consensus label while
    # keeping the surrounding architecture (embedding dim, actor/critic) identical.
    #   cola     -> real DINO-style ConsensusBuilder (learned view-invariant label)
    #   random   -> fixed random per-agent label, never changes (RandomLabelConsensusBuilder)
    #   shuffled -> real CB labels permuted across agents per batch element (ShuffledLabelConsensusBuilder)
    #   no_cola  -> all-zero label (NullConsensusBuilder)
    parser.add_argument(
        "--cb_variant",
        type=str,
        default="cola",
        choices=["cola", "random", "shuffled", "no_cola"],
        help="Consensus-builder content variant for control-ablation studies.",
    )

    # Optional history-aware path (classic path remains default)
    parser.add_argument("--use_history_path", action="store_true")
    parser.add_argument(
        "--history_encoder",
        type=str,
        default="gru",
        choices=["identity", "gru", "window", "transformer"],
    )
    parser.add_argument("--history_window", type=int, default=10)
    parser.add_argument("--history_gru_layers", type=int, default=1)
    parser.add_argument("--history_transformer_layers", type=int, default=2)
    parser.add_argument("--history_transformer_heads", type=int, default=4)

    # Optimization
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau_polyak", type=float, default=0.01)
    parser.add_argument("--lr_cb", type=float, default=3e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-2)
    parser.add_argument("--lr_critic", type=float, default=1e-2)
    # Embedding (one-hot -> dense) learning rate. Kept lower than the critic lr by
    # default since a shared module trained by all critics can destabilize at 1e-2.
    parser.add_argument("--lr_emb", type=float, default=1e-3)
    # Teacher-temperature warmup for the consensus builder (anti-collapse). Start
    # value <= 0 disables it (constant tau_teacher, the paper default).
    parser.add_argument("--cb_tau_teacher_warmup_start", type=float, default=0.0)
    parser.add_argument("--cb_tau_teacher_warmup_steps", type=int, default=0)

    # Replay and loop
    parser.add_argument("--buffer_capacity", type=int, default=250000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=1024)
    parser.add_argument("--train_freq", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=2000000)
    parser.add_argument("--noise_std_init", type=float, default=0.3)
    parser.add_argument("--noise_std_min", type=float, default=0.05)
    parser.add_argument("--noise_decay", type=float, default=0.9999)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--reward_window", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=20000)

    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=20)

    # Logging / WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cola-marl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group for comparing COLA vs baseline.")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated WandB tags.")
    parser.add_argument("--api_key_path", type=str, default="src/apiKey.txt")

    # Runtime and outputs
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_model_path", type=str, default="models/final_cola_model.pth")
    

    return parser


def _maybe_create_wandb_logger(args, config_payload: dict) -> Optional[WandbLogger]:
    if not args.use_wandb:
        return None

    key_path = args.api_key_path
    if not os.path.isabs(key_path):
        key_path = os.path.join(REPO_ROOT, key_path)

    tags = None
    if args.wandb_tags:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    # Default group: algorithm family so WandB can overlay COLA vs baseline.
    group = args.wandb_group
    if group is None:
        group = "maddpg_{}".format(args.scenario)

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
    """Return a unique checkpoint path per training run.

    Filename example:
      final_cola_model_simple_spread_20260421_143215.pth
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(base_save_path)
    stem, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".pth"
    unique_name = "{}_{}_{}{}".format(stem, scenario, timestamp, ext)
    return os.path.join(os.path.dirname(base_save_path), unique_name)


def _build_history_encoder(args, obs_dim: int):
    if args.history_encoder == "identity":
        if args.history_window != 1:
            raise ValueError("identity history encoder requires --history_window 1.")
        return IdentityHistoryEncoder(obs_dim=obs_dim)

    if args.history_encoder == "gru":
        return GRUHistoryEncoder(
            obs_dim=obs_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.history_gru_layers,
        )

    if args.history_encoder == "window":
        return WindowConcatEncoder(
            obs_dim=obs_dim,
            window=args.history_window,
            out_dim=args.hidden_dim,
        )

    if args.history_encoder == "transformer":
        return TransformerHistoryEncoder(
            obs_dim=obs_dim,
            out_dim=args.hidden_dim,
            nhead=args.history_transformer_heads,
            num_layers=args.history_transformer_layers,
            max_len=max(100, args.history_window),
        )

    raise ValueError("Unsupported history encoder: {}".format(args.history_encoder))


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
        obs_mask=args.obs_mask,
        disable_comm=args.disable_comm,
    )

    if args.use_history_path:
        replay_buffer = SequenceReplayBuffer(
            capacity=args.buffer_capacity,
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            state_dim=env.state_dim,
            window=args.history_window,
            device=device,
        )
        history_encoder = _build_history_encoder(args, env.obs_dim).to(device)
        consensus_builder = HistoryAwareConsensusBuilder(
            encoder=history_encoder,
            k=args.k,
            mlp_hidden=args.hidden_dim,
        ).to(device)
    else:
        replay_buffer = ReplayBuffer(
            capacity=args.buffer_capacity,
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            state_dim=env.state_dim,
            device=device,
        )
        # --no_cola is kept as a backward-compatible alias for the no_cola variant.
        cb_variant = "no_cola" if args.no_cola else args.cb_variant

        # All variants share K, embedding dim, hidden sizes, temperatures, and EMA
        # params so the only thing that differs across conditions is the *content*
        # of the consensus label, not the surrounding architecture.
        if cb_variant == "no_cola":
            consensus_builder = NullConsensusBuilder(k=args.k).to(device)
        elif cb_variant == "random":
            consensus_builder = RandomLabelConsensusBuilder(
                n_agents=env.n_agents,
                k=args.k,
                seed=args.seed,
            ).to(device)
        elif cb_variant == "shuffled":
            consensus_builder = ShuffledLabelConsensusBuilder(
                obs_dim=env.obs_dim,
                k=args.k,
                hidden_dim=args.hidden_dim,
                tau_teacher_warmup_start=args.cb_tau_teacher_warmup_start,
                tau_teacher_warmup_steps=args.cb_tau_teacher_warmup_steps,
            ).to(device)
        else:  # "cola"
            consensus_builder = ConsensusBuilder(
                obs_dim=env.obs_dim,
                k=args.k,
                hidden_dim=args.hidden_dim,
                tau_teacher_warmup_start=args.cb_tau_teacher_warmup_start,
                tau_teacher_warmup_steps=args.cb_tau_teacher_warmup_steps,
            ).to(device)

    embedding_layer = ConsensusEmbedding(
        k=args.k,
        emb_dim=args.emb_dim,
    ).to(device)

    actors = [
        Actor(
            obs_dim=env.obs_dim,
            emb_dim=args.emb_dim,
            action_dim=env.action_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        for _ in range(env.n_agents)
    ]

    critics = [
        Critic(
            state_dim=env.state_dim,
            n_agents=env.n_agents,
            emb_dim=args.emb_dim,
            action_dim=env.action_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        for _ in range(env.n_agents)
    ]

    target_actors = [copy.deepcopy(actor).to(device) for actor in actors]
    target_critics = [copy.deepcopy(critic).to(device) for critic in critics]

    if args.use_history_path:
        opt_cb = torch.optim.Adam(
            list(consensus_builder.student_encoder.parameters())
            + list(consensus_builder.student_head.parameters()),
            lr=args.lr_cb,
        )
    else:
        opt_cb = torch.optim.Adam(consensus_builder.student.parameters(), lr=args.lr_cb)

    opt_actors = [torch.optim.Adam(actor.parameters(), lr=args.lr_actor) for actor in actors]
    opt_critics = [torch.optim.Adam(critic.parameters(), lr=args.lr_critic) for critic in critics]
    # Train the one-hot -> dense embedding end-to-end with the critic loss (as in
    # the COLA paper). Shared across all variants so the architecture is identical
    # in the control ablation; only the consensus label content differs.
    opt_emb = torch.optim.Adam(embedding_layer.parameters(), lr=args.lr_emb)

    if args.use_history_path:
        updater = HistoryAwareMADDPGUpdater(
            consensus_builder=consensus_builder,
            embedding_layer=embedding_layer,
            actors=actors,
            critics=critics,
            target_actors=target_actors,
            target_critics=target_critics,
            opt_cb=opt_cb,
            opt_actors=opt_actors,
            opt_critics=opt_critics,
            gamma=args.gamma,
            tau_polyak=args.tau_polyak,
        )
    else:
        updater = MADDPGUpdater(
            consensus_builder=consensus_builder,
            embedding_layer=embedding_layer,
            actors=actors,
            critics=critics,
            target_actors=target_actors,
            target_critics=target_critics,
            opt_cb=opt_cb,
            opt_actors=opt_actors,
            opt_critics=opt_critics,
            gamma=args.gamma,
            tau_polyak=args.tau_polyak,
            opt_emb=opt_emb,
        )

    loop_config = COLATrainingConfig(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        train_freq=args.train_freq,
        batch_size=args.batch_size,
        noise_std_init=args.noise_std_init,
        noise_std_min=args.noise_std_min,
        noise_decay=args.noise_decay,
        log_interval=args.log_interval,
        reward_window=args.reward_window,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    # Surface the effective variant (after the --no_cola alias) for WandB grouping.
    if not args.use_history_path:
        run_config["cb_variant"] = "no_cola" if args.no_cola else args.cb_variant
    wandb_logger = _maybe_create_wandb_logger(args, run_config)

    def _on_log(record: dict) -> None:
        print(
            "[train] step={step:>8d}  ep_return={episode_return:>8.3f}"
            "  episodes={episode_count:>5d}  noise={noise_std:.4f}".format(**record)
        )
        if wandb_logger is not None:
            # Structured WandB panels: periodic greedy-eval keys go under eval/,
            # everything else under train/. Periodic eval is logged at every
            # eval_interval step, so eval/episode_return renders as a CURVE
            # (the paper-style learning curve), not a single end-of-run point.
            step = int(record["step"])
            wandb_record = {}
            for k, v in record.items():
                if k == "step":
                    continue
                if k.startswith("eval_"):
                    wandb_record["eval/" + k[len("eval_"):]] = v
                else:
                    wandb_record["train/" + k] = v
            wandb_logger.log_metrics(wandb_record, step=step)

    if args.use_history_path:
        window_manager = ObservationWindowManager(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            window=args.history_window,
            device=device,
        )
        training_loop = HistoryAwareCOLATrainingLoop(
            env=env,
            replay_buffer=replay_buffer,
            window_manager=window_manager,
            consensus_builder=consensus_builder,
            embedding_layer=embedding_layer,
            actors=actors,
            updater=updater,
            config=loop_config,
            on_log=_on_log,
        )
    else:
        training_loop = COLATrainingLoop(
            env=env,
            replay_buffer=replay_buffer,
            consensus_builder=consensus_builder,
            embedding_layer=embedding_layer,
            actors=actors,
            updater=updater,
            config=loop_config,
            on_log=_on_log,
        )

    train_result = training_loop.run()

    if args.use_history_path:
        eval_window_manager = ObservationWindowManager(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            window=args.history_window,
            device=device,
        )
        evaluator = HistoryAwarePolicyEvaluator(
            env=env,
            window_manager=eval_window_manager,
            consensus_builder=consensus_builder,
            embedding_layer=embedding_layer,
            actors=actors,
        )
    else:
        evaluator = PolicyEvaluator(
            env=env,
            consensus_builder=consensus_builder,
            embedding_layer=embedding_layer,
            actors=actors,
        )
    eval_metrics = evaluator.evaluate(n_episodes=args.eval_episodes)

    if wandb_logger is not None:
        # Send evaluation metrics with eval/ prefix.
        eval_wandb = {"eval/" + k: v for k, v in eval_metrics.items()}
        wandb_logger.log_metrics(eval_wandb, step=int(train_result["total_steps"]))
        wandb_logger.finish(
            {
                "total_steps": train_result["total_steps"],
                "train_steps": train_result["train_steps"],
                "final_noise_std": train_result["final_noise_std"],
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
        "actors": [a.state_dict() for a in actors],
        "critics": [c.state_dict() for c in critics],
        "target_actors": [a.state_dict() for a in target_actors],
        "target_critics": [c.state_dict() for c in target_critics],
    }
    torch.save(checkpoint, save_path)

    summary = {
        "train_result": train_result,
        "eval_metrics": eval_metrics,
        "model_path": save_path,
        "device": device,
    }

    stem = os.path.splitext(save_path)[0]

    # Full summary mirrors the final console JSON and includes all periodic logs.
    full_summary_path = stem + "_full_summary.json"
    with open(full_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # JSONL log file is convenient for large runs and streaming tools.
    logs_path = stem + "_logs.jsonl"
    with open(logs_path, "w", encoding="utf-8") as f:
        for record in train_result.get("logs", []):
            f.write(json.dumps(record) + "\n")

    # Sidecar metadata helps watcher/video pipelines discover fresh checkpoints.
    metadata_path = stem + ".json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": save_path,
                "scenario": args.scenario,
                "n_agents": env.n_agents,
                "obs_dim": env.obs_dim,
                "action_dim": env.action_dim,
                "state_dim": env.state_dim,
                "device": device,
                "train_result": {
                    "total_steps": train_result["total_steps"],
                    "train_steps": train_result["train_steps"],
                    "final_noise_std": train_result["final_noise_std"],
                },
                "eval_metrics": eval_metrics,
                "full_summary_path": full_summary_path,
                "logs_path": logs_path,
            },
            f,
            indent=2,
        )

    summary["metadata_path"] = metadata_path
    summary["full_summary_path"] = full_summary_path
    summary["logs_path"] = logs_path

    print("Training completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
