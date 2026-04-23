"""Watcher for running trained checkpoints and recording rollout videos."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import torch

from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.history_aware_builder import HistoryAwareConsensusBuilder
from cola_framework.encoders.gru_encoder import GRUHistoryEncoder
from cola_framework.encoders.identity_encoder import IdentityHistoryEncoder
from cola_framework.encoders.transformer_encoder import TransformerHistoryEncoder
from cola_framework.encoders.window_encoder import WindowConcatEncoder
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.policies.actor import Actor
from cola_framework.utils.window_manager import ObservationWindowManager


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _args_get(run_args: Any, key: str, default: Any) -> Any:
    if isinstance(run_args, dict):
        return run_args.get(key, default)
    return getattr(run_args, key, default)


def _is_history_checkpoint(run_args: Any, consensus_state: Dict[str, torch.Tensor]) -> bool:
    configured = bool(_args_get(run_args, "use_history_path", False))
    if configured:
        return True
    return any(name.startswith("student_encoder.") for name in consensus_state.keys())


def _build_history_encoder(run_args: Any, obs_dim: int):
    history_encoder = str(_args_get(run_args, "history_encoder", "gru"))
    hidden_dim = int(_args_get(run_args, "hidden_dim", 64))
    history_window = int(_args_get(run_args, "history_window", 10))

    if history_encoder == "identity":
        if history_window != 1:
            raise ValueError("identity history encoder requires history_window=1 in checkpoint args.")
        return IdentityHistoryEncoder(obs_dim=obs_dim)

    if history_encoder == "gru":
        return GRUHistoryEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=int(_args_get(run_args, "history_gru_layers", 1)),
        )

    if history_encoder == "window":
        return WindowConcatEncoder(
            obs_dim=obs_dim,
            window=history_window,
            out_dim=hidden_dim,
        )

    if history_encoder == "transformer":
        return TransformerHistoryEncoder(
            obs_dim=obs_dim,
            out_dim=hidden_dim,
            nhead=int(_args_get(run_args, "history_transformer_heads", 4)),
            num_layers=int(_args_get(run_args, "history_transformer_layers", 2)),
            max_len=max(100, history_window),
        )

    raise ValueError("Unsupported history encoder in checkpoint args: {}".format(history_encoder))


def find_latest_checkpoint(models_dir: str, suffix: str = ".pth") -> str:
    """Return newest checkpoint file path in models_dir."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError("Models directory not found: {}".format(models_dir))

    candidates = []
    for name in os.listdir(models_dir):
        if name.endswith(suffix):
            path = os.path.join(models_dir, name)
            if os.path.isfile(path):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError("No checkpoint files found in {}".format(models_dir))

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


class PolicyWatcher:
    """Loads a trained checkpoint and runs/records deterministic rollouts."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        models_dir: str = "models",
        device: str = "auto",
        scenario_override: Optional[str] = None,
        n_agents_override: Optional[int] = None,
        max_cycles_override: Optional[int] = None,
        render_mode: str = "rgb_array",
    ) -> None:
        self.device = _resolve_device(device)

        if checkpoint_path is None:
            self.checkpoint_path = find_latest_checkpoint(models_dir=models_dir, suffix=".pth")
        else:
            self.checkpoint_path = checkpoint_path

        if not os.path.isabs(self.checkpoint_path):
            self.checkpoint_path = os.path.abspath(self.checkpoint_path)
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(self.checkpoint_path))

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if "args" not in checkpoint:
            raise ValueError("Checkpoint does not contain expected 'args' field.")
        if "consensus_builder" not in checkpoint:
            raise ValueError("Checkpoint does not contain expected 'consensus_builder' state.")

        run_args = checkpoint["args"]
        scenario = scenario_override or _args_get(run_args, "scenario", "simple_spread")
        n_agents = int(n_agents_override or _args_get(run_args, "n_agents", 3))
        max_cycles = int(max_cycles_override or _args_get(run_args, "max_cycles", 100))

        self.env = MPEWrapper(
            scenario=scenario,
            n_agents=n_agents,
            max_cycles=max_cycles,
            device=self.device,
            render_mode=render_mode,
        )

        k = int(_args_get(run_args, "k", 4))
        emb_dim = int(_args_get(run_args, "emb_dim", 16))
        hidden_dim = int(_args_get(run_args, "hidden_dim", 64))

        consensus_state = checkpoint["consensus_builder"]
        self.use_history_path = _is_history_checkpoint(run_args, consensus_state)
        self.window_manager = None

        if self.use_history_path:
            history_window = int(_args_get(run_args, "history_window", 10))
            history_encoder = _build_history_encoder(run_args, self.env.obs_dim).to(self.device)
            self.consensus_builder = HistoryAwareConsensusBuilder(
                encoder=history_encoder,
                k=k,
                mlp_hidden=hidden_dim,
            ).to(self.device)
            self.window_manager = ObservationWindowManager(
                n_agents=self.env.n_agents,
                obs_dim=self.env.obs_dim,
                window=history_window,
                device=self.device,
            )
        else:
            self.consensus_builder = ConsensusBuilder(
                obs_dim=self.env.obs_dim,
                k=k,
                hidden_dim=hidden_dim,
            ).to(self.device)
        self.embedding_layer = ConsensusEmbedding(k=k, emb_dim=emb_dim).to(self.device)
        self.actors = [
            Actor(
                obs_dim=self.env.obs_dim,
                emb_dim=emb_dim,
                action_dim=self.env.action_dim,
                hidden_dim=hidden_dim,
            ).to(self.device)
            for _ in range(self.env.n_agents)
        ]

        self.consensus_builder.load_state_dict(consensus_state)
        self.embedding_layer.load_state_dict(checkpoint["embedding_layer"])

        if len(checkpoint["actors"]) != self.env.n_agents:
            raise ValueError(
                "Checkpoint actor count ({}) does not match environment n_agents ({}).".format(
                    len(checkpoint["actors"]),
                    self.env.n_agents,
                )
            )

        for i, actor_state in enumerate(checkpoint["actors"]):
            self.actors[i].load_state_dict(actor_state)

        self.consensus_builder.eval()
        self.embedding_layer.eval()
        for actor in self.actors:
            actor.eval()

    @torch.no_grad()
    def _act(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_history_path:
            if self.window_manager is None:
                raise RuntimeError("window_manager must be initialized for history-aware checkpoints.")
            obs_seq = self.window_manager.get()
            consensus = self.consensus_builder.infer(obs_seq)
        else:
            consensus = self.consensus_builder.infer(obs)

        cemb = self.embedding_layer(consensus)
        actions = torch.stack(
            [self.actors[a](obs[a], cemb[a]) for a in range(self.env.n_agents)],
            dim=0,
        )
        return actions.clamp(-1.0, 1.0)

    def watch(
        self,
        n_episodes: int = 1,
        output_dir: str = "eval_videos",
        fps: int = 20,
        seed: Optional[int] = None,
        save_video: bool = True,
        video_prefix: str = "watch",
    ) -> Dict[str, object]:
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_stem = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
        episode_returns = []
        video_paths = []

        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            obs, _ = self.env.reset(seed=ep_seed)

            if self.use_history_path:
                if self.window_manager is None:
                    raise RuntimeError("window_manager must be initialized for history-aware checkpoints.")
                self.window_manager.reset()
                self.window_manager.push(obs)

            done = False
            ep_return = 0.0
            frames = []
            video_enabled = save_video

            while not done:
                if video_enabled:
                    frame = self.env.render_frame()
                    if frame is not None:
                        frames.append(frame)
                    else:
                        # Keep rollout running even if rendering is unavailable.
                        video_enabled = False

                actions = self._act(obs)
                obs, _, rewards, dones = self.env.step(actions)

                if self.use_history_path:
                    self.window_manager.push(obs)

                ep_return += float(rewards.mean().item())
                done = bool(dones.all().item())

            if video_enabled and frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = "{}_{}_ep{:03d}_{}.mp4".format(
                    video_prefix,
                    checkpoint_stem,
                    ep,
                    timestamp,
                )
                video_path = os.path.join(output_dir, filename)
                imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
                video_paths.append(video_path)

            episode_returns.append(ep_return)

        mean_return = float(sum(episode_returns) / max(1, len(episode_returns)))
        return {
            "checkpoint_path": self.checkpoint_path,
            "episodes": n_episodes,
            "mean_episode_return": mean_return,
            "episode_returns": episode_returns,
            "video_paths": video_paths,
        }

    def close(self) -> None:
        self.env.close()
