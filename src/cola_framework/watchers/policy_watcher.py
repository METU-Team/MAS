"""Watcher for running trained checkpoints and recording rollout videos."""

import os
from datetime import datetime
from typing import Dict, List, Optional

import imageio.v2 as imageio
import torch

from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.policies.actor import Actor


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


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

        run_args = checkpoint["args"]
        scenario = scenario_override or run_args.get("scenario", "simple_spread")
        n_agents = int(n_agents_override or run_args.get("n_agents", 3))
        max_cycles = int(max_cycles_override or run_args.get("max_cycles", 100))

        self.env = MPEWrapper(
            scenario=scenario,
            n_agents=n_agents,
            max_cycles=max_cycles,
            device=self.device,
            render_mode=render_mode,
        )

        k = int(run_args.get("k", 4))
        emb_dim = int(run_args.get("emb_dim", 16))
        hidden_dim = int(run_args.get("hidden_dim", 64))

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

        self.consensus_builder.load_state_dict(checkpoint["consensus_builder"])
        self.embedding_layer.load_state_dict(checkpoint["embedding_layer"])
        for i, actor_state in enumerate(checkpoint["actors"]):
            self.actors[i].load_state_dict(actor_state)

        self.consensus_builder.eval()
        self.embedding_layer.eval()
        for actor in self.actors:
            actor.eval()

    @torch.no_grad()
    def _act(self, obs: torch.Tensor) -> torch.Tensor:
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
