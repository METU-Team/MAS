"""Watch a trained COLA-MAPPO checkpoint and optionally record rollout videos.

Loads the most recent .pth in models/ by default, or pass --checkpoint_path.
Actions are selected deterministically: tanh(mean) of each GaussianActor.

Example:
  PYTHONPATH=src python watch_mappo.py --checkpoint_path models/my_run.pth \\
    --episodes 3 --use_virtual_display
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.policies.gaussian_actor import GaussianActor


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _find_latest_checkpoint(models_dir: str, prefix: str = "final_mappo") -> Optional[str]:
    if not os.path.isdir(models_dir):
        return None
    candidates = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".pth") and prefix in f
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch trained COLA-MAPPO policies.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="eval_videos")
    parser.add_argument("--video_prefix", type=str, default="mappo")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--use_virtual_display", action="store_true")
    return parser


@torch.no_grad()
def _run_episode(env, cb, emb, actors, seed):
    obs, _ = env.reset(seed=seed)
    frames = []
    ep_return = 0.0
    done = False

    while not done:
        frame = env.render_frame()
        if frame is not None:
            frames.append(frame)

        consensus = cb.infer(obs)
        cemb = emb(consensus)
        actions = torch.stack(
            [actors[a](obs[a], cemb[a]) for a in range(env.n_agents)],
            dim=0,
        ).clamp(-1.0, 1.0)

        obs, _, rewards, dones = env.step(actions)
        ep_return += float(rewards.mean().item())
        done = bool(dones.all().item())

    return ep_return, frames


def main() -> None:
    args = _build_parser().parse_args()

    models_dir = args.models_dir
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(REPO_ROOT, models_dir)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = _find_latest_checkpoint(models_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No MAPPO checkpoint found in {}. "
            "Pass --checkpoint_path explicitly.".format(models_dir)
        )
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, checkpoint_path)

    print("Loading checkpoint:", checkpoint_path)
    device = _resolve_device(args.device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})

    # ── Rebuild environment ───────────────────────────────────────────────────
    env = MPEWrapper(
        scenario=saved_args.get("scenario", "simple_spread"),
        n_agents=saved_args.get("n_agents", 3),
        max_cycles=saved_args.get("max_cycles", 100),
        device=device,
        render_mode="rgb_array" if not args.no_video else None,
    )

    # ── Rebuild modules ───────────────────────────────────────────────────────
    k = saved_args.get("k", 4)
    emb_dim = saved_args.get("emb_dim", 16)
    hidden_dim = saved_args.get("hidden_dim", 64)
    log_std_init = saved_args.get("log_std_init", -0.5)
    no_cola = saved_args.get("no_cola", False)

    if no_cola:
        cb = NullConsensusBuilder(k=k).to(device)
    else:
        cb = ConsensusBuilder(obs_dim=env.obs_dim, k=k, hidden_dim=hidden_dim).to(device)
    cb.load_state_dict(ckpt["consensus_builder"])
    cb.eval()

    emb = ConsensusEmbedding(k=k, emb_dim=emb_dim).to(device)
    emb.load_state_dict(ckpt["embedding_layer"])
    emb.eval()

    actors: List[GaussianActor] = []
    for i, state_dict in enumerate(ckpt["actors"]):
        actor = GaussianActor(
            obs_dim=env.obs_dim, emb_dim=emb_dim,
            action_dim=env.action_dim, hidden_dim=hidden_dim,
            log_std_init=log_std_init,
        ).to(device)
        actor.load_state_dict(state_dict)
        actor.eval()
        actors.append(actor)

    # ── Virtual display ───────────────────────────────────────────────────────
    display = None
    if args.use_virtual_display and not args.no_video:
        if not os.environ.get("DISPLAY"):
            try:
                from pyvirtualdisplay import Display
                display = Display(visible=0, size=(1400, 900))
                display.start()
                print("Virtual display started.")
            except Exception as e:
                print("Virtual display failed:", e)

    os.makedirs(args.output_dir if os.path.isabs(args.output_dir)
                else os.path.join(REPO_ROOT, args.output_dir), exist_ok=True)
    output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                  else os.path.join(REPO_ROOT, args.output_dir))

    # ── Run episodes ──────────────────────────────────────────────────────────
    all_returns = []
    video_paths = []
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        for ep_idx in range(args.episodes):
            seed = (args.seed + ep_idx) if args.seed is not None else None
            ep_return, frames = _run_episode(env, cb, emb, actors, seed)
            all_returns.append(ep_return)
            print("Episode {}/{}: return={:.3f}  frames={}".format(
                ep_idx + 1, args.episodes, ep_return, len(frames)))

            if frames and not args.no_video:
                import imageio
                video_path = os.path.join(
                    output_dir,
                    "{}_{}_{}_ep{}.mp4".format(args.video_prefix, stem, ts, ep_idx)
                )
                imageio.mimsave(video_path, frames, fps=args.fps)
                video_paths.append(video_path)
                print("  Saved:", video_path)
    finally:
        env.close()
        if display is not None:
            display.stop()

    result = {
        "checkpoint_path": checkpoint_path,
        "mean_episode_return": float(sum(all_returns) / max(1, len(all_returns))),
        "episode_returns": all_returns,
        "video_paths": video_paths,
    }
    print("\nWatch completed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
