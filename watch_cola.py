"""Watch a trained COLA checkpoint and optionally record rollout videos.

Defaults to newest .pth in models directory. You may also pass --checkpoint_path.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.watchers.policy_watcher import PolicyWatcher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch trained COLA policies.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default="models")

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="eval_videos")
    parser.add_argument("--video_prefix", type=str, default="watch")
    parser.add_argument("--no_video", action="store_true")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Optional overrides for environment restoration.
    parser.add_argument("--scenario", type=str, default=None, choices=["simple_spread", "simple_tag"])
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--max_cycles", type=int, default=None)

    # Useful on headless Linux when recording videos.
    parser.add_argument("--use_virtual_display", action="store_true")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, checkpoint_path)

    models_dir = args.models_dir
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(REPO_ROOT, models_dir)

    display = None
    if args.use_virtual_display and (not args.no_video):
        if os.environ.get("DISPLAY", "") == "":
            try:
                from pyvirtualdisplay import Display

                display = Display(visible=0, size=(1400, 900))
                display.start()
                print("Virtual display started for video rendering.")
            except Exception as exc:
                print("Virtual display could not start, continuing without it: {}".format(exc))

    watcher = PolicyWatcher(
        checkpoint_path=checkpoint_path,
        models_dir=models_dir,
        device=args.device,
        scenario_override=args.scenario,
        n_agents_override=args.n_agents,
        max_cycles_override=args.max_cycles,
        render_mode="rgb_array",
    )

    try:
        result = watcher.watch(
            n_episodes=args.episodes,
            output_dir=args.output_dir,
            fps=args.fps,
            seed=args.seed,
            save_video=(not args.no_video),
            video_prefix=args.video_prefix,
        )
    finally:
        watcher.close()
        if display is not None:
            display.stop()

    print("Watch completed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
