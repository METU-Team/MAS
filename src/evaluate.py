"""
src/evaluate.py
===============
Eğitilmiş QMIX modelini yükleyip GRF ortamında greedy (epsilon=0)
test koşusu yapar ve oyun videosunu diske kaydeder.

Kullanım
--------
    cd masProje
    source grf_venv39/bin/activate
    python -m src.evaluate                           # varsayılan best_model.pth
    python -m src.evaluate --model models/checkpoint_1000.pth
    python -m src.evaluate --episodes 3 --video-dir eval_videos
"""

import os
import sys
import argparse
import shutil
import subprocess
from datetime import datetime
import numpy as np
import torch
import cv2

# Proje kökünü path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs.grf_env import make_grf_env
from src.agents.qmix_agent import QMIXAgent


def _get_frame(env_wrapper):
    """GRF ortamindan RGB frame alir; desteklenmiyorsa None dondurur."""
    try:
        return env_wrapper.env.render(mode="rgb_array")
    except Exception:
        return None


def _create_video_writer(video_path: str, first_frame: np.ndarray, fps: int = 15):
    """Ilk frame boyutuna gore MP4 writer olusturur."""
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(video_path, fourcc, fps, (width, height))


def _make_whatsapp_compatible_mp4(input_path: str) -> str:
    """
    Videoyu WhatsApp uyumlu MP4'e donusturur.
    Hedef: H.264 + yuv420p + faststart
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return input_path

    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_wa.mp4"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return input_path

    return output_path if os.path.exists(output_path) else input_path


def evaluate_and_record(
    model_path: str,
    env_name: str = "academy_3_vs_1_with_keeper",
    num_agents: int = 3,
    obs_dim: int = 115,
    n_actions: int = 19,
    hidden_dim: int = 64,
    mixing_dim: int = 32,
    max_steps: int = 400,
    num_episodes: int = 1,
    video_dir: str = "eval_videos",
    fps: int = 15,
    record_video: bool = False,
    whatsapp_compatible: bool = True,
    show_window: bool = False,
    disable_3d: bool = True,
    device: str = "cpu",
):
    """
    Kaydedilmiş QMIX modelini yükler, greedy politika ile oynatır
    ve GRF'in dahili video kaydedicisi ile videoyu diske yazar.

    Parametreler
    ----------
    model_path   : ``best_model.pth`` veya checkpoint dosyasının yolu.
    num_episodes : Kaç bölüm oynatılacak (her biri ayrı video).
    video_dir    : Videoların kaydedileceği klasör.
    """
    state_dim = obs_dim * num_agents
    video_dir = os.path.join(PROJECT_ROOT, video_dir)
    os.makedirs(video_dir, exist_ok=True)

    print("=" * 65)
    print("  QMIX Değerlendirme & Video Kayıt")
    print("=" * 65)
    print(f"  Model       : {model_path}")
    print(f"  Senaryo     : {env_name}")
    print(f"  Bölüm sayısı: {num_episodes}")
    print(f"  Video klasör: {video_dir}")
    print(f"  Video kaydi : {record_video}")
    print(f"  WA uyumlu   : {whatsapp_compatible}")
    print(f"  Pencere     : {show_window}")
    print(f"  Disable 3D  : {disable_3d}")
    print(f"  Cihaz       : {device}")
    print("=" * 65)

    # Headless modu zorla: pencere acilmasini engelle.
    if not show_window:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ.pop("DISPLAY", None)

    # ── 1) Ortamı oluştur ──
    env = make_grf_env(
        env_name=env_name,
        num_agents=num_agents,
        representation="simple115v2",
        render=show_window,
        disable_3d=disable_3d,
        write_video=False,
        dump_dir=video_dir,
        device=device,
    )

    # ── 2) Ajanı oluştur ve ağırlıkları yükle ──
    agent = QMIXAgent(
        num_agents=num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        mixing_dim=mixing_dim,
        device=device,
    )

    abs_model_path = (
        model_path
        if os.path.isabs(model_path)
        else os.path.join(PROJECT_ROOT, model_path)
    )
    agent.load(abs_model_path)
    # Epsilon'u 0 yap – tamamen greedy
    agent.eps = 0.0
    print(f"  Model yüklendi. eps={agent.eps}")

    # ── 3) Bölümleri oynat ──
    all_returns = []
    wins = 0

    for ep in range(1, num_episodes + 1):
        agent_obs, state = env.reset()
        agent.init_hidden(batch_size=1)
        last_actions = torch.zeros(num_agents, dtype=torch.long)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"episode_{ep:03d}_{timestamp}.mp4")
        writer = None

        episode_return = 0.0
        steps = 0
        won = False

        for _ in range(max_steps):
            if record_video:
                frame = _get_frame(env)
                if frame is not None:
                    if writer is None:
                        writer = _create_video_writer(video_path, frame, fps=fps)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            actions, _ = agent.select_actions(
                agent_obs, last_actions, evaluate=True
            )
            next_obs, next_state, rewards, done, info = env.step(actions)

            step_reward = rewards.mean().item()
            episode_return += step_reward
            steps += 1

            agent_obs = next_obs
            state = next_state
            last_actions = torch.as_tensor(actions, dtype=torch.long)

            if done:
                if step_reward > 0:
                    won = True
                break

        # Son frame'i de yazmayi dene
        if record_video:
            frame = _get_frame(env)
            if writer is not None and frame is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if writer is not None:
            writer.release()

        shared_video_path = video_path
        if record_video and writer is not None and whatsapp_compatible:
            shared_video_path = _make_whatsapp_compatible_mp4(video_path)

        all_returns.append(episode_return)
        if won:
            wins += 1

        result_str = "GOL ★" if won else "Bitti"
        if record_video:
            print(
                f"  Bölüm {ep}/{num_episodes}  "
                f"return={episode_return:+.3f}  steps={steps:3d}  "
                f"sonuç={result_str}  video={os.path.basename(shared_video_path)}"
            )
        else:
            print(
                f"  Bölüm {ep}/{num_episodes}  "
                f"return={episode_return:+.3f}  steps={steps:3d}  "
                f"sonuç={result_str}"
            )

    env.close()

    # ── 4) Özet ──
    mean_return = float(np.mean(all_returns))
    win_rate = wins / num_episodes

    print("=" * 65)
    print(f"  Ortalama return : {mean_return:+.3f}")
    print(f"  Win rate        : {win_rate:.2%}  ({wins}/{num_episodes})")
    print(f"  Videolar        : {video_dir}")
    print("=" * 65)

    return mean_return, win_rate


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="QMIX GRF Değerlendirme & Video")
    p.add_argument(
        "--model", type=str, default="models/best_model.pth",
        help="Yüklenecek model dosyasının yolu.",
    )
    p.add_argument("--episodes", type=int, default=1,
                    help="Kaç bölüm oynatılacak.")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--video-dir", type=str, default="eval_videos",
                    help="Video çıktı klasörü.")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--mixing-dim", type=int, default=32)
    p.add_argument("--fps", type=int, default=15,
                    help="Kaydedilecek videonun FPS degeri.")
    p.add_argument(
        "--record-video",
        action="store_true",
        help="Acilirsa mp4 kaydi yapar; verilmezse tamamen headless calisir.",
    )
    p.add_argument(
        "--no-whatsapp-compatible",
        action="store_false",
        dest="whatsapp_compatible",
        help="WhatsApp icin H.264 donusumu kapatilir.",
    )
    p.set_defaults(whatsapp_compatible=True)
    p.add_argument(
        "--show-window",
        action="store_true",
        help="Acilirsa oyun penceresi gosterilir; varsayilan kapali.",
    )
    p.add_argument(
        "--disable-3d",
        action="store_true",
        default=True,
        help="GRF 3D rendering'i kapatir (varsayilan acik).",
    )
    p.add_argument(
        "--enable-3d",
        action="store_false",
        dest="disable_3d",
        help="GRF 3D rendering'i ac.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_and_record(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        video_dir=args.video_dir,
        hidden_dim=args.hidden_dim,
        mixing_dim=args.mixing_dim,
        fps=args.fps,
        record_video=args.record_video,
        whatsapp_compatible=args.whatsapp_compatible,
        show_window=args.show_window,
        disable_3d=args.disable_3d,
    )
