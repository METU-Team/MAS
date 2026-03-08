"""
src/train.py
============
QMIX eğitim döngüsü – Weights & Biases (wandb) entegrasyonu,
model checkpointing ve periyodik test (evaluation) koşuları ile.

Kullanım
--------
    cd masProje
    source grf_venv39/bin/activate
    python -m src.train                # varsayılan parametrelerle
    python -m src.train --episodes 5000 --eval-interval 200
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import wandb

# Proje kökünü path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs.grf_env import make_grf_env
from src.agents.qmix_agent import QMIXAgent
from src.utils.replay_buffer import EpisodeReplayBuffer


# ═══════════════════════════════════════════════════════════════════════════
# Varsayılan hiperparametreler
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    # Ortam
    "env_name": "academy_3_vs_1_with_keeper",
    "num_agents": 3,
    "obs_dim": 115,
    "n_actions": 19,
    "representation": "simple115v2",
    # Ağ mimarisi
    "hidden_dim": 64,
    "mixing_dim": 32,
    # Eğitim
    "total_episodes": 2000,
    "max_steps": 400,
    "batch_size": 16,
    "buffer_capacity": 5000,
    "lr": 5e-4,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 50_000,
    "train_interval": 1,          # Kaç bölümde bir öğrenme yapılacak
    "target_update_interval": 200,  # Kaç bölümde bir hedef ağ güncelleme
    # Değerlendirme
    "eval_interval": 100,          # Kaç bölümde bir test koşusu
    "eval_episodes": 10,           # Test koşusundaki bölüm sayısı
    # Checkpoint
    "checkpoint_interval": 1000,   # Düzenli checkpoint aralığı
    "model_dir": "models",
}


# ═══════════════════════════════════════════════════════════════════════════
# Bir bölüm oynama (veri toplama)
# ═══════════════════════════════════════════════════════════════════════════
def run_episode(env, agent, replay_buffer, max_steps, evaluate=False):
    """
    Tek bir bölüm oynar.

    Dönüş
    ------
    episode_return : float – bölüm boyunca toplanan toplam ödül.
    episode_steps  : int   – bölüm adım sayısı.
    won            : bool  – gol atılarak kazanılmış mı.
    """
    agent_obs, state = env.reset()
    agent.init_hidden(batch_size=1)
    last_actions = torch.zeros(agent.num_agents, dtype=torch.long)

    episode_return = 0.0
    episode_steps = 0
    won = False

    for _ in range(max_steps):
        actions, _ = agent.select_actions(
            agent_obs, last_actions, evaluate=evaluate
        )
        next_obs, next_state, rewards, done, info = env.step(actions)

        # Takım ödülü (tüm ajanların ortalaması)
        step_reward = rewards.mean().item()
        episode_return += step_reward

        if not evaluate and replay_buffer is not None:
            replay_buffer.push_transition(
                obs=agent_obs,
                state=state,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                next_state=next_state,
                done=done,
            )

        agent_obs = next_obs
        state = next_state
        last_actions = torch.as_tensor(actions, dtype=torch.long)
        episode_steps += 1

        if done:
            # GRF'te ödül >0 → gol atıldı
            if step_reward > 0:
                won = True
            break

    if not evaluate and replay_buffer is not None:
        replay_buffer.finish_episode()

    return episode_return, episode_steps, won


# ═══════════════════════════════════════════════════════════════════════════
# Değerlendirme (Evaluation)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(env, agent, num_episodes, max_steps):
    """
    Greedy politika ile birden fazla test bölümü oynar.

    Dönüş
    ------
    mean_return : float – ortalama bölüm ödülü.
    win_rate    : float – kazanma yüzdesi [0, 1].
    """
    returns = []
    wins = 0

    for _ in range(num_episodes):
        ep_return, _, won = run_episode(
            env, agent, replay_buffer=None,
            max_steps=max_steps, evaluate=True,
        )
        returns.append(ep_return)
        if won:
            wins += 1

    mean_return = float(np.mean(returns))
    win_rate = wins / num_episodes
    return mean_return, win_rate


# ═══════════════════════════════════════════════════════════════════════════
# Ana eğitim döngüsü
# ═══════════════════════════════════════════════════════════════════════════
def train(config: dict):
    """Ana eğitim fonksiyonu."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = config["obs_dim"] * config["num_agents"]

    # ── 1) WandB başlat ──
    wandb.init(
        project="MAS",
        config=config,
        name=f"qmix_{config['env_name']}",
    )
    print("=" * 65)
    print("  QMIX Eğitimi – Google Research Football")
    print("=" * 65)
    print(f"  Cihaz         : {device}")
    print(f"  Senaryo       : {config['env_name']}")
    print(f"  Toplam bölüm  : {config['total_episodes']}")
    print(f"  WandB proje   : MAS")
    print("=" * 65)

    # ── 2) Ortam oluştur ──
    env = make_grf_env(
        env_name=config["env_name"],
        num_agents=config["num_agents"],
        representation=config["representation"],
        render=False,
        write_video=False,
        device=device,
    )

    # ── 3) QMIX ajanını başlat ──
    agent = QMIXAgent(
        num_agents=config["num_agents"],
        obs_dim=config["obs_dim"],
        state_dim=state_dim,
        n_actions=config["n_actions"],
        hidden_dim=config["hidden_dim"],
        mixing_dim=config["mixing_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        eps_start=config["eps_start"],
        eps_end=config["eps_end"],
        eps_decay_steps=config["eps_decay_steps"],
        device=device,
    )
    print(f"  Agent         : {agent}")

    # ── 4) Replay Buffer ──
    replay_buffer = EpisodeReplayBuffer(capacity=config["buffer_capacity"])

    # ── 5) Model dizini ──
    model_dir = os.path.join(PROJECT_ROOT, config["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    best_win_rate = -1.0
    train_losses = []

    # ══════════════════════════════════════════════════════════════════════
    # Eğitim döngüsü
    # ══════════════════════════════════════════════════════════════════════
    start_time = time.time()

    for episode in range(1, config["total_episodes"] + 1):
        # ── Veri topla (bir bölüm oyna) ──
        ep_return, ep_steps, ep_won = run_episode(
            env, agent, replay_buffer,
            max_steps=config["max_steps"], evaluate=False,
        )

        # ── Eğitim (replay buffer yeterince doluysa) ──
        loss = None
        if (episode % config["train_interval"] == 0
                and replay_buffer.can_sample(config["batch_size"])):
            batch = replay_buffer.sample(config["batch_size"])
            loss = agent.train_step(batch)
            train_losses.append(loss)

        # ── Hedef ağ güncelleme ──
        if episode % config["target_update_interval"] == 0:
            agent.update_targets()
            print(f"  [Episode {episode}] Hedef ağlar güncellendi.")

        # ── WandB eğitim logları ──
        log_dict = {
            "train/episode_return": ep_return,
            "train/episode_steps": ep_steps,
            "train/episode_won": int(ep_won),
            "train/epsilon": agent.eps,
            "train/buffer_size": len(replay_buffer),
        }
        if loss is not None:
            log_dict["train/loss"] = loss
        wandb.log(log_dict, step=episode)

        # ── Terminal bilgi çıktısı (her 50 bölümde) ──
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            avg_loss = (
                np.mean(train_losses[-50:]) if train_losses else 0.0
            )
            print(
                f"  [Episode {episode:5d}/{config['total_episodes']}] "
                f"return={ep_return:+.3f}  steps={ep_steps:3d}  "
                f"eps={agent.eps:.3f}  loss={avg_loss:.4f}  "
                f"buffer={len(replay_buffer)}  "
                f"elapsed={elapsed:.0f}s"
            )

        # ═══════════════════════════════════════════════════════════════
        # Periyodik değerlendirme (Evaluation)
        # ═══════════════════════════════════════════════════════════════
        if episode % config["eval_interval"] == 0:
            mean_return, win_rate = evaluate(
                env, agent,
                num_episodes=config["eval_episodes"],
                max_steps=config["max_steps"],
            )

            wandb.log({
                "eval/mean_episode_return": mean_return,
                "eval/win_rate": win_rate,
                "mean_episode_return": mean_return,
                "win_rate": win_rate,
            }, step=episode)

            print(
                f"  ★ [Eval @ {episode}]  "
                f"mean_return={mean_return:+.3f}  "
                f"win_rate={win_rate:.2%}"
            )

            # ── Best model checkpointing ──
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = os.path.join(model_dir, "best_model.pth")
                agent.save(best_path)
                print(
                    f"    ✓ Yeni en iyi model kaydedildi! "
                    f"win_rate={win_rate:.2%} → {best_path}"
                )
                wandb.run.summary["best_win_rate"] = best_win_rate

        # ═══════════════════════════════════════════════════════════════
        # Düzenli checkpoint (her N bölümde)
        # ═══════════════════════════════════════════════════════════════
        if episode % config["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(
                model_dir, f"checkpoint_{episode}.pth"
            )
            agent.save(ckpt_path)
            print(f"    ✓ Checkpoint kaydedildi → {ckpt_path}")

    # ══════════════════════════════════════════════════════════════════════
    # Eğitim sonu
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    print("=" * 65)
    print(f"  Eğitim tamamlandı. Toplam süre: {total_time:.0f}s")
    print(f"  En iyi win_rate: {best_win_rate:.2%}")
    print("=" * 65)

    # Son model kaydet
    final_path = os.path.join(model_dir, "final_model.pth")
    agent.save(final_path)
    print(f"  Son model kaydedildi → {final_path}")

    wandb.finish()
    env.close()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="QMIX GRF Eğitimi")
    p.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["total_episodes"],
                    help="Toplam eğitim bölümü sayısı.")
    p.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"],
                    help="Bir bölümdeki maksimum adım sayısı.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--eval-interval", type=int,
                    default=DEFAULT_CONFIG["eval_interval"],
                    help="Kaç bölümde bir test koşusu yapılacak.")
    p.add_argument("--eval-episodes", type=int,
                    default=DEFAULT_CONFIG["eval_episodes"])
    p.add_argument("--checkpoint-interval", type=int,
                    default=DEFAULT_CONFIG["checkpoint_interval"])
    p.add_argument("--target-update", type=int,
                    default=DEFAULT_CONFIG["target_update_interval"])
    p.add_argument("--eps-decay-steps", type=int,
                    default=DEFAULT_CONFIG["eps_decay_steps"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "total_episodes": args.episodes,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "checkpoint_interval": args.checkpoint_interval,
        "target_update_interval": args.target_update,
        "eps_decay_steps": args.eps_decay_steps,
    })
    train(cfg)
