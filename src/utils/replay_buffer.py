"""
utils/replay_buffer.py
======================
QMIX eğitimi için bölüm bazlı (episode-level) Replay Buffer.

QMIX'te GRU gizli durumları bölüm boyunca taşındığından, deneyim
tekrarı bölüm parçaları (episode chunks) üzerinden yapılır. Bu buffer
her zaman adımındaki geçişi (transition) biriktirip, ardından tüm
bölümü tek bir kayıt olarak saklar.
"""

import random
from collections import deque
import numpy as np
import torch


class EpisodeReplayBuffer:
    """
    Bölüm bazlı replay buffer.

    Her bölüm şu alanları içeren bir sözlük (dict) olarak saklanır:
        obs          : (T, num_agents, obs_dim)
        state        : (T, state_dim)
        actions      : (T, num_agents)
        rewards      : (T, num_agents)
        next_obs     : (T, num_agents, obs_dim)
        next_state   : (T, state_dim)
        dones        : (T,)

    T = bölüm uzunluğu (adım sayısı), bölümden bölüme değişir.

    Parametreler
    ----------
    capacity : Saklanacak maksimum bölüm sayısı.
    """

    def __init__(self, capacity: int = 5000):
        self.buffer: deque[dict] = deque(maxlen=capacity)

        # Geçici olarak tek bölüm boyunca geçişleri biriktiren listeler
        self._current_obs: list = []
        self._current_state: list = []
        self._current_actions: list = []
        self._current_rewards: list = []
        self._current_next_obs: list = []
        self._current_next_state: list = []
        self._current_dones: list = []

    # ── Bölüm içi geçiş biriktirme ──

    def push_transition(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: np.ndarray,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Tek bir zaman adımı geçişini geçici listeye ekler."""
        self._current_obs.append(obs.cpu())
        self._current_state.append(state.cpu())
        self._current_actions.append(
            torch.as_tensor(actions, dtype=torch.long)
        )
        self._current_rewards.append(rewards.cpu())
        self._current_next_obs.append(next_obs.cpu())
        self._current_next_state.append(next_state.cpu())
        self._current_dones.append(float(done))

    def finish_episode(self):
        """Biriktirilen geçişleri tek bir bölüm kaydı olarak buffer'a ekler."""
        if not self._current_obs:
            return

        episode = {
            "obs": torch.stack(self._current_obs),           # (T, N, obs)
            "state": torch.stack(self._current_state),       # (T, state)
            "actions": torch.stack(self._current_actions),   # (T, N)
            "rewards": torch.stack(self._current_rewards),   # (T, N)
            "next_obs": torch.stack(self._current_next_obs),
            "next_state": torch.stack(self._current_next_state),
            "dones": torch.tensor(self._current_dones),      # (T,)
        }
        self.buffer.append(episode)
        self._clear_temp()

    def _clear_temp(self):
        self._current_obs.clear()
        self._current_state.clear()
        self._current_actions.clear()
        self._current_rewards.clear()
        self._current_next_obs.clear()
        self._current_next_state.clear()
        self._current_dones.clear()

    # ── Örnekleme ──

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Rastgele ``batch_size`` adet bölüm seçer ve aynı uzunluğa (max_T)
        pad'leyip yığın (batch) olarak döndürür.

        Dönüş anahtarları (tümü Tensor):
            obs         : (B, T, N, obs_dim)
            state       : (B, T, state_dim)
            actions     : (B, T, N)
            rewards     : (B, T, N)
            next_obs    : (B, T, N, obs_dim)
            next_state  : (B, T, state_dim)
            dones       : (B, T)
            mask        : (B, T)  – geçerli zaman adımları 1, pad 0
        """
        episodes = random.sample(list(self.buffer), batch_size)
        max_t = max(ep["dones"].size(0) for ep in episodes)

        batch = {k: [] for k in [
            "obs", "state", "actions", "rewards",
            "next_obs", "next_state", "dones", "mask",
        ]}

        for ep in episodes:
            t = ep["dones"].size(0)
            pad = max_t - t

            mask = torch.ones(max_t)
            if pad > 0:
                mask[t:] = 0.0

            batch["mask"].append(mask)
            for key in ["obs", "state", "actions", "rewards",
                        "next_obs", "next_state", "dones"]:
                tensor = ep[key]
                if pad > 0:
                    pad_shape = (pad, *tensor.shape[1:])
                    tensor = torch.cat(
                        [tensor, torch.zeros(pad_shape)], dim=0
                    )
                batch[key].append(tensor)

        return {k: torch.stack(v) for k, v in batch.items()}

    def __len__(self):
        return len(self.buffer)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
