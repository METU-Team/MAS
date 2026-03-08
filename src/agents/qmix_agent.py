"""
agents/qmix_agent.py
====================
QMIX multi-agent kontrol sınıfı.

Bu sınıf, paylaşımlı (shared) GRU tabanlı ajan ağlarını ve
QMIX karıştırma ağını bir arada tutar. Epsilon-greedy eylem
seçimi ve hedef ağ (target network) yönetimini sağlar.

Eğitim döngüsü bu sınıfın dışında yer alır; burada yalnızca
ağ yapıları, eylem seçimi ve hedef ağ güncelleme metotları
tanımlanmıştır.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from src.networks.agent_network import AgentNetwork
from src.networks.mixing_network import QMIXMixingNetwork


class QMIXAgent:
    """
    QMIX çoklu ajan kontrolcüsü.

    Parametreler
    ----------
    num_agents  : Kontrol edilen ajan sayısı.
    obs_dim     : Bireysel gözlem boyutu (simple115v2 → 115).
    state_dim   : Global state boyutu (obs_dim * num_agents).
    n_actions   : Ayrık eylem sayısı (GRF → 19).
    hidden_dim  : GRU / ajan ağı gizli boyutu.
    mixing_dim  : Karıştırma ağı gizli boyutu.
    lr          : Öğrenme oranı.
    gamma       : İndirim faktörü.
    eps_start   : Epsilon başlangıç değeri.
    eps_end     : Epsilon minimum değeri.
    eps_decay_steps : Epsilon'un lineer olarak azalacağı adım sayısı.
    device      : ``'cpu'`` veya ``'cuda'``.
    """

    def __init__(
        self,
        num_agents: int = 3,
        obs_dim: int = 115,
        state_dim: int = 345,
        n_actions: int = 19,
        hidden_dim: int = 64,
        mixing_dim: int = 32,
        lr: float = 5e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = torch.device(device)

        # ── Epsilon-greedy parametreleri ──
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self._step_count = 0

        # ── Ajan ağı (parametre paylaşımlı) ──
        self.agent_net = AgentNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # ── Karıştırma (mixing) ağı ──
        self.mixing_net = QMIXMixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim,
            mixing_dim=mixing_dim,
        ).to(self.device)

        # ── Hedef ağlar (target networks) ──
        self.target_agent_net = copy.deepcopy(self.agent_net)
        self.target_mixing_net = copy.deepcopy(self.mixing_net)
        self.target_agent_net.requires_grad_(False)
        self.target_mixing_net.requires_grad_(False)

        # ── Optimizer ──
        self.params = list(self.agent_net.parameters()) + list(
            self.mixing_net.parameters()
        )
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

        # ── Bölüm boyunca taşınan GRU gizli durumları ──
        self._hidden: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Gizli durum yönetimi
    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int = 1):
        """Yeni bölüm başında tüm ajanların gizli durumlarını sıfırlar."""
        # (num_agents, batch_size, hidden_dim)
        self._hidden = self.agent_net.init_hidden(batch_size).unsqueeze(0).expand(
            self.num_agents, -1, -1
        ).contiguous().to(self.device)

    # ------------------------------------------------------------------
    # Epsilon-greedy eylem seçimi
    # ------------------------------------------------------------------
    def _decay_epsilon(self):
        """Lineer epsilon azaltma."""
        self._step_count += 1
        fraction = min(1.0, self._step_count / self.eps_decay_steps)
        self.eps = self.eps_start + fraction * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def select_actions(
        self,
        agent_obs: torch.Tensor,
        last_actions: torch.Tensor,
        evaluate: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Epsilon-greedy politikası ile eylem seçer.

        Parametreler
        ----------
        agent_obs    : (num_agents, obs_dim) – her ajanın gözlemi.
        last_actions : (num_agents,) – önceki adımda seçilen eylemler (int).
        evaluate     : True ise keşif (exploration) yapılmaz (greedy).

        Dönüş
        ------
        actions      : np.ndarray (num_agents,)  – seçilen eylem indeksleri.
        q_values_all : Tensor (num_agents, n_actions) – hesaplanan Q değerleri.
        """
        if self._hidden is None:
            self.init_hidden(batch_size=1)

        # One-hot kodlama
        last_onehot = torch.zeros(
            self.num_agents, self.n_actions, device=self.device
        )
        last_onehot.scatter_(
            1, last_actions.unsqueeze(1).long().to(self.device), 1.0
        )

        agent_obs = agent_obs.to(self.device)
        q_values_list = []
        new_hiddens = []

        for i in range(self.num_agents):
            obs_i = agent_obs[i].unsqueeze(0)           # (1, obs_dim)
            la_i = last_onehot[i].unsqueeze(0)           # (1, n_actions)
            h_i = self._hidden[i]                        # (1, hidden_dim)

            q_i, h_new = self.agent_net(obs_i, la_i, h_i)
            q_values_list.append(q_i.squeeze(0))         # (n_actions,)
            new_hiddens.append(h_new)

        self._hidden = torch.stack(new_hiddens, dim=0)   # (n_agents, 1, hid)
        q_values_all = torch.stack(q_values_list, dim=0)  # (n_agents, n_act)

        # Greedy eylemler
        greedy_actions = q_values_all.argmax(dim=-1).cpu().numpy()

        if evaluate:
            return greedy_actions, q_values_all

        # Epsilon-greedy keşif
        actions = greedy_actions.copy()
        for i in range(self.num_agents):
            if np.random.rand() < self.eps:
                actions[i] = np.random.randint(0, self.n_actions)

        self._decay_epsilon()
        return actions, q_values_all

    # ------------------------------------------------------------------
    # Eğitim adımı (QMIX loss)
    # ------------------------------------------------------------------
    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Bir mini-yığın (batch) bölüm verisi üzerinden QMIX TD-loss
        hesaplar ve geri yayılım uygular.

        Parametreler
        ----------
        batch : EpisodeReplayBuffer.sample() çıktısı.
            obs        : (B, T, N, obs_dim)
            state      : (B, T, state_dim)
            actions    : (B, T, N)
            rewards    : (B, T, N)
            next_obs   : (B, T, N, obs_dim)
            next_state : (B, T, state_dim)
            dones      : (B, T)
            mask       : (B, T)

        Dönüş
        ------
        loss : float – ortalama TD-loss değeri.
        """
        obs = batch["obs"].to(self.device)             # (B, T, N, obs)
        state = batch["state"].to(self.device)         # (B, T, S)
        actions = batch["actions"].to(self.device)     # (B, T, N)
        rewards = batch["rewards"].to(self.device)     # (B, T, N)
        next_obs = batch["next_obs"].to(self.device)   # (B, T, N, obs)
        next_state = batch["next_state"].to(self.device)
        dones = batch["dones"].to(self.device)         # (B, T)
        mask = batch["mask"].to(self.device)           # (B, T)

        B, T, N = obs.shape[:3]

        # Takım ödülü: ajanların aldığı ödülün ortalaması
        team_reward = rewards.mean(dim=-1)             # (B, T)

        # ── İleri geçiş: tüm zaman adımları için Q-değerlerini hesapla ──
        # In-place yazımları engellemek için zaman/adım sonuçlarını listede toplayıp
        # sonradan stack ediyoruz.
        chosen_qs_steps = []
        target_max_qs_steps = []

        # GRU gizli durumları – her ajan için ayrı tensor referansı
        hidden = [
            self.agent_net.init_hidden(B).to(self.device)
            for _ in range(N)
        ]
        target_hidden = [
            self.target_agent_net.init_hidden(B).to(self.device)
            for _ in range(N)
        ]

        for t in range(T):
            # Önceki eylemler (t=0 → sıfır)
            if t == 0:
                last_onehot = torch.zeros(B, N, self.n_actions,
                                          device=self.device)
            else:
                last_onehot = torch.zeros(B, N, self.n_actions,
                                          device=self.device)
                last_onehot.scatter_(
                    2, actions[:, t - 1].unsqueeze(-1).long(), 1.0
                )

            # -- Next step önceki eylemler (current actions) --
            next_last_onehot = torch.zeros(B, N, self.n_actions,
                                           device=self.device)
            next_last_onehot.scatter_(
                2, actions[:, t].unsqueeze(-1).long(), 1.0
            )

            step_chosen_qs = []
            step_target_max_qs = []
            new_hidden = []
            new_target_hidden = []

            for i in range(N):
                # Canlı ağ
                q_i, h_new = self.agent_net(
                    obs[:, t, i],                       # (B, obs)
                    last_onehot[:, i],                  # (B, act)
                    hidden[i],                          # (B, hid)
                )
                new_hidden.append(h_new)
                step_chosen_qs.append(q_i.gather(
                    1, actions[:, t, i].unsqueeze(1).long()
                ).squeeze(1))

                # Hedef ağ
                with torch.no_grad():
                    tq_i, th_new = self.target_agent_net(
                        next_obs[:, t, i],
                        next_last_onehot[:, i],
                        target_hidden[i],
                    )
                    new_target_hidden.append(th_new)
                    step_target_max_qs.append(tq_i.max(dim=1).values)

            hidden = new_hidden
            target_hidden = new_target_hidden
            chosen_qs_steps.append(torch.stack(step_chosen_qs, dim=1))
            target_max_qs_steps.append(torch.stack(step_target_max_qs, dim=1))

        chosen_qs = torch.stack(chosen_qs_steps, dim=1)              # (B, T, N)
        target_max_qs = torch.stack(target_max_qs_steps, dim=1)      # (B, T, N)

        # ── Mixing ──
        # chosen_qs   : (B, T, N) → her adımda bireysel Q değerleri
        q_tot = self.mixing_net(
            chosen_qs.view(B * T, N), state.view(B * T, -1)
        ).view(B, T)                                        # (B, T)

        with torch.no_grad():
            target_q_tot = self.target_mixing_net(
                target_max_qs.view(B * T, N),
                next_state.view(B * T, -1),
            ).view(B, T)

            targets = team_reward + self.gamma * (1 - dones) * target_q_tot

        # ── Masked MSE loss ──
        td_error = (q_tot - targets) ** 2
        loss = (td_error * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Hedef ağ güncelleme
    # ------------------------------------------------------------------
    def update_targets(self):
        """Hedef ağları (target networks) canlı ağlardan kopyalar (hard update)."""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    # ------------------------------------------------------------------
    # Kaydetme / Yükleme
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Model ağırlıklarını diske kaydeder."""
        torch.save(
            {
                "agent_net": self.agent_net.state_dict(),
                "mixing_net": self.mixing_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "eps": self.eps,
                "step_count": self._step_count,
            },
            path,
        )

    def load(self, path: str):
        """Model ağırlıklarını diskten yükler."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.agent_net.load_state_dict(ckpt["agent_net"])
        self.mixing_net.load_state_dict(ckpt["mixing_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps = ckpt["eps"]
        self._step_count = ckpt["step_count"]
        self.update_targets()

    # ------------------------------------------------------------------
    # Bilgi
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"QMIXAgent(agents={self.num_agents}, obs={self.obs_dim}, "
            f"act={self.n_actions}, hidden={self.hidden_dim}, "
            f"eps={self.eps:.3f})"
        )
