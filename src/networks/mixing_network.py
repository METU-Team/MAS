"""
networks/mixing_network.py
==========================
QMIX Mixing Network – bireysel ajan Q-değerlerini global state
bilgisi ile birleştirerek Q_tot üretir.

Orijinal makale: Rashid et al., "QMIX: Monotonic Value Function
Factorisation for Deep Multi-Agent Reinforcement Learning", ICML 2018.

Temel kısıtlama: ∂Q_tot / ∂Q_i ≥ 0  (monotonluk)
Bu kısıtlama, karıştırma ağının ağırlıklarının negatif-olmayan
olması ile sağlanır (``torch.abs`` ile).
"""

import torch
import torch.nn as nn


class QMIXMixingNetwork(nn.Module):
    """
    State-dependent hyper-network ile karıştırma ağı.

    Bireysel ajan Q-değerlerini (Q_1, Q_2, …, Q_n) alır ve global
    state vektörü kullanılarak üretilen ağırlıklarla doğrusal olmayan
    bir şekilde birleştirip tek bir Q_tot skaler değeri döndürür.

    Parametreler
    ----------
    num_agents : Ajan sayısı.
    state_dim  : Global state vektörü boyutu.
    mixing_dim : Karıştırma katmanının gizli boyutu.
    """

    def __init__(self, num_agents: int = 3, state_dim: int = 345,
                 mixing_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_dim = mixing_dim

        # ---------- Hyper-network'ler ----------
        # İlk katman ağırlıkları: state → (num_agents * mixing_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, num_agents * mixing_dim),
        )
        # İlk katman bias: state → (mixing_dim,)
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)

        # İkinci katman ağırlıkları: state → (mixing_dim * 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, mixing_dim),
        )
        # İkinci katman bias: state-dependent skaler
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        İleri geçiş.

        Parametreler
        ----------
        agent_qs : (batch, num_agents) – her ajanın seçili eylem Q-değeri
        state    : (batch, state_dim)  – global state vektörü

        Dönüş
        ------
        q_tot : (batch, 1) – toplam Q değeri
        """
        batch_size = agent_qs.size(0)

        # --- Birinci karıştırma katmanı ---
        w1 = torch.abs(self.hyper_w1(state))               # monotonlük kısıtı
        w1 = w1.view(batch_size, self.num_agents, self.mixing_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_dim)

        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)  # (B, 1, N)
        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)        # (B, 1, mix)

        # --- İkinci karıştırma katmanı ---
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.mixing_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2    # (B, 1, 1)
        q_tot = q_tot.squeeze(-1)              # (B, 1)

        return q_tot
