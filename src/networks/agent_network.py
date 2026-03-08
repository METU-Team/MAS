"""
networks/agent_network.py
=========================
GRU tabanlı ajan ağı – QMIX mimarisi için.

Her ajan, kısmi gözlemini (observation) ve önceki eylemini (one-hot)
girdi olarak alır; GRU gizli durumu (hidden state) bölüm boyunca
taşınarak zamansal bağımlılıkları yakalar.

Girdi  : obs (obs_dim) + last_action_onehot (n_actions)
Çıktı  : Her eylem için Q değeri (n_actions)
"""

import torch
import torch.nn as nn


class AgentNetwork(nn.Module):
    """
    Tek bir ajanın Q-fonksiyonunu temsil eden GRU tabanlı ağ.
    QMIX'te tüm ajanlar bu ağı parametre paylaşımlı (shared) kullanır.

    Parametreler
    ----------
    obs_dim    : Gözlem vektörü boyutu (simple115v2 → 115).
    n_actions  : Ayrık eylem sayısı (GRF → 19).
    hidden_dim : GRU gizli katman boyutu.
    """

    def __init__(self, obs_dim: int = 115, n_actions: int = 19,
                 hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Girdi boyutu: gözlem + önceki eylem (one-hot)
        input_dim = obs_dim + n_actions

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor, last_action_onehot: torch.Tensor,
                hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        İleri geçiş.

        Parametreler
        ----------
        obs               : (batch, obs_dim)
        last_action_onehot: (batch, n_actions)
        hidden            : (batch, hidden_dim) – GRU gizli durumu

        Dönüş
        ------
        q_values : (batch, n_actions) – her eylem için Q değeri
        hidden   : (batch, hidden_dim) – güncellenmiş gizli durum
        """
        x = torch.cat([obs, last_action_onehot], dim=-1)   # (batch, input_dim)
        x = torch.relu(self.fc1(x))                        # (batch, hidden_dim)
        hidden = self.gru(x, hidden)                       # (batch, hidden_dim)
        q_values = self.fc2(hidden)                        # (batch, n_actions)
        return q_values, hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Sıfır başlangıç gizli durumu üretir."""
        return torch.zeros(batch_size, self.hidden_dim)
