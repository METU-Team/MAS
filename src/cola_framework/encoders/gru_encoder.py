"""GRU-based history encoder."""

import torch
import torch.nn as nn

from cola_framework.interfaces.history_encoder import HistoryEncoderModule


class GRUHistoryEncoder(nn.Module, HistoryEncoderModule):
    """Encodes observation history with a GRU and returns last hidden state."""

    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.out_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        if obs_seq.ndim != 3:
            raise ValueError("obs_seq must have shape [B, T, obs_dim].")
        if obs_seq.shape[2] != self.obs_dim:
            raise ValueError("Last dimension of obs_seq must match obs_dim.")
        _, h_n = self.gru(obs_seq)
        return h_n[-1]
