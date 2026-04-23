"""Window concatenation history encoder."""

import torch
import torch.nn as nn

from cola_framework.interfaces.history_encoder import HistoryEncoderModule


class WindowConcatEncoder(nn.Module, HistoryEncoderModule):
    """Concatenates a fixed observation window and projects to out_dim."""

    def __init__(self, obs_dim: int, window: int, out_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.window = window
        self.proj = nn.Sequential(
            nn.Linear(obs_dim * window, out_dim),
            nn.ReLU(),
        )

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        if obs_seq.ndim != 3:
            raise ValueError("obs_seq must have shape [B, T, obs_dim].")
        if obs_seq.shape[1] != self.window:
            raise ValueError("WindowConcatEncoder got unexpected T.")
        if obs_seq.shape[2] != self.obs_dim:
            raise ValueError("Last dimension of obs_seq must match obs_dim.")

        batch_size = obs_seq.shape[0]
        flat = obs_seq.reshape(batch_size, self.window * self.obs_dim)
        return self.proj(flat)
