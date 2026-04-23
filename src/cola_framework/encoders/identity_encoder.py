"""Identity history encoder for T=1 fallback behaviour."""

import torch
import torch.nn as nn

from cola_framework.interfaces.history_encoder import HistoryEncoderModule


class IdentityHistoryEncoder(nn.Module, HistoryEncoderModule):
    """No-op encoder.

    Expects sequence length T=1 and returns the single timestep features.
    This is useful as a baseline that mimics single-step consensus behaviour.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.out_dim = obs_dim

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        if obs_seq.ndim != 3:
            raise ValueError("obs_seq must have shape [B, T, obs_dim].")
        if obs_seq.shape[1] != 1:
            raise ValueError("IdentityHistoryEncoder requires T=1.")
        if obs_seq.shape[2] != self.obs_dim:
            raise ValueError("Last dimension of obs_seq must match obs_dim.")
        return obs_seq[:, 0, :]
