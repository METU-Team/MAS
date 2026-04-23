"""Transformer-based history encoder."""

import torch
import torch.nn as nn

from cola_framework.interfaces.history_encoder import HistoryEncoderModule


class TransformerHistoryEncoder(nn.Module, HistoryEncoderModule):
    """Encodes observation sequences via self-attention and mean pooling."""

    def __init__(
        self,
        obs_dim: int,
        out_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 100,
    ) -> None:
        super().__init__()
        if out_dim % nhead != 0:
            raise ValueError("out_dim must be divisible by nhead.")

        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.max_len = max_len

        self.input_proj = nn.Linear(obs_dim, out_dim)
        self.pos_emb = nn.Embedding(max_len, out_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dim_feedforward=out_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        if obs_seq.ndim != 3:
            raise ValueError("obs_seq must have shape [B, T, obs_dim].")
        if obs_seq.shape[2] != self.obs_dim:
            raise ValueError("Last dimension of obs_seq must match obs_dim.")

        batch_size, seq_len, _ = obs_seq.shape
        if seq_len > self.max_len:
            raise ValueError("Sequence length exceeds max_len for positional embedding.")

        x = self.input_proj(obs_seq)
        pos_ids = torch.arange(seq_len, device=obs_seq.device)
        x = x + self.pos_emb(pos_ids).unsqueeze(0)
        x = self.transformer(x)
        return x.mean(dim=1)
