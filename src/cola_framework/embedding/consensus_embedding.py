"""Consensus embedding layer for COLA."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cola_framework.interfaces.embedding import ConsensusEmbeddingModule


class ConsensusEmbedding(nn.Module, ConsensusEmbeddingModule):
    """Projects one-hot consensus labels into a dense embedding space.

    Input can be [B, n_agents] or [n_agents] integer labels.
    Output mirrors input leading dimensions with emb_dim as last dimension.
    """

    def __init__(self, k: int, emb_dim: int = 16) -> None:
        super().__init__()
        self.k = k
        self.emb_dim = emb_dim
        self.proj = nn.Linear(k, emb_dim, bias=False)

    def forward(self, consensus: torch.Tensor) -> torch.Tensor:
        """Convert consensus indices to dense vectors.

        consensus: int tensor of shape [B, n_agents] or [n_agents]
        returns: float tensor of shape [B, n_agents, emb_dim] or [n_agents, emb_dim]
        """
        if consensus.dtype != torch.int64 and consensus.dtype != torch.int32:
            raise ValueError("consensus must be an integer tensor (int64/int32).")
        if consensus.ndim not in (1, 2):
            raise ValueError("consensus must have shape [n_agents] or [B, n_agents].")

        one_hot = F.one_hot(consensus.long(), num_classes=self.k).float()

        # Detach here to guarantee RL gradients stop at the discrete labels and
        # never flow back to the consensus-builder computation graph.
        return self.proj(one_hot.detach())
