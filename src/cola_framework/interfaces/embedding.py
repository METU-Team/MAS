"""Abstract contract for consensus embedding modules."""

from abc import ABC, abstractmethod

import torch


class ConsensusEmbeddingModule(ABC):
    """Maps discrete consensus labels to dense vectors."""

    @abstractmethod
    def forward(self, consensus: torch.Tensor) -> torch.Tensor:
        """Return dense embeddings for consensus labels."""
