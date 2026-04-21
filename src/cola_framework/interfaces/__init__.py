"""Public interface contracts for framework modules."""

from cola_framework.interfaces.consensus import ConsensusModule
from cola_framework.interfaces.embedding import ConsensusEmbeddingModule
from cola_framework.interfaces.environment import MultiAgentEnvironment
from cola_framework.interfaces.replay_buffer import ExperienceReplay

__all__ = [
	"ConsensusModule",
	"ConsensusEmbeddingModule",
	"MultiAgentEnvironment",
	"ExperienceReplay",
]
