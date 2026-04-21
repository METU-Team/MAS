"""Public interface contracts for framework modules."""

from cola_framework.interfaces.consensus import ConsensusModule
from cola_framework.interfaces.critic import CriticModule
from cola_framework.interfaces.embedding import ConsensusEmbeddingModule
from cola_framework.interfaces.environment import MultiAgentEnvironment
from cola_framework.interfaces.evaluator import EvaluatorModule
from cola_framework.interfaces.logger import MetricsLoggerModule
from cola_framework.interfaces.policy import PolicyModule
from cola_framework.interfaces.replay_buffer import ExperienceReplay
from cola_framework.interfaces.training_loop import TrainingLoopModule
from cola_framework.interfaces.updater import UpdateModule

__all__ = [
	"ConsensusModule",
	"CriticModule",
	"ConsensusEmbeddingModule",
	"EvaluatorModule",
	"MetricsLoggerModule",
	"MultiAgentEnvironment",
	"PolicyModule",
	"ExperienceReplay",
	"TrainingLoopModule",
	"UpdateModule",
]
