"""COLA framework package.

This package is organized so each major training component can be developed,
tested, and replaced independently.
"""

from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.interfaces.environment import MultiAgentEnvironment
from cola_framework.buffers.replay_buffer import ReplayBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.policies.actor import Actor
from cola_framework.critics.centralized_critic import Critic
from cola_framework.trainers.maddpg_updater import MADDPGUpdater
from cola_framework.loops.cola_training_loop import COLATrainingConfig, COLATrainingLoop
from cola_framework.evaluation.policy_evaluator import PolicyEvaluator
from cola_framework.monitoring.wandb_logger import WandbLogger

__all__ = [
	"MPEWrapper",
	"MultiAgentEnvironment",
	"ReplayBuffer",
	"ConsensusBuilder",
	"ConsensusEmbedding",
	"Actor",
	"Critic",
	"MADDPGUpdater",
	"COLATrainingConfig",
	"COLATrainingLoop",
	"PolicyEvaluator",
	"WandbLogger",
]
