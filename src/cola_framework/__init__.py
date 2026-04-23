"""COLA framework package.

This package is organized so each major training component can be developed,
tested, and replaced independently.
"""

from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.interfaces.environment import MultiAgentEnvironment
from cola_framework.buffers.replay_buffer import ReplayBuffer
from cola_framework.buffers.sequence_buffer import SequenceReplayBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.history_aware_builder import HistoryAwareConsensusBuilder
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.policies.actor import Actor
from cola_framework.critics.centralized_critic import Critic
from cola_framework.trainers.maddpg_updater import MADDPGUpdater
from cola_framework.trainers.history_aware_maddpg_updater import HistoryAwareMADDPGUpdater
from cola_framework.loops.cola_training_loop import COLATrainingConfig, COLATrainingLoop
from cola_framework.loops.history_aware_training_loop import HistoryAwareCOLATrainingLoop
from cola_framework.evaluation.policy_evaluator import PolicyEvaluator
from cola_framework.evaluation.history_aware_policy_evaluator import HistoryAwarePolicyEvaluator
from cola_framework.monitoring.wandb_logger import WandbLogger
from cola_framework.watchers.policy_watcher import PolicyWatcher, find_latest_checkpoint
from cola_framework.utils.window_manager import ObservationWindowManager
from cola_framework.encoders.identity_encoder import IdentityHistoryEncoder
from cola_framework.encoders.gru_encoder import GRUHistoryEncoder
from cola_framework.encoders.window_encoder import WindowConcatEncoder
from cola_framework.encoders.transformer_encoder import TransformerHistoryEncoder

__all__ = [
	"MPEWrapper",
	"MultiAgentEnvironment",
	"ReplayBuffer",
	"SequenceReplayBuffer",
	"ConsensusBuilder",
	"HistoryAwareConsensusBuilder",
	"ConsensusEmbedding",
	"Actor",
	"Critic",
	"MADDPGUpdater",
	"HistoryAwareMADDPGUpdater",
	"COLATrainingConfig",
	"COLATrainingLoop",
	"HistoryAwareCOLATrainingLoop",
	"PolicyEvaluator",
	"HistoryAwarePolicyEvaluator",
	"WandbLogger",
	"PolicyWatcher",
	"find_latest_checkpoint",
	"ObservationWindowManager",
	"IdentityHistoryEncoder",
	"GRUHistoryEncoder",
	"WindowConcatEncoder",
	"TransformerHistoryEncoder",
]
