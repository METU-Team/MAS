"""Training loop implementations."""

from cola_framework.loops.cola_training_loop import COLATrainingConfig, COLATrainingLoop
from cola_framework.loops.history_aware_training_loop import HistoryAwareCOLATrainingLoop

from cola_framework.loops.mappo_training_loop import MAPPOTrainingConfig, MAPPOTrainingLoop

__all__ = ["COLATrainingConfig", "COLATrainingLoop", "HistoryAwareCOLATrainingLoop", "MAPPOTrainingConfig", "MAPPOTrainingLoop"]
