"""Training update implementations."""

from cola_framework.trainers.maddpg_updater import MADDPGUpdater
from cola_framework.trainers.history_aware_maddpg_updater import HistoryAwareMADDPGUpdater

from cola_framework.trainers.mappo_updater import MAPPOUpdater
from cola_framework.trainers.qmix_updater import QMIXUpdater

__all__ = ["MADDPGUpdater", "HistoryAwareMADDPGUpdater", "MAPPOUpdater", "QMIXUpdater"]
