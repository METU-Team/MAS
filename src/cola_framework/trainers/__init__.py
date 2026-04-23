"""Training update implementations."""

from cola_framework.trainers.maddpg_updater import MADDPGUpdater
from cola_framework.trainers.history_aware_maddpg_updater import HistoryAwareMADDPGUpdater

__all__ = ["MADDPGUpdater", "HistoryAwareMADDPGUpdater"]
