"""Abstract contract for experiment metrics logging backends."""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class MetricsLoggerModule(ABC):
    """Receives scalar metrics during training/evaluation."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, object], step: Optional[int] = None) -> None:
        """Log one metrics record."""

    @abstractmethod
    def finish(self, summary: Optional[Dict[str, object]] = None) -> None:
        """Finalize logger resources and optionally publish summary data."""
