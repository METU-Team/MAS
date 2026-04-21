"""Weights & Biases metrics logger for COLA experiments."""

import numbers
from typing import Dict, Optional

from cola_framework.interfaces.logger import MetricsLoggerModule


class WandbLogger(MetricsLoggerModule):
    """Logs metrics to WandB and authenticates using an API key file."""

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, object]] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        mode: str = "online",
        api_key_path: str = "src/apiKey.txt",
        wandb_module=None,
    ) -> None:
        self._wandb = wandb_module
        if self._wandb is None:
            import wandb as _wandb

            self._wandb = _wandb

        self._run = None

        if mode != "disabled":
            api_key = self._read_api_key(api_key_path)
            # Login once at startup; key is never printed or logged.
            self._wandb.login(key=api_key, relogin=True)

        self._run = self._wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            config=config,
            tags=tags,
            group=group,
            mode=mode,
        )

    def _read_api_key(self, api_key_path: str) -> str:
        with open(api_key_path, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key:
                    return key
        raise ValueError("WandB API key file is empty.")

    def _flatten_metrics(self, metrics: Dict[str, object]) -> Dict[str, float]:
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                flat[key] = float(value)
                continue

            if isinstance(value, (list, tuple)):
                numeric_items = []
                for idx, item in enumerate(value):
                    if isinstance(item, numbers.Number):
                        item_f = float(item)
                        flat["{}/{}".format(key, idx)] = item_f
                        numeric_items.append(item_f)
                if numeric_items:
                    flat["{}/mean".format(key)] = sum(numeric_items) / len(numeric_items)

        return flat

    def log_metrics(self, metrics: Dict[str, object], step: Optional[int] = None) -> None:
        payload = self._flatten_metrics(metrics)
        if payload:
            self._wandb.log(payload, step=step)

    def finish(self, summary: Optional[Dict[str, object]] = None) -> None:
        if self._run is None:
            return
        if summary:
            summary_payload = self._flatten_metrics(summary)
            self._run.summary.update(summary_payload)
        self._run.finish()
