"""Structured experiment logger for reproducibility and tracking."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Log experiment events, metrics, and artifacts to disk in structured format.

    Usage:
        exp = ExperimentLogger("experiments/run_001")
        exp.log_config(config)
        for epoch in range(epochs):
            exp.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=epoch)
        exp.log_artifact("model.pt", model_path)
        exp.finalize()
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)

        self._metrics: List[Dict[str, Any]] = []
        self._config: Dict[str, Any] = {}
        self._artifacts: List[Dict[str, str]] = []
        self._events: List[Dict[str, Any]] = []

        logger.info(f"Experiment logger initialized: {self.run_dir}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self._config = config
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for a given training step."""
        entry = {"step": step, "timestamp": datetime.datetime.now().isoformat()}
        entry.update(metrics)
        self._metrics.append(entry)

    def log_event(self, event_type: str, details: str) -> None:
        """Log a freeform event (e.g., 'learning_rate_change', 'early_stop')."""
        self._events.append({
            "type": event_type,
            "details": details,
            "timestamp": datetime.datetime.now().isoformat(),
        })

    def log_artifact(self, name: str, path: str) -> None:
        """Register an artifact file (model checkpoint, plot, etc.)."""
        self._artifacts.append({"name": name, "path": str(path)})

    def finalize(self) -> Path:
        """Write all accumulated data to disk and return run directory.

        Creates:
            - metrics.json: All logged metrics
            - events.json: All logged events
            - summary.json: Run summary with config, artifacts, and timestamps
        """
        # Metrics
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self._metrics, f, indent=2)

        # Events
        with open(self.run_dir / "events.json", "w") as f:
            json.dump(self._events, f, indent=2)

        # Summary
        summary = {
            "run_id": self.run_id,
            "config": self._config,
            "artifacts": self._artifacts,
            "total_steps": len(self._metrics),
            "total_events": len(self._events),
            "started": self._metrics[0]["timestamp"] if self._metrics else None,
            "finished": datetime.datetime.now().isoformat(),
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Experiment finalized: {self.run_dir}")
        return self.run_dir

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get accumulated metrics as a list of dicts."""
        return self._metrics.copy()
