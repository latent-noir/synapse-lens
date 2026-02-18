"""Application state model — central state management with Qt signals.

This is the 'Model' in MVC. All state changes emit signals that
views can connect to for automatic UI updates.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal


class AppState(QObject):
    """Central application state with signal-based notifications."""

    # ── Signals ──────────────────────────────────────────────────────────
    model_changed = pyqtSignal(str)            # model architecture name
    dataset_changed = pyqtSignal(str)          # dataset name
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    epoch_completed = pyqtSignal(int, dict)    # epoch, metrics
    training_completed = pyqtSignal(dict)      # full history
    status_updated = pyqtSignal(str)           # status message
    config_changed = pyqtSignal(dict)          # full config dict

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model_name: str = "mlp"
        self._dataset_name: str = "mnist"
        self._is_training: bool = False
        self._history: Dict[str, List[float]] = {}
        self._config: Dict[str, Any] = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 10,
            "dropout": 0.2,
            "hidden_sizes": [256, 128],
            "device": "cpu",
        }
        self._model = None
        self._comparison_models: Dict[str, Any] = {}

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value
        self.model_changed.emit(value)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value: str):
        self._dataset_name = value
        self.dataset_changed.emit(value)

    @property
    def is_training(self) -> bool:
        return self._is_training

    @is_training.setter
    def is_training(self, value: bool):
        self._is_training = value
        if value:
            self.training_started.emit()
        else:
            self.training_stopped.emit()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()

    def update_config(self, **kwargs):
        self._config.update(kwargs)
        self.config_changed.emit(self._config)

    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history

    @history.setter
    def history(self, value: Dict[str, List[float]]):
        self._history = value

    def add_comparison_model(self, name: str, model, history: Dict):
        """Store a model for comparison."""
        self._comparison_models[name] = {"model": model, "history": history}

    @property
    def comparison_models(self) -> Dict[str, Any]:
        return self._comparison_models
