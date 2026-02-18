"""
Shared Types
============
Common type aliases and dataclasses used across the nnviz package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainingSnapshot:
    """A snapshot of training state at a particular epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    model_state: dict[str, Any] | None = None


@dataclass
class LayerData:
    """Data extracted from a specific model layer."""
    name: str
    layer_type: str
    weight_shape: tuple[int, ...] | None = None
    weight_norm: float | None = None
    grad_norm: float | None = None


@dataclass
class ExperimentResult:
    """Result of a complete training experiment."""
    config_name: str
    model_type: str
    final_loss: float
    final_accuracy: float
    epochs_trained: int
    history: dict[str, list[float]] = field(default_factory=dict)
