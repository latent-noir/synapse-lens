"""Visualization backend abstraction.

Allows switching between matplotlib (for static/Qt rendering) and Plotly
(for web/Dash rendering). All visualization functions should work through
a backend interface, making the system extensible to future rendering engines.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch.nn as nn


class VisualizationBackend(ABC):
    """Abstract interface for visualization backends."""

    @abstractmethod
    def plot_weight_heatmap(self, model: nn.Module, layer_name: Optional[str] = None) -> Any:
        """Render weight heatmaps for model layers."""
        ...

    @abstractmethod
    def plot_weight_distribution(self, model: nn.Module) -> Any:
        """Render weight distribution histograms."""
        ...

    @abstractmethod
    def plot_gradient_flow(self, model: nn.Module) -> Any:
        """Render gradient flow bar chart."""
        ...

    @abstractmethod
    def plot_gradient_histogram(self, gradients: Dict[str, Any]) -> Any:
        """Render gradient distribution histograms."""
        ...

    @abstractmethod
    def plot_activation_grid(self, activations: Any, layer_name: str) -> Any:
        """Render activation map grid."""
        ...

    @abstractmethod
    def plot_activation_distribution(self, activations: Dict[str, Any]) -> Any:
        """Render activation distribution histograms."""
        ...

    @abstractmethod
    def plot_training_curves(self, history: Dict[str, List[float]]) -> Any:
        """Render training loss/accuracy curves."""
        ...

    @abstractmethod
    def plot_confusion_matrix(self, y_true: Any, y_pred: Any, class_names: List[str]) -> Any:
        """Render a confusion matrix."""
        ...


# ── Backend Registry ─────────────────────────────────────────────────────────
_backends: Dict[str, type] = {}
_active_backend: Optional[VisualizationBackend] = None


def register_backend(name: str, backend_class: type) -> None:
    """Register a visualization backend by name."""
    _backends[name] = backend_class


def set_backend(name: str, **kwargs) -> None:
    """Set the active visualization backend."""
    global _active_backend
    if name not in _backends:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_backends.keys())}")
    _active_backend = _backends[name](**kwargs)


def get_backend() -> VisualizationBackend:
    """Get the active visualization backend. Defaults to matplotlib."""
    global _active_backend
    if _active_backend is None:
        # Lazy import to avoid circular imports
        from nnviz.visualization.matplotlib_backend import MatplotlibBackend
        _active_backend = MatplotlibBackend()
    return _active_backend
