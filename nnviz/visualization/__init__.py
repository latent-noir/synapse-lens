"""Visualization sub-package — Plotly-based charts for weights, gradients, activations, and metrics."""

from nnviz.visualization.weights import plot_weight_heatmap, plot_weight_distribution
from nnviz.visualization.gradients import plot_gradient_flow, plot_gradient_histogram
from nnviz.visualization.activations import plot_activation_grid
from nnviz.visualization.metrics import plot_training_curves

__all__ = [
    "plot_weight_heatmap",
    "plot_weight_distribution",
    "plot_gradient_flow",
    "plot_gradient_histogram",
    "plot_activation_grid",
    "plot_training_curves",
]
