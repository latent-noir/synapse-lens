"""Tests for visualization functions."""

import plotly.graph_objects as go
from nnviz.visualization.weights import plot_weight_heatmap, plot_weight_distribution
from nnviz.visualization.metrics import plot_training_curves


class TestWeightVisualization:
    def test_heatmap_returns_figure(self, simple_mlp):
        fig = plot_weight_heatmap(simple_mlp)
        assert isinstance(fig, go.Figure)

    def test_distribution_returns_figure(self, simple_mlp):
        fig = plot_weight_distribution(simple_mlp)
        assert isinstance(fig, go.Figure)


class TestMetricsVisualization:
    def test_training_curves(self):
        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "train_accuracy": [0.5, 0.7, 0.85],
        }
        fig = plot_training_curves(history)
        assert isinstance(fig, go.Figure)
