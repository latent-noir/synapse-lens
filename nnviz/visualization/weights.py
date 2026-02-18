"""
Weight Visualization
====================
Heatmaps and distribution plots for model weight matrices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nnviz.visualization.utils import DEFAULT_LAYOUT, apply_layout


def plot_weight_heatmap(
    model: nn.Module,
    layer_name: str | None = None,
    colorscale: str = "RdBu",
) -> go.Figure:
    """Plot heatmaps of weight matrices for all (or a specific) linear/conv layers.

    Args:
        model: The PyTorch model.
        layer_name: Optional specific layer name to plot.
        colorscale: Plotly colorscale name.

    Returns:
        A Plotly Figure with weight heatmaps.
    """
    layers = _get_weight_layers(model, layer_name)
    n = len(layers)
    if n == 0:
        return _empty_figure("No weight layers found")

    fig = make_subplots(rows=1, cols=n, subplot_titles=[name for name, _ in layers])

    for i, (name, param) in enumerate(layers, 1):
        weights = param.detach().cpu().numpy()
        # For Conv2d, flatten to 2D
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        fig.add_trace(
            go.Heatmap(
                z=weights,
                colorscale=colorscale,
                showscale=(i == n),
                name=name,
            ),
            row=1,
            col=i,
        )

    fig.update_layout(title="Weight Heatmaps", **DEFAULT_LAYOUT)
    return fig


def plot_weight_distribution(
    model: nn.Module,
    layer_name: str | None = None,
    num_bins: int = 50,
) -> go.Figure:
    """Plot histograms of weight value distributions.

    Args:
        model: The PyTorch model.
        layer_name: Optional specific layer name.
        num_bins: Number of histogram bins.

    Returns:
        A Plotly Figure with weight distribution histograms.
    """
    layers = _get_weight_layers(model, layer_name)
    if not layers:
        return _empty_figure("No weight layers found")

    fig = go.Figure()
    for name, param in layers:
        weights = param.detach().cpu().numpy().flatten()
        fig.add_trace(go.Histogram(x=weights, nbinsx=num_bins, name=name, opacity=0.7))

    fig.update_layout(
        title="Weight Distributions",
        xaxis_title="Weight Value",
        yaxis_title="Count",
        barmode="overlay",
        **DEFAULT_LAYOUT,
    )
    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_weight_layers(
    model: nn.Module, layer_name: str | None
) -> list[tuple[str, torch.Tensor]]:
    """Extract named weight parameters from linear and conv layers."""
    result = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if layer_name is not None and name != layer_name:
            continue
        if module.weight is not None:
            result.append((name, module.weight))
    return result


def _empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(**DEFAULT_LAYOUT)
    return fig
