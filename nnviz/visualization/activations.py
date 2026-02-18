"""
Activation Visualization
========================
Per-layer activation grid plots and distribution charts from hook data.
"""

from __future__ import annotations

import math

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nnviz.engine.hooks import HookRecord
from nnviz.visualization.utils import DEFAULT_LAYOUT


def plot_activation_grid(
    record: HookRecord,
    max_channels: int = 16,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Plot a grid of activation maps from a forward hook record.

    For 4-D tensors (batch, channels, H, W), shows the first sample's
    channel activations.  For 2-D tensors (batch, features), shows a
    single heatmap row.

    Args:
        record: A HookRecord from a forward hook.
        max_channels: Maximum number of channels to display.
        colorscale: Plotly colorscale.

    Returns:
        A Plotly Figure with the activation grid.
    """
    data = record.data
    if data.dim() == 4:
        # Conv2d output: (B, C, H, W) — take first sample
        activations = data[0].numpy()  # (C, H, W)
        n = min(activations.shape[0], max_channels)
        cols = min(n, 4)
        rows = math.ceil(n / cols)

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Ch {i}" for i in range(n)])
        for i in range(n):
            r, c = divmod(i, cols)
            fig.add_trace(
                go.Heatmap(
                    z=activations[i],
                    colorscale=colorscale,
                    showscale=False,
                ),
                row=r + 1,
                col=c + 1,
            )
        fig.update_layout(title=f"Activations: {record.layer_name}", **DEFAULT_LAYOUT)
        return fig

    elif data.dim() == 2:
        # Linear output: (B, features) — show first sample as a bar chart
        activations = data[0].numpy()
        fig = go.Figure(go.Bar(
            y=activations,
            marker_color="teal",
        ))
        fig.update_layout(
            title=f"Activations: {record.layer_name}",
            xaxis_title="Neuron Index",
            yaxis_title="Activation Value",
            **DEFAULT_LAYOUT,
        )
        return fig

    else:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Unsupported tensor shape: {list(data.shape)}",
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(**DEFAULT_LAYOUT)
        return fig


def plot_activation_distribution(
    record: HookRecord,
    num_bins: int = 50,
) -> go.Figure:
    """Plot a histogram of activation values from a hook record.

    Args:
        record: A HookRecord from a forward hook.
        num_bins: Number of histogram bins.

    Returns:
        A Plotly Figure with the activation distribution.
    """
    values = record.data.numpy().flatten()
    fig = go.Figure(go.Histogram(x=values, nbinsx=num_bins, marker_color="darkcyan"))
    fig.update_layout(
        title=f"Activation Distribution: {record.layer_name}",
        xaxis_title="Activation Value",
        yaxis_title="Count",
        **DEFAULT_LAYOUT,
    )
    return fig
