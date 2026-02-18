"""
Gradient Visualization
======================
Gradient flow charts and per-layer gradient distribution histograms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go

from nnviz.visualization.utils import DEFAULT_LAYOUT


def plot_gradient_flow(model: nn.Module) -> go.Figure:
    """Plot the average gradient magnitude per layer (gradient flow).

    Useful for diagnosing vanishing/exploding gradients.
    Must be called after a backward pass.

    Args:
        model: The PyTorch model (after .backward()).

    Returns:
        A Plotly Figure showing gradient magnitudes per layer.
    """
    layer_names = []
    avg_grads = []
    max_grads = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layer_names.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    if not layer_names:
        fig = go.Figure()
        fig.add_annotation(
            text="No gradients found — run .backward() first",
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(**DEFAULT_LAYOUT)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layer_names,
        y=avg_grads,
        name="Avg Gradient",
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        x=layer_names,
        y=max_grads,
        name="Max Gradient",
        marker_color="firebrick",
        opacity=0.6,
    ))

    fig.update_layout(
        title="Gradient Flow",
        xaxis_title="Layer",
        yaxis_title="Gradient Magnitude",
        barmode="group",
        xaxis_tickangle=-45,
        **DEFAULT_LAYOUT,
    )
    return fig


def plot_gradient_histogram(
    model: nn.Module,
    layer_name: str | None = None,
    num_bins: int = 50,
) -> go.Figure:
    """Plot histograms of gradient value distributions.

    Args:
        model: The PyTorch model (after .backward()).
        layer_name: Optional layer name filter.
        num_bins: Number of histogram bins.

    Returns:
        A Plotly Figure with gradient distributions.
    """
    fig = go.Figure()
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if layer_name is not None and name != layer_name:
            continue

        grads = param.grad.detach().cpu().numpy().flatten()
        fig.add_trace(go.Histogram(x=grads, nbinsx=num_bins, name=name, opacity=0.7))

    fig.update_layout(
        title="Gradient Distributions",
        xaxis_title="Gradient Value",
        yaxis_title="Count",
        barmode="overlay",
        **DEFAULT_LAYOUT,
    )
    return fig
