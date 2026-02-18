"""
Training Metrics Visualization
==============================
Loss curves, accuracy curves, and learning rate schedules.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nnviz.visualization.utils import DEFAULT_LAYOUT


def plot_training_curves(
    history: dict[str, list[float]],
) -> go.Figure:
    """Plot training loss and accuracy curves on a dual-axis chart.

    Args:
        history: Dict with keys like 'train_loss', 'train_accuracy'.

    Returns:
        A Plotly Figure with training curves.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "train_loss" in history:
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_loss"],
                name="Loss",
                mode="lines+markers",
                line=dict(color="firebrick", width=2),
            ),
            secondary_y=False,
        )

    if "train_accuracy" in history:
        epochs = list(range(1, len(history["train_accuracy"]) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_accuracy"],
                name="Accuracy",
                mode="lines+markers",
                line=dict(color="steelblue", width=2),
            ),
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    fig.update_layout(title="Training Curves", **DEFAULT_LAYOUT)
    return fig


def plot_learning_rate(lr_history: list[float]) -> go.Figure:
    """Plot the learning rate schedule over training steps.

    Args:
        lr_history: List of learning rate values per step.

    Returns:
        A Plotly Figure showing the LR schedule.
    """
    fig = go.Figure(go.Scatter(
        x=list(range(len(lr_history))),
        y=lr_history,
        mode="lines",
        line=dict(color="darkorange", width=2),
    ))
    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        **DEFAULT_LAYOUT,
    )
    return fig
