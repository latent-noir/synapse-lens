"""Jupyter notebook widget utilities for building interactive experiment dashboards.

Provides factory functions for ipywidgets that integrate with the nnviz engine.
Designed to be imported and used inside Jupyter notebooks.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


def create_architecture_selector(
    architectures: List[str] = None,
) -> widgets.Dropdown:
    """Create a dropdown for selecting model architecture."""
    if architectures is None:
        from nnviz.engine.models import ModelFactory
        architectures = ModelFactory.available()

    return widgets.Dropdown(
        options=architectures,
        value=architectures[0],
        description="Model:",
        style={"description_width": "80px"},
    )


def create_dataset_selector(
    datasets: List[str] = None,
) -> widgets.Dropdown:
    """Create a dropdown for selecting dataset."""
    if datasets is None:
        from nnviz.data.registry import DatasetRegistry
        datasets = DatasetRegistry.available()

    return widgets.Dropdown(
        options=datasets,
        value=datasets[0] if datasets else None,
        description="Dataset:",
        style={"description_width": "80px"},
    )


def create_hyperparameter_panel() -> Dict[str, widgets.Widget]:
    """Create a panel of hyperparameter controls.

    Returns:
        Dict mapping parameter name → widget.
    """
    return {
        "learning_rate": widgets.FloatLogSlider(
            value=0.001, base=10, min=-5, max=-1, step=0.1,
            description="LR:", readout_format=".1e",
            style={"description_width": "80px"},
        ),
        "batch_size": widgets.Dropdown(
            options=[16, 32, 64, 128, 256],
            value=64,
            description="Batch:",
            style={"description_width": "80px"},
        ),
        "epochs": widgets.IntSlider(
            value=10, min=1, max=100, step=1,
            description="Epochs:",
            style={"description_width": "80px"},
        ),
        "dropout": widgets.FloatSlider(
            value=0.2, min=0.0, max=0.8, step=0.05,
            description="Dropout:",
            style={"description_width": "80px"},
        ),
        "hidden_sizes": widgets.Text(
            value="256,128",
            description="Hidden:",
            style={"description_width": "80px"},
            placeholder="e.g. 256,128,64",
        ),
    }


def create_training_controls(
    on_train: Callable,
    on_stop: Callable,
    on_reset: Callable,
) -> widgets.HBox:
    """Create train/stop/reset buttons.

    Args:
        on_train: Callback when 'Train' is clicked.
        on_stop: Callback when 'Stop' is clicked.
        on_reset: Callback when 'Reset' is clicked.
    """
    train_btn = widgets.Button(
        description="▶ Train",
        button_style="success",
        icon="play",
    )
    stop_btn = widgets.Button(
        description="⏹ Stop",
        button_style="danger",
        icon="stop",
    )
    reset_btn = widgets.Button(
        description="↻ Reset",
        button_style="warning",
        icon="refresh",
    )

    train_btn.on_click(lambda _: on_train())
    stop_btn.on_click(lambda _: on_stop())
    reset_btn.on_click(lambda _: on_reset())

    return widgets.HBox([train_btn, stop_btn, reset_btn])


def create_live_plot_output() -> widgets.Output:
    """Create an Output widget for live matplotlib plots."""
    return widgets.Output(layout=widgets.Layout(
        width="100%",
        min_height="400px",
        border="1px solid #444",
    ))


def create_status_bar() -> widgets.HTML:
    """Create an HTML status bar for displaying training progress."""
    return widgets.HTML(
        value='<div style="padding:8px; background:#1a1a2e; color:#0ff; '
              'border-radius:4px; font-family:monospace;">'
              'Ready — Select model and dataset to begin</div>'
    )


def update_status(status_bar: widgets.HTML, message: str, color: str = "#0ff") -> None:
    """Update the status bar with a message."""
    status_bar.value = (
        f'<div style="padding:8px; background:#1a1a2e; color:{color}; '
        f'border-radius:4px; font-family:monospace;">{message}</div>'
    )


def create_progress_bar() -> widgets.FloatProgress:
    """Create a training progress bar."""
    return widgets.FloatProgress(
        value=0, min=0, max=100,
        description="Progress:",
        bar_style="info",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="100%"),
    )


def build_experiment_dashboard() -> widgets.VBox:
    """Build a complete experiment control panel.

    Returns a VBox containing all widgets. Training logic must be
    wired up externally in the notebook.
    """
    arch_sel = create_architecture_selector()
    dataset_sel = create_dataset_selector()
    hp = create_hyperparameter_panel()
    plot_output = create_live_plot_output()
    status = create_status_bar()
    progress = create_progress_bar()

    # Placeholder callbacks
    controls = create_training_controls(
        on_train=lambda: update_status(status, "Training started...", "#0f0"),
        on_stop=lambda: update_status(status, "Training stopped.", "#f00"),
        on_reset=lambda: update_status(status, "Ready — Reset complete.", "#0ff"),
    )

    # Layout
    selector_row = widgets.HBox([arch_sel, dataset_sel])
    hp_box = widgets.VBox(list(hp.values()))

    header = widgets.HTML(
        '<h2 style="color:#0ff; font-family:monospace; margin:0;">'
        '🧠 Neural Network Experiment Dashboard</h2>'
    )

    dashboard = widgets.VBox([
        header,
        selector_row,
        hp_box,
        controls,
        progress,
        status,
        plot_output,
    ], layout=widgets.Layout(padding="10px"))

    # Attach widgets for external access
    dashboard.arch_selector = arch_sel
    dashboard.dataset_selector = dataset_sel
    dashboard.hyperparameters = hp
    dashboard.plot_output = plot_output
    dashboard.status_bar = status
    dashboard.progress_bar = progress

    return dashboard
