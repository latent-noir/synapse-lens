"""
Dashboard Callbacks
===================
Reactive Dash callbacks connecting controls to visualization panels.
"""

from __future__ import annotations

import logging

from dash import Input, Output, State, callback, html, dcc
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ── Shared state (for demo — in production, use a proper store) ──────────────
_state: dict = {
    "model": None,
    "history": {},
    "hook_manager": None,
}


@callback(
    Output("tab-content", "children"),
    Input("viz-tabs", "value"),
)
def render_tab(tab: str):
    """Route tab selection to the appropriate visualization."""
    if tab == "weights":
        return _render_weights_tab()
    elif tab == "gradients":
        return _render_gradients_tab()
    elif tab == "activations":
        return _render_activations_tab()
    elif tab == "metrics":
        return _render_metrics_tab()
    return html.P("Select a tab above.")


@callback(
    Output("training-status", "children"),
    Input("train-button", "n_clicks"),
    State("model-selector", "value"),
    State("dataset-selector", "value"),
    State("epoch-slider", "value"),
    State("lr-input", "value"),
    prevent_initial_call=True,
)
def start_training(n_clicks: int, model_name: str, dataset: str, epochs: int, lr: float):
    """Handle the Train button click — train the selected model."""
    from nnviz.engine.models import ModelFactory
    from nnviz.engine.trainer import Trainer, TrainerConfig
    from nnviz.data.loaders import get_dataloader

    try:
        model = ModelFactory.create(model_name)
        config = TrainerConfig(epochs=epochs, learning_rate=lr)
        trainer = Trainer(model, config=config)
        train_loader = get_dataloader(dataset, train=True, batch_size=64)

        history = trainer.train(train_loader)

        _state["model"] = model
        _state["history"] = history
        _state["hook_manager"] = trainer.hook_manager

        return html.Div([
            html.P(f"✅ Training complete — {epochs} epochs", className="success"),
            html.P(f"Final loss: {history['train_loss'][-1]:.4f}"),
            html.P(f"Final accuracy: {history['train_accuracy'][-1]:.4f}"),
        ])
    except Exception as e:
        logger.exception("Training failed")
        return html.P(f"❌ Error: {e}", className="error")


# ── Tab Renderers ────────────────────────────────────────────────────────────

def _render_weights_tab():
    model = _state.get("model")
    if model is None:
        return html.P("Train a model first to visualize weights.")

    from nnviz.visualization.weights import plot_weight_heatmap, plot_weight_distribution
    return html.Div([
        dcc.Graph(figure=plot_weight_heatmap(model)),
        dcc.Graph(figure=plot_weight_distribution(model)),
    ])


def _render_gradients_tab():
    model = _state.get("model")
    if model is None:
        return html.P("Train a model first to visualize gradients.")

    from nnviz.visualization.gradients import plot_gradient_flow, plot_gradient_histogram
    return html.Div([
        dcc.Graph(figure=plot_gradient_flow(model)),
        dcc.Graph(figure=plot_gradient_histogram(model)),
    ])


def _render_activations_tab():
    hook_manager = _state.get("hook_manager")
    if hook_manager is None:
        return html.P("Train a model first to visualize activations.")

    from nnviz.visualization.activations import plot_activation_grid
    records = hook_manager.get_records()
    forward_records = [r for r in records if r.hook_type == "forward"]

    if not forward_records:
        return html.P("No activation records captured.")

    # Show the most recent activation per layer
    latest = {}
    for r in forward_records:
        latest[r.layer_name] = r

    graphs = []
    for layer_name, record in latest.items():
        graphs.append(dcc.Graph(figure=plot_activation_grid(record)))

    return html.Div(graphs)


def _render_metrics_tab():
    history = _state.get("history", {})
    if not history:
        return html.P("Train a model first to see metrics.")

    from nnviz.visualization.metrics import plot_training_curves
    return dcc.Graph(figure=plot_training_curves(history))
