"""
Dashboard Layouts
=================
Component composition and page structure for the Dash UI.
"""

from __future__ import annotations

from dash import dcc, html


def create_layout() -> html.Div:
    """Build the main dashboard layout with sidebar controls and tabbed panels."""
    return html.Div(
        className="dashboard-container",
        children=[
            # ── Header ───────────────────────────────────────────────────
            html.Header(
                className="dashboard-header",
                children=[
                    html.H1("🧠 Synapse Lens"),
                    html.P("Real-time weight, gradient & activation inspection"),
                ],
            ),

            # ── Main Content ─────────────────────────────────────────────
            html.Div(
                className="dashboard-body",
                children=[
                    # Sidebar
                    html.Aside(
                        className="sidebar",
                        children=[
                            html.H3("Controls"),

                            html.Label("Model Architecture"),
                            dcc.Dropdown(
                                id="model-selector",
                                options=[
                                    {"label": "MLP", "value": "mlp"},
                                    {"label": "CNN", "value": "cnn"},
                                ],
                                value="mlp",
                            ),

                            html.Label("Dataset"),
                            dcc.Dropdown(
                                id="dataset-selector",
                                options=[
                                    {"label": "MNIST", "value": "mnist"},
                                ],
                                value="mnist",
                            ),

                            html.Label("Epochs"),
                            dcc.Slider(
                                id="epoch-slider",
                                min=1,
                                max=50,
                                step=1,
                                value=10,
                                marks={i: str(i) for i in range(0, 51, 10)},
                            ),

                            html.Label("Learning Rate"),
                            dcc.Input(
                                id="lr-input",
                                type="number",
                                value=0.001,
                                step=0.0001,
                                min=0.00001,
                                max=1.0,
                            ),

                            html.Button(
                                "Train",
                                id="train-button",
                                n_clicks=0,
                                className="btn-primary",
                            ),

                            html.Div(id="training-status", className="status-panel"),
                        ],
                    ),

                    # Main panel with tabs
                    html.Main(
                        className="main-panel",
                        children=[
                            dcc.Tabs(
                                id="viz-tabs",
                                value="weights",
                                children=[
                                    dcc.Tab(label="Weights", value="weights"),
                                    dcc.Tab(label="Gradients", value="gradients"),
                                    dcc.Tab(label="Activations", value="activations"),
                                    dcc.Tab(label="Training Metrics", value="metrics"),
                                ],
                            ),
                            html.Div(id="tab-content", className="tab-content"),
                        ],
                    ),
                ],
            ),
        ],
    )
