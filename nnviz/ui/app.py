"""
Dash Application
=================
Main entry point for the interactive web dashboard.
"""

from __future__ import annotations

import dash
from dash import html

from nnviz.ui.layouts import create_layout


def create_app(debug: bool = False) -> dash.Dash:
    """Create and configure the Dash application.

    Args:
        debug: Enable Dash debug mode with hot reloading.

    Returns:
        A configured Dash app instance.
    """
    app = dash.Dash(
        __name__,
        title="Synapse Lens",
        update_title="Loading...",
        suppress_callback_exceptions=True,
    )

    app.layout = create_layout()

    # Register callbacks (imported here to avoid circular imports)
    from nnviz.ui import callbacks as _callbacks  # noqa: F401

    return app


def run(host: str = "127.0.0.1", port: int = 8050, debug: bool = True) -> None:
    """Launch the dashboard server.

    Args:
        host: Host address to bind to.
        port: Port number.
        debug: Enable debug mode.
    """
    app = create_app(debug=debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
