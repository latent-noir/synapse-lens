"""
Visualization Utilities
=======================
Shared layout defaults, color palettes, and export helpers for Plotly figures.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


# ── Default Layout ───────────────────────────────────────────────────────────

DEFAULT_LAYOUT: dict[str, Any] = {
    "template": "plotly_dark",
    "font": dict(family="Inter, sans-serif", size=12),
    "margin": dict(l=60, r=30, t=60, b=60),
    "height": 500,
}


def apply_layout(fig: go.Figure, **overrides: Any) -> go.Figure:
    """Apply the default layout to a figure with optional overrides."""
    layout = {**DEFAULT_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


def export_figure(fig: go.Figure, path: str, format: str = "png", scale: int = 2) -> None:
    """Export a Plotly figure to a file.

    Args:
        fig: The figure to export.
        path: Output file path.
        format: Image format ('png', 'svg', 'pdf', 'html').
        scale: Resolution multiplier for raster formats.
    """
    if format == "html":
        fig.write_html(path)
    else:
        fig.write_image(path, format=format, scale=scale)
