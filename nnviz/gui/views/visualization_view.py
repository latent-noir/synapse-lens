"""Visualization view — live weight, gradient, and activation panels."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QComboBox, QHBoxLayout,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from nnviz.gui.models.app_state import AppState


class MatplotlibWidget(QWidget):
    """Embeds a matplotlib Figure inside a QWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(facecolor="#16213e")
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)

    def update_figure(self, fig: Figure):
        """Replace the displayed figure."""
        self.figure = fig
        self.canvas.figure = fig
        self.canvas.draw()


class VisualizationView(QWidget):
    """Tabbed visualization panel for weights, gradients, activations, and metrics."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Sub-tabs ─────────────────────────────────────────────────────
        self.viz_tabs = QTabWidget()

        # Weight panel
        self.weight_canvas = MatplotlibWidget()
        self.viz_tabs.addTab(self.weight_canvas, "⚖️ Weights")

        # Gradient panel
        self.gradient_canvas = MatplotlibWidget()
        self.viz_tabs.addTab(self.gradient_canvas, "📉 Gradients")

        # Activation panel
        self.activation_canvas = MatplotlibWidget()
        self.viz_tabs.addTab(self.activation_canvas, "🔥 Activations")

        # Metrics panel
        self.metrics_canvas = MatplotlibWidget()
        self.viz_tabs.addTab(self.metrics_canvas, "📈 Metrics")

        layout.addWidget(self.viz_tabs)

        # Styling
        self.viz_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #16213e; }
            QTabBar::tab { background: #1a1a2e; color: #b0b0b0; padding: 6px 12px; }
            QTabBar::tab:selected { background: #533483; color: white; }
        """)

    def update_weights(self, fig: Figure):
        self.weight_canvas.update_figure(fig)

    def update_gradients(self, fig: Figure):
        self.gradient_canvas.update_figure(fig)

    def update_activations(self, fig: Figure):
        self.activation_canvas.update_figure(fig)

    def update_metrics(self, fig: Figure):
        self.metrics_canvas.update_figure(fig)
