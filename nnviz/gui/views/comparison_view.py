"""Comparison view — side-by-side model comparison tab."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nnviz.gui.models.app_state import AppState


class ComparisonView(QWidget):
    """Model comparison tab with summary table and charts."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("⚖️ Model Comparison")
        header.setStyleSheet("color: #0ff; font-size: 16px; font-weight: bold; padding: 8px;")
        layout.addWidget(header)

        # ── Summary Table ────────────────────────────────────────────────
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Model", "Parameters", "Final Loss", "Accuracy", "Epochs"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget { background: #16213e; color: #e0e0e0;
                          border: 1px solid #333; gridline-color: #333; }
            QHeaderView::section { background: #0f3460; color: white;
                                  padding: 4px; border: 1px solid #333; }
        """)
        layout.addWidget(self.table)

        # ── Comparison Chart ─────────────────────────────────────────────
        self.chart_figure = Figure(facecolor="#16213e")
        self.chart_canvas = FigureCanvas(self.chart_figure)
        layout.addWidget(self.chart_canvas)

        # ── Refresh Button ───────────────────────────────────────────────
        self.refresh_btn = QPushButton("🔄 Refresh Comparison")
        self.refresh_btn.setStyleSheet(
            "QPushButton { background: #9b59b6; color: white; padding: 8px 20px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #8e44ad; }"
        )
        self.refresh_btn.clicked.connect(self._refresh)
        layout.addWidget(self.refresh_btn)

    def _refresh(self):
        """Refresh the comparison table and chart from state."""
        models = self.state.comparison_models
        self.table.setRowCount(len(models))

        for row, (name, info) in enumerate(models.items()):
            model = info["model"]
            history = info["history"]
            params = sum(p.numel() for p in model.parameters())
            final_loss = history.get("train_loss", [float("nan")])[-1] if history.get("train_loss") else float("nan")
            accuracy = history.get("train_accuracy", [0])[-1] if history.get("train_accuracy") else 0

            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{params:,}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{final_loss:.4f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{accuracy:.2%}"))
            self.table.setItem(row, 4, QTableWidgetItem(str(len(history.get("train_loss", [])))))

        # Update chart
        self.chart_figure.clear()
        if models:
            ax = self.chart_figure.add_subplot(111)
            ax.set_facecolor("#16213e")
            for name, info in models.items():
                h = info["history"]
                if "train_loss" in h:
                    ax.plot(h["train_loss"], label=name, linewidth=2)
            ax.set_xlabel("Epoch", color="white")
            ax.set_ylabel("Loss", color="white")
            ax.set_title("Training Loss Comparison", color="white")
            ax.legend()
            ax.tick_params(colors="white")
            ax.grid(True, alpha=0.3)

        self.chart_canvas.draw()
