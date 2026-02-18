"""Main window — application shell with menu bar, toolbar, and tabbed interface."""

from __future__ import annotations
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QToolBar,
    QAction, QStatusBar, QVBoxLayout, QWidget, QMessageBox,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from nnviz.gui.models.app_state import AppState
from nnviz.gui.views.training_view import TrainingView
from nnviz.gui.views.visualization_view import VisualizationView
from nnviz.gui.views.comparison_view import ComparisonView
from nnviz.gui.controllers.training_controller import TrainingController
from nnviz.gui.controllers.visualization_controller import VisualizationController


class MainWindow(QMainWindow):
    """Main application window with tabbed panels and toolbar."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Synapse Lens")
        self.setMinimumSize(1200, 800)

        # ── State Model ──────────────────────────────────────────────────
        self.state = AppState(self)

        # ── Views ────────────────────────────────────────────────────────
        self.training_view = TrainingView(self.state, self)
        self.visualization_view = VisualizationView(self.state, self)
        self.comparison_view = ComparisonView(self.state, self)

        # ── Controllers ──────────────────────────────────────────────────
        self.training_ctrl = TrainingController(self.state, self.training_view, self)
        self.viz_ctrl = VisualizationController(self.state, self.visualization_view, self)

        # ── Tab Widget ───────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.addTab(self.training_view, "🏋️ Training")
        self.tabs.addTab(self.visualization_view, "📊 Visualization")
        self.tabs.addTab(self.comparison_view, "⚖️ Comparison")
        self.setCentralWidget(self.tabs)

        # ── Menu Bar ─────────────────────────────────────────────────────
        self._create_menu_bar()

        # ── Status Bar ───────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.state.status_updated.connect(self.status_bar.showMessage)
        self.state.status_updated.emit("Ready")

        # ── Styling ──────────────────────────────────────────────────────
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QTabWidget::pane { border: 1px solid #333; background: #16213e; }
            QTabBar::tab { background: #0f3460; color: #e0e0e0; padding: 8px 16px;
                           border: 1px solid #333; margin-right: 2px; }
            QTabBar::tab:selected { background: #533483; color: white; }
            QStatusBar { background: #0f3460; color: #0ff; font-family: monospace; }
            QMenuBar { background: #0f3460; color: #e0e0e0; }
            QMenuBar::item:selected { background: #533483; }
            QMenu { background: #16213e; color: #e0e0e0; }
            QMenu::item:selected { background: #533483; }
        """)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        new_action = QAction("&New Experiment", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_experiment)
        file_menu.addAction(new_action)

        export_action = QAction("&Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _new_experiment(self):
        self.state.is_training = False
        self.state.status_updated.emit("New experiment — configure and train")

    def _export_results(self):
        from nnviz.analysis.export import export_experiment_json
        if self.state.history:
            path = export_experiment_json(
                "experiments/gui_results.json",
                config=self.state.config,
                history=self.state.history,
                metrics={"note": "exported from GUI"},
            )
            self.state.status_updated.emit(f"Exported to {path}")
        else:
            self.state.status_updated.emit("No results to export")

    def _show_about(self):
        QMessageBox.about(
            self,
            "About nnviz",
            "Synapse Lens\n\n"
            "A production-grade tool for real-time neural network inspection.\n"
            "Built with PyTorch, C++, Plotly, and PyQt5.",
        )


def launch_gui():
    """Launch the PyQt5 GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_gui()
