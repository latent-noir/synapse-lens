"""Visualization controller — updates visualization panels when training completes."""

from __future__ import annotations

from nnviz.gui.models.app_state import AppState
from nnviz.visualization.matplotlib_backend import MatplotlibBackend


class VisualizationController:
    """Controller connecting AppState to VisualizationView.

    Listens for training_completed signal and refreshes all visualization panels.
    """

    def __init__(self, state: AppState, view, parent=None):
        self.state = state
        self.view = view
        self.viz = MatplotlibBackend()

        # Connect signals
        self.state.training_completed.connect(self._on_training_completed)

    def _on_training_completed(self, history: dict):
        """Update all visualization panels after training finishes."""
        # Find the model from the training controller (via parent)
        main_window = self.view.parent()
        if main_window and hasattr(main_window, 'training_ctrl') and hasattr(main_window.training_ctrl, '_model'):
            model = main_window.training_ctrl._model

            # Weight visualization
            try:
                fig = self.viz.plot_weight_heatmap(model)
                self.view.update_weights(fig)
            except Exception:
                pass

            # Gradient visualization
            try:
                fig = self.viz.plot_gradient_flow(model)
                self.view.update_gradients(fig)
            except Exception:
                pass

        # Metrics visualization
        try:
            fig = self.viz.plot_training_curves(history)
            self.view.update_metrics(fig)
        except Exception:
            pass
