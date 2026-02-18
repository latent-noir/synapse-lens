"""Weight evolution tracking across training epochs."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class WeightTracker:
    """Track weight statistics across training epochs for evolution analysis.

    Usage:
        tracker = WeightTracker(model)
        for epoch in range(num_epochs):
            train(model, ...)
            tracker.snapshot(epoch)

        fig = tracker.plot_evolution()
    """

    def __init__(self, model: nn.Module, layers: Optional[List[str]] = None):
        """
        Args:
            model: Model to track.
            layers: Specific layer names to track. If None, tracks all weight layers.
        """
        self.model = model
        self.tracked_layers = layers
        self.history: Dict[str, Dict[str, List[float]]] = {}

        # Initialize tracking for relevant layers
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            if layers is not None and name not in layers:
                continue
            self.history[name] = {
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
                "l2_norm": [],
                "epochs": [],
            }

    def snapshot(self, epoch: int) -> None:
        """Record weight statistics for the current epoch."""
        for name, param in self.model.named_parameters():
            if name not in self.history:
                continue
            w = param.detach().cpu().numpy().flatten()
            stats = self.history[name]
            stats["mean"].append(float(np.mean(w)))
            stats["std"].append(float(np.std(w)))
            stats["min"].append(float(np.min(w)))
            stats["max"].append(float(np.max(w)))
            stats["l2_norm"].append(float(np.linalg.norm(w)))
            stats["epochs"].append(epoch)

    def plot_evolution(self, metric: str = "mean") -> Figure:
        """Plot weight evolution across epochs.

        Args:
            metric: One of 'mean', 'std', 'min', 'max', 'l2_norm'.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for name, stats in self.history.items():
            if not stats["epochs"]:
                continue
            ax.plot(stats["epochs"], stats[metric], label=name, marker="o", markersize=3)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Weight Evolution — {metric.replace('_', ' ').title()}", fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_all_metrics(self) -> Figure:
        """Plot all tracked metrics in a 2x3 grid."""
        metrics = ["mean", "std", "min", "max", "l2_norm"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for name, stats in self.history.items():
                if not stats["epochs"]:
                    continue
                ax.plot(stats["epochs"], stats[metric], label=name, linewidth=1)
            ax.set_title(metric.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].axis("off")  # Hide unused subplot
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=7, ncol=2)
        fig.suptitle("Weight Evolution — All Metrics", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    def get_data(self) -> Dict[str, Dict[str, List[float]]]:
        """Get raw tracking data for JSON export."""
        return deepcopy(self.history)
