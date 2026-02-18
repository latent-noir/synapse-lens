"""Matplotlib visualization backend.

This is the base implementation that can also render into Qt widgets
via matplotlib's Qt5Agg backend. To use with Qt:
    import matplotlib
    matplotlib.use('Qt5Agg')
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import torch
import torch.nn as nn

from nnviz.visualization.base import VisualizationBackend, register_backend


class MatplotlibBackend(VisualizationBackend):
    """Matplotlib-based visualization backend, extensible to Qt via Qt5Agg."""

    def __init__(self, style: str = "dark_background", figsize: tuple = (10, 6)):
        self.style = style
        self.figsize = figsize
        plt.style.use(style)

    def _new_figure(self, **kwargs) -> tuple:
        figsize = kwargs.pop("figsize", self.figsize)
        return plt.subplots(figsize=figsize, **kwargs)

    # ── Weight Visualization ─────────────────────────────────────────────

    def plot_weight_heatmap(self, model: nn.Module, layer_name: Optional[str] = None) -> Figure:
        layers = [(n, p) for n, p in model.named_parameters() if "weight" in n and p.dim() >= 2]
        if layer_name:
            layers = [(n, p) for n, p in layers if layer_name in n]

        if not layers:
            fig, ax = self._new_figure()
            ax.text(0.5, 0.5, "No weight layers found", ha="center", va="center",
                    fontsize=14, color="white")
            ax.set_axis_off()
            return fig

        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, param) in zip(axes, layers):
            w = param.detach().cpu().numpy()
            if w.ndim > 2:
                w = w.reshape(w.shape[0], -1)  # Flatten conv filters
            im = ax.imshow(w, aspect="auto", cmap="viridis")
            ax.set_title(name, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("Weight Heatmaps", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_weight_distribution(self, model: nn.Module) -> Figure:
        fig, ax = self._new_figure()
        for name, param in model.named_parameters():
            if "weight" in name:
                w = param.detach().cpu().numpy().flatten()
                ax.hist(w, bins=50, alpha=0.5, label=name, density=True)

        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density")
        ax.set_title("Weight Distributions")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        return fig

    # ── Gradient Visualization ───────────────────────────────────────────

    def plot_gradient_flow(self, model: nn.Module) -> Figure:
        fig, ax = self._new_figure()
        layers_info = []
        for name, param in model.named_parameters():
            if param.grad is not None and "weight" in name:
                grad = param.grad.detach().cpu().numpy()
                layers_info.append((name, np.abs(grad).mean(), np.abs(grad).max()))

        if not layers_info:
            ax.text(0.5, 0.5, "No gradients available (run backward first)",
                    ha="center", va="center", fontsize=12, color="white")
            ax.set_axis_off()
            return fig

        names = [l[0] for l in layers_info]
        means = [l[1] for l in layers_info]
        maxes = [l[2] for l in layers_info]
        x = range(len(names))

        ax.bar(x, maxes, alpha=0.3, color="cyan", label="Max Gradient")
        ax.bar(x, means, alpha=0.8, color="lime", label="Mean Gradient")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Gradient Magnitude")
        ax.set_title("Gradient Flow")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_gradient_histogram(self, gradients: Dict[str, Any]) -> Figure:
        fig, ax = self._new_figure()
        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                grad = grad.detach().cpu().numpy().flatten()
            ax.hist(grad, bins=50, alpha=0.5, label=name, density=True)

        ax.set_xlabel("Gradient Value")
        ax.set_ylabel("Density")
        ax.set_title("Gradient Distributions")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    # ── Activation Visualization ─────────────────────────────────────────

    def plot_activation_grid(self, activations: Any, layer_name: str) -> Figure:
        if isinstance(activations, torch.Tensor):
            act = activations.detach().cpu().numpy()
        else:
            act = np.array(activations)

        if act.ndim == 4:  # Conv layer: (batch, channels, H, W)
            act = act[0]  # Take first sample
            n = min(16, act.shape[0])
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
            axes = np.array(axes).flatten()
            for i in range(n):
                axes[i].imshow(act[i], cmap="inferno")
                axes[i].set_title(f"Ch {i}", fontsize=8)
                axes[i].axis("off")
            for i in range(n, len(axes)):
                axes[i].axis("off")
        else:
            fig, ax = self._new_figure()
            if act.ndim >= 2:
                act = act[0]
            ax.bar(range(len(act)), act, color="cyan", alpha=0.7)
            ax.set_title(f"Activations: {layer_name}")

        fig.suptitle(f"Activation Grid — {layer_name}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_activation_distribution(self, activations: Dict[str, Any]) -> Figure:
        fig, ax = self._new_figure()
        for name, act in activations.items():
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy().flatten()
            ax.hist(act, bins=50, alpha=0.5, label=name, density=True)

        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Density")
        ax.set_title("Activation Distributions")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    # ── Training Metrics ─────────────────────────────────────────────────

    def plot_training_curves(self, history: Dict[str, List[float]]) -> Figure:
        fig, ax1 = self._new_figure()
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        # Loss (left y-axis)
        if "train_loss" in history:
            ax1.plot(epochs, history["train_loss"], "c-o", label="Train Loss", markersize=3)
        if "val_loss" in history:
            ax1.plot(epochs, history["val_loss"], "m-s", label="Val Loss", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="cyan")
        ax1.tick_params(axis="y", labelcolor="cyan")

        # Accuracy (right y-axis)
        if "train_accuracy" in history or "val_accuracy" in history:
            ax2 = ax1.twinx()
            if "train_accuracy" in history:
                ax2.plot(epochs, history["train_accuracy"], "g-^", label="Train Acc", markersize=3)
            if "val_accuracy" in history:
                ax2.plot(epochs, history["val_accuracy"], "y-v", label="Val Acc", markersize=3)
            ax2.set_ylabel("Accuracy", color="lime")
            ax2.tick_params(axis="y", labelcolor="lime")
            ax2.legend(loc="center right", fontsize=8)

        ax1.set_title("Training Curves", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig

    # ── Confusion Matrix ─────────────────────────────────────────────────

    def plot_confusion_matrix(self, y_true: Any, y_pred: Any, class_names: List[str]) -> Figure:
        from sklearn.metrics import confusion_matrix as sk_cm

        cm = sk_cm(y_true, y_pred)
        fig, ax = self._new_figure(figsize=(8, 8))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)

        # Annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8, color=color)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        return fig


# Register with the backend system
register_backend("matplotlib", MatplotlibBackend)
