"""Architecture comparison dashboard — compare multiple trained models side by side."""

from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def compare_architectures(
    models: Dict[str, nn.Module],
    histories: Dict[str, Dict[str, List[float]]],
    dataloader: DataLoader,
    num_classes: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare multiple architectures by performance, parameter count, and speed.

    Args:
        models: Dict mapping model name → nn.Module.
        histories: Dict mapping model name → training history.
        dataloader: Test DataLoader for evaluation.
        num_classes: Number of output classes.
        device: Device for inference.

    Returns:
        Dict with comparison tables and figures.
    """
    results: Dict[str, Any] = {"summary": [], "figures": []}

    for name, model in models.items():
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Evaluate accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        accuracy = correct / total if total > 0 else 0.0

        # Final training loss
        history = histories.get(name, {})
        final_loss = history.get("train_loss", [float("nan")])[-1] if history.get("train_loss") else float("nan")

        results["summary"].append({
            "name": name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "test_accuracy": round(accuracy, 4),
            "final_train_loss": round(final_loss, 4),
            "num_epochs": len(history.get("train_loss", [])),
        })

    # ── Generate comparison plots ────────────────────────────────────────
    # 1. Accuracy comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [r["name"] for r in results["summary"]]
    accs = [r["test_accuracy"] for r in results["summary"]]
    params = [r["total_params"] for r in results["summary"]]

    axes[0].barh(names, accs, color=["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b"][:len(names)])
    axes[0].set_xlabel("Test Accuracy")
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_xlim(0, 1)

    # 2. Parameter count comparison
    axes[1].barh(names, params, color=["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b"][:len(names)])
    axes[1].set_xlabel("Total Parameters")
    axes[1].set_title("Model Size")

    # 3. Training curves overlay
    for name in names:
        h = histories.get(name, {})
        if "train_loss" in h:
            axes[2].plot(h["train_loss"], label=name, linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Training Loss Comparison")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Architecture Comparison Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    results["figures"].append(fig)

    return results
