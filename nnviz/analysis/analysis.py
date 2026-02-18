"""Confusion matrix and classification report utilities."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str = "cpu",
) -> np.ndarray:
    """Compute confusion matrix by running inference on a DataLoader.

    Returns:
        np.ndarray of shape (num_classes, num_classes).
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                cm[t, p] += 1

    return cm


def classification_report(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a classification report from a confusion matrix.

    Returns:
        Dict with per-class precision, recall, f1, support, plus macro/weighted averages.
    """
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    report: Dict[str, Any] = {"per_class": {}, "macro_avg": {}, "weighted_avg": {}}

    precisions, recalls, f1s, supports = [], [], [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report["per_class"][class_names[i]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": int(support),
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total_support = sum(supports)
    weights = [s / total_support for s in supports] if total_support > 0 else [0] * num_classes

    report["macro_avg"] = {
        "precision": round(np.mean(precisions), 4),
        "recall": round(np.mean(recalls), 4),
        "f1_score": round(np.mean(f1s), 4),
    }
    report["weighted_avg"] = {
        "precision": round(sum(p * w for p, w in zip(precisions, weights)), 4),
        "recall": round(sum(r * w for r, w in zip(recalls, weights)), 4),
        "f1_score": round(sum(f * w for f, w in zip(f1s, weights)), 4),
    }
    report["accuracy"] = round(np.trace(cm) / cm.sum(), 4) if cm.sum() > 0 else 0.0
    report["total_samples"] = int(total_support)

    return report
