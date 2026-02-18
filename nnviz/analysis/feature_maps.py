"""Feature map extraction and visualization."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def extract_feature_maps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layers: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Extract feature maps from specified (or all conv) layers.

    Args:
        model: PyTorch model.
        input_tensor: Input tensor of shape (1, C, H, W) or (C, H, W).
        target_layers: Layer names to capture. If None, captures all Conv2d layers.

    Returns:
        Dict mapping layer name → activation tensor.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    feature_maps: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook_fn

    for name, module in model.named_modules():
        if target_layers is not None:
            if name in target_layers:
                hooks.append(module.register_forward_hook(make_hook(name)))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return feature_maps


def plot_feature_maps(
    feature_maps: Dict[str, torch.Tensor],
    max_channels: int = 16,
) -> List[Figure]:
    """Visualize feature maps as image grids.

    Args:
        feature_maps: Output from extract_feature_maps().
        max_channels: Maximum number of channels to display per layer.

    Returns:
        List of matplotlib Figure objects.
    """
    figures = []

    for layer_name, fmap in feature_maps.items():
        act = fmap[0].numpy()  # First sample: (channels, H, W)
        n = min(max_channels, act.shape[0])
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        fig.suptitle(f"Feature Maps — {layer_name}", fontsize=12, fontweight="bold")
        axes = np.array(axes).flatten()

        for i in range(n):
            axes[i].imshow(act[i], cmap="inferno")
            axes[i].set_title(f"Ch {i}", fontsize=7)
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()
        figures.append(fig)

    return figures
