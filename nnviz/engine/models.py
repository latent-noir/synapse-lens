"""
Model Definitions
=================
Pre-built neural network architectures for experimentation.
All models accept a configuration dict and are registered in the ModelFactory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


class SimpleMLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers.

    Args:
        input_size: Number of input features (e.g., 784 for MNIST).
        hidden_sizes: List of hidden layer widths.
        num_classes: Number of output classes.
        activation: Activation function class (default: ReLU).
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] | None = None,
        num_classes: int = 10,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers: list[nn.Module] = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                activation(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))


class SimpleCNN(nn.Module):
    """Simple convolutional neural network for image classification.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class ModelFactory:
    """Registry-based factory for creating models by name.

    Usage:
        >>> model = ModelFactory.create("mlp", input_size=784)
        >>> ModelFactory.register("custom", MyCustomModel)
    """

    _registry: dict[str, type[nn.Module]] = {
        "mlp": SimpleMLP,
        "cnn": SimpleCNN,
    }

    @classmethod
    def register(cls, name: str, model_class: type[nn.Module]) -> None:
        """Register a new model architecture."""
        cls._registry[name.lower()] = model_class

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Create a model instance by registered name."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown model '{name}'. Available: {available}"
            )
        return cls._registry[name_lower](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """List all registered model names."""
        return sorted(cls._registry.keys())


class AnomalyMLP(nn.Module):
    """MLP for tabular anomaly / intrusion detection.

    Designed for non-image data with batch normalization for
    stable training on high-dimensional numeric features.

    Args:
        input_size: Number of input features (e.g., 41 for NSL-KDD).
        hidden_sizes: List of hidden layer widths.
        num_classes: 2 for binary, 5 for multi-class.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int = 41,
        hidden_sizes: list[int] | None = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        layers: list[nn.Module] = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Register AnomalyMLP
ModelFactory.register("anomaly_mlp", AnomalyMLP)

