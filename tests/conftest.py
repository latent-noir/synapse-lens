"""Shared pytest fixtures for the nnviz test suite."""

import pytest
import torch

from nnviz.engine.models import SimpleMLP, SimpleCNN


@pytest.fixture
def simple_mlp():
    """A small MLP for testing."""
    return SimpleMLP(input_size=784, hidden_sizes=[64, 32], num_classes=10, dropout=0.0)


@pytest.fixture
def simple_cnn():
    """A small CNN for testing."""
    return SimpleCNN(in_channels=1, num_classes=10, dropout=0.0)


@pytest.fixture
def dummy_mnist_batch():
    """A fake MNIST-like batch of data."""
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    return images, labels
