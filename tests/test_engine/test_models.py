"""Tests for model definitions and the ModelFactory."""

import pytest
import torch

from nnviz.engine.models import SimpleMLP, SimpleCNN, ModelFactory


class TestSimpleMLP:
    def test_output_shape(self, simple_mlp, dummy_mnist_batch):
        images, _ = dummy_mnist_batch
        output = simple_mlp(images)
        assert output.shape == (8, 10)

    def test_forward_no_error(self, simple_mlp):
        x = torch.randn(4, 784)
        output = simple_mlp(x)
        assert output.shape == (4, 10)


class TestSimpleCNN:
    def test_output_shape(self, simple_cnn, dummy_mnist_batch):
        images, _ = dummy_mnist_batch
        output = simple_cnn(images)
        assert output.shape == (8, 10)


class TestModelFactory:
    def test_create_mlp(self):
        model = ModelFactory.create("mlp")
        assert isinstance(model, SimpleMLP)

    def test_create_cnn(self):
        model = ModelFactory.create("cnn")
        assert isinstance(model, SimpleCNN)

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("nonexistent")

    def test_available(self):
        available = ModelFactory.available()
        assert "mlp" in available
        assert "cnn" in available
