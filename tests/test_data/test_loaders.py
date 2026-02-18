"""Tests for data loading utilities."""

from nnviz.data.registry import DatasetRegistry


class TestDatasetRegistry:
    def test_mnist_registered(self):
        assert "mnist" in DatasetRegistry.available()

    def test_unknown_dataset(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetRegistry.get("nonexistent")
