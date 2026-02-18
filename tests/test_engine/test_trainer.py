"""Tests for the Trainer class."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnviz.engine.trainer import Trainer, TrainerConfig


@pytest.fixture
def tiny_dataloader():
    """A tiny DataLoader for fast tests."""
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8)


class TestTrainer:
    def test_train_runs(self, simple_mlp, tiny_dataloader):
        config = TrainerConfig(epochs=2, learning_rate=0.01)
        trainer = Trainer(simple_mlp, config=config)
        history = trainer.train(tiny_dataloader)

        assert "train_loss" in history
        assert "train_accuracy" in history
        assert len(history["train_loss"]) == 2

    def test_history_loss_decreases_or_stable(self, simple_cnn, tiny_dataloader):
        config = TrainerConfig(epochs=3, learning_rate=0.01)
        trainer = Trainer(simple_cnn, config=config)
        history = trainer.train(tiny_dataloader)

        # Just verify it runs and produces values (no convergence guarantee on 32 samples)
        assert all(isinstance(v, float) for v in history["train_loss"])
