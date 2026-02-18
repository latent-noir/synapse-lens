"""
Trainer
=======
Configurable training loop with callback and hook integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from nnviz.engine.hooks import HookManager
from nnviz.engine.callbacks import Callback

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the training loop."""
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"
    log_interval: int = 100


class Trainer:
    """Runs the training loop, manages hooks and callbacks.

    Args:
        model: The PyTorch model to train.
        config: Training configuration.
        optimizer: Optional optimizer override (defaults to Adam).
        callbacks: List of Callback instances to invoke during training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig | None = None,
        optimizer: Optimizer | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer or Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.callbacks = callbacks or []
        self.hook_manager = HookManager(self.model)
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
        }

    def train(self, train_loader: DataLoader) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Training history dict with loss and accuracy per epoch.
        """
        self.model.train()
        self._notify_callbacks("on_train_start", trainer=self)

        for epoch in range(1, self.config.epochs + 1):
            self._notify_callbacks("on_epoch_start", epoch=epoch, trainer=self)
            epoch_loss, correct, total = 0.0, 0, 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                if (batch_idx + 1) % self.config.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

                self._notify_callbacks(
                    "on_batch_end",
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    trainer=self,
                )

            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total if total > 0 else 0.0
            self.history["train_loss"].append(avg_loss)
            self.history["train_accuracy"].append(accuracy)

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} — "
                f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            self._notify_callbacks(
                "on_epoch_end",
                epoch=epoch,
                loss=avg_loss,
                accuracy=accuracy,
                trainer=self,
            )

        self._notify_callbacks("on_train_end", trainer=self)
        return self.history

    def _notify_callbacks(self, event: str, **kwargs: Any) -> None:
        """Dispatch an event to all registered callbacks."""
        for cb in self.callbacks:
            handler = getattr(cb, event, None)
            if handler is not None:
                handler(**kwargs)
