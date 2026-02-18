"""
Training Callbacks
==================
Protocol-based callback system for the training loop.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@runtime_checkable
class Callback(Protocol):
    """Protocol defining the callback interface for the Trainer."""

    def on_train_start(self, **kwargs: Any) -> None: ...
    def on_train_end(self, **kwargs: Any) -> None: ...
    def on_epoch_start(self, **kwargs: Any) -> None: ...
    def on_epoch_end(self, **kwargs: Any) -> None: ...
    def on_batch_end(self, **kwargs: Any) -> None: ...


class LoggingCallback:
    """Logs training progress to the Python logger."""

    def on_train_start(self, **kwargs: Any) -> None:
        logger.info("Training started.")

    def on_train_end(self, **kwargs: Any) -> None:
        logger.info("Training complete.")

    def on_epoch_start(self, epoch: int = 0, **kwargs: Any) -> None:
        logger.info(f"── Epoch {epoch} ──")

    def on_epoch_end(
        self, epoch: int = 0, loss: float = 0.0, accuracy: float = 0.0, **kwargs: Any
    ) -> None:
        logger.info(f"Epoch {epoch} — Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

    def on_batch_end(self, **kwargs: Any) -> None:
        pass  # Batch-level logging handled by Trainer


class CheckpointCallback:
    """Saves model checkpoints at the end of each epoch.

    Args:
        save_dir: Directory to save checkpoints to.
        save_best_only: If True, only save when loss improves.
    """

    def __init__(self, save_dir: str = "checkpoints", save_best_only: bool = True) -> None:
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self._best_loss = float("inf")

    def on_train_start(self, **kwargs: Any) -> None:
        import os
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_end(self, **kwargs: Any) -> None:
        pass

    def on_epoch_start(self, **kwargs: Any) -> None:
        pass

    def on_epoch_end(
        self, epoch: int = 0, loss: float = 0.0, trainer: Any = None, **kwargs: Any
    ) -> None:
        if trainer is None:
            return

        if self.save_best_only and loss >= self._best_loss:
            return

        self._best_loss = loss
        path = f"{self.save_dir}/model_epoch_{epoch}.pt"
        torch.save(trainer.model.state_dict(), path)
        logger.info(f"Checkpoint saved: {path}")

    def on_batch_end(self, **kwargs: Any) -> None:
        pass
