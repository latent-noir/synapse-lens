"""
CLI: Train a model
==================
Run a training experiment from a configuration file.

Usage:
    python scripts/train.py                          # Use default config
    python scripts/train.py --config configs/mnist_experiment.yaml
"""

from __future__ import annotations

import argparse
import logging

from nnviz.utils.logging import setup_logging
from nnviz.utils.config import load_config
from nnviz.engine.models import ModelFactory
from nnviz.engine.trainer import Trainer, TrainerConfig
from nnviz.engine.callbacks import LoggingCallback, CheckpointCallback
from nnviz.data.loaders import get_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML experiment config file.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    logger = logging.getLogger("nnviz")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Model: {config.model.type} | Dataset: {config.data.name}")
    logger.info(f"Epochs: {config.training.epochs} | LR: {config.training.learning_rate}")

    # Create model
    model_kwargs = {"dropout": config.model.dropout}
    if config.model.type == "mlp":
        model_kwargs["hidden_sizes"] = config.model.hidden_sizes
    model = ModelFactory.create(config.model.type, **model_kwargs)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loader
    train_loader = get_dataloader(
        dataset_name=config.data.name,
        train=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        data_dir=config.data.data_dir,
    )

    # Configure trainer
    trainer_config = TrainerConfig(
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        device=config.training.device,
        log_interval=config.training.log_interval,
    )
    callbacks = [
        LoggingCallback(),
        CheckpointCallback(
            save_dir=config.training.checkpoint_dir,
            save_best_only=config.training.save_best_only,
        ),
    ]
    trainer = Trainer(model, config=trainer_config, callbacks=callbacks)

    # Train
    history = trainer.train(train_loader)
    logger.info(
        f"Training complete — Final loss: {history['train_loss'][-1]:.4f}, "
        f"Accuracy: {history['train_accuracy'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
