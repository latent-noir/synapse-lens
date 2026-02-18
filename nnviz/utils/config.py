"""
Configuration
=============
YAML-based configuration loading with dataclass validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: str = "mnist"
    batch_size: int = 64
    num_workers: int = 0
    data_dir: str = "./data"
    augment: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    type: str = "mlp"
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML config file. Uses the default config if None.

    Returns:
        A validated ExperimentConfig instance.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return ExperimentConfig()

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return ExperimentConfig(
        name=raw.get("name", "default"),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
