"""
Model Registry
==============
Serialization and persistence of trained model checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn as nn

from nnviz.engine.models import ModelFactory

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Save, load, and list model checkpoints with metadata.

    Args:
        root_dir: Root directory for storing checkpoints.
    """

    METADATA_FILE = "registry.json"

    def __init__(self, root_dir: str = "checkpoints") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.root / self.METADATA_FILE
        self._metadata: dict[str, Any] = self._load_metadata()

    def save(
        self,
        model: nn.Module,
        name: str,
        model_type: str,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a model checkpoint with metadata."""
        path = self.root / f"{name}.pt"
        torch.save(model.state_dict(), path)

        self._metadata[name] = {
            "model_type": model_type,
            "path": str(path),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }
        self._save_metadata()
        logger.info(f"Model saved: {name} → {path}")
        return path

    def load(self, name: str, **model_kwargs: Any) -> nn.Module:
        """Load a model by registered name."""
        if name not in self._metadata:
            raise KeyError(f"Model '{name}' not found in registry.")

        entry = self._metadata[name]
        model = ModelFactory.create(entry["model_type"], **model_kwargs)
        model.load_state_dict(torch.load(entry["path"], weights_only=True))
        logger.info(f"Model loaded: {name}")
        return model

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models with metadata."""
        return [{"name": k, **v} for k, v in self._metadata.items()]

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata_path.exists():
            return json.loads(self._metadata_path.read_text())
        return {}

    def _save_metadata(self) -> None:
        self._metadata_path.write_text(json.dumps(self._metadata, indent=2))
