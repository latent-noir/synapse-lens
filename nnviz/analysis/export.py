"""JSON export and import for experiment reproducibility."""

from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import json
import datetime


def export_experiment_json(
    filepath: str | Path,
    config: Dict[str, Any],
    history: Dict[str, Any],
    metrics: Dict[str, Any],
    model_info: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Export a complete experiment result to a JSON file.

    Args:
        filepath: Output file path.
        config: Experiment configuration (hyperparameters, dataset, etc.).
        history: Training history (loss, accuracy per epoch).
        metrics: Final evaluation metrics (accuracy, F1, etc.).
        model_info: Optional model architecture details.
        metadata: Optional extra metadata.

    Returns:
        Path to the saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    experiment = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": _serialize(config),
        "training_history": _serialize(history),
        "evaluation_metrics": _serialize(metrics),
    }

    if model_info:
        experiment["model_info"] = _serialize(model_info)
    if metadata:
        experiment["metadata"] = _serialize(metadata)

    with open(filepath, "w") as f:
        json.dump(experiment, f, indent=2, default=str)

    return filepath


def load_experiment_json(filepath: str | Path) -> Dict[str, Any]:
    """Load a previously exported experiment JSON file.

    Returns:
        Dict with all experiment data.
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        return json.load(f)


def _serialize(obj: Any) -> Any:
    """Recursively serialize objects to JSON-safe types."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
