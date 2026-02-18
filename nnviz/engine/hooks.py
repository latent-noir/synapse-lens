"""
Hook Manager
============
Registers and manages PyTorch forward/backward hooks for capturing
intermediate activations, gradients, and weight snapshots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HookRecord:
    """A single captured snapshot from a hook."""
    layer_name: str
    hook_type: str  # "forward" or "backward"
    data: torch.Tensor
    step: int = 0


class HookManager:
    """Manages forward and backward hooks on a PyTorch model.

    Captures activations (forward hooks) and gradients (backward hooks)
    for each registered layer.  Data is stored in an append-only buffer
    that can be queried by the visualization layer.

    Args:
        model: The PyTorch model to instrument.
        target_layers: Optional list of layer names to hook.  If None,
            hooks are registered on all ``nn.Linear`` and ``nn.Conv2d`` layers.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: list[str] | None = None,
    ) -> None:
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._records: list[HookRecord] = []
        self._step = 0
        self._target_layers = target_layers
        self._register_hooks()

    # ── Public API ───────────────────────────────────────────────────────────

    def get_records(self, layer_name: str | None = None) -> list[HookRecord]:
        """Retrieve captured hook records, optionally filtered by layer."""
        if layer_name is None:
            return list(self._records)
        return [r for r in self._records if r.layer_name == layer_name]

    def get_latest(self, layer_name: str | None = None) -> dict[str, HookRecord]:
        """Get the most recent record per layer (or for a specific layer)."""
        latest: dict[str, HookRecord] = {}
        for r in self._records:
            if layer_name is not None and r.layer_name != layer_name:
                continue
            key = f"{r.layer_name}_{r.hook_type}"
            latest[key] = r
        return latest

    def clear(self) -> None:
        """Clear all captured records."""
        self._records.clear()
        self._step = 0

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def increment_step(self) -> None:
        """Advance the global step counter (call once per batch)."""
        self._step += 1

    # ── Internals ────────────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Walk the model and register hooks on matching layers."""
        for name, module in self.model.named_modules():
            if self._target_layers is not None:
                if name not in self._target_layers:
                    continue
            elif not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            logger.debug(f"Registering hooks on layer: {name}")

            # Forward hook — captures output activations
            handle_fwd = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self._handles.append(handle_fwd)

            # Backward hook — captures gradient w.r.t. output
            handle_bwd = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )
            self._handles.append(handle_bwd)

    def _make_forward_hook(self, layer_name: str):
        """Create a forward hook closure for the given layer."""
        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._records.append(
                HookRecord(
                    layer_name=layer_name,
                    hook_type="forward",
                    data=output.detach().cpu(),
                    step=self._step,
                )
            )
        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create a backward hook closure for the given layer."""
        def hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            if grad_output[0] is not None:
                self._records.append(
                    HookRecord(
                        layer_name=layer_name,
                        hook_type="backward",
                        data=grad_output[0].detach().cpu(),
                        step=self._step,
                    )
                )
        return hook
