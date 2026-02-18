"""Tests for the HookManager."""

import torch
from nnviz.engine.hooks import HookManager


class TestHookManager:
    def test_records_captured(self, simple_mlp, dummy_mnist_batch):
        hm = HookManager(simple_mlp)
        images, labels = dummy_mnist_batch

        output = simple_mlp(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()

        records = hm.get_records()
        assert len(records) > 0

        # Should have both forward and backward records
        types = {r.hook_type for r in records}
        assert "forward" in types
        assert "backward" in types

        hm.remove_hooks()

    def test_clear(self, simple_mlp, dummy_mnist_batch):
        hm = HookManager(simple_mlp)
        images, labels = dummy_mnist_batch
        simple_mlp(images)

        assert len(hm.get_records()) > 0
        hm.clear()
        assert len(hm.get_records()) == 0
        hm.remove_hooks()
