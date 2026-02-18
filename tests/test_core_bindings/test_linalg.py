"""Tests for C++ core bindings (requires nnviz_core to be built)."""

import pytest


@pytest.fixture
def core_available():
    """Skip if C++ core is not built."""
    try:
        import nnviz_core
        return nnviz_core
    except ImportError:
        pytest.skip("nnviz_core not built — skipping C++ binding tests")


class TestLinalgBindings:
    def test_frobenius_norm(self, core_available):
        import numpy as np
        m = np.eye(3)
        result = core_available.linalg.frobenius_norm(m)
        assert abs(result - np.sqrt(3.0)) < 1e-10

    def test_matmul(self, core_available):
        import numpy as np
        a = np.eye(3)
        b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = core_available.linalg.matmul(a, b)
        assert np.allclose(result, b)


class TestStatisticsBindings:
    def test_mean(self, core_available):
        result = core_available.statistics.mean([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(result - 3.0) < 1e-10

    def test_histogram(self, core_available):
        result = core_available.statistics.histogram([0.1, 0.5, 0.9, 1.5, 2.0], 5)
        assert "bin_edges" in result
        assert "counts" in result
        assert len(result["bin_edges"]) == 6


class TestActivationBindings:
    def test_relu(self, core_available):
        result = core_available.activations.relu([-1.0, 0.0, 1.0])
        assert result == [0.0, 0.0, 1.0]

    def test_softmax_sums_to_one(self, core_available):
        result = core_available.activations.softmax([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-10
