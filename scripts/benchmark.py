"""
CLI: Benchmark C++ vs Python Operations
========================================
Compares the performance of C++ core operations against
equivalent pure-Python / NumPy implementations.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import time
import argparse
import numpy as np


def _time_it(fn, *args, iterations: int = 1000) -> float:
    """Time a function over multiple iterations, return avg milliseconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000  # ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark C++ core vs NumPy.")
    parser.add_argument("--size", type=int, default=1000, help="Vector/matrix size")
    parser.add_argument("--iterations", type=int, default=1000, help="Timing iterations")
    args = parser.parse_args()

    n = args.size
    iters = args.iterations

    print(f"\n{'='*60}")
    print(f"  NNVIZ Benchmark — size={n}, iterations={iters}")
    print(f"{'='*60}\n")

    data = np.random.randn(n).tolist()

    # ── NumPy baselines ──────────────────────────────────────────
    np_data = np.array(data)
    np_mean_ms = _time_it(np.mean, np_data, iterations=iters)
    np_std_ms = _time_it(np.std, np_data, iterations=iters)

    print(f"  {'Operation':<30} {'NumPy (ms)':>12} {'C++ (ms)':>12} {'Speedup':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")

    # ── C++ core (if available) ──────────────────────────────────
    try:
        from nnviz_core import statistics as cpp_stats

        cpp_mean_ms = _time_it(cpp_stats.mean, data, iterations=iters)
        cpp_std_ms = _time_it(cpp_stats.stddev, data, iterations=iters)

        print(f"  {'mean':<30} {np_mean_ms:>12.4f} {cpp_mean_ms:>12.4f} {np_mean_ms/cpp_mean_ms:>9.1f}x")
        print(f"  {'stddev':<30} {np_std_ms:>12.4f} {cpp_std_ms:>12.4f} {np_std_ms/cpp_std_ms:>9.1f}x")

    except ImportError:
        print("  ⚠️  C++ core (nnviz_core) not built — skipping C++ benchmarks.")
        print(f"  {'mean (NumPy only)':<30} {np_mean_ms:>12.4f}")
        print(f"  {'stddev (NumPy only)':<30} {np_std_ms:>12.4f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
