# Architecture

> Full system architecture for the Synapse Lens.  
> For a quick overview, see [README.md](README.md).

---

## Layered Architecture

| Layer | Language | Responsibility |
|---|---|---|
| **Core Engine** | C++17 + pybind11 | High-performance matrix ops, statistical computations, activation kernels |
| **ML Engine** | Python / PyTorch | Model definitions, training loop, hook management, checkpoint registry |
| **Data Layer** | Python / torchvision | Dataset loading, transform pipelines, dataset registry |
| **Visualization** | Python / Plotly | Weight heatmaps, gradient flow, activation grids, training curves |
| **UI** | Python / Dash | Interactive web dashboard with sidebar controls and tabbed panels |
| **Utilities** | Python | YAML config, structured logging, shared dataclass types |

---

## Data Flow

1. **User** selects model, dataset, and hyperparameters in the Dash UI
2. **UI → Engine**: `Trainer.train()` is invoked with config
3. **Engine** registers hooks via `HookManager`, runs the training loop
4. **HookManager** captures forward/backward snapshots per layer per batch
5. **Engine → Core** (optional): fast C++ statistics for histograms, norms
6. **Visualization** builds Plotly figures from hook records and model state
7. **UI** displays updated figures in the tabbed dashboard

---

## Extension Points

| Area | Mechanism |
|---|---|
| New datasets | `DatasetRegistry.register("name", factory_fn)` |
| New models | Add class + `ModelFactory.register("name", cls)` |
| New visualizations | Add module in `nnviz/visualization/`, return `go.Figure` |
| New C++ ops | Add `core/include/nnviz/*.hpp` + source + bind in `bind_core.cpp` |
| New dashboard tabs | Add layout in `layouts.py`, wire callback in `callbacks.py` |

---

## Build System

- **C++**: CMake 3.20+ with Eigen 3.4, pybind11, GoogleTest
- **Python**: `pyproject.toml` (PEP 621) with setuptools backend
- **Testing**: pytest (Python), GoogleTest + CTest (C++)
