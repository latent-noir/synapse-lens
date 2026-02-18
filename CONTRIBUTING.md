# Contributing to Synapse Lens

Thank you for your interest in contributing! We welcome bug reports, feature requests, and pull requests to improve the project.

## 🛠️ Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/Start-Coding-AI/neural-network-visualizer.git
    cd neural-network-visualizer
    ```
3.  **Install dependencies** in editable mode:
    ```bash
    pip install -e ".[dev]"
    ```
4.  **Create a branch** for your feature:
    ```bash
    git checkout -b feature/my-awesome-feature
    ```

## 🧪 Running Tests

Please ensure all tests pass before submitting a PR.

### Python Tests
```bash
pytest tests/
```

### C++ Tests
If modifying the core engine:
```bash
mkdir -p build && cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

## 🎨 Code Style

- **Python**: We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/). Please run `black .` and `flake8` before committing.
- **C++**: We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
- **Docstrings**: All classes and functions should have Google-style docstrings.

## 📝 Pull Request Guidelines

1.  Provide a clear description of the problem and verify that your changes fix it.
2.  Include unit tests for any new functionality.
3.  Update documentation if applicable.
4.  Keep PRs focused on a single change.

## 🐛 Reporting Bugs

Please open an issue on GitHub with:
- A clear description of the bug.
- Steps to reproduce.
- Expected vs. actual behavior.
- Environment details (OS, Python version, etc.).

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
