# ─── nnviz Docker Image ──────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml setup.py ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY . .

# Build C++ core
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DNNVIZ_BUILD_TESTS=OFF && \
    cmake --build build && \
    cp build/core/nnviz_core*.so nnviz/ 2>/dev/null || true

# Expose dashboard port
EXPOSE 8050

CMD ["python", "scripts/visualize.py", "--host", "0.0.0.0"]
