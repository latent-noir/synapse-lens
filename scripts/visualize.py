"""
CLI: Launch Visualization Dashboard
====================================
Start the interactive Dash web dashboard.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --port 8080 --debug
"""

from __future__ import annotations

import argparse

from nnviz.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the nnviz dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    setup_logging()
    from nnviz.ui.app import run
    run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
