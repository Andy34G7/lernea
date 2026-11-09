"""
CLI entry point for running the Tokenisers Lab Benchmark Dashboard.
Usage:
    python -m tokenisers.dashboard --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser

from tokenisers.dashboard.server import DashboardServer, start_server


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tokenisers.dashboard",
        description="Artificial Analysis-style Benchmark Intelligence Dashboard for Tokenisers Lab",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        choices=DashboardServer.ALLOWED_HOSTS,
        help="Interface to bind (default: 127.0.0.1, or HOST env var; 0.0.0.0 for public deploy)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8765")),
        help="Port number to bind (default: 8765, or PORT env var)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Train playground tokenizers on the SYNTHETIC offline corpus instead of real data",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard in the default web browser upon launch",
    )

    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print("=" * 72)
    print("  TOKENISERS LAB — ALGORITHMIC BENCHMARK INTELLIGENCE DASHBOARD")
    print(f"  Live UI available at: {url}")
    print("  Endpoints:")
    print(f"    - Overview & Visual Dashboard: {url}")
    print(f"    - Benchmarks Unified JSON API: {url}api/benchmarks")
    print(f"    - Live Tokenizer Playground API: {url}api/tokenize")
    print("=" * 72)

    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        start_server(host=args.host, port=args.port, blocking=True, offline=args.offline)
    except KeyboardInterrupt:
        print("\nExiting dashboard.")
        sys.exit(0)


if __name__ == "__main__":
    main()
