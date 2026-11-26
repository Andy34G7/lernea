"""
Build the static benchmark site.
Usage:
    python -m tokenisers.dashboard                 # writes site/data.js
    python -m http.server -d site 8765             # preview at http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenisers.dashboard.build import REPO_ROOT, build_site


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tokenisers.dashboard", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "site", help="Site directory (default: site/)")
    parser.add_argument("--skip-playground", action="store_true",
                        help="Keep the playground block of the existing data.js instead of retraining its tokenizers")
    parser.add_argument("--offline", action="store_true",
                        help="Train playground tokenizers on the SYNTHETIC offline corpus instead of real data")
    args = parser.parse_args()
    path = build_site(args.out, playground=not args.skip_playground, offline=args.offline)
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
