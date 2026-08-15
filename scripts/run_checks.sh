#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/verify_release.py
python -m unittest discover -s tests -v
