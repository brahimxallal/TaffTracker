#!/usr/bin/env bash
# CI pipeline: lint, type-check, and test.
# Usage: bash scripts/ci.sh
set -euo pipefail

echo "=== ruff check ==="
python -m ruff check src/ tests/ scripts/

echo "=== ruff format --check ==="
python -m ruff format --check src/ tests/ scripts/

echo "=== mypy ==="
python -m mypy src/ --ignore-missing-imports --no-error-summary

echo "=== pytest ==="
python -m pytest tests/ -x -q --ignore=tests/test_capture_process.py

echo "=== All checks passed ==="
