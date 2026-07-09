#!/usr/bin/env bash
set -euo pipefail
cd /srv/claude-bot/worktrees/WasmAgent_wasmagent-train-replay
pip install -e ".[dev]" --break-system-packages -q 2>&1; echo "PIP_EXIT=$?"
python3 -m ruff check train_replay tests 2>&1; echo "RUFF_EXIT=$?"
python3 -m mypy train_replay 2>&1; echo "MYPY_EXIT=$?"
pytest tests/ -x 2>&1; echo "PYTEST_EXIT=$?"
git status --short 2>&1; echo "GITSTATUS_EXIT=$?"
