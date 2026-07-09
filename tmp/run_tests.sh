#!/bin/bash
cd /srv/claude-bot/worktrees/WasmAgent_wasmagent-train-replay
pip install -e ".[dev]" --break-system-packages -q 2>&1
echo "PIP EXIT: $?"
python3 -m ruff check train_replay tests 2>&1
echo "RUFF EXIT: $?"
python3 -m mypy train_replay 2>&1
echo "MYPY EXIT: $?"
pytest tests/ -x -v 2>&1
echo "PYTEST EXIT: $?"
