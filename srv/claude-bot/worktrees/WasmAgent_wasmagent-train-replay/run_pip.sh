#!/bin/bash
set -e
cd /srv/claude-bot/worktrees/WasmAgent_wasmagent-train-replay
pip install -e ".[dev]" --break-system-packages -q 2>&1
echo "EXIT_CODE: $?"
