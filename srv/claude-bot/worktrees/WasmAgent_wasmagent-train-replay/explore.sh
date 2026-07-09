#!/bin/bash
cd /srv/claude-bot/worktrees/WasmAgent_wasmagent-train-replay
git status --short
echo "===GIT LOG==="
git log --oneline -5
echo "===FILES==="
find . -not -path './.git/*' -type f | sort
