#!/bin/bash
cd /srv/claude-bot/worktrees/WasmAgent_wasmagent-train-replay
git log --oneline -10 2>&1
echo "---"
git status --short 2>&1
echo "---"
git diff HEAD -- docs/15-milestones.md 2>&1 | head -100
