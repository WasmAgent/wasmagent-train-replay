#!/bin/bash
echo "HOME: $HOME"
echo "PWD: $PWD"
ls -la /srv/claude-bot/worktrees/ 2>/dev/null || echo "no srv path"
ls -la /srv/ 2>/dev/null || echo "no srv"
find / -maxdepth 4 -name "wasmagent-train-replay" -type d 2>/dev/null | head -20
