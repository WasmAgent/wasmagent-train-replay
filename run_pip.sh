#!/bin/bash
pip install -e ".[dev]" --break-system-packages -q 2>&1
echo "PIP EXIT: $?"
