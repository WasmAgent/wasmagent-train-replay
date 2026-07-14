#!/usr/bin/env python3
"""Run the tests and output results."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-x", "-v"],
    capture_output=False,
)
sys.exit(result.returncode)
