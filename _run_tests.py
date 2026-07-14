import subprocess
import sys

# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-x", "-v"],
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr, file=sys.stderr)
print(f"exit code: {result.returncode}")
