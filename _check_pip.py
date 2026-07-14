import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "list"],
    capture_output=True,
    text=True,
)
print(result.stdout[:2000])
print("...", file=sys.stderr)
