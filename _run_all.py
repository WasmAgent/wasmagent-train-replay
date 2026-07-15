import subprocess, sys

# Run pip install
print("=== pip install ===")
proc = subprocess.Popen(
    [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "--user", "-q"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
stdout, stderr = proc.communicate()
print(f"exit: {proc.returncode}")
if stdout: print(stdout[-500:])
if stderr: print(stderr[-500:])
