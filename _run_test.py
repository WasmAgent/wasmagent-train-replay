import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]", "--break-system-packages", "-q"], capture_output=True, text=True)
print("pip stdout:", r.stdout[:200] if r.stdout else "")
print("pip stderr:", r.stderr[:200] if r.stderr else "")
print("pip rc:", r.returncode)
