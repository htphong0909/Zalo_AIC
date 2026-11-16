import subprocess
import sys
import os

log_path = "/content/sft.log"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== SFT Run Launcher Initialized ===\n")

proc = subprocess.Popen(
    [sys.executable, "-u", "/content/colab_run_sft_apex.py"],
    stdout=open(log_path, "a", encoding="utf-8", buffering=1),
    stderr=subprocess.STDOUT,
    start_new_session=True,
    cwd="/content",
)

print(f"SFT Training Job launched successfully with PID: {proc.pid}")
print(f"Log file streaming to: {log_path}")
