import os
import subprocess

log_path = "/content/sft.log"
print("=== SFT Run Status Monitor ===")

# Check process status
ps_res = subprocess.run(["pgrep", "-f", "colab_run_sft_apex.py"], capture_output=True, text=True)
pids = ps_res.stdout.strip().split()
if pids and pids[0]:
    print(f"Status: RUNNING (PID: {', '.join(pids)})")
else:
    print("Status: FINISHED / NOT_RUNNING")

# Check GPU status
try:
    smi_res = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    if smi_res.stdout.strip():
        parts = [p.strip() for p in smi_res.stdout.strip().split(",")]
        print(f"GPU Utilization: {parts[0]}% | VRAM: {parts[1]} MB / {parts[2]} MB | Temp: {parts[3]}C")
except Exception as e:
    print(f"GPU check notice: {e}")

# Read tail of log file
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"\n--- Log Output (last {min(35, len(lines))} lines) ---")
    for line in lines[-35:]:
        print(line, end="")
else:
    print("Log file not found yet.")
