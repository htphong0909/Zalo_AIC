#!/usr/bin/env python3
"""
Production-grade, self-healing Colab execution runner for ZALO_AIC.
Hardcoded auto-check for 'Connection was lost' and WebSocket disconnects.
Guarantees caller never hangs and handles kernel recovery deterministically.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "scripts" / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

DEFAULT_SESSION = os.environ.get("COLAB_SESSION", "buzzasr")
DEFAULT_GPU = os.environ.get("COLAB_GPU", "L4")

# ==============================================================================
# HARDCODED AUTO-CHECK SIGNATURES (Independent of Agent Prompts)
# ==============================================================================
LOST_CONNECTION_SIGNATURES: List[str] = [
    "Connection was lost",
    "RuntimeError: Connection was lost",
    "jupyter_kernel_client",
    "WebSocketClosedError",
    "websocket connection closed",
    "ConnectionResetError",
    "RemoteDisconnected",
    "Read timed out",
    "Connection aborted",
    "ConnectionError",
    "TimeoutError",
    "BrokenPipeError",
    "Server disconnected",
]


def check_lost_connection(text: Optional[str]) -> Tuple[bool, str]:
    """Hardcoded auto-check for Colab WebSocket / connection drops.

    Inspects text (stdout/stderr/traceback) against known disconnect signatures.
    Returns:
        (is_lost, matching_signature)
    """
    if not text:
        return False, ""
    lower = str(text).lower()
    for sig in LOST_CONNECTION_SIGNATURES:
        if sig.lower() in lower:
            return True, sig
    return False, ""


def reset_session_kernel(session: str = DEFAULT_SESSION) -> None:
    """Clears stale kernel_id/session_id in colab-cli cache so colab-cli acquires a fresh kernel."""
    try:
        p = Path.home() / ".config" / "colab-cli" / "sessions.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if session in data:
                data[session]["kernel_id"] = None
                data[session]["session_id"] = None
                p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                print(f"[colab_runner] Reset stale kernel_id for session '{session}'.", flush=True)
    except Exception as e:
        print(f"[colab_runner] Warning: could not reset session kernel: {e}", flush=True)


def restart_kernel(session: str = DEFAULT_SESSION, timeout: int = 30) -> bool:
    """Restarts the Colab session kernel to clear dead WebSocket state."""
    print(f"[colab_runner] Invoking 'colab restart-kernel -s {session}'...", flush=True)
    try:
        res = subprocess.run(
            ["colab", "restart-kernel", "-s", session],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        reset_session_kernel(session)
        time.sleep(3)
        return res.returncode == 0
    except Exception as e:
        print(f"[colab_runner] Error restarting kernel: {e}", flush=True)
        reset_session_kernel(session)
        return False


def package_codebase(output_zip: Path) -> int:
    """Safely packages code without risk of recursive zip bombs or heavy directories."""
    exclude_dirs = {
        '.git', '__pycache__', '.pytest_cache', 'archive', 'temp',
        'reports', 'data', 'weights', 'models', '.cache', 'scratch',
        '.user_uploaded', '.system_generated'
    }
    if output_zip.exists():
        output_zip.unlink()

    file_count = 0
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['src', 'tests', 'configs', 'scripts']:
            folder_path = ROOT / folder
            if not folder_path.exists():
                continue
            for p in folder_path.rglob('*'):
                if (
                    p.is_file()
                    and p.suffix != '.zip'
                    and not any(part in exclude_dirs for part in p.parts)
                ):
                    z.write(p, p.relative_to(ROOT))
                    file_count += 1

    size_kb = output_zip.stat().st_size / 1024.0
    print(f"[colab_runner] Packaged {file_count} files ({size_kb:.1f} KB) into {output_zip.name}", flush=True)
    return file_count


def ping_session(session: str = DEFAULT_SESSION, timeout: int = 15) -> bool:
    """Pings Colab kernel. Returns True if alive and responsive."""
    ping_script = SCRATCH / "ping.py"
    ping_script.write_text('print("COLAB_PONG", flush=True)\n', encoding="utf-8")
    sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    for attempt in range(2):
        try:
            res = subprocess.run(
                ["colab", "exec", "-s", session, "-f", str(ping_script), "--timeout", str(timeout)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=sub_env,
                timeout=timeout + 5,
            )
            if res.returncode == 0 and "COLAB_PONG" in res.stdout:
                return True
        except Exception as e:
            print(f"[colab_runner] Ping attempt {attempt+1} failed: {e}", flush=True)
        if attempt == 0:
            reset_session_kernel(session)
            time.sleep(2)
    return False


def ensure_session(session: str = DEFAULT_SESSION, gpu: str = DEFAULT_GPU) -> bool:
    """Verifies session exists in `colab sessions`; if pruned/terminated, creates it anew."""
    try:
        res = subprocess.run(["colab", "sessions"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if session not in res.stdout:
            print(f"[colab_runner] Session '{session}' missing or pruned! Creating new session (GPU={gpu})...", flush=True)
            new_res = subprocess.run(["colab", "new", "-s", session, "--gpu", gpu], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=150)
            if new_res.returncode == 0:
                print(f"[colab_runner] Session '{session}' created successfully.", flush=True)
                time.sleep(5)
                return True
            else:
                print(f"[colab_runner] Failed to create session '{session}': {new_res.stderr}", flush=True)
                return False
        return True
    except Exception as e:
        print(f"[colab_runner] Error checking session: {e}", flush=True)
        return False


def upload_file(
    local_path: str | Path,
    remote_path: str,
    *,
    session: str = DEFAULT_SESSION,
    max_retries: int = 3,
    timeout: int = 180,
    gpu: str = DEFAULT_GPU,
) -> subprocess.CompletedProcess:
    """Uploads a local file to Colab with retries on HTTP 500 or WebSocket disconnects."""
    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(f"Local upload source not found: {src}")
    args = ["colab", "upload", "-s", session, str(src), remote_path]
    sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    last_res: subprocess.CompletedProcess | None = None
    for attempt in range(1, max_retries + 1):
        ensure_session(session, gpu=gpu)
        try:
            res = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=sub_env,
                timeout=timeout,
            )
            last_res = res
            if res.returncode == 0:
                return res
            combined = f"{res.stdout}\n{res.stderr}"
            print(
                f"[colab_runner] Upload attempt {attempt}/{max_retries} failed ({res.returncode}): "
                f"{combined.strip()[:300]}",
                flush=True,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"[colab_runner] Upload timeout on attempt {attempt}/{max_retries}: {exc}", flush=True)
        if attempt < max_retries:
            time.sleep(3 * attempt)
    if last_res is not None:
        return last_res
    return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="Upload timed out")


def run_command(
    cmd_or_file: str,
    session: str = DEFAULT_SESSION,
    is_file: bool = False,
    timeout: int = 300,
    max_retries: int = 3,
    env: Optional[List[str]] = None,
    stream: bool = True,
    gpu: str = "L4",
) -> subprocess.CompletedProcess:
    """Runs a command or script on Colab with real-time streaming and auto-recovery for lost connection.

    ``env`` is a list of ``KEY=VALUE`` strings set in the remote kernel (e.g. ``HF_TOKEN=...``). Values may be
    secrets, so they are never printed and are stripped from timeout tracebacks.
    """
    args = ["colab", "exec", "-s", session]
    if is_file:
        args.extend(["-f", cmd_or_file])
    else:
        args.append(cmd_or_file)
    args.extend(["--timeout", str(timeout)])
    for kv in env or []:
        args.extend(["--env", kv])

    sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    last_res = None

    for attempt in range(1, max_retries + 1):
        # 0. Pre-flight session existence
        ensure_session(session, gpu=gpu)

        try:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=sub_env,
                bufsize=1,
            )
            line_q: queue.Queue[str | None] = queue.Queue()

            def _read_stdout() -> None:
                try:
                    if proc.stdout:
                        for ln in iter(proc.stdout.readline, ""):
                            line_q.put(ln)
                finally:
                    line_q.put(None)

            reader_t = threading.Thread(target=_read_stdout, daemon=True)
            reader_t.start()

            output_lines = []
            start_t = time.time()
            is_lost = False
            sig = ""
            eof = False

            while True:
                try:
                    line = line_q.get(timeout=0.1)
                except queue.Empty:
                    line = ""

                if line is None:
                    eof = True
                    if proc.poll() is not None:
                        break
                    continue

                if line:
                    output_lines.append(line)
                    if stream:
                        print(line, end="", flush=True)
                    has_lost, detected_sig = check_lost_connection(line)
                    if has_lost:
                        is_lost = True
                        sig = detected_sig
                        proc.terminate()
                        break
                elif eof and proc.poll() is not None:
                    break
                elif time.time() - start_t > timeout + 30:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    reader_t.join(timeout=2)
                    raise subprocess.TimeoutExpired(cmd=args, timeout=timeout + 30)

            while not line_q.empty():
                rem = line_q.get_nowait()
                if rem:
                    output_lines.append(rem)
                    if stream:
                        print(rem, end="", flush=True)
                    has_lost, detected_sig = check_lost_connection(rem)
                    if has_lost:
                        is_lost = True
                        sig = detected_sig

            try:
                ret_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                ret_code = proc.wait(timeout=5)
            reader_t.join(timeout=2)
            combined_output = "".join(output_lines)
            res = subprocess.CompletedProcess(args=args, returncode=ret_code, stdout=combined_output, stderr="")

            if is_lost:
                print(
                    f"\n[COLAB AUTO-CHECK] DETECTED: Connection lost ('{sig}') on attempt {attempt}/{max_retries}!",
                    flush=True,
                )
                if attempt < max_retries:
                    print(
                        f"[COLAB AUTO-CHECK] Self-healing: restarting kernel or session '{session}' and retrying in 5s...",
                        flush=True,
                    )
                    restart_kernel(session)
                    ensure_session(session, gpu=gpu)
                    ping_ok = ping_session(session, timeout=15)
                    print(f"[colab_runner] Post-restart ping status: {ping_ok}", flush=True)
                    time.sleep(5)
                    continue
                else:
                    print(f"[COLAB AUTO-CHECK] FATAL: Exhausted all {max_retries} retries.", flush=True)
                    return res

            if res.returncode == 0:
                return res

            # Non-zero returncode without connection drop
            last_res = res
            if attempt < max_retries:
                time.sleep(3)
                continue
            return res

        except subprocess.TimeoutExpired:
            print(
                f"\n[COLAB AUTO-CHECK] Subprocess timeout expired after {timeout + 30}s (attempt {attempt}/{max_retries})!",
                flush=True,
            )
            if attempt < max_retries:
                restart_kernel(session)
                ensure_session(session, gpu=gpu)
                time.sleep(5)
                continue
            raise RuntimeError(
                f"colab exec timed out after {timeout + 30}s on session '{session}' (all {max_retries} attempts)"
            ) from None

    return last_res


def run_remote_py(
    session: str,
    code: str,
    *,
    env: Optional[List[str]] = None,
    timeout: int = 120,
    gpu: str = DEFAULT_GPU,
) -> str:
    """Execute an inline Python snippet on Colab via a temporary script file."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        tf_path = Path(tf.name)
    try:
        res = run_command(
            str(tf_path),
            session=session,
            is_file=True,
            timeout=timeout,
            max_retries=2,
            env=env,
            stream=True,
            gpu=gpu,
        )
        if res.returncode != 0:
            raise RuntimeError(f"colab_runner.run_command failed ({res.returncode}): {res.stderr or res.stdout}")
        return res.stdout or ""
    finally:
        tf_path.unlink(missing_ok=True)


def run_referee(
    loop_id: int,
    hypothesis: str,
    session: str = DEFAULT_SESSION,
    max_samples: int = 150,
    adapter_path: Optional[str] = None,
    baseline: bool = False,
) -> int:
    """Executes Phase 3 and Phase 4 with hardcoded packaging, upload, evaluation, and logging."""
    print(f"\n=== [COLAB RUNNER] Starting Automated Dual-Referee for Loop #{loop_id:03d} ===", flush=True)
    raw_log_path = SCRATCH / "temp_referee_raw.log"
    with open(raw_log_path, "w", encoding="utf-8") as raw_f:
        raw_f.write(f"=== REFEREE RUN LOG FOR LOOP {loop_id} ===\nHypothesis: {hypothesis}\n\n")

    # 1. Package Codebase
    pkg_zip = SCRATCH / "code_latest.zip"
    package_codebase(pkg_zip)

    # 2. Upload and Unzip on Colab
    print(f"[colab_runner] Uploading codebase to Colab session '{session}'...", flush=True)
    up_res = subprocess.run(
        ["colab", "upload", "-s", session, str(pkg_zip), "/content/code_latest.zip"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if up_res.returncode != 0:
        print(f"[colab_runner] Error uploading zip: {up_res.stderr}", flush=True)
        return up_res.returncode

    unzip_script = SCRATCH / "unzip_code.py"
    unzip_script.write_text(
        "import os, zipfile, shutil\n"
        "from pathlib import Path\n"
        "os.makedirs('/content/ZALO_AIC', exist_ok=True)\n"
        "with zipfile.ZipFile('/content/code_latest.zip', 'r') as z:\n"
        "    z.extractall('/content/ZALO_AIC')\n"
        "eval_data = Path('/content/eval_data')\n"
        "data_target = Path('/content/ZALO_AIC/data')\n"
        "if eval_data.exists() and not data_target.exists():\n"
        "    try:\n"
        "        os.symlink(eval_data / 'data', data_target)\n"
        "        print('Symlinked /content/eval_data/data -> /content/ZALO_AIC/data', flush=True)\n"
        "    except Exception:\n"
        "        shutil.copytree(eval_data / 'data', data_target)\n"
        "        print('Copied /content/eval_data/data -> /content/ZALO_AIC/data', flush=True)\n"
        "print('Unpacked successfully to /content/ZALO_AIC', flush=True)\n",
        encoding="utf-8",
    )
    run_command(str(unzip_script), session=session, is_file=True, timeout=60)

    # Clean local scratch result files so stale runs cannot leak
    local_train_json = SCRATCH / "temp_train_res.json"
    local_dev_json = SCRATCH / "temp_dev_res.json"
    if local_train_json.exists():
        local_train_json.unlink()
    if local_dev_json.exists():
        local_dev_json.unlink()

    # 3. Execute Referee 1 (Glass-Box Train) on Colab
    print(f"[colab_runner] Executing Referee 1 (Train Glass-Box, {max_samples} samples) on Colab...", flush=True)
    adapter_arg = f", '--adapter-path', '{adapter_path}'" if adapter_path else ""
    eval_train_script = SCRATCH / "eval_train_runner.py"
    eval_train_script.write_text(
        "import os, sys, subprocess\n"
        "os.chdir('/content/ZALO_AIC')\n"
        "sys.path.insert(0, '/content/ZALO_AIC')\n"
        "if os.path.exists('scripts/scratch/temp_train_res.json'):\n"
        "    os.remove('scripts/scratch/temp_train_res.json')\n"
        "cmd = ['python', 'src/eval/train_asr_evaluator.py', '--max-samples', str(" + str(max_samples) + "), '--output', 'scripts/scratch/temp_train_res.json'" + adapter_arg + "]\n"
        "res = subprocess.run(cmd, capture_output=True, text=True)\n"
        "sys.stdout.write(res.stdout)\n"
        "sys.stderr.write(res.stderr)\n"
        "if res.returncode != 0:\n"
        "    raise RuntimeError(f'Referee 1 failed with exit code {res.returncode}')\n",
        encoding="utf-8",
    )

    t0 = time.time()
    res1 = run_command(str(eval_train_script), session=session, is_file=True, timeout=300)
    with open(raw_log_path, "a", encoding="utf-8") as raw_f:
        raw_f.write(f"\n--- REFEREE 1 (TRAIN) ---\nReturn code: {res1.returncode}\n{res1.stdout}\n{res1.stderr}\n")

    if res1.returncode != 0 or "RuntimeError: Referee 1 failed" in (res1.stdout or "") or "RuntimeError: Referee 1 failed" in (res1.stderr or ""):
        print(f"[colab_runner] Referee 1 failed: {res1.stderr or res1.stdout}", flush=True)
        return 1

    # Download temp_train_res.json
    subprocess.run(
        ["colab", "download", "-s", session, "/content/ZALO_AIC/scripts/scratch/temp_train_res.json", str(local_train_json)],
        capture_output=True,
        timeout=120,
    )
    if not local_train_json.exists():
        print(f"[colab_runner] Referee 1 output not found at {local_train_json}", flush=True)
        return 1

    # 4. Execute Referee 2 (Sealed Blind Dev) on Colab
    print(f"[colab_runner] Executing Referee 2 (Sealed Blind Dev, 299 samples) on Colab...", flush=True)
    eval_dev_script = SCRATCH / "eval_dev_runner.py"
    eval_dev_script.write_text(
        "import os, sys, subprocess\n"
        "os.chdir('/content/ZALO_AIC')\n"
        "sys.path.insert(0, '/content/ZALO_AIC')\n"
        "if os.path.exists('scripts/scratch/temp_dev_res.json'):\n"
        "    os.remove('scripts/scratch/temp_dev_res.json')\n"
        "cmd = ['python', 'src/eval/blind_dev_evaluator.py', '--output', 'scripts/scratch/temp_dev_res.json'" + adapter_arg + "]\n"
        "res = subprocess.run(cmd, capture_output=True, text=True)\n"
        "sys.stdout.write(res.stdout)\n"
        "sys.stderr.write(res.stderr)\n"
        "if res.returncode != 0:\n"
        "    raise RuntimeError(f'Referee 2 failed with exit code {res.returncode}')\n",
        encoding="utf-8",
    )

    res2 = run_command(str(eval_dev_script), session=session, is_file=True, timeout=300)
    with open(raw_log_path, "a", encoding="utf-8") as raw_f:
        raw_f.write(f"\n--- REFEREE 2 (DEV) ---\nReturn code: {res2.returncode}\n{res2.stdout}\n{res2.stderr}\n")

    if res2.returncode != 0 or "RuntimeError: Referee 2 failed" in (res2.stdout or "") or "RuntimeError: Referee 2 failed" in (res2.stderr or ""):
        print(f"[colab_runner] Referee 2 failed: {res2.stderr or res2.stdout}", flush=True)
        return 1

    # Download temp_dev_res.json
    subprocess.run(
        ["colab", "download", "-s", session, "/content/ZALO_AIC/scripts/scratch/temp_dev_res.json", str(local_dev_json)],
        capture_output=True,
        timeout=120,
    )
    if not local_dev_json.exists():
        print(f"[colab_runner] Referee 2 output not found at {local_dev_json}", flush=True)
        return 1

    # 5. Finalize via loop_logger.py
    print(f"[colab_runner] Finalizing loop #{loop_id} telemetry via loop_logger.py...", flush=True)
    fin_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "loop_logger.py"),
        "finalize",
        "--loop", str(loop_id),
        "--hypothesis", hypothesis,
        "--train-json", str(local_train_json),
        "--dev-json", str(local_dev_json),
        "--raw-log", str(raw_log_path),
    ]
    if baseline:
        fin_cmd.append("--baseline")

    fin_res = subprocess.run(fin_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(fin_res.stdout)
    if fin_res.stderr:
        print(fin_res.stderr, file=sys.stderr)
    return fin_res.returncode


def scan_log(log_path_or_text: str) -> int:
    """Standalone CLI utility to scan a log file or string for lost connection."""
    p = Path(log_path_or_text)
    if p.exists() and p.is_file():
        content = p.read_text(encoding="utf-8", errors="replace")
    else:
        content = log_path_or_text

    is_lost, sig = check_lost_connection(content)
    if is_lost:
        print(f"[COLAB AUTO-CHECK] SCAN RESULT: FOUND connection lost signature: '{sig}'")
        return 1
    else:
        print("[COLAB AUTO-CHECK] SCAN RESULT: CLEAN (No lost connection detected)")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-healing Colab runner with hardcoded lost connection auto-check")
    sub = parser.add_subparsers(dest="cmd")

    p_ping = sub.add_parser("ping", help="Ping Colab session kernel")
    p_ping.add_argument("-s", "--session", default=DEFAULT_SESSION)

    p_restart = sub.add_parser("restart-kernel", help="Restart Colab session kernel")
    p_restart.add_argument("-s", "--session", default=DEFAULT_SESSION)

    p_pkg = sub.add_parser("package", help="Package codebase to zip")
    p_pkg.add_argument("-o", "--output", default=str(SCRATCH / "code_latest.zip"))

    p_scan = sub.add_parser("scan-log", help="Scan log file for lost connection signatures")
    p_scan.add_argument("path", help="Path to log file to scan")

    p_exec = sub.add_parser("exec", help="Execute script on Colab with auto-check & auto-retry")
    p_exec.add_argument("script", help="Script file path to execute on Colab")
    p_exec.add_argument("-s", "--session", default=DEFAULT_SESSION)
    p_exec.add_argument("-t", "--timeout", type=int, default=300)
    p_exec.add_argument("-r", "--retries", type=int, default=3)

    p_ref = sub.add_parser("referee", help="Automated Dual-Referee evaluation pipeline on Colab")
    p_ref.add_argument("--loop", type=int, required=True, help="Loop ID number")
    p_ref.add_argument("--hypothesis", type=str, required=True, help="Loop hypothesis string")
    p_ref.add_argument("-s", "--session", default=DEFAULT_SESSION)
    p_ref.add_argument("--max-samples", type=int, default=150)
    p_ref.add_argument("--adapter-path", type=str, default=None)
    p_ref.add_argument("--baseline", action="store_true")

    args = parser.parse_args()

    if args.cmd == "ping":
        ok = ping_session(args.session)
        print(f"Session {args.session} alive: {ok}")
        sys.exit(0 if ok else 1)
    elif args.cmd == "restart-kernel":
        ok = restart_kernel(args.session)
        print(f"Kernel restarted: {ok}")
        sys.exit(0 if ok else 1)
    elif args.cmd == "package":
        package_codebase(Path(args.output))
    elif args.cmd == "scan-log":
        sys.exit(scan_log(args.path))
    elif args.cmd == "exec":
        res = run_command(args.script, session=args.session, is_file=True, timeout=args.timeout, max_retries=args.retries)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)
    elif args.cmd == "referee":
        code = run_referee(
            loop_id=args.loop,
            hypothesis=args.hypothesis,
            session=args.session,
            max_samples=args.max_samples,
            adapter_path=args.adapter_path,
            baseline=args.baseline,
        )
        sys.exit(code)
    else:
        parser.print_help()
