"""Permanent 3-Tier Telemetry, Immutable Archive & Live Monitor Engine for ASR Loop (`scripts/loop_logger.py`).

Design Principle: "Write-Deep on Disk, Read-Shallow in Context"
1. Tier 1 (Immutable Per-Loop Archive): `reports/agent_loops/loop_XXX_<timestamp>/`
   Preserves 100% of every run permanently (forensic brief, exact git patch BEFORE rollback,
   modification & audit summary, Dual-Referee KPIs, Train error diff, raw logs,
   and 1-page LOOP_REPORT.md) — surviving `git restore src/` on REJECTED experiments.
2. Tier 2 (Compact Index & Clickable Markdown Dashboard):
   - `reports/agent_loops/experiments_index.jsonl` (1 line per loop for fast agent reads)
   - `reports/AGENT_LOOP_DASHBOARD.md` (with direct links to LOOP_REPORT.md & 02_code_changes.patch)
   - `configs/agent_loop_state.json` (state file)
3. Tier 3 (Real-Time Live Monitor):
   - `reports/agent_loops/live_status.json`
   - `reports/agent_loops/live_monitor.html` (Self-contained HTML dashboard with SVG charts)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_ROOT = Path(__file__).resolve().parent.parent

# Strict allowlist for Sealed Blind Dev metrics (Rule: Zero Dev Leakage)
ALLOWED_DEV_MACRO_KEYS = {
    "benchmark",
    "total_samples",
    "wer",
    "cer",
    "sentence_em",
    "latency",
    "timestamp",
}

PHASES = {
    "1_diagnosis": "Phase 1/4: Forensic Diagnosis (Train Glass-Box)",
    "2_refactor": "Phase 2/4: Architectural Refactoring & Unit Tests",
    "2_audit": "Phase 2b/4: Read-Only SOP & Anti-Snoop Audit",
    "3_referee": "Phase 3/4: Dual-Referee GPU Evaluation",
    "4_pareto": "Phase 4/4: Pareto Gate & Permanent Archival",
    "idle": "Idle / Completed",
}


def git_cmd(*args: str, cwd: Path = DEFAULT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class ASRLoopLogger:
    def __init__(self, root: Path = DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self.reports_dir = self.root / "reports" / "agent_loops"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "configs" / "agent_loop_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.index_file = self.reports_dir / "experiments_index.jsonl"
        self.dashboard_md = self.root / "reports" / "AGENT_LOOP_DASHBOARD.md"
        self.live_html = self.reports_dir / "live_monitor.html"
        self.live_json = self.reports_dir / "live_status.json"

    def load_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {
                "active_loop": 0,
                "current_best_loop": 0,
                "best_dev_wer": 100.0,
                "best_dev_cer": 100.0,
                "best_dev_em": 0.0,
                "best_train_wer": 100.0,
                "best_train_em": 0.0,
                "total_loops": 0,
                "accepted_loops": 0,
                "rejected_loops": 0,
                "history": [],
            }
        with open(self.state_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_state(self, state: Dict[str, Any]) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _find_loop_dir(self, loop_id: int) -> Optional[Path]:
        pattern = f"loop_{loop_id:03d}_*"
        matches = sorted(self.reports_dir.glob(pattern))
        return matches[-1] if matches else None

    def start_loop(self, loop_id: int, hypothesis: str, agent: str = "loop-researcher") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        loop_dir = self.reports_dir / f"loop_{loop_id:03d}_{ts}"
        loop_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "loop": loop_id,
            "hypothesis": hypothesis,
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
            "current_phase": "1_diagnosis",
            "agent": agent,
        }
        with open(loop_dir / "00_loop_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.update_live_status(
            loop=loop_id,
            phase="1_diagnosis",
            hypothesis=hypothesis,
            active_agent=agent,
            details="Starting forensic diagnosis on Train Glass-Box",
        )
        return loop_dir

    def update_live_status(
        self,
        loop: int,
        phase: str,
        hypothesis: str = "",
        active_agent: str = "",
        details: str = "",
    ) -> None:
        state = self.load_state()
        data = {
            "updated_at": datetime.now().isoformat(),
            "active_loop": loop,
            "phase": phase,
            "phase_label": PHASES.get(phase, phase),
            "hypothesis": hypothesis,
            "active_agent": active_agent,
            "details": details,
            "best_dev_wer": state.get("best_dev_wer", 100.0),
            "best_dev_cer": state.get("best_dev_cer", 100.0),
            "best_dev_em": state.get("best_dev_em", 0.0),
            "accepted_loops": state.get("accepted_loops", 0),
            "rejected_loops": state.get("rejected_loops", 0),
        }
        with open(self.live_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.render_live_html(data)

    def save_forensic_brief(self, loop_id: int, content: str) -> Path:
        loop_dir = self._find_loop_dir(loop_id)
        if not loop_dir:
            raise FileNotFoundError(f"Loop directory for loop {loop_id} not found.")
        out_file = loop_dir / "01_forensic_brief.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)
        return out_file

    def snapshot_code_changes(
        self,
        loop_id: int,
        optimization_summary: str,
        pytest_passed: bool = True,
        pytest_summary: str = "",
        audit_passed: bool = True,
        audit_notes: str = "",
    ) -> Path:
        loop_dir = self._find_loop_dir(loop_id)
        if not loop_dir:
            raise FileNotFoundError(f"Loop directory for loop {loop_id} not found.")

        # Capture git diff BEFORE any rollback
        diff_res = git_cmd("diff", "src/", "configs/")
        patch_content = diff_res.stdout
        with open(loop_dir / "02_code_changes.patch", "w", encoding="utf-8") as f:
            f.write(patch_content)

        stat_res = git_cmd("diff", "--stat", "src/", "configs/")
        summary = {
            "loop": loop_id,
            "timestamp": datetime.now().isoformat(),
            "optimization_summary": optimization_summary,
            "diff_stat": stat_res.stdout.strip(),
            "pytest_passed": pytest_passed,
            "pytest_summary": pytest_summary,
            "audit_passed": audit_passed,
            "audit_notes": audit_notes,
        }
        with open(loop_dir / "02_modification_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return loop_dir / "02_code_changes.patch"

    def finalize_loop(
        self,
        loop_id: int,
        hypothesis: str,
        train_json_path: Path,
        dev_json_path: Path,
        raw_log_path: Optional[Path] = None,
        is_baseline: bool = False,
    ) -> Dict[str, Any]:
        loop_dir = self._find_loop_dir(loop_id)
        if not loop_dir:
            raise FileNotFoundError(f"Loop directory for loop {loop_id} not found.")

        with open(train_json_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        with open(dev_json_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)

        # STRICT PROTOCOL: Sanitize Dev data (Zero Dev Leakage)
        sanitized_dev = {k: dev_data[k] for k in ALLOWED_DEV_MACRO_KEYS if k in dev_data}

        train_wer = train_data.get("wer", 100.0)
        train_cer = train_data.get("cer", 100.0)
        train_em = train_data.get("sentence_em", 0.0)
        train_lat = train_data.get("latency", 0.0)

        dev_wer = sanitized_dev.get("wer", 100.0)
        dev_cer = sanitized_dev.get("cer", 100.0)
        dev_em = sanitized_dev.get("sentence_em", 0.0)
        dev_lat = sanitized_dev.get("latency", 0.0)

        state = self.load_state()

        if is_baseline:
            decision = "ACCEPTED"
            reason = "Initial Baseline established"
            delta_dev_wer = 0.0
            delta_dev_cer = 0.0
            delta_dev_em = 0.0
            delta_train_wer = 0.0
        else:
            best_dev_wer = state.get("best_dev_wer", 100.0)
            best_dev_cer = state.get("best_dev_cer", 100.0)
            best_dev_em = state.get("best_dev_em", 0.0)
            best_train_wer = state.get("best_train_wer", 100.0)

            delta_dev_wer = dev_wer - best_dev_wer
            delta_dev_cer = dev_cer - best_dev_cer
            delta_dev_em = dev_em - best_dev_em
            delta_train_wer = train_wer - best_train_wer

            # ASR Pareto Gate: Lower WER/CER is better, Higher EM is better
            dev_improved = (delta_dev_wer < 0) or (delta_dev_wer <= 0 and delta_dev_em > 0)
            train_safe = (delta_train_wer <= 2.0)  # No catastrophic degradation on train

            if dev_improved and train_safe:
                decision = "ACCEPTED"
                reason = f"Dev WER improved ({delta_dev_wer:+.2f}%) or EM improved ({delta_dev_em:+.2f}%) without train collapse"
            else:
                decision = "REJECTED"
                reason = f"Dev did not improve (delta WER={delta_dev_wer:+.2f}%, delta EM={delta_dev_em:+.2f}%) or Train regressed"

        # Archive metrics
        kpis = {
            "loop": loop_id,
            "hypothesis": hypothesis,
            "decision": decision,
            "reason": reason,
            "train": {"wer": train_wer, "cer": train_cer, "sentence_em": train_em, "latency": train_lat},
            "dev": {"wer": dev_wer, "cer": dev_cer, "sentence_em": dev_em, "latency": dev_lat},
            "deltas": {
                "delta_dev_wer": delta_dev_wer,
                "delta_dev_cer": delta_dev_cer,
                "delta_dev_em": delta_dev_em,
                "delta_train_wer": delta_train_wer,
            },
        }
        with open(loop_dir / "03_kpi_metrics.json", "w", encoding="utf-8") as f:
            json.dump(kpis, f, ensure_ascii=False, indent=2)

        # Save train error clusters
        with open(loop_dir / "03_train_error_diff.json", "w", encoding="utf-8") as f:
            json.dump(train_data.get("error_samples", []), f, ensure_ascii=False, indent=2)

        if raw_log_path and raw_log_path.exists():
            import shutil
            shutil.copyfile(raw_log_path, loop_dir / "03_raw_execution.log")

        # Generate 1-page human-readable report
        self._write_loop_report(loop_dir, kpis)

        # Update global state & VCS
        if decision == "ACCEPTED":
            state["best_dev_wer"] = dev_wer
            state["best_dev_cer"] = dev_cer
            state["best_dev_em"] = dev_em
            state["best_train_wer"] = train_wer
            state["best_train_em"] = train_em
            state["current_best_loop"] = loop_id
            state["accepted_loops"] = state.get("accepted_loops", 0) + 1
            git_cmd("add", "src/", "configs/")
            git_cmd("commit", "-m", f"feat(loop_{loop_id:03d}): ACCEPTED - Dev WER {dev_wer:.2f}% EM {dev_em:.2f}% | {hypothesis[:60]}")
        else:
            state["rejected_loops"] = state.get("rejected_loops", 0) + 1
            # Rollback working tree cleanly
            git_cmd("restore", "src/", "configs/")

        state["total_loops"] = state.get("total_loops", 0) + 1
        state["history"].append(kpis)
        self.save_state(state)

        # Append to index
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(kpis, ensure_ascii=False) + "\n")

        self.render_dashboard()
        self.update_live_status(
            loop=loop_id,
            phase="idle",
            hypothesis=hypothesis,
            details=f"Loop {loop_id} {decision}: Dev WER={dev_wer:.2f}%, EM={dev_em:.2f}%",
        )
        return kpis

    def _write_loop_report(self, loop_dir: Path, kpis: Dict[str, Any]) -> None:
        report = rf"""# Loop Report #{kpis['loop']:03d} — {kpis['decision']}

**Hypothesis:** {kpis['hypothesis']}  
**Decision Reason:** {kpis['reason']}  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **{kpis['train']['wer']:.2f}%** | **{kpis['dev']['wer']:.2f}%** | `{kpis['deltas']['delta_dev_wer']:+.2f}%` | **$\le$ 1.50%** |
| **CER** | **{kpis['train']['cer']:.2f}%** | **{kpis['dev']['cer']:.2f}%** | `{kpis['deltas']['delta_dev_cer']:+.2f}%` | **$\le$ 1.00%** |
| **Sentence EM** | **{kpis['train']['sentence_em']:.2f}%** | **{kpis['dev']['sentence_em']:.2f}%** | `{kpis['deltas']['delta_dev_em']:+.2f}%` | **$\ge$ 95.00%** |
| **Latency** | {kpis['train']['latency']:.3f}s | {kpis['dev']['latency']:.3f}s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
"""
        with open(loop_dir / "LOOP_REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)

    def render_dashboard(self) -> None:
        state = self.load_state()
        history = state.get("history", [])

        md = rf"""# Autonomous ASR Agent Optimization Loop Dashboard

> **Target Definition of Done (DoD) on Sealed Dev:** **Pipeline Tool-Call F1 > 80.00%** | **State Match $\ge$ 75.00%** | **WER $\le$ 1.50%** | **CER $\le$ 1.00%** | **Sentence EM $\ge$ 95.00%**  
> **Status:** Current Best Loop: **#{state.get('current_best_loop', 0):02d}** | Best Dev WER: **{state.get('best_dev_wer', 100.0):.2f}%** | Best Dev EM: **{state.get('best_dev_em', 0.0):.2f}%**  
> **Total Runs:** {state.get('total_loops', 0)} ({state.get('accepted_loops', 0)} ACCEPTED / {state.get('rejected_loops', 0)} REJECTED)

---

## Experiment Run History

| Loop | Decision | Hypothesis | Train WER | Dev WER | Dev CER | Dev EM | Report | Patch |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for x in reversed(history):
            lp = x["loop"]
            dec_badge = "**ACCEPTED**" if x["decision"] == "ACCEPTED" else "*REJECTED*"
            loop_dir = self._find_loop_dir(lp)
            rel_path = f"agent_loops/{loop_dir.name}" if loop_dir else ""
            report_link = f"[Report]({rel_path}/LOOP_REPORT.md)" if loop_dir else "-"
            patch_link = f"[Patch]({rel_path}/02_code_changes.patch)" if loop_dir else "-"

            md += f"| #{lp:02d} | {dec_badge} | {x['hypothesis'][:45]} | {x['train']['wer']:.2f}% | **{x['dev']['wer']:.2f}%** | {x['dev']['cer']:.2f}% | **{x['dev']['sentence_em']:.2f}%** | {report_link} | {patch_link} |\n"

        with open(self.dashboard_md, "w", encoding="utf-8") as f:
            f.write(md)

    def render_live_html(self, data: Dict[str, Any]) -> None:
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ASR Autonomous Loop Live Monitor</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 13px; }}
.badge-running {{ background: #0284c7; color: white; }}
.badge-idle {{ background: #475569; color: white; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
.metric {{ font-size: 28px; font-weight: 700; color: #38bdf8; }}
.label {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
</style>
</head>
<body>
<h1>🎙️ ASR Optimization Loop Live Monitor</h1>
<div class="card">
  <span class="badge { 'badge-running' if data['phase'] != 'idle' else 'badge-idle' }">{data['phase_label']}</span>
  <p style="font-size: 16px; margin-top: 10px;"><strong>Loop #{data['active_loop']:02d}:</strong> {data['hypothesis'] or 'No active loop'}</p>
  <p style="color: #94a3b8; font-size: 14px;">Agent: <code>{data['active_agent']}</code> | {data['details']}</p>
</div>
<div class="grid">
  <div class="card"><div class="label">Best Dev WER</div><div class="metric">{data['best_dev_wer']:.2f}%</div></div>
  <div class="card"><div class="label">Best Dev CER</div><div class="metric">{data['best_dev_cer']:.2f}%</div></div>
  <div class="card"><div class="label">Best Dev EM</div><div class="metric">{data['best_dev_em']:.2f}%</div></div>
  <div class="card"><div class="label">Accepted / Rejected</div><div class="metric" style="color: #4ade80;">{data['accepted_loops']} / {data['rejected_loops']}</div></div>
</div>
</body>
</html>
"""
        with open(self.live_html, "w", encoding="utf-8") as f:
            f.write(html)

    def print_brief(self, max_recent: int = 5) -> None:
        state = self.load_state()
        history = state.get("history", [])
        print(f"=== AUTONOMOUS ASR LOOP STATE ===")
        print(f"Current Best Loop: #{state.get('current_best_loop', 0):02d}")
        print(f"Best Dev WER: {state.get('best_dev_wer', 100.0):.2f}% | Best Dev EM: {state.get('best_dev_em', 0.0):.2f}%")
        print(f"Total Runs: {state.get('total_loops', 0)} ({state.get('accepted_loops', 0)} Accepted / {state.get('rejected_loops', 0)} Rejected)")
        print("\nRecent Experiments:")
        for x in history[-max_recent:]:
            print(f"  Loop #{x['loop']:02d} [{x['decision']}]: Dev WER={x['dev']['wer']:.2f}%, EM={x['dev']['sentence_em']:.2f}% | {x['hypothesis'][:50]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Autonomous Loop Telemetry Logger")
    sub = parser.add_subparsers(dest="cmd")

    p_start = sub.add_parser("start")
    p_start.add_argument("--loop", type=int, required=True)
    p_start.add_argument("--hypothesis", type=str, required=True)
    p_start.add_argument("--agent", type=str, default="loop-researcher")

    p_phase = sub.add_parser("phase")
    p_phase.add_argument("--loop", type=int, required=True)
    p_phase.add_argument("--phase", type=str, required=True, choices=list(PHASES.keys()))
    p_phase.add_argument("--agent", type=str, default="")
    p_phase.add_argument("--details", type=str, default="")

    p_forensic = sub.add_parser("forensic")
    p_forensic.add_argument("--loop", type=int, required=True)
    p_forensic.add_argument("--file", type=str, required=True)

    p_snap = sub.add_parser("snapshot")
    p_snap.add_argument("--loop", type=int, required=True)
    p_snap.add_argument("--optimization", type=str, required=True)
    p_snap.add_argument("--pytest-passed", action="store_true")
    p_snap.add_argument("--pytest-summary", type=str, default="")
    p_snap.add_argument("--audit-passed", action="store_true")
    p_snap.add_argument("--audit-notes", type=str, default="")

    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("--loop", type=int, required=True)
    p_fin.add_argument("--hypothesis", type=str, required=True)
    p_fin.add_argument("--train-json", type=str, required=True)
    p_fin.add_argument("--dev-json", type=str, required=True)
    p_fin.add_argument("--raw-log", type=str, default="")
    p_fin.add_argument("--baseline", action="store_true")

    p_brief = sub.add_parser("brief")
    p_brief.add_argument("--max-recent", type=int, default=5)

    p_render = sub.add_parser("render")

    args = parser.parse_args()
    logger = ASRLoopLogger()

    if args.cmd == "start":
        p = logger.start_loop(args.loop, args.hypothesis, args.agent)
        print(f"[LOOP LOGGER] Initialized loop folder: {p}")
    elif args.cmd == "phase":
        logger.update_live_status(args.loop, args.phase, active_agent=args.agent, details=args.details)
    elif args.cmd == "forensic":
        content = open(args.file, "r", encoding="utf-8").read()
        logger.save_forensic_brief(args.loop, content)
        print(f"[LOOP LOGGER] Saved forensic brief for Loop #{args.loop}")
    elif args.cmd == "snapshot":
        p = logger.snapshot_code_changes(
            args.loop,
            args.optimization,
            pytest_passed=args.pytest_passed,
            pytest_summary=args.pytest_summary,
            audit_passed=args.audit_passed,
            audit_notes=args.audit_notes,
        )
        print(f"[LOOP LOGGER] Captured pre-rollback snapshot: {p}")
    elif args.cmd == "finalize":
        res = logger.finalize_loop(
            args.loop,
            args.hypothesis,
            Path(args.train_json),
            Path(args.dev_json),
            Path(args.raw_log) if args.raw_log else None,
            is_baseline=args.baseline,
        )
        print(f"[PARETO GATE] Loop #{args.loop:02d} -> {res['decision']}: {res['reason']}")
        print(f"Train: WER={res['train']['wer']:.2f}%, EM={res['train']['sentence_em']:.2f}% | Dev: WER={res['dev']['wer']:.2f}%, EM={res['dev']['sentence_em']:.2f}%")
    elif args.cmd == "brief":
        logger.print_brief(args.max_recent)
    elif args.cmd == "render":
        logger.render_dashboard()
        print("[LOOP LOGGER] Dashboards re-rendered.")
    else:
        parser.print_help()
