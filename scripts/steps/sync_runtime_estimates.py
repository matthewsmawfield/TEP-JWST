#!/usr/bin/env python3
"""
Sync last-run runtime estimates into the live step-file header comments.

This utility reads `results/outputs/pipeline_summary.json` from the most
recent successful canonical run and stamps each live `scripts/steps/step_*.py`
file with a single header comment immediately after the shebang.

Usage:
    python scripts/steps/sync_runtime_estimates.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEPS_DIR = PROJECT_ROOT / "scripts" / "steps"
SUMMARY_PATH = PROJECT_ROOT / "results" / "outputs" / "pipeline_summary.json"
HEADER_PREFIX = "# Estimated runtime from last full canonical run"


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 0.1:
        return "<0.1s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def _format_run_at_label(run_at: str | None) -> str:
    if not run_at:
        return "unknown UTC"
    try:
        dt = datetime.datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    except ValueError:
        return run_at
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _load_summary() -> tuple[str, str, dict[str, float]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing pipeline summary: {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError(
            "pipeline_summary.json does not record a PASS full run; rerun the canonical pipeline first"
        )

    run_at = _format_run_at_label(summary.get("run_at"))
    total_elapsed_s = summary.get("total_elapsed_s")
    if not isinstance(total_elapsed_s, (int, float)):
        raise RuntimeError("pipeline_summary.json is missing numeric total_elapsed_s")

    step_elapsed_s: dict[str, float] = {}
    for item in summary.get("steps", []):
        name = item.get("name")
        elapsed_s = item.get("elapsed_s")
        if isinstance(name, str) and isinstance(elapsed_s, (int, float)):
            step_elapsed_s[name] = float(elapsed_s)

    if not step_elapsed_s:
        raise RuntimeError("pipeline_summary.json does not contain per-step elapsed times")

    return run_at, _fmt_elapsed(float(total_elapsed_s)), step_elapsed_s


def _sync_step_header(step_path: Path, run_at_label: str, elapsed_label: str, total_label: str) -> bool:
    lines = step_path.read_text().splitlines(keepends=True)
    header_line = (
        f"{HEADER_PREFIX} ({run_at_label}; full pipeline {total_label}): {elapsed_label}.\n"
    )

    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    changed = False

    while insert_at < len(lines) and lines[insert_at].startswith(HEADER_PREFIX):
        if lines[insert_at] != header_line:
            changed = True
        del lines[insert_at]

    if insert_at >= len(lines) or lines[insert_at] != header_line:
        lines.insert(insert_at, header_line)
        changed = True

    if changed:
        step_path.write_text("".join(lines))
    return changed


def main() -> int:
    run_at_label, total_label, step_elapsed_s = _load_summary()
    updated = 0
    missing = []

    for step_name, elapsed_s in sorted(step_elapsed_s.items()):
        step_path = STEPS_DIR / step_name
        if not step_path.exists():
            missing.append(step_name)
            continue
        if _sync_step_header(step_path, run_at_label, _fmt_elapsed(elapsed_s), total_label):
            updated += 1

    print(f"Runtime header sync source: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Last full canonical run: {run_at_label}")
    print(f"Full pipeline runtime: {total_label}")
    print(f"Step files updated: {updated}")
    if missing:
        print("Missing live step files:")
        for name in missing:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
