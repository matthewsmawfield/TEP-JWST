#!/usr/bin/env python3
"""
TEP-JWST: Core Evidence Fast-Path

Runs only the four independent lines of evidence (L1-L4) in ~5 minutes.
This is the recommended entry point for first-time users and reviewers.

Lines of Evidence:
  L1: Dust–Γ_t correlation (z > 8, ρ = +0.62)
  L2: Inside-out core screening (JADES colour gradients)
  L3: Mass–sSFR inversion at z > 7
  L4: Dynamical-mass anomaly resolution in the live RUBIES-like kinematic regime

Usage:
    python scripts/run_core_evidence.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.pipeline_runner import run_pipeline

CORE_EVIDENCE_STEPS = [
    # ── Data foundation ───────────────────────────────────────────────────
    "step_001_uncover_load.py",          # UNCOVER DR4: 2,315 galaxies
    "step_002_tep_model.py",             # Γ_t computation for all sources

    # ── L1: Dust–Γ_t correlation (primary) ───────────────────────────────
    "step_006_thread5_z8_dust.py",       # z > 8 dust anomaly (ρ = +0.59)
    "step_030_z8_dust_prediction.py",    # Quantitative dust prediction tests

    # ── L2: Inside-out core screening ────────────────────────────────────
    "step_037_resolved_gradients.py",    # Resolved core screening (JADES)
    "step_139_colour_gradient_steiger.py", # Steiger Z-test: t_eff vs M*

    # ── L3: Mass–sSFR inversion ───────────────────────────────────────────
    "step_004_thread1_z7_inversion.py",  # z > 7 inversion (Δρ = +0.25)

    # ── L4: Dynamical mass comparison ─────────────────────────────────────
    "step_117_dynamical_mass_comparison.py", # live regime-level kinematic consistency test
    "step_143_mass_proxy_breaker.py",    # 3-test mass-proxy degeneracy break

    # ── Concordance summary ───────────────────────────────────────────────
    "step_162_l1_l3_independence.py",    # α₀ concordance (5 observables)
]


def main():
    results = run_pipeline(
        pipeline_name="CORE EVIDENCE (L1–L4)",
        steps=CORE_EVIDENCE_STEPS,
        description="Four independent lines of evidence for TEP (~5 minutes)",
    )
    print("\nKey outputs:")
    print("  results/outputs/step_006_*.json  — L1: Dust–Γ_t correlation")
    print("  results/outputs/step_037_*.json  — L2: Core screening")
    print("  results/outputs/step_004_*.json  — L3: sSFR inversion")
    print("  results/outputs/step_117_*.json  — L4: Dynamical masses")
    print("  results/outputs/step_162_*.json  — α₀ concordance")

    n_fail = sum(1 for r in results if r["status"] != "PASS")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
