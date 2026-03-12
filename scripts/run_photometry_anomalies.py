#!/usr/bin/env python3
"""
TEP-JWST: Photometry Anomalies Pipeline

Runs analyses related to dust, UV slopes, mass-to-light ratios, and
SFR-age consistency. Targeted at observers focused on SED fitting and
photometric properties.

Usage:
    python scripts/run_photometry_anomalies.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.pipeline_runner import run_pipeline

PHOTOMETRY_STEPS = [
    # ── Data foundation ───────────────────────────────────────────────────
    "step_001_uncover_load.py",           # UNCOVER DR4 load and quality cuts
    "step_002_tep_model.py",              # Γ_t computation

    # ── Dust and SED analysis ─────────────────────────────────────────────
    "step_006_thread5_z8_dust.py",        # z > 8 dust anomaly (ρ = +0.59)
    "step_014_jwst_uv_slope.py",          # UV slope β analysis
    "step_017_ml_ratio.py",               # Mass-to-light ratio
    "step_030_z8_dust_prediction.py",     # Quantitative dust predictions
    "step_129_dust_models.py",            # Alternative dust physics models

    # ── SFR and age consistency ───────────────────────────────────────────
    "step_045_sfr_age_consistency.py",    # SFR-age consistency
    "step_044_metallicity_age_decoupling.py", # Metallicity-age decoupling

    # ── Multi-survey replication ──────────────────────────────────────────
    "step_032_ceers_replication.py",      # CEERS independent replication
    "step_034_cosmosweb_replication.py",  # COSMOS-Web replication
    "step_153_cosmos2025_sed_analysis.py", # COSMOS2025 LePHARE SED
    "step_157_cosmos2025_ssfr_inversion.py", # COSMOS2025 sSFR inversion

    # ── Summary ───────────────────────────────────────────────────────────
    "step_008_summary.py",
]


def main():
    results = run_pipeline(
        pipeline_name="PHOTOMETRY ANOMALIES",
        steps=PHOTOMETRY_STEPS,
        description="Dust, UV slopes, M/L ratios, and SFR-age consistency",
    )
    n_fail = sum(1 for r in results if r["status"] != "PASS")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
