#!/usr/bin/env python3
"""
TEP-JWST Step 130: Cross-survey systematic error budget (CEERS + UNCOVER + COSMOS-Web)

Cross-survey systematic error budget (CEERS + UNCOVER + COSMOS-Web)


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)

STEP_NUM  = "130"  # Pipeline step number (sequential 001-176)
STEP_NAME = "cross_survey_systematics"  # Cross-survey systematic error budget: Cochran's Q test for heterogeneity across UNCOVER/CEERS/COSMOS-Web surveys

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH   = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

import json as _json
import numpy as np
from scipy.stats import chi2  # Chi-squared distribution for heterogeneity test (Cochran's Q)

OUTPUTS = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (alias)

# Known rho values from individual survey steps
SURVEY_RESULTS = [
    ("UNCOVER",     "step_020_uncover_partial_correlation.json", "rho",      "p"),
    ("CEERS",       "step_032_ceers_replication.json",           "mass_dust.rho", "mass_dust.p"),
    ("COSMOS-Web",  "step_034_cosmosweb_replication.json",       "mass_dust.rho", "mass_dust.p"),
]


def _get_nested(d, key):
    """Get nested key like 'mass_dust.rho' from dict."""
    parts = key.split(".")
    val = d
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return None
    return float(val) if val is not None else None


def run():
    print_status(f"STEP {STEP_NUM}: Cross-survey systematic error budget", "INFO")

    survey_data = []
    for name, fname, rho_key, p_key in SURVEY_RESULTS:
        fpath = OUTPUTS / fname
        if not fpath.exists():
            logger.warning(f"  Missing: {fname}")
            continue
        try:
            d = _json.loads(fpath.read_text())
            rho = _get_nested(d, rho_key)
            p   = _get_nested(d, p_key)
            n_key = "n" if "n" in d else "ceers_n" if "ceers_n" in d else "cosmosweb_n" if "cosmosweb_n" in d else None
            n = d.get(n_key, None) if n_key else None
            if n is None:
                n = _get_nested(d, "mass_dust.n")
            if rho is not None:
                survey_data.append({"survey": name, "rho": rho, "p": p, "n": n})
                logger.info(f"  {name}: rho={rho:.3f}, p={p:.2e}, N={n}")
        except Exception as e:
            logger.warning(f"  Could not read {fname}: {e}")

    if not survey_data:
        result = {
            "step": STEP_NUM, "name": STEP_NAME,
            "status": "SKIPPED_NO_DATA",
            "note": "No survey step outputs found.",
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as f:
            _json.dump(result, f, indent=2)
        return result

    rhos = np.array([s["rho"] for s in survey_data], dtype=float)
    rhos = rhos[np.isfinite(rhos)]

    # Cross-survey statistics
    rho_mean  = float(np.mean(rhos))
    rho_std   = float(np.std(rhos))
    rho_min   = float(np.min(rhos))
    rho_max   = float(np.max(rhos))
    rho_range = float(rho_max - rho_min)

    # Systematic floor: inter-survey scatter as fraction of signal
    sys_frac = float(rho_std / rho_mean) if rho_mean != 0 else float("nan")

    # Cochran's Q heterogeneity test (if we have Ns)
    q_stat, q_p = float("nan"), float("nan")
    ns = [s["n"] for s in survey_data if s["n"] is not None]
    if len(survey_data) >= 2 and len(ns) == len(survey_data):
        weights = np.array([float(n) for n in ns])
        rho_wt  = float(np.average(rhos, weights=weights))
        Q = float(np.sum(weights * (rhos - rho_wt) ** 2))
        df_q = len(rhos) - 1
        q_p  = float(1 - chi2.cdf(Q, df_q))
        q_stat = Q
        logger.info(f"  Cochran's Q = {Q:.3f}, p = {q_p:.3f} (df={df_q})")

    logger.info(
        f"  Cross-survey rho: mean={rho_mean:.3f} +/-{rho_std:.3f} "
        f"(range {rho_range:.3f}), systematic floor={sys_frac:.1%}"
    )

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "Cross-survey systematic error budget",
        "surveys":    survey_data,
        "rho_mean":   rho_mean,
        "rho_std":    rho_std,
        "rho_min":    rho_min,
        "rho_max":    rho_max,
        "rho_range":  rho_range,
        "systematic_frac": sys_frac,
        "cochran_Q":  q_stat,
        "cochran_p":  q_p,
        "conclusion": (
            f"Cross-survey rho scatter: {rho_std:.3f} ({sys_frac:.1%} of signal). "
            f"Cochran Q-test: p={q_p:.3f} (consistent={'yes' if q_p > 0.05 else 'borderline'}). "
            f"Systematic floor < {rho_std:.3f} dex."
        ) if np.isfinite(rho_std) else "Insufficient surveys for comparison.",
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        _json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete. rho scatter={rho_std:.4f}", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
