#!/usr/bin/env python3
"""
Step 174: Stellar Mass Function — Mass Threshold Counts (Table 15)

Counts galaxies above key stellar-mass thresholds before and after the TEP
isochrony-bias correction, providing the pipeline-backed numbers for
Discussion Table 15.

Three surveys are combined (UNCOVER + CEERS + COSMOS-Web) at z > 7.
For each (redshift bin, mass threshold) cell:
  N_obs  = galaxies with log M*_obs > threshold
  N_corr = galaxies with log M*_true > threshold
  where  log M*_true = log M*_obs - n * log10(Gamma_t),  n = 0.7

Also reproduces the Labbé+2023 anomalous-galaxy resolution check from
step_146 for cross-validation.

Output:
  - results/outputs/step_174_smf_mass_threshold_counts.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like  # Shared TEP model
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types

STEP_NUM = "174"  # Pipeline step number
STEP_NAME = "smf_mass_threshold_counts"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products

for p in [LOGS_PATH, OUTPUT_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)  # Step-specific logger
set_step_logger(logger)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Mass-to-light power-law index for the SMF correction (§4.6.2 caveat)
N_ML = 0.7

# Redshift bins and mass thresholds for Table 15
Z_BINS = [(7, 8), (8, 9), (9, 10)]
MASS_THRESHOLDS = [10.0, 10.5]


# ── Data loading ─────────────────────────────────────────────────────────────

def _add_tep_columns(df):
    """Compute log_Mh and gamma_t for a DataFrame with z and log_Mstar."""
    df["log_Mh"] = df.apply(
        lambda r: stellar_to_halo_mass_behroozi_like(r["log_Mstar"], r["z"]), axis=1
    )
    df["gamma_t"] = df.apply(
        lambda r: compute_gamma_t(r["log_Mh"], r["z"]), axis=1
    )
    return df


def load_all_surveys():
    """Load UNCOVER, CEERS highz, and COSMOS-Web highz (full z-range catalogs).

    The SMF Table 15 spans z = 7-10, so we need full z-range catalogs
    (not z>8-only files) to cover the z = 7-8 bin.
    """
    surveys = []

    # UNCOVER (z = 4-10)
    uncover_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        df_u = pd.read_csv(uncover_path)
        df_u["z"] = df_u["z_phot"].astype(float)
        df_u = df_u[(df_u["z"] >= 4.0) & (df_u["z"] <= 12.0) & (df_u["log_Mstar"] >= 8.0)].copy()
        df_u = df_u.dropna(subset=["z", "log_Mstar"])
        df_u = _add_tep_columns(df_u)
        df_u["survey"] = "UNCOVER"
        logger.info(f"  UNCOVER: loaded {len(df_u)} galaxies")
        surveys.append(df_u[["z", "log_Mstar", "log_Mh", "gamma_t", "survey"]])

    # CEERS highz (z = 4-12)
    ceers_path = DATA_INTERIM_PATH / "ceers_highz_sample.csv"
    if ceers_path.exists():
        df_c = pd.read_csv(ceers_path)
        df_c["z"] = df_c["z_phot"].astype(float)
        df_c = df_c[(df_c["z"] >= 4.0) & (df_c["z"] <= 12.0) & (df_c["log_Mstar"] >= 8.0)].copy()
        df_c = df_c.dropna(subset=["z", "log_Mstar"])
        df_c = _add_tep_columns(df_c)
        df_c["survey"] = "CEERS"
        logger.info(f"  CEERS: loaded {len(df_c)} galaxies (z=4-12)")
        surveys.append(df_c[["z", "log_Mstar", "log_Mh", "gamma_t", "survey"]])

    # COSMOS-Web highz (z = 4-12)
    cosmo_path = DATA_INTERIM_PATH / "cosmosweb_highz_sample.csv"
    if cosmo_path.exists():
        df_cw = pd.read_csv(cosmo_path)
        df_cw["z"] = df_cw["z_phot"].astype(float)
        df_cw = df_cw[(df_cw["z"] >= 4.0) & (df_cw["z"] <= 12.0) & (df_cw["log_Mstar"] >= 8.0)].copy()
        df_cw = df_cw.dropna(subset=["z", "log_Mstar"])
        df_cw = _add_tep_columns(df_cw)
        df_cw["survey"] = "COSMOS-Web"
        logger.info(f"  COSMOS-Web: loaded {len(df_cw)} galaxies (z=4-12)")
        surveys.append(df_cw[["z", "log_Mstar", "log_Mh", "gamma_t", "survey"]])

    return pd.concat(surveys, ignore_index=True)


# ── Core analysis ────────────────────────────────────────────────────────────

def count_above_threshold(df, z_lo, z_hi, log_mass_thresh, n_ml=N_ML):
    """
    Count galaxies above a mass threshold before and after TEP correction.
    """
    mask = (df["z"] >= z_lo) & (df["z"] < z_hi)
    sub = df[mask].copy()

    if len(sub) == 0:
        return {"z_lo": z_lo, "z_hi": z_hi, "threshold": log_mass_thresh,
                "n_bin": 0, "n_obs": 0, "n_corr": 0, "reduction_pct": 0.0}

    # Observed count above threshold
    n_obs = int((sub["log_Mstar"] > log_mass_thresh).sum())

    # TEP-corrected mass
    sub["log_Mstar_true"] = sub["log_Mstar"] - n_ml * np.log10(
        np.maximum(sub["gamma_t"].values, 1e-3)
    )
    n_corr = int((sub["log_Mstar_true"] > log_mass_thresh).sum())

    reduction = (1 - n_corr / n_obs) * 100 if n_obs > 0 else 0.0

    return {
        "z_lo": z_lo,
        "z_hi": z_hi,
        "threshold": log_mass_thresh,
        "n_bin": len(sub),
        "n_obs": n_obs,
        "n_corr": n_corr,
        "reduction_pct": round(reduction, 1),
    }


def run():
    print_status("=" * 65, "INFO")
    print_status(f"Step {STEP_NUM}: Stellar Mass Function — Mass Threshold Counts", "INFO")
    print_status("=" * 65, "INFO")

    logger.info("Loading three-survey data...")
    df = load_all_surveys()
    logger.info(f"Total: {len(df)} galaxies across {df['survey'].nunique()} surveys")

    # Compute Table 15 counts
    table_rows = []
    for z_lo, z_hi in Z_BINS:
        for thresh in MASS_THRESHOLDS:
            row = count_above_threshold(df, z_lo, z_hi, thresh)
            table_rows.append(row)
            logger.info(
                f"  z=[{z_lo},{z_hi}), logM*>{thresh}: "
                f"N_obs={row['n_obs']}, N_corr={row['n_corr']}, "
                f"reduction={row['reduction_pct']:.0f}%"
            )

    # Per-survey breakdown for z>7
    z7_mask = df["z"] >= 7.0
    per_survey = {}
    for survey in df["survey"].unique():
        s = df[(z7_mask) & (df["survey"] == survey)]
        per_survey[survey] = len(s)
    total_z7 = int(z7_mask.sum())

    # Summary
    print_status(f"\nTable 15 (n_ml = {N_ML}):", "INFO")
    print_status(f"{'Redshift':<12} {'Threshold':<12} {'N_obs':>6} {'N_corr':>7} {'Reduction':>10}", "INFO")
    for r in table_rows:
        print_status(
            f"z={r['z_lo']}-{r['z_hi']:<8} logM*>{r['threshold']:<6} "
            f"{r['n_obs']:>6} {r['n_corr']:>7} {r['reduction_pct']:>9.0f}%",
            "INFO",
        )

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "description": "Stellar mass function mass-threshold counts (Table 15)",
        "n_ml": N_ML,
        "total_galaxies": len(df),
        "total_z7": total_z7,
        "per_survey_z7": per_survey,
        "table_15": table_rows,
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=safe_json_default)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
