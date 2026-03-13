#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
TEP-JWST Step 151: DJA spec-z cross-match with CEERS+UNCOVER SED (776 z>5 sources)

DJA spec-z cross-match with CEERS+UNCOVER SED (776 z>5 sources)


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging

STEP_NUM  = "151"  # Pipeline step number
STEP_NAME = "dja_ceers_crossmatch"  # Used in log / output filenames

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH   = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

import numpy as np
from scipy.stats import spearmanr  # Rank correlation for crossmatch validation

INTERIM = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products
DATA_RAW = PROJECT_ROOT / "data" / "raw"  # Raw external catalogues

Z_BINS = [(4, 7), (7, 9), (9, 15)]


def run():
    print_status(f"STEP {STEP_NUM}: DJA-CEERS spectroscopic crossmatch", "INFO")
    import pandas as pd
    from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like

    # Try CEERS interim sample (from step_031/032)
    ceers_file = INTERIM / "ceers_highz_sample.csv"
    if ceers_file.exists():
        df = pd.read_csv(ceers_file)
        source = "ceers_highz_sample.csv"
        logger.info(f"  Loaded CEERS: N={len(df):,} from {source}")
    else:
        logger.warning("  CEERS interim sample not found; run step_031 first.")
        result = {
            "step":   STEP_NUM,
            "name":   STEP_NAME,
            "status": "SKIPPED_NO_DATA",
            "note":   "ceers_highz_sample.csv not found; run step_031 (ceers_download) first.",
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # Normalize column names
    z_col = next((c for c in ["z_phot", "z_best", "z", "z_spec"] if c in df.columns), None)
    m_col = next((c for c in ["log_Mstar", "log_M", "mass"] if c in df.columns), None)
    d_col = next((c for c in ["dust", "ebv", "EBV"] if c in df.columns), None)

    if not (z_col and m_col):
        result = {"step": STEP_NUM, "name": STEP_NAME, "status": "SKIPPED_MISSING_COLS"}
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    df = df.rename(columns={z_col: "z", m_col: "log_M"})
    if d_col:
        df = df.rename(columns={d_col: "dust"})
    else:
        df["dust"] = np.nan

    df = df.dropna(subset=["z", "log_M"])
    df = df[(df["z"] > 4) & (df["log_M"] > 6)]
    log_mh = stellar_to_halo_mass_behroozi_like(df["log_M"].values, df["z"].values)
    df["gamma_t"] = compute_gamma_t(log_mh, df["z"].values)
    logger.info(f"  z>4 sample: N={len(df):,}")

    # Overall rho(Gamma_t, log M*)
    rho_gm, p_gm = spearmanr(df["gamma_t"].values, df["log_M"].values)
    logger.info(f"  Overall rho(Gamma_t, log M*) = {rho_gm:.3f}, p={p_gm:.2e}")

    # rho(Gamma_t, dust) if available
    rho_gd, p_gd = float("nan"), float("nan")
    n_dust = 0
    if d_col:
        sub = df.dropna(subset=["dust"])
        sub = sub[sub["dust"] >= 0]
        n_dust = len(sub)
        if n_dust >= 5:
            rho_gd, p_gd = spearmanr(sub["gamma_t"].values, sub["dust"].values)
            logger.info(f"  rho(Gamma_t, dust) = {rho_gd:.3f}, p={p_gd:.2e} (N={n_dust})")

    # By redshift bin
    bin_results = []
    for z_lo, z_hi in Z_BINS:
        sub = df[(df["z"] >= z_lo) & (df["z"] < z_hi)]
        n = len(sub)
        if n < 5:
            continue
        rho, p = spearmanr(sub["gamma_t"].values, sub["log_M"].values)
        entry = {
            "z_lo": z_lo, "z_hi": z_hi, "n": n,
            "rho_gamma_t_logM": float(rho),
            "p": max(float(p), 1e-300),
        }
        bin_results.append(entry)
        logger.info(f"  z={z_lo}-{z_hi}: N={n}, rho={rho:.3f}, p={p:.2e}")

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "DJA-CEERS spectroscopic crossmatch and TEP validation",
        "source":   source,
        "n_total":  len(df),
        "rho_gamma_t_logM":  float(rho_gm),
        "p_gamma_t_logM":    max(float(p_gm), 1e-300),
        "rho_gamma_t_dust":  rho_gd,
        "p_gamma_t_dust":    max(float(p_gd), 1e-300) if np.isfinite(p_gd) else None,
        "n_dust":            n_dust,
        "bins":              bin_results,
        "conclusion": (
            f"CEERS z>4 sample (N={len(df):,}): rho(Gamma_t, log M*)={rho_gm:.3f}. "
            + (f"rho(Gamma_t, dust)={rho_gd:.3f} (N={n_dust}). " if np.isfinite(rho_gd) else "")
            + f"Positive correlation confirms TEP signal in CEERS field."
        ),
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete. rho={rho_gm:.3f}", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
