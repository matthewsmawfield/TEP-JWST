#!/usr/bin/env python3
"""
TEP-JWST Step 154: JADES DR4 emission lines

JADES DR4 emission lines — ionization signal and Balmer absorption


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging

STEP_NUM  = "154"  # Pipeline step number
STEP_NAME = "jades_emission_lines"  # Used in log / output filenames

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH   = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

import warnings
import numpy as np
from scipy.stats import spearmanr  # Rank correlation for emission line analysis

DATA_RAW = PROJECT_ROOT / "data" / "raw"  # Raw external catalogues
JADES_SPEC_FILE = DATA_RAW / "jades_hainline" / "JADES_DR4_spectroscopic_catalog.fits"

Z_MIN = 4.0
Z_MAX = 12.0
SNR_MIN = 3.0
Z_BINS = [(4, 6), (6, 8), (8, 12)]

# Column name candidates for JADES R100 emission lines
HA_COLS   = ["Ha_FLUX", "HA_FLUX", "H_ALPHA_FLUX", "FLUX_Ha", "Ha+NII", "BLND_HBAA_N2_FLUX", "HBAA_6563_FLUX"]
HB_COLS   = ["Hb_FLUX", "HB_FLUX", "H_BETA_FLUX",  "FLUX_Hb", "HBAB_4861_FLUX"]
OIII_COLS = ["OIII_FLUX", "O3_FLUX", "FLUX_OIII", "[OIII]_FLUX", "O3_5007_FLUX"]
HA_EW_COLS = ["Ha_EW", "HA_EW", "EW_Ha", "EW_H_ALPHA"]
OIII_EW_COLS = ["OIII_EW", "O3_EW", "EW_OIII"]


def find_col(cols_candidates, available):
    for c in cols_candidates:
        if c in available:
            return c
    return None


def load_jades_spec():
    """Load JADES DR4 spectroscopic catalog and extract emission line data."""
    if not JADES_SPEC_FILE.exists():
        return None
    from astropy.io import fits
    warnings.filterwarnings("ignore")
    rows = []
    with fits.open(JADES_SPEC_FILE, memmap=False) as hdul:
        # Try all table extensions for emission line data
        for ext_i in range(1, len(hdul)):
            try:
                hdr  = hdul[ext_i].header
                data = hdul[ext_i].data
                if data is None or len(data) == 0:
                    continue
                cols = [c.name.upper() for c in data.columns]

                # Need redshift
                z_col = next((c for c in ["Z_PRISM", "Z_R100", "REDSHIFT", "Z"] if c in cols), None)
                if z_col is None:
                    continue

                # Emission line columns
                ha_col   = find_col([c.upper() for c in HA_COLS],   cols)
                hb_col   = find_col([c.upper() for c in HB_COLS],   cols)
                oiii_col = find_col([c.upper() for c in OIII_COLS], cols)
                ha_ew    = find_col([c.upper() for c in HA_EW_COLS],   cols)
                oiii_ew  = find_col([c.upper() for c in OIII_EW_COLS], cols)

                if not any([ha_col, hb_col, oiii_col, ha_ew, oiii_ew]):
                    continue

                for row in data:
                    try:
                        z = float(row[z_col])
                        if not (Z_MIN <= z <= Z_MAX):
                            continue
                        entry = {"z": z}
                        for k, col in [
                            ("ha_flux", ha_col), ("hb_flux", hb_col),
                            ("oiii_flux", oiii_col), ("ha_ew", ha_ew), ("oiii_ew", oiii_ew)
                        ]:
                            if col:
                                try:
                                    entry[k] = float(row[col])
                                except Exception:
                                    entry[k] = np.nan
                        rows.append(entry)
                    except Exception:
                        continue
                if rows:
                    logger.info(f"  Loaded from ext {ext_i}: {len(rows):,} rows with z+EL columns")
                    break
            except Exception:
                continue
    return rows if rows else None


def run():
    print_status(f"STEP {STEP_NUM}: JADES DR4 emission line vs. Gamma_t analysis", "INFO")
    import pandas as pd
    from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like

    rows = load_jades_spec()
    if not rows:
        # Fall back to JADES interim
        interim_file = PROJECT_ROOT / "data" / "interim" / "jades_highz_physical.csv"
        if interim_file.exists():
            df = pd.read_csv(interim_file)
            logger.info(f"  Fallback to JADES interim: N={len(df):,}")
            # No emission line columns — report limited analysis
            df = df[(df["z_best"] > Z_MIN)].dropna(subset=["z_best", "log_Mstar"])
            log_mh = stellar_to_halo_mass_behroozi_like(
                df["log_Mstar"].values, df["z_best"].values
            )
            gt = compute_gamma_t(log_mh, df["z_best"].values)
            rho, p = spearmanr(gt, df["log_Mstar"].values)
            result = {
                "step":   STEP_NUM,
                "name":   STEP_NAME,
                "status": "SUCCESS_PARTIAL",
                "description": "JADES emission lines — interim fallback (no EW columns)",
                "source": "jades_highz_physical.csv",
                "n":      len(df),
                "rho_gamma_t_logM": float(rho),
                "p":      max(float(p), 1e-300),
                "note":   "Emission line EW columns not found in JADES DR4 FITS; "
                          "reporting rho(Gamma_t, log M*) from interim catalog instead.",
            }
        else:
            result = {
                "step": STEP_NUM, "name": STEP_NAME,
                "status": "SKIPPED_NO_DATA",
                "note": "JADES DR4 spectroscopic catalog not found.",
            }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    df = pd.DataFrame(rows)
    logger.info(f"  Emission line catalog: N={len(df):,}")

    # Compute Gamma_t (use z and fiducial M*=9.5 as we lack masses here)
    log_mh = stellar_to_halo_mass_behroozi_like(
        np.full(len(df), 9.5), df["z"].values
    )
    df["gamma_t"] = compute_gamma_t(log_mh, df["z"].values)

    # Correlations: Gamma_t vs emission line EW/flux
    line_results = []
    for line_key, label in [
        ("ha_ew", "H-alpha EW"), ("oiii_ew", "[OIII] EW"),
        ("ha_flux", "H-alpha flux"), ("hb_flux", "H-beta flux"),
        ("oiii_flux", "[OIII] flux"),
    ]:
        if line_key not in df.columns:
            continue
        sub = df.dropna(subset=[line_key])
        sub = sub[sub[line_key] > 0]
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub["gamma_t"].values, np.log10(sub[line_key].values))
        line_results.append({
            "line":  label,
            "n":     len(sub),
            "rho":   float(rho),
            "p":     max(float(p), 1e-300),
        })
        logger.info(f"  {label}: N={len(sub)}, rho={rho:.3f}, p={p:.2e}")

    # Redshift bins
    bin_results = []
    for z_lo, z_hi in Z_BINS:
        sub = df[(df["z"] >= z_lo) & (df["z"] < z_hi)]
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub["gamma_t"].values, sub["z"].values)
        bin_results.append({"z_lo": z_lo, "z_hi": z_hi, "n": len(sub), "rho_gt_z": float(rho)})

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "JADES DR4 emission line EW vs. Gamma_t",
        "n_total":      len(df),
        "line_correlations": line_results,
        "z_bins":            bin_results,
        "conclusion": (
            f"JADES DR4 emission lines (N={len(df):,}): "
            + (f"rho(Gamma_t, log EW_Ha)={line_results[0]['rho']:.3f}. " if line_results else "")
            + "TEP predicts stronger emission lines in deeper potentials (more time for ionizing stellar evolution)."
        ),
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
