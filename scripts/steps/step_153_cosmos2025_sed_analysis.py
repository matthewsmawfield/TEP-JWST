#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 39.8s.
"""
TEP-JWST Step 153: COSMOS2025 LePHARE SED analysis (Shuntov+2025)

Full z>4 redshift-binned dust-Gamma_t correlation analysis using the
COSMOS-Web LePHARE catalog (COSMOSWeb_mastercatalog_v1_lephare.fits).
Produces Table B4a statistics: rho(Gamma_t, E(B-V)) by redshift bin,
and the partial correlation controlling for M* and z across N~37,965
valid galaxy SED fits.

Data: COSMOSWeb_mastercatalog_v1_lephare.fits (Shuntov et al. 2025)
Requires: data/raw/COSMOSWeb_mastercatalog_v1_lephare.fits

Author: Matthew L. Smawfield
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import (
    compute_gamma_t,
    stellar_to_halo_mass_behroozi_like,
)
from scripts.utils.p_value_utils import format_p_value
from scripts.utils.rank_stats import partial_rank_correlation

STEP_NUM  = "153"
STEP_NAME = "cosmos2025_sed_analysis"

DATA_PATH   = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH   = PROJECT_ROOT / "logs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

COSMOSWEB_FITS = DATA_PATH / "COSMOSWeb_mastercatalog_v1_lephare.fits"

# Redshift bins for Table B4a
Z_BINS = [
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0),
    (9.0, 10.0),
    (10.0, 13.0),
]


def partial_spearman(x, y, controls):
    """Spearman partial correlation of x and y after regressing out controls."""
    rho, p, _ = partial_rank_correlation(x, y, controls)
    return float(rho), float(p)


def bootstrap_ci(x, y, n_boot=200, seed=42):
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos.append(float(r))
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def load_cosmos_catalog():
    """Load and filter COSMOS-Web LePHARE catalog for valid galaxy SED fits."""
    from astropy.io import fits
    from astropy.cosmology import Planck18 as cosmo

    if not COSMOSWEB_FITS.exists():
        logger.error(f"COSMOS-Web FITS not found: {COSMOSWEB_FITS}")
        return None

    logger.info(f"Loading {COSMOSWEB_FITS.name} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(COSMOSWEB_FITS) as hdul:
            t = hdul[1].data
            df = pd.DataFrame({
                "z_phot":   t["zfinal"].astype(float),
                "log_Mstar": t["mass_med"].astype(float),
                "ebv":       t["ebv_minchi2"].astype(float),
                "ssfr":      t["ssfr_med"].astype(float),
                "sfr":       t["sfr_med"].astype(float),
                "galaxy_type": t["type"].astype(int),
            })

    logger.info(f"Total sources: {len(df):,}")

    # Filter: galaxies only (type=0), valid z, valid SED parameters
    df = df[
        (df["galaxy_type"] == 0) &
        (df["z_phot"] > 0) & (df["z_phot"] < 15) &
        (df["log_Mstar"] > -90) &
        (df["ebv"] > -900) &
        (df["ssfr"] > -90) &
        (df["log_Mstar"] > 6.0)
    ].copy()

    logger.info(f"Valid galaxy SED fits: {len(df):,}")

    # Compute TEP quantities
    df["log_Mh"] = stellar_to_halo_mass_behroozi_like(
        df["log_Mstar"].values, df["z_phot"].values
    )
    df["gamma_t"] = compute_gamma_t(df["log_Mh"].values, df["z_phot"].values)
    df["t_cosmic"] = cosmo.age(df["z_phot"].values).value
    df["t_eff"] = np.clip(df["t_cosmic"] * df["gamma_t"], 1e-4, None)
    df["log_ssfr"] = df["ssfr"]  # Already log10 from LePHARE

    return df


def analyze_redshift_bin(sub, z_lo, z_hi):
    """Compute Spearman rho(Gamma_t, E(B-V)) and partial rho for a redshift bin."""
    n = len(sub)
    if n < 10:
        return {"n": n, "note": "insufficient_sample"}

    x = sub["gamma_t"].values
    y = sub["ebv"].values

    rho, p = spearmanr(x, y)
    p = max(float(p), 1e-300)

    ci_lo, ci_hi = bootstrap_ci(x, y, n_boot=200)

    # Partial rho controlling for M* and z
    try:
        p_rho, p_p = partial_spearman(
            x, y,
            controls=[sub["log_Mstar"].values, sub["z_phot"].values]
        )
        p_p = max(float(p_p), 1e-300)
    except Exception:
        p_rho, p_p = float("nan"), float("nan")

    return {
        "z_lo": z_lo,
        "z_hi": z_hi,
        "n": n,
        "rho": float(rho),
        "p": p,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "partial_rho": p_rho,
        "partial_p":   p_p,
    }


def run():
    print_status(f"STEP {STEP_NUM}: COSMOS2025 LePHARE SED analysis (Table B4a)", "INFO")

    df = load_cosmos_catalog()
    if df is None or len(df) == 0:
        result = {
            "step": STEP_NUM, "name": STEP_NAME,
            "status": "FAILED_NO_DATA",
            "note": f"COSMOS-Web FITS not found at {COSMOSWEB_FITS}. "
                    "Run step_033_cosmosweb_download.py first.",
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    # z>4 baseline
    df4 = df[df["z_phot"] > 4].copy()
    logger.info(f"z>4 sample: N = {len(df4):,}")

    # Redshift-binned analysis
    bins_results = []
    for z_lo, z_hi in Z_BINS:
        sub = df[(df["z_phot"] >= z_lo) & (df["z_phot"] < z_hi)].copy()
        res = analyze_redshift_bin(sub, z_lo, z_hi)
        bins_results.append(res)
        logger.info(
            f"  z={z_lo:.0f}-{z_hi:.0f}: N={res.get('n',0):,}, "
            f"rho={res.get('rho', float('nan')):.3f}, "
            f"partial_rho={res.get('partial_rho', float('nan')):.3f}"
        )

    # Combined z>7 and z>8
    sub7 = df[df["z_phot"] > 7].copy()
    sub8 = df[df["z_phot"] > 8].copy()
    sub9 = df[(df["z_phot"] >= 9.0) & (df["z_phot"] < 13.0)].copy()
    z7_res = analyze_redshift_bin(sub7, 7.0, 99.0)
    z7_res["label"] = "z>7_combined"
    z8_res = analyze_redshift_bin(sub8, 8.0, 99.0)
    z8_res["label"] = "z>8_combined"
    z9_res = analyze_redshift_bin(sub9, 9.0, 13.0)
    z9_res["label"] = "z=9-13_combined"

    # Full z>4 partial correlation
    if len(df4) > 50:
        full_partial_rho, full_partial_p = partial_spearman(
            df4["gamma_t"].values, df4["ebv"].values,
            controls=[df4["log_Mstar"].values, df4["z_phot"].values]
        )
        full_partial_p = max(full_partial_p, 1e-300)
    else:
        full_partial_rho, full_partial_p = float("nan"), float("nan")

    logger.info(f"Full z>4 partial rho: {full_partial_rho:.3f} (p={full_partial_p:.2e})")

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "COSMOS2025 LePHARE SED analysis — Table B4a",
        "n_total_z4": len(df4),
        "redshift_bins": bins_results,
        "z7_combined":  z7_res,
        "z8_combined":  z8_res,
        "z9_combined":  z9_res,
        "full_partial_rho":  full_partial_rho,
        "full_partial_p":    full_partial_p,
        "data_source": "COSMOSWeb_mastercatalog_v1_lephare.fits (Shuntov et al. 2025)",
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete. N(z>4)={len(df4):,}, "
                 f"partial rho(z>4)={full_partial_rho:.3f}", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
