#!/usr/bin/env python3
"""
TEP-JWST Step 134: TEP prediction error budget

TEP prediction error budget — alpha_0 uncertainty propagation


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM  = "134"
STEP_NAME = "prediction_errors"

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH   = PROJECT_ROOT / "logs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

import numpy as np
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like, ALPHA_0, ALPHA_UNCERTAINTY

# Fiducial galaxy parameters for sensitivity calculations
FIDUCIAL_LOG_MSTAR = np.array([8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
FIDUCIAL_Z        = np.array([5, 6, 7, 8, 9, 10])
ALPHA_GRID        = np.linspace(
    max(0.01, ALPHA_0 - 3 * ALPHA_UNCERTAINTY),
    ALPHA_0 + 3 * ALPHA_UNCERTAINTY,
    50
)


def gamma_sensitivity(log_mstar, z):
    """Compute dGamma_t/d_alpha_0 at fiducial alpha_0."""
    log_mh = stellar_to_halo_mass_behroozi_like(
        np.array([log_mstar]), np.array([z])
    )[0]
    g_hi = compute_gamma_t(
        np.array([log_mh]), np.array([z]),
        alpha_0=ALPHA_0 + 0.01
    )[0]
    g_lo = compute_gamma_t(
        np.array([log_mh]), np.array([z]),
        alpha_0=ALPHA_0 - 0.01
    )[0]
    return float((g_hi - g_lo) / 0.02)


def run():
    print_status(f"STEP {STEP_NUM}: TEP prediction error budget (alpha_0 uncertainty)", "INFO")

    # 1. alpha_0 sensitivity of Gamma_t at fiducial galaxies
    sensitivity_rows = []
    for m in FIDUCIAL_LOG_MSTAR:
        for z in FIDUCIAL_Z:
            log_mh = stellar_to_halo_mass_behroozi_like(np.array([m]), np.array([z]))[0]
            g_fid  = compute_gamma_t(np.array([log_mh]), np.array([z]))[0]
            dg_da  = gamma_sensitivity(m, z)
            delta_g = abs(dg_da) * ALPHA_UNCERTAINTY
            sensitivity_rows.append({
                "log_Mstar": float(m),
                "z":          float(z),
                "gamma_t_fid": float(g_fid),
                "dGamma_dalpha": float(dg_da),
                "delta_gamma_1sigma": float(delta_g),
                "frac_uncertainty": float(delta_g / g_fid) if g_fid > 0 else float("nan"),
            })

    # 2. Propagate alpha_0 uncertainty through rho prediction
    # Use analytic approximation: d(rho)/d(alpha_0) via finite difference on Gamma_t grid
    # Load COSMOS-Web data for numerical estimate
    import pandas as pd
    from scipy.stats import spearmanr
    DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
    rho_vs_alpha = []
    cw_file = DATA_INTERIM / "cosmosweb_highz_sample.csv"
    if cw_file.exists():
        df = pd.read_csv(cw_file)
        df = df.dropna(subset=["z_phot", "log_Mstar"]).copy()
        df = df[(df["z_phot"] > 4) & (df["log_Mstar"] > 6)]
        if "dust" in df.columns and len(df) > 10:
            df_valid = df[df["dust"] >= 0].copy()
            log_mh = stellar_to_halo_mass_behroozi_like(
                df_valid["log_Mstar"].values, df_valid["z_phot"].values
            )
            for a in ALPHA_GRID:
                gt = compute_gamma_t(log_mh, df_valid["z_phot"].values, alpha_0=a)
                rho, _ = spearmanr(gt, df_valid["dust"].values)
                rho_vs_alpha.append({"alpha": float(a), "rho": float(rho)})

    if rho_vs_alpha:
        alphas = [r["alpha"] for r in rho_vs_alpha]
        rhos   = [r["rho"]   for r in rho_vs_alpha]
        # Finite difference around alpha_0
        idx0 = np.argmin(np.abs(np.array(alphas) - ALPHA_0))
        drho_da = float(
            np.gradient(np.array(rhos), np.array(alphas))[idx0]
        )
        delta_rho_1sigma = abs(drho_da) * ALPHA_UNCERTAINTY
        rho_at_alpha0 = rhos[idx0]
        logger.info(
            f"  d(rho)/d(alpha_0) = {drho_da:.3f}"
            f"  delta_rho(1-sigma) = {delta_rho_1sigma:.4f}"
        )
    else:
        drho_da = float("nan")
        delta_rho_1sigma = float("nan")
        rho_at_alpha0 = float("nan")
        rho_vs_alpha = []

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "TEP prediction error budget — alpha_0 uncertainty propagation",
        "alpha_0":            float(ALPHA_0),
        "alpha_0_uncertainty": float(ALPHA_UNCERTAINTY),
        "gamma_t_sensitivity": sensitivity_rows[:12],
        "rho_vs_alpha": rho_vs_alpha[:20],
        "drho_dalpha":         drho_da,
        "delta_rho_1sigma":    delta_rho_1sigma,
        "rho_at_alpha0":       rho_at_alpha0,
        "conclusion": (
            f"1-sigma alpha_0 uncertainty (+/-{ALPHA_UNCERTAINTY}) "
            f"propagates to delta_rho = +/-{delta_rho_1sigma:.4f}."
        ) if np.isfinite(delta_rho_1sigma) else "Could not compute delta_rho.",
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete. delta_rho(1sigma)={delta_rho_1sigma:.4f}", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
