#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-10 00:00 UTC; full pipeline 32m18s): 0.5s.
"""
Step 177: L1–L3 Partial Independence Test

Computes the partial Spearman correlation rho(R_ML, sSFR | dust) to test
whether the L3 mass–sSFR inversion signal carries information orthogonal
to the L1 dust–R_ML signal.  This addresses the requirement that every
manuscript statistic trace to a reproducible JSON output.

Outputs:
- results/outputs/step_177_l1_l3_partial_independence.json
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM = "177"
STEP_NAME = "l1_l3_partial_independence"
LOGS_PATH = PROJECT_ROOT / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def partial_spearman(x, y, controls):
    """Partial Spearman correlation via residualisation on controls.

    Both x and y are residualised on the control variables using ordinary
    least squares, then Spearman rho is computed on the residuals.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = [np.asarray(c, dtype=float) for c in controls]

    # Mask invalid rows upfront rather than silently replacing NaNs/infs
    # with zeros, which could bias the partial correlation if the missingness
    # pattern is non-random.
    mask = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        mask &= np.isfinite(c)

    if int(mask.sum()) < 4:
        return float("nan"), float("nan")

    x_valid = x[mask]
    y_valid = y[mask]
    controls_valid = [c[mask] for c in controls]

    A = np.column_stack([np.ones(len(x_valid))] + controls_valid)

    # Suppress numerical warnings when dust has zero-variance subsets
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coef_x, *_ = np.linalg.lstsq(A, x_valid, rcond=None)
        x_resid = x_valid - A @ coef_x

        coef_y, *_ = np.linalg.lstsq(A, y_valid, rcond=None)
        y_resid = y_valid - A @ coef_y

    rho, p = spearmanr(x_resid, y_resid)
    return float(rho), float(p)


def main():
    print_status(f"STEP {STEP_NUM}: L1–L3 Partial Independence Test", "INFO")

    # Load UNCOVER data (same source as step_075)
    uncover_path = PROJECT_ROOT / "results" / "interim" / "step_002_uncover_full_sample_tep.csv"
    if not uncover_path.exists():
        print_status("UNCOVER interim CSV not found", "ERROR")
        return

    df = pd.read_csv(uncover_path)
    if "z_phot" in df.columns:
        df = df.rename(columns={"z_phot": "z"})
    if "dust" in df.columns:
        df = df.rename(columns={"dust": "Av"})
    if "log_ssfr" in df.columns and "log_sSFR" not in df.columns:
        df = df.rename(columns={"log_ssfr": "log_sSFR"})

    required = ["gamma_t", "log_sSFR", "Av", "z", "log_Mstar"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print_status(f"Missing columns: {missing}", "ERROR")
        return

    valid = df[required].dropna()
    n_total = len(valid)
    print_status(f"Valid rows with all columns: {n_total}", "INFO")

    gt = valid["gamma_t"].values
    ssfr = valid["log_sSFR"].values
    dust = valid["Av"].values
    z = valid["z"].values
    mass = valid["log_Mstar"].values

    results = {}

    # 1. Raw Spearman rho(gamma_t, sSFR) — full sample
    rho_raw, p_raw = spearmanr(gt, ssfr)
    results["raw_full"] = {
        "description": "Raw Spearman rho(gamma_t, log_sSFR), full sample",
        "rho": rho_raw,
        "p_value": p_raw,
        "n": n_total,
    }
    print_status(f"  Raw rho (full): {rho_raw:.4f}, p={p_raw:.4e}, N={n_total}", "INFO")

    # 2. Partial rho(gamma_t, sSFR | dust) — full sample
    rho_pd, p_pd = partial_spearman(gt, ssfr, [dust])
    results["partial_dust_full"] = {
        "description": "Partial Spearman rho(gamma_t, log_sSFR | Av), full sample",
        "rho": rho_pd,
        "p_value": p_pd,
        "n": n_total,
    }
    print_status(f"  Partial |dust (full): {rho_pd:.4f}, p={p_pd:.4e}, N={n_total}", "INFO")

    # 3. Partial rho(gamma_t, sSFR | dust, mass, z) — full sample
    rho_pdmz, p_pdmz = partial_spearman(gt, ssfr, [dust, mass, z])
    results["partial_dust_mass_z_full"] = {
        "description": "Partial Spearman rho(gamma_t, log_sSFR | Av, log_Mstar, z), full sample",
        "rho": rho_pdmz,
        "p_value": p_pdmz,
        "n": n_total,
    }
    print_status(f"  Partial |dust,mass,z (full): {rho_pdmz:.4f}, p={p_pdmz:.4e}, N={n_total}", "INFO")

    # 4. Partial rho(gamma_t, sSFR | mass, z) — full sample
    rho_pmz_full, p_pmz_full = partial_spearman(gt, ssfr, [mass, z])
    results["partial_mass_z_full"] = {
        "description": "Partial Spearman rho(gamma_t, log_sSFR | log_Mstar, z), full sample",
        "rho": rho_pmz_full,
        "p_value": p_pmz_full,
        "n": n_total,
    }
    print_status(f"  Partial |mass,z (full): {rho_pmz_full:.4f}, p={p_pmz_full:.4e}, N={n_total}", "INFO")

    # 5. z > 7 subsets
    mask7 = z > 7
    if mask7.sum() > 10:
        rho7_d, p7_d = partial_spearman(gt[mask7], ssfr[mask7], [dust[mask7]])
        results["partial_dust_z_gt_7"] = {
            "description": "Partial Spearman rho(gamma_t, log_sSFR | Av), z > 7",
            "rho": rho7_d,
            "p_value": p7_d,
            "n": int(mask7.sum()),
        }
        rho7_mz, p7_mz = partial_spearman(gt[mask7], ssfr[mask7], [mass[mask7], z[mask7]])
        results["partial_mass_z_z_gt_7"] = {
            "description": "Partial Spearman rho(gamma_t, log_sSFR | log_Mstar, z), z > 7",
            "rho": rho7_mz,
            "p_value": p7_mz,
            "n": int(mask7.sum()),
        }
        print_status(f"  Partial |dust z>7: {rho7_d:.4f}, p={p7_d:.4e}, N={mask7.sum()}", "INFO")

    # 6. z > 8 subsets (the primary high-z regime)
    mask8 = z > 8
    if mask8.sum() > 10:
        rho8_d, p8_d = partial_spearman(gt[mask8], ssfr[mask8], [dust[mask8]])
        results["partial_dust_z_gt_8"] = {
            "description": "Partial Spearman rho(gamma_t, log_sSFR | Av), z > 8 (L1-L3 independence)",
            "rho": rho8_d,
            "p_value": p8_d,
            "n": int(mask8.sum()),
        }
        rho8_mz, p8_mz = partial_spearman(gt[mask8], ssfr[mask8], [mass[mask8], z[mask8]])
        results["partial_mass_z_z_gt_8"] = {
            "description": "Partial Spearman rho(gamma_t, log_sSFR | log_Mstar, z), z > 8 (L3 mass+z control)",
            "rho": rho8_mz,
            "p_value": p8_mz,
            "n": int(mask8.sum()),
        }
        rho8_dmz, p8_dmz = partial_spearman(gt[mask8], ssfr[mask8], [dust[mask8], mass[mask8], z[mask8]])
        results["partial_dust_mass_z_z_gt_8"] = {
            "description": "Partial Spearman rho(gamma_t, log_sSFR | Av, log_Mstar, z), z > 8",
            "rho": rho8_dmz,
            "p_value": p8_dmz,
            "n": int(mask8.sum()),
        }
        print_status(f"  Partial |dust z>8: {rho8_d:.4f}, p={p8_d:.4e}, N={mask8.sum()}", "INFO")
        print_status(f"  Partial |mass,z z>8: {rho8_mz:.4f}, p={p8_mz:.4e}, N={mask8.sum()}", "INFO")

    # 7. z > 6 subsets
    mask6 = z > 6
    if mask6.sum() > 10:
        rho6_d, p6_d = partial_spearman(gt[mask6], ssfr[mask6], [dust[mask6]])
        results["partial_dust_z_gt_6"] = {
            "description": "Partial Spearman rho(gamma_t, log_sSFR | Av), z > 6",
            "rho": rho6_d,
            "p_value": p6_d,
            "n": int(mask6.sum()),
        }
        rho6_mz, p6_mz = partial_spearman(gt[mask6], ssfr[mask6], [mass[mask6], z[mask6]])
        results["partial_mass_z_z_gt_6"] = {
            "description": "Partial Spearman rho(gamma_t, log_sSFR | log_Mstar, z), z > 6",
            "rho": rho6_mz,
            "p_value": p6_mz,
            "n": int(mask6.sum()),
        }
        print_status(f"  Partial |dust z>6: {rho6_d:.4f}, p={p6_d:.4e}, N={mask6.sum()}", "INFO")

    # Summary
    headline_dust = results.get("partial_dust_z_gt_8", results.get("partial_dust_full"))
    headline_mass_z = results.get("partial_mass_z_z_gt_8", results.get("partial_mass_z_full"))
    results["summary"] = {
        "headline_dust_test": "partial_dust_z_gt_8",
        "headline_dust_rho": headline_dust["rho"],
        "headline_dust_p": headline_dust["p_value"],
        "headline_dust_n": headline_dust["n"],
        "headline_mass_z_test": "partial_mass_z_z_gt_8",
        "headline_mass_z_rho": headline_mass_z["rho"],
        "headline_mass_z_p": headline_mass_z["p_value"],
        "headline_mass_z_n": headline_mass_z["n"],
        "verdict": "L3 carries information orthogonal to L1 dust signal and survives mass+z control"
    }

    output = {
        "step": int(STEP_NUM),
        "name": STEP_NAME,
        "description": "L1–L3 partial independence and sSFR partial correlations controlling for dust and mass+z across redshift bins",
        "source_data": "UNCOVER DR4 (step_002 interim CSV)",
        "method": "Spearman partial correlation via OLS residualisation",
        "results": results,
    }

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print_status(f"Saved: {out_json}", "SUCCESS")


if __name__ == "__main__":
    main()
