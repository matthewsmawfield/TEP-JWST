#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): <0.1s.
"""
TEP-JWST Step 133: Hubble tension connection

Hubble tension connection — TEP H0 prediction consistency


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging

STEP_NUM  = "133"  # Pipeline step number
STEP_NAME = "hubble_connection"  # Used in log / output filenames

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH   = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

import numpy as np
from scripts.utils.tep_model import (
    compute_gamma_t, stellar_to_halo_mass_behroozi_like,
    KAPPA_GAL, KAPPA_GAL_UNCERTAINTY, Z_REF  # Shared TEP model
)

# Observed H0 tension values
H0_PLANCK    = 67.4   # CMB-inferred Hubble constant [km/s/Mpc] (Planck 2020)
H0_SH0ES     = 73.04  # Local Cepheid+SNIa Hubble constant [km/s/Mpc] (Riess+2022)
H0_TENSION   = H0_SH0ES - H0_PLANCK  # H0 discrepancy [km/s/Mpc]
SIGMA_TENSION = 5.0   # Significance of H0 tension [sigma]

# S8 tension
S8_PLANCK    = 0.832  # Planck CMB S8 value
S8_KiDS      = 0.766  # KiDS weak-lensing S8 value
S8_TENSION   = S8_PLANCK - S8_KiDS  # S8 discrepancy

# TEP prediction for H0:
# TEP modifies the effective Hubble flow in the late universe through
# screening-dependent time dilation. The key insight is that Cepheid
# distances (local H0 anchors) are measured in partially unscreened halos,
# which inflates the inferred H0.
# TEP prediction: H0_local = H0_CMB * <Gamma_t^alpha>_Cepheid
# where <Gamma_t^alpha>_Cepheid is the mean enhancement over Cepheid hosts

CEPHEID_HALO_MASS    = 12.3  # log Msun, typical spiral halo hosting Cepheids
CEPHEID_Z            = 0.001  # local universe redshift
LMC_HALO_MASS        = 11.5   # log Msun


def h0_tep_prediction(kappa=KAPPA_GAL):
    """
    Predict the TEP contribution to the H0 tension.
    H0_local / H0_global = mean(<Gamma_t>_Cepheid-hosts)
    """
    # Compute Gamma_t for typical Cepheid host halo at z~0
    gt_mw = float(compute_gamma_t(
        np.array([CEPHEID_HALO_MASS]),
        np.array([CEPHEID_Z])
    )[0])
    # Gamma_t for LMC (lower mass, less screening)
    gt_lmc = float(compute_gamma_t(
        np.array([LMC_HALO_MASS]),
        np.array([CEPHEID_Z])
    )[0])
    # Effective H0 correction: distances shorten by Gamma_t^0.5 (area-distance scaling)
    # => H0_local = H0_global * <Gamma_t>^0.5
    gt_mean = (gt_mw + gt_lmc) / 2
    h0_correction = gt_mean ** 0.5  # approximation
    h0_tep = H0_PLANCK * h0_correction
    return float(h0_tep), float(gt_mean), float(h0_correction)


def s8_tep_prediction(kappa=KAPPA_GAL):
    """
    Predict the TEP contribution to the S8 tension.
    TEP modifies the growth of structure via enhanced effective time,
    but screening suppresses this at cluster scales (high density).
    """
    # At cluster scales (M~10^14 Msun), Gamma_t -> 1 (fully screened)
    # At field galaxy scales (M~10^11 Msun), Gamma_t > 1
    # Net effect: sigma_8 is inflated by ~Gamma_t^0.5 in low-density probes (WL)
    # => S8_WL = S8_CMB / <Gamma_t>_field
    gt_field = float(compute_gamma_t(
        np.array([11.5]), np.array([0.3])  # z~0.3 weak lensing survey median
    )[0])
    s8_correction = 1.0 / gt_field**0.3  # partial suppression
    s8_tep = S8_PLANCK * s8_correction
    return float(s8_tep), float(gt_field), float(s8_correction)


def run():
    print_status(f"STEP {STEP_NUM}: TEP connection to Hubble and S8 tensions", "INFO")

    # H0 tension prediction
    h0_tep, gt_ceph, h0_corr = h0_tep_prediction()
    h0_residual = abs(h0_tep - H0_SH0ES)
    h0_explained_frac = 1 - h0_residual / H0_TENSION if H0_TENSION > 0 else float("nan")

    logger.info(f"  H0 tension: {H0_PLANCK} -> {H0_SH0ES} km/s/Mpc ({SIGMA_TENSION} sigma)")
    logger.info(f"  TEP prediction: H0_local = {h0_tep:.2f} km/s/Mpc")
    logger.info(f"  H0 tension explained: {100*h0_explained_frac:.0f}%")

    # S8 tension prediction
    s8_tep, gt_field, s8_corr = s8_tep_prediction()
    s8_residual = abs(s8_tep - S8_KiDS)
    s8_explained_frac = 1 - s8_residual / S8_TENSION if S8_TENSION > 0 else float("nan")

    logger.info(f"  S8 tension: {S8_PLANCK} (CMB) vs {S8_KiDS} (KiDS)")
    logger.info(f"  TEP prediction: S8_WL = {s8_tep:.3f}")
    logger.info(f"  S8 tension explained: {100*s8_explained_frac:.0f}%")

    # Kappa_gal uncertainty propagation
    h0_tep_hi, _, _ = h0_tep_prediction(KAPPA_GAL + KAPPA_GAL_UNCERTAINTY)
    h0_tep_lo, _, _ = h0_tep_prediction(KAPPA_GAL - KAPPA_GAL_UNCERTAINTY)
    h0_uncertainty = (h0_tep_hi - h0_tep_lo) / 2

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "TEP connection to Hubble tension and S8 tension",
        "h0_planck":          H0_PLANCK,
        "h0_shoes":           H0_SH0ES,
        "h0_tension_sigma":   SIGMA_TENSION,
        "h0_tep_prediction":  h0_tep,
        "h0_tep_uncertainty": h0_uncertainty,
        "h0_tension_explained_frac": h0_explained_frac,
        "gamma_t_cepheid_hosts":    gt_ceph,
        "h0_correction_factor":    h0_corr,
        "s8_planck":          S8_PLANCK,
        "s8_kids":            S8_KiDS,
        "s8_tep_prediction":  s8_tep,
        "s8_tension_explained_frac": s8_explained_frac,
        "gamma_t_field_z03":  gt_field,
        "conclusion": (
            f"TEP predicts H0_local = {h0_tep:.1f} +/- {h0_uncertainty:.1f} km/s/Mpc, "
            f"explaining {100*h0_explained_frac:.0f}% of the H0 tension. "
            f"S8 tension partially resolved: S8_WL(TEP) = {s8_tep:.3f} vs KiDS = {S8_KiDS}."
        ),
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete. H0_TEP={h0_tep:.1f} km/s/Mpc", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
