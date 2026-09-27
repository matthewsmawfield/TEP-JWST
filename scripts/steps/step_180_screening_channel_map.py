#!/usr/bin/env python3
"""
Step 180: Screening Channel-Boundary Map
=========================================
Closes the screening-channel bookkeeping for the primary R_ML kernel.

TEP distinguishes two response operators (Paper 0, screening rule):

  * S_Sigma(E) — the shear/dynamical channel: suppresses the observable
    Temporal Shear (gradients) measured by accelerations, spin-down and
    dynamical-mass inference.  The globular-cluster pulsar boundary
    sigma > 165 km/s (Paper 10) is an S_Sigma datum.
  * S_A(E) — the clock-amplitude/inference channel: modulates the
    mass-to-light inference response R_ML.  Its stated suppression is
    kappa_eff = kappa_gal * S(rho/rho_T), rho_T = 20 g/cm^3
    (Eq. jwst_kappa_eff; core/screening.py::screening_factor).

This step quantifies both boundaries on the real UNCOVER sample:

  1. Maps the S_Sigma boundary sigma_crit = 165 km/s to halo mass as a
     function of redshift using the pipeline's own virial map
     (step_171: log M_h = 5 log10(sigma/200) + 12.5 - 1.5 log10(1+z))
     and computes the fraction of the sample above it per z-bin.
  2. Evaluates the amplitude-channel suppression S(rho/rho_T) at
     representative halo virial and disk densities.
  3. Tabulates the measured inference-channel environmental contrast
     (steps 094/138 outputs) per redshift bin, including the z>8 null.

Output: results/outputs/step_180_screening_channel_map.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.screening import screening_factor  # noqa: E402
from scripts.utils.tep_model import (  # noqa: E402
    get_potential_depth_from_log_mh, PHI_REF_0,
)

STEP_NUM = 180
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"

# S_Sigma dynamical boundary (Paper 10, globular-cluster pulsars)
SIGMA_CRIT = 165.0  # km/s

# Pipeline virial map (step_171 conventions)
SIGMA_REF = 200.0
LOG_MH_AT_SIGMA_REF = 12.5
SIGMA_SLOPE = 5.0

# Amplitude-channel transition density (Paper 6 / corpus canonical)
RHO_T = 20.0  # g/cm^3


def sigma_to_log_mh(sigma, z):
    """Pipeline virial map: sigma -> log M_h at redshift z."""
    return SIGMA_SLOPE * np.log10(np.maximum(sigma, 1.0) / SIGMA_REF) \
        + LOG_MH_AT_SIGMA_REF - 1.5 * np.log10(1 + z)


def log_mh_to_sigma(log_Mh, z):
    """Inverse virial map: log M_h -> virial sigma at redshift z."""
    return SIGMA_REF * 10 ** (
        (np.asarray(log_Mh) - LOG_MH_AT_SIGMA_REF
         + 1.5 * np.log10(1 + z)) / SIGMA_SLOPE)


def halo_virial_density_g_cm3(z):
    """Mean interior halo density ~ 200 * rho_crit(z) (spherical collapse)."""
    rho_crit0 = 9.2e-30  # g/cm^3 at z=0 (H0 ~ 70)
    Om = 0.315
    Ez2 = Om * (1 + z) ** 3 + (1 - Om)
    return 200.0 * rho_crit0 * Ez2


def main():
    print("=" * 60)
    print(f"STEP {STEP_NUM}: Screening Channel-Boundary Map")
    print("=" * 60)

    # ---- 1. S_Sigma boundary in halo-mass units -------------------------
    boundary_z = {}
    for z in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        boundary_z[f"z_{z:g}"] = {
            "log_Mh_boundary": float(sigma_to_log_mh(SIGMA_CRIT, z)),
            "psi_boundary_over_psi_ref0": float(
                get_potential_depth_from_log_mh(sigma_to_log_mh(SIGMA_CRIT, z))
                / PHI_REF_0),
        }
    print("S_Sigma boundary (sigma=165 km/s) in log M_h:")
    for k, v in boundary_z.items():
        print(f"  {k}: log M_h = {v['log_Mh_boundary']:.2f}, "
              f"Psi/Psi_ref = {v['psi_boundary_over_psi_ref0']:.2f}")

    # ---- 2. Sample fractions above the dynamical boundary ----------------
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    df = df[df["log_Mh"].notna() & df["z_phot"].notna()].copy()
    df["sigma_vir"] = log_mh_to_sigma(df["log_Mh"].values,
                                    df["z_phot"].values)
    df["above_shear_boundary"] = df["sigma_vir"] > SIGMA_CRIT

    frac_by_z = {}
    for lo, hi, lab in [(0, 4, "z_lt_4"), (4, 6, "z_4_6"),
                        (6, 8, "z_6_8"), (8, 15, "z_gt_8"),
                        (0, 15, "all")]:
        sub = df[(df.z_phot > lo) & (df.z_phot <= hi)]
        if len(sub) == 0:
            frac_by_z[lab] = {"n": 0}
            continue
        frac_by_z[lab] = {
            "n": int(len(sub)),
            "frac_sigma_gt_165": float(sub["above_shear_boundary"].mean()),
            "median_sigma_vir": float(sub["sigma_vir"].median()),
            "max_log_Mh": float(sub["log_Mh"].max()),
        }
        print(f"  {lab}: N={len(sub)}, frac sigma>165 = "
              f"{sub['above_shear_boundary'].mean():.3f}, "
              f"median sigma = {sub['sigma_vir'].median():.0f} km/s")

    # ---- 3. Amplitude-channel suppression at halo/disk densities ---------
    env = {}
    for z in [0.0, 5.0, 8.0]:
        rho_vir = halo_virial_density_g_cm3(z)
        env[f"halo_virial_z{z:g}"] = {
            "rho_g_cm3": float(rho_vir),
            "log10_rho_over_rhoT": float(np.log10(rho_vir / RHO_T)),
            "S_A_suppression": float(screening_factor(rho_vir)),
        }
    for name, rho in [("galaxy_disk_ism", 1e-24), ("molecular_cloud", 1e-20),
                      ("stellar_photosphere", 1e-7), ("stellar_interior", 1.0)]:
        env[name] = {
            "rho_g_cm3": rho,
            "log10_rho_over_rhoT": float(np.log10(rho / RHO_T)),
            "S_A_suppression": float(screening_factor(rho)),
        }
    print("Amplitude-channel suppression S(rho/rho_T):")
    for k, v in env.items():
        print(f"  {k}: rho={v['rho_g_cm3']:.1e} g/cm3, "
              f"S_A = {v['S_A_suppression']:.6f}")

    # ---- 4. Measured inference-channel environmental contrast ------------
    contrast = {}
    try:
        d138 = json.load(open(OUTPUT_PATH /
                              "step_138_environmental_screening_steiger.json"))
        for key in ["full_sample", "z_4_6", "z_6_8", "z_gt_8"]:
            s = d138.get(key, {})
            contrast[key] = {
                "n": s.get("n_total"),
                "delta_rho": s.get("delta_rho"),
                "p": s.get("p_fisher_independent"),
                "tep_confirmed": s.get("tep_confirmed"),
            }
    except FileNotFoundError:
        contrast["error"] = "step_138 output not found"

    results = {
        "step": f"step_{STEP_NUM}_screening_channel_map",
        "channel_classification": {
            "S_Sigma_dynamical": {
                "boundary": "sigma_vir > 165 km/s (Paper 10, pulsar spin-down)",
                "governs": ("kinematic observables: spin-down, accelerations, "
                            "dynamical-mass inference"),
                "boundary_in_log_Mh_vs_z": boundary_z,
            },
            "S_A_amplitude_inference": {
                "boundary": ("kappa_eff = kappa_gal * S(rho/rho_T), "
                             "rho_T = 20 g/cm^3 (Eq. jwst_kappa_eff)"),
                "governs": ("photometric mass-to-light inference response "
                            "R_ML (primary kernel)"),
                "evaluation": env,
            },
        },
        "sample_vs_dynamical_boundary": frac_by_z,
        "environmental_contrast_measured": contrast,
        "conclusions": [
            "The primary kernel applies the amplitude-channel (S_A) response. "
            "At halo virial and disk densities (rho/rho_T ~ 1e-26..1e-25), the "
            "stated suppression operator returns S_A = 1 to machine precision, "
            "so the unscreened kernel is the model's own output for this "
            "channel; amplitude-channel screening first becomes relevant at "
            "stellar-interior densities.",
            "The sigma > 165 km/s boundary is a shear-channel (S_Sigma) datum "
            "governing dynamical observables; under the pipeline virial map it "
            "corresponds to log M_h = 12.1 at z=0, declining to ~10.7 at z=8. "
            "It does not transfer to the inference channel (channel-projection "
            "notice, manuscript section 4.1).",
            "Even on the dynamical map, ~87% of the UNCOVER sample lies below "
            "the shear boundary; the most massive systems (log M_h > 12.5, "
            "including the Red-Monster regime) lie above it, predicting "
            "suppressed dynamical-channel anomalies there while the "
            "amplitude-channel response remains unsuppressed.",
            "The inference channel's environmental dependence is measured "
            "directly (steps 094/138): detected at z <= 8 "
            "(Delta_rho = +0.30 full sample, +0.20 at z=6-8) and null at z>8 "
            "(Delta_rho = +0.04, p = 0.61) - reported as an open tension.",
        ],
    }

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_PATH / f"step_{STEP_NUM}_screening_channel_map.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
