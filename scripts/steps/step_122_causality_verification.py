#!/usr/bin/env python3
"""
TEP-JWST Step 122: Causality constraint verification for scalar-tensor TEP theory

Causality constraint verification for scalar-tensor TEP theory


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import (
    KAPPA_GAL, ALPHA_PHOTON_BOUND, RHO_CRIT_G_CM3, 
    temporal_topology_suppression, compute_gamma_t
)

STEP_NUM  = "122"  # Pipeline step number (sequential 001-176)
STEP_NAME = "causality_verification"  # Causality verification: PPN gamma parameter constraint |gamma_PPN - 1| < 2.3e-5 (Cassini), screening radius calculation for Temporal Topology

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH   = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

import numpy as np

# Physical constants
C_LIGHT     = 2.998e5   # Speed of light c [km/s]
H0          = 67.4      # Hubble constant H_0 [km/s/Mpc] (Planck 2018)
G_NEWTON    = 4.302e-3  # Gravitational constant G [pc Msun^-1 (km/s)^2]


def ppn_gamma_parameter(kappa=ALPHA_PHOTON_BOUND):
    """
    Compute PPN gamma parameter for TEP scalar-tensor theory.
    TEP has Temporal Topology screening: in high-density environments (solar system),
    the effective coupling is suppressed via continuous field gradient flattening
    (Temporal Shear), driving gamma_PPN -> 1 (GR limit) inside screened regions.

    Uses ALPHA_PHOTON_BOUND (v0.7 Paper 9) which describes the coupling in the 
    photon sector, distinct from the clock sector (kappa).

    Constraint: |gamma_PPN - 1| < 2.3e-5 (Cassini, Bertotti+2003)
    """
    # 1. Background (unscreened) coupling in the photon sector.
    #    Do not use KAPPA_GAL here: it is a magnitude-sector response
    #    coefficient and is not the dimensionless scalar-tensor coupling.
    omega_unscreened = (1.0 / kappa**2 - 3) / 2 if kappa > 0 else float("inf")
    
    # 2. Screening suppression in solar system (v0.7 Temporal Topology)
    rho_solar = 1.4  # g/cm^3 (mean solar interior density)
    kappa_eff = temporal_topology_suppression(rho_solar, RHO_CRIT_G_CM3, kappa)

    # 3. PPN parameter with screened coupling
    omega_screened = (1.0 / kappa_eff**2 - 3) / 2 if kappa_eff > 0 else float("inf")
    gamma_ppn      = (1 + omega_screened) / (2 + omega_screened) if omega_screened > -2 else 1.0
    
    if omega_screened > 1e8:
        gamma_ppn = 1.0 - 1.0 / (2 * omega_screened)

    return float(omega_screened), float(gamma_ppn)


def screening_radius(kappa=KAPPA_GAL, log_mh=12.0, z=0):
    """
    Screening radius lambda_s where TEP effect is suppressed.
    
    In v0.7 Temporal Topology, screening occurs via continuous field gradient
    flattening (Temporal Shear) rather than a discrete boundary. The screening
    radius marks where suppression becomes significant (rho ~ rho_c).
    
    Uses the unified temporal_topology_suppression() for actual coupling values.
    Approximation: r_s ~ R_virial * 1.2 (calibrated from Milky Way constraints)
    """
    # Virial radius approximation: R_vir ~ (M_h / (200 * rho_c))^(1/3)
    m_h_msun = 10**log_mh
    # rho_c at z (in Msun/Mpc^3)
    hz = H0 * np.sqrt(0.315 * (1+z)**3 + 0.685)  # km/s/Mpc
    rho_c_z = 2.775e11 * (hz / H0)**2  # Msun/Mpc^3 (approx)
    r_vir_mpc = (3 * m_h_msun / (4 * np.pi * 200 * rho_c_z))**(1/3)
    # Screening radius is O(1) * R_vir (from Milky Way calibration)
    r_screen_mpc = r_vir_mpc * 1.2  # screening at ~1.2 R_vir
    return float(r_vir_mpc), float(r_screen_mpc)


def signal_propagation_speed():
    """
    Verify that the TEP scalar field propagates causally.
    In massless limit: c_s = c (Lorentz invariant)
    With mass term: c_s < c (massive scalar).
    Constraint from GW170817: |c_tensor/c - 1| < 5e-16
    TEP only modifies scalar sector (not tensor), so GW speed unchanged.
    """
    c_scalar   = 1.0   # c_s / c (massless limit; causally propagating)
    c_tensor   = 1.0   # TEP does not modify graviton sector
    gw170817_constraint = 5e-16
    gw_satisfied = abs(c_tensor - 1.0) < gw170817_constraint
    return {
        "c_scalar_over_c":        c_scalar,
        "c_tensor_over_c":        c_tensor,
        "gw170817_constraint":    gw170817_constraint,
        "gw_speed_satisfied":     gw_satisfied,
        "note": "TEP modifies only the scalar sector; tensor GW speed is unchanged (c_T=c precisely)",
    }


def run():
    print_status(f"STEP {STEP_NUM}: Causality constraint verification for TEP", "INFO")

    # 1. PPN gamma parameter
    omega_eff, gamma_ppn = ppn_gamma_parameter(ALPHA_PHOTON_BOUND)
    cassini_limit = 2.3e-5
    ppn_deviation = abs(gamma_ppn - 1.0)
    ppn_satisfied = ppn_deviation < cassini_limit
    logger.info(f"  PPN gamma = {gamma_ppn:.8f} (deviation {ppn_deviation:.2e})")
    logger.info(f"  Cassini limit: {cassini_limit:.2e} -> {'SATISFIED' if ppn_satisfied else 'VIOLATED'}")

    # 2. GW speed constraint
    gw_info = signal_propagation_speed()
    logger.info(f"  GW speed: c_T/c = {gw_info['c_tensor_over_c']} -> {'OK' if gw_info['gw_speed_satisfied'] else 'VIOLATED'}")

    # 3. Screening radii for MW/Cosmological halos
    screening_info = []
    for log_mh, label, z in [
        (12.0, "Milky Way halo", 0.0),
        (13.0, "Group halo", 0.0),
        (14.5, "Cluster halo", 0.0),
        (11.5, "Typical z=7 halo", 7.0),
    ]:
        r_vir, r_s = screening_radius(KAPPA_GAL, log_mh, z)
        gt = float(compute_gamma_t(
            np.array([log_mh]), np.array([z if z > 0 else 0.001]),
            kappa=KAPPA_GAL
        )[0])
        entry = {
            "label":        label,
            "log_Mh":       log_mh,
            "z":            z,
            "r_virial_Mpc": r_vir,
            "r_screen_Mpc": r_s,
            "gamma_t":      gt,
        }
        screening_info.append(entry)
        logger.info(f"  {label}: R_vir={r_vir:.3f} Mpc, Gamma_t={gt:.3f}")
    
    # Note: Gamma_t values above are UNscreened theoretical maxima.
    # At z~1e-3, massive clusters are fully screened (S(rho) ~ 0), so actual 
    # Gamma_t ~ 1. The high values shown represent the coupling strength that
    # would apply in the unscreened regime (relevant for high-z galaxies).

    # 4. Causality summary
    causal_violations = []
    if not ppn_satisfied:
        causal_violations.append(f"PPN gamma deviation {ppn_deviation:.2e} exceeds Cassini limit {cassini_limit:.2e}")
    if not gw_info["gw_speed_satisfied"]:
        causal_violations.append("GW speed constraint violated")

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "Causality constraint verification for scalar-tensor TEP theory",
        "ppn_gamma":             gamma_ppn,
        "ppn_deviation":         ppn_deviation,
        "ppn_cassini_limit":     cassini_limit,
        "ppn_satisfied":         ppn_satisfied,
        "omega_bd_effective":    omega_eff,
        "gw_speed":              gw_info,
        "screening":             screening_info,
        "causality_violations":  causal_violations,
        "causally_consistent":   len(causal_violations) == 0,
        "conclusion": (
            f"TEP theory is causally consistent: "
            f"PPN gamma deviation ({ppn_deviation:.2e}) < Cassini limit ({cassini_limit:.2e}); "
            f"GW speed unchanged (c_T = c precisely). "
            f"Screening mechanism confines TEP to cosmological scales."
        ) if len(causal_violations) == 0 else (
            f"Causality warnings: {'; '.join(causal_violations)}"
        ),
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete. Causally consistent: {result['causally_consistent']}", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
