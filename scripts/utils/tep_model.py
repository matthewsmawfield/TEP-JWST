#!/usr/bin/env python3
"""
TEP Universal Model Utilities (v0.8 Jakarta/Kos)

Shared functions for computing TEP quantities across all research papers:
COS (Paper 10), H0 (Paper 11), JWST (Paper 12), WB (Paper 13).

This module provides a unified point of truth for:
1. Universal Couplings (KAPPA_GAL, ALPHA_INT)
2. Positive Potential-Depth Proxies and Observable Channel Responses
3. Screening Mechanisms (Temporal Topology)
4. Kinematic Profile Models (Wide Binaries)

Author: Matthew L. Smawfield
Date: April 2026
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from core import constants as tep_const

# =============================================================================
# 1. UNIVERSAL COUPLINGS & CONSTANTS
# =============================================================================

# CANONICAL OBSERVABLE RESPONSE COEFFICIENT (Paper 11)
# Measured from Cepheid period-luminosity residuals.
# Units: Magnitudes [mag]
KAPPA_GAL = tep_const.KAPPA_GAL
KAPPA_GAL_UNCERTAINTY = tep_const.KAPPA_GAL_UNCERTAINTY

# PHOTON-SECTOR BOUND (Paper 9 / Cassini PPN)
# This is a dimensionless photon-sector coupling bound. It is deliberately
# separate from KAPPA_GAL, which is a magnitude-sector response coefficient.
ALPHA_PHOTON_BOUND = 0.003

# STELLAR EVOLUTION INDEX
# M/L ~ t^n from stellar isochrones.
ALPHA_NUCLEAR = tep_const.ALPHA_NUCLEAR

# POTENTIAL PARAMETERS
LOG_MH_REF = tep_const.LOG_MH_REF
PHI_REF_0 = tep_const.PHI_REF_0    # Dimensionless Phi/c^2 for 10^12 Msun halo at z=0
Z_REF = tep_const.Z_REF

# SCREENING SCALES (from core.constants)
RHO_CRIT_G_CM3 = tep_const.RHO_C

# PHYSICAL CONSTANTS
C_LIGHT_KM_S = 2.99792458e5
G_NEWTON_PC_MSUN = 4.30091e-3  # (pc/Msun) * (km/s)^2
G_AU = 887.1                   # (km/s)^2 * AU / M_sun

# =============================================================================
# 2. POTENTIAL DEPTH AND OBSERVABLE RESPONSE
# =============================================================================

def get_potential_depth_from_log_mh(log_Mh):
    """Compute the positive dimensionless virial depth Psi = |Phi|/c^2 at z=0."""
    return 1.6e-7 * (10**log_Mh / 1e12)**(2/3)

def get_phi_from_log_mh(log_Mh):
    """Backward-compatible alias returning positive potential depth, not signed Phi."""
    return get_potential_depth_from_log_mh(log_Mh)

def get_halo_potential(log_Mh):
    """Backward-compatible alias returning positive potential depth."""
    return get_potential_depth_from_log_mh(log_Mh)

def tep_alpha(z, kappa=KAPPA_GAL):
    """Redshift-dependent observable response coefficient κ(z)."""
    return kappa * np.sqrt(1 + z)

def compute_ml_response_from_depth(psi, z, kappa=None, n=ALPHA_NUCLEAR):
    """
    Compute the positive mass-to-light inference response from potential depth.

    R_ML = exp[ K_gal * (Psi - Psi_ref,0) * sqrt(1+z) ],
    where K_gal = kappa * ln(10) / (2.5*n). The kappa argument is a
    magnitude-sector observable response coefficient, not the conformal factor,
    a local proper-time ratio, or a bare scalar coupling.
    """
    eff_kappa = KAPPA_GAL if kappa is None else kappa
    k_exp = (eff_kappa * np.log(10)) / (2.5 * n)
    argument = k_exp * (np.asarray(psi) - PHI_REF_0) * np.sqrt(1 + np.asarray(z))
    return np.exp(argument)

def compute_gamma_t_from_phi(phi, z, kappa=None, n=ALPHA_NUCLEAR):
    """Backward-compatible alias for compute_ml_response_from_depth."""
    return compute_ml_response_from_depth(phi, z, kappa=kappa, n=n)

def compute_ml_response(log_Mh, z, kappa=None, n=ALPHA_NUCLEAR):
    """Compute the observable mass-to-light response from halo mass and redshift."""
    psi = get_potential_depth_from_log_mh(log_Mh)
    return compute_ml_response_from_depth(psi, z, kappa=kappa, n=n)

def compute_gamma_t(log_Mh, z, kappa=None, n=ALPHA_NUCLEAR):
    """Backward-compatible alias for compute_ml_response."""
    return compute_ml_response(log_Mh, z, kappa=kappa, n=n)

# =============================================================================
# 3. KINEMATIC & SCREENING MODELS
# =============================================================================

def tep_screening_model(s, r_s, alpha):
    """Canonical TEP exponential screening model for velocities."""
    return 1.0 + alpha * (1.0 - np.exp(-s / r_s))

def temporal_topology_suppression(rho, rho_c=RHO_CRIT_G_CM3, kappa_bare=KAPPA_GAL):
    """Continuous Temporal Topology suppression factor."""
    x = np.log10(rho / rho_c) / 0.5
    suppression = 1.0 / (1.0 + np.exp(x))
    return kappa_bare * suppression

# =============================================================================
# 4. MASS CORRECTIONS & BIAS
# =============================================================================

def ml_inference_bias(response, n=ALPHA_NUCLEAR):
    """Mass inference bias M_obs/M_true = R_ML^n."""
    return np.power(np.maximum(response, 0.01), n)

def isochrony_mass_bias(gamma_t, n=ALPHA_NUCLEAR):
    """Backward-compatible alias for ml_inference_bias."""
    return ml_inference_bias(gamma_t, n)

def correct_stellar_mass(log_Mstar, response, n=ALPHA_NUCLEAR):
    """Correct an observed logarithmic stellar mass using the M/L response."""
    return log_Mstar - np.log10(ml_inference_bias(response, n))


def compute_ml_response_self_consistent(log_Mstar_obs, z, kappa=None, n=ALPHA_NUCLEAR,
                                         tol=1e-4, max_iter=200, damping=0.3):
    """
    Solve the M*–Mh–R_ML relation self-consistently.

    The single-pass pipeline computes R_ML from the observed (biased) mass,
    then corrects the mass once.  But R_ML itself depends on the halo mass,
    which depends on the stellar mass.  The self-consistent solution iterates:

        M_h = AM(M*_true, z)
        R_ML = f(M_h, z)
        M*_true = M*_obs - n * log10(R_ML)

    to convergence.  For typical high-z galaxies (M* < 10) the single-pass
    and iterated solutions differ by < 2%, but at M* > 10.5 the difference
    can reach 10–60% because the abundance-matching slope amplifies the
    feedback loop.

    Damped fixed-point iteration (damping=0.3) is used because the undamped
    update oscillates between two fixed points for high-mass galaxies where
    the SMHM slope amplifies the feedback.  R_ML is clipped to [0.01, 100]
    inside the loop to keep the iteration bounded for extreme masses where
    the exponential response would otherwise diverge.  Convergence is
    assessed on the relative change in log_M_true (which is bounded) rather
    than on R_ML directly (which can span many orders of magnitude).

    Returns (R_ML_iterated, log_Mstar_true_iterated).
    """
    log_mstar = np.asarray(log_Mstar_obs, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    n_arr = np.broadcast_to(n, log_mstar.shape).astype(float) if np.ndim(n) else np.full_like(log_mstar, float(n))
    log_m_true = log_mstar.copy()
    log_m_true_prev = log_mstar.copy() + 1.0  # ensure first iteration runs
    for _ in range(max_iter):
        log_mh = stellar_to_halo_mass_behroozi_like(log_m_true, z_arr)
        r_ml = compute_ml_response(log_mh, z_arr, kappa=kappa, n=n_arr)
        # Clip R_ML to physical range to keep iteration bounded
        r_ml = np.clip(r_ml, 0.01, 100.0)
        log_m_true_new = log_mstar - n_arr * np.log10(r_ml)
        log_m_true = (1.0 - damping) * log_m_true + damping * log_m_true_new
        # Converge on log_M_true (bounded) rather than R_ML (can be extreme)
        delta = np.max(np.abs(log_m_true - log_m_true_prev))
        if delta < tol:
            break
        log_m_true_prev = log_m_true.copy()
    # Final R_ML from converged mass (clipped for safety)
    log_mh = stellar_to_halo_mass_behroozi_like(log_m_true, z_arr)
    r_ml = compute_ml_response(log_mh, z_arr, kappa=kappa, n=n_arr)
    r_ml = np.clip(r_ml, 0.01, 100.0)
    return r_ml, log_m_true

def stellar_to_halo_mass(log_Mstar, z=None):
    """
    Simple abundance-matching proxy for high-z JWST analyses.

    Upgraded from a simple +2.0 dex offset to use a Behroozi-like empirical 
    SMHM relation proxy to ensure redshift and mass dependence are rigorously 
    accounted for.
    """
    if z is None:
        # Fallback for older scripts that don't provide redshift
        return np.asarray(log_Mstar) + 2.0
    return stellar_to_halo_mass_behroozi_like(log_Mstar, z)

def stellar_to_halo_mass_behroozi_like(log_Mstar, z):
    """Empirical SMHM relation proxy for high-z.

    WARNING: This is a simplified linear proxy for the Behroozi+2019 SMHM
    relation, calibrated approximately for z~4-8.  At z>10 it is an
    EXTRAPOLATION with no calibration anchor.  The stellar-halo mass
    relation at z>10 is poorly constrained by observations, and
    abundance-matching extrapolations in this regime can produce
    physically pathological baryon-conversion efficiencies.  Results at
    z>10 should be treated as indicative only, not as calibrated evidence.
    Multiple independent SHMRs and a baryon-fraction ceiling should be
    used for any z>10 quantitative claim.
    """
    log_ratio = -1.8 - 0.1 * (np.asarray(log_Mstar) - 10) - 0.05 * (np.asarray(z) - 5)
    return np.asarray(log_Mstar) - log_ratio

def compute_ml_response_from_mstar(log_Mstar, z, kappa=None):
    """Compute the M/L response after mapping stellar mass to a halo-mass proxy."""
    return compute_ml_response(stellar_to_halo_mass(log_Mstar, z), z, kappa=kappa)

def compute_gamma_t_from_mstar(log_Mstar, z, kappa=None):
    """Backward-compatible alias for compute_ml_response_from_mstar."""
    return compute_ml_response_from_mstar(log_Mstar, z, kappa=kappa)

def correct_age_ratio(age_ratio, response):
    """Correct an inferred age ratio using the observable response proxy."""
    return np.asarray(age_ratio) / np.asarray(response)

def compute_inferred_time_proxy(t_cosmic, response):
    """Compute the observer-side inferred-time proxy t_proxy = t_cosmic R_ML."""
    return np.maximum(np.asarray(t_cosmic) * np.asarray(response), 0.001)

def compute_effective_time(t_cosmic, gamma_t):
    """Backward-compatible alias for compute_inferred_time_proxy."""
    return compute_inferred_time_proxy(t_cosmic, gamma_t)

# =============================================================================
# 5. COSMOLOGY UTILS
# =============================================================================

def cosmic_time_gyr(z, H0=67.4, Om=0.315):
    """Cosmic time at redshift z in Gyr (flat LCDM)."""
    z = np.atleast_1d(z)
    H0_s = H0 * 1e3 / 3.0857e22
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + (1-Om)))
    results = [integrate.quad(integrand, zi, np.inf)[0] / H0_s / 3.156e16 for zi in z]
    return np.array(results) if len(results) > 1 else results[0]

def _cosmic_time_gyr(z, H0=67.4, Om=0.315, OL=0.685):
    """Backward-compatible flat-LCDM cosmic time helper."""
    return cosmic_time_gyr(z, H0=H0, Om=Om)

def compute_inferred_time_from_halo(log_Mh, z, kappa=None):
    """Compute the observer-side inferred-time proxy from halo mass and redshift."""
    response = compute_ml_response(log_Mh, z, kappa=kappa)
    return compute_inferred_time_proxy(_cosmic_time_gyr(z), response)

def compute_t_eff(log_Mh, z, kappa=None):
    """Backward-compatible alias for compute_inferred_time_from_halo."""
    return compute_inferred_time_from_halo(log_Mh, z, kappa=kappa)
