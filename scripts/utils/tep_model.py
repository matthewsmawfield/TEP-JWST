#!/usr/bin/env python3
"""
TEP Universal Model Utilities (v0.8 Jakarta/Kos)

Shared functions for computing TEP quantities across all research papers:
COS (Paper 10), H0 (Paper 11), JWST (Paper 12), WB (Paper 13).

This module provides a unified point of truth for:
1. Universal Couplings (KAPPA_GAL, ALPHA_INT)
2. Chronological Enhancement (Gamma_t)
3. Screening Mechanisms (Temporal Topology)
4. Kinematic Profile Models (Wide Binaries)

Author: Matthew L. Smawfield
Date: April 2026
"""

import numpy as np
import pandas as pd
from scipy import integrate

# =============================================================================
# 1. UNIVERSAL COUPLINGS & CONSTANTS
# =============================================================================

# CANONICAL OBSERVABLE RESPONSE COEFFICIENT (Paper 11)
# Measured from Cepheid period-luminosity residuals.
# Units: Magnitudes [mag]
KAPPA_GAL = 9.6e5
KAPPA_GAL_UNCERTAINTY = 4.0e5

# PHOTON-SECTOR BOUND (Paper 9 / Cassini PPN)
# This is a dimensionless photon-sector coupling bound. It is deliberately
# separate from KAPPA_GAL, which is a magnitude-sector response coefficient.
ALPHA_PHOTON_BOUND = 0.003

# STELLAR EVOLUTION INDEX
# M/L ~ t^n from stellar isochrones.
ALPHA_NUCLEAR = 0.7

# POTENTIAL PARAMETERS
LOG_MH_REF = 12.0
PHI_REF_0 = 1.6e-7    # Dimensionless Phi/c^2 for 10^12 Msun halo at z=0
Z_REF = 5.5

# SCREENING SCALES
RHO_CRIT_G_CM3 = 20.0  # Temporal Topology saturation scale (g/cm^3)

# PHYSICAL CONSTANTS
C_LIGHT_KM_S = 2.99792458e5
G_NEWTON_PC_MSUN = 4.30091e-3  # (pc/Msun) * (km/s)^2
G_AU = 887.1                   # (km/s)^2 * AU / M_sun

# =============================================================================
# 2. CHRONOLOGICAL ENHANCEMENT (Gamma_t)
# =============================================================================

def get_phi_from_log_mh(log_Mh):
    """Compute dimensionless virial potential Phi/c^2 at z=0."""
    return 1.6e-7 * (10**log_Mh / 1e12)**(2/3)

def get_halo_potential(log_Mh):
    """Backward-compatible alias for the canonical z=0 virial potential."""
    return get_phi_from_log_mh(log_Mh)

def tep_alpha(z, kappa=KAPPA_GAL):
    """Redshift-dependent observable response coefficient κ(z)."""
    return kappa * np.sqrt(1 + z)

def compute_gamma_t_from_phi(phi, z, kappa=None, n=ALPHA_NUCLEAR):
    """
    Compute TEP chronological enhancement from dimensionless potential Phi/c^2.

    Gamma_t = exp[ K_gal * (Phi - Phi_ref,0) * sqrt(1+z) ],
    where K_gal = kappa * ln(10) / (2.5*n).  The kappa argument is the
    magnitude-sector observable response coefficient, not a bare scalar
    coupling.
    """
    eff_kappa = KAPPA_GAL if kappa is None else kappa
    k_exp = (eff_kappa * np.log(10)) / (2.5 * n)
    argument = k_exp * (np.asarray(phi) - PHI_REF_0) * np.sqrt(1 + np.asarray(z))
    return np.exp(argument)

def compute_gamma_t(log_Mh, z, kappa=None, n=ALPHA_NUCLEAR):
    """
    Compute TEP chronological enhancement factor (Potential-Linear Form).
    
    Gamma_t = exp[ K * (Phi - Phi_ref) * sqrt(1+z) ]
    where K = kappa * ln(10) / (2.5 * n)
    """
    phi = get_phi_from_log_mh(log_Mh)
    return compute_gamma_t_from_phi(phi, z, kappa=kappa, n=n)

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

def isochrony_mass_bias(gamma_t, n=ALPHA_NUCLEAR):
    """M/L ratio bias: M_obs / M_true = Gamma_t^n."""
    return np.power(np.maximum(gamma_t, 0.01), n)

def correct_stellar_mass(log_Mstar, gamma_t, n=ALPHA_NUCLEAR):
    """Apply TEP correction to observed stellar mass."""
    return log_Mstar - np.log10(isochrony_mass_bias(gamma_t, n))

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
    """Empirical SMHM relation proxy for high-z."""
    log_ratio = -1.8 - 0.1 * (np.asarray(log_Mstar) - 10) + 0.05 * (np.asarray(z) - 5)
    return np.asarray(log_Mstar) - log_ratio

def compute_gamma_t_from_mstar(log_Mstar, z, kappa=None):
    """Compute Gamma_t after mapping stellar mass to a halo-mass proxy."""
    return compute_gamma_t(stellar_to_halo_mass(log_Mstar, z), z, kappa=kappa)

def correct_age_ratio(age_ratio, gamma_t):
    """Apply TEP correction to an observed stellar-age/cosmic-age ratio."""
    return np.asarray(age_ratio) / np.asarray(gamma_t)

def compute_effective_time(t_cosmic, gamma_t):
    """Effective stellar-population time t_eff = t_cosmic * Gamma_t."""
    return np.maximum(np.asarray(t_cosmic) * np.asarray(gamma_t), 0.001)

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

def compute_t_eff(log_Mh, z, kappa=None):
    """Compute TEP effective time from halo mass and redshift."""
    gamma_t = compute_gamma_t(log_Mh, z, kappa=kappa)
    return compute_effective_time(_cosmic_time_gyr(z), gamma_t)
