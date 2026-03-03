#!/usr/bin/env python3
"""
TEP Model Utilities

Shared functions for computing TEP quantities across all pipeline steps.
This eliminates code duplication and ensures consistency.

TEP Model (Exponential Form from Paper 1):
    Gamma_t = exp[alpha(z) * (2/3) * (log_Mh - log_Mh_ref(z)) * z_factor]
    
    where:
    - alpha(z) = alpha_0 * sqrt(1 + z)
    - alpha_0 = 0.58 ± 0.16 (from Cepheid calibration, Paper 12)
    - log_Mh_ref(z) = LOG_MH_REF - 1.5 * log10(1+z)  [Fixed potential depth scaling]
    - z_factor = (1 + z) / (1 + z_ref)
    - z_ref = 5.5 (reference redshift)

Author: Matthew L. Smawfield
Date: January 2026
"""

import numpy as np

# =============================================================================
# TEP MODEL CONSTANTS
# =============================================================================

ALPHA_0 = 0.58
ALPHA_UNCERTAINTY = 0.16
LOG_MH_REF = 12.0
Z_REF = 5.5
RHO_CRIT_G_CM3 = 20.0  # Critical screening density in g/cm^3

# =============================================================================
# TEP MODEL FUNCTIONS
# =============================================================================

def tep_alpha(z, alpha_0=ALPHA_0):
    """
    Redshift-dependent TEP coupling.
    
    alpha(z) = alpha_0 * sqrt(1 + z)
    
    This scaling comes from the theoretical expectation that the
    TEP coupling increases with the background field strength,
    which scales as sqrt(1 + z) in the early universe.
    
    Parameters
    ----------
    z : float or array
        Redshift
    alpha_0 : float
        Base coupling constant (default: 0.58)
        
    Returns
    -------
    float or array
        Redshift-dependent coupling
    """
    return alpha_0 * np.sqrt(1 + z)


def compute_gamma_t(log_Mh, z, alpha_0=ALPHA_0):
    """
    Compute TEP chronological enhancement factor (Exponential Form).
    
    Gamma_t = exp[alpha(z) * (2/3) * delta_log_Mh * z_factor]
    
    where:
    - delta_log_Mh = log_Mh - log_Mh_ref(z)
    - log_Mh_ref(z) = 12.0 - 1.5 * log10(1+z)  [Fixed Potential Depth Scaling]
    - z_factor = (1 + z) / (1 + z_ref)
    
    The exponential form is derived from the conformal factor A(phi) = exp(2*beta*phi/M_Pl)
    in the underlying TEP theory (Paper 1). This ensures:
    - Gamma_t > 0 always (no negative time)
    - Gamma_t > 1: Enhanced proper time (deeper potential)
    - Gamma_t < 1: Suppressed proper time (shallower potential)  
    - Gamma_t = 1: Reference potential (log_Mh = log_Mh_ref(z))
    
    For small arguments, exp(x) ≈ 1 + x, recovering the linear approximation.
    
    Parameters
    ----------
    log_Mh : float or array
        Log10 of halo mass in solar masses
    z : float or array
        Redshift
    alpha_0 : float
        Base coupling constant (default: 0.58)
        
    Returns
    -------
    float or array
        Chronological enhancement factor Gamma_t
    """
    alpha_z = tep_alpha(z, alpha_0)
    
    # Redshift-dependent reference mass (Fixed Sigma Scaling)
    # M_ref ~ (1+z)^(-1.5) for fixed potential depth
    log_mh_ref_z = LOG_MH_REF - 1.5 * np.log10(1 + z)
    
    delta_log_Mh = log_Mh - log_mh_ref_z
    z_factor = (1 + z) / (1 + Z_REF)
    
    argument = alpha_z * (2/3) * delta_log_Mh * z_factor
    return np.exp(argument)


def stellar_to_halo_mass(log_Mstar, z=None):
    """
    Simple abundance matching proxy for high-z.
    
    log_Mh ~ log_Mstar + 2.0
    
    This is a rough proxy sufficient for ranking and first-order TEP estimates.
    
    Parameters
    ----------
    log_Mstar : float or array
        Log10 of stellar mass
    z : float or array, optional
        Redshift (unused in this simple proxy, but kept for interface)
        
    Returns
    -------
    float or array
        Log10 of halo mass
    """
    return log_Mstar + 2.0


def stellar_to_halo_mass_behroozi_like(log_Mstar, z):
    log_ratio_base = -1.8
    mass_term = -0.1 * (log_Mstar - 10)
    z_term = 0.05 * (z - 5)
    log_ratio = log_ratio_base + mass_term + z_term
    return log_Mstar - log_ratio


def compute_gamma_t_from_mstar(log_Mstar, z, alpha_0=ALPHA_0):
    """
    Compute Gamma_t from stellar mass using simple abundance matching.
    
    log_Mh = stellar_to_halo_mass(log_Mstar)
    
    Parameters
    ----------
    log_Mstar : float or array
        Log10 of stellar mass in solar masses
    z : float or array
        Redshift
    alpha_0 : float
        Base coupling constant (default: 0.58)
        
    Returns
    -------
    float or array
        Chronological enhancement factor Gamma_t
    """
    log_Mh = stellar_to_halo_mass(log_Mstar, z)
    return compute_gamma_t(log_Mh, z, alpha_0)


def isochrony_mass_bias(gamma_t, n=0.7):
    """
    Mass-to-light ratio bias from isochrony assumption.
    
    M/L_apparent / M/L_true = Gamma_t^n
    
    This comes from the stellar population aging: older populations
    have higher M/L, and the M/L scales approximately as t^n.
    
    The power-law index n depends on redshift and metallicity:
    - n ~ 0.9 at z=4-6 (moderate metallicity, constant SFH)
    - n ~ 0.5 at z>6 (low metallicity; adopted for primary high-z analysis)
    - n ~ 0.7 is a reasonable global default
    
    With exponential Gamma_t, values are always positive, so no special handling needed.
    
    Parameters
    ----------
    gamma_t : float or array
        Chronological enhancement factor
    n : float
        M/L power-law index (default: 0.7)
        
    Returns
    -------
    float or array
        Mass-to-light ratio bias factor
    """
    return np.power(np.maximum(gamma_t, 0.01), n)


def compute_effective_time(t_cosmic, gamma_t):
    """
    Compute effective time experienced by stellar populations.
    
    t_eff = t_cosmic × Gamma_t
    
    Parameters
    ----------
    t_cosmic : float or array
        Cosmic time in Gyr
    gamma_t : float or array
        Chronological enhancement factor
        
    Returns
    -------
    float or array
        Effective time in Gyr
    """
    t_eff = t_cosmic * gamma_t
    return np.maximum(t_eff, 0.001)  # Ensure positive


def _cosmic_time_gyr(z, H0=67.4, Om=0.315, OL=0.685):
    """Cosmic time at redshift z in Gyr (flat LCDM)."""
    from scipy import integrate as _integrate
    z = np.atleast_1d(np.asarray(z, dtype=float))
    H0_s = H0 * 1e3 / 3.0857e22  # km/s/Mpc -> 1/s
    results = np.empty_like(z)
    for i, zi in enumerate(z):
        def integrand(zp):
            return 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + OL))
        val, _ = _integrate.quad(integrand, zi, np.inf)
        results[i] = val / H0_s / 3.156e16  # seconds -> Gyr
    return results if len(results) > 1 else results[0]


def compute_t_eff(log_Mh, z, alpha_0=ALPHA_0):
    """
    Compute TEP effective time t_eff = t_cosmic(z) * Gamma_t.

    Parameters
    ----------
    log_Mh : array
        Log10 halo mass
    z : array
        Redshift
    alpha_0 : float
        Coupling constant

    Returns
    -------
    array
        Effective time in Gyr
    """
    gamma_t = compute_gamma_t(log_Mh, z, alpha_0)
    t_cosmic = _cosmic_time_gyr(z)
    return np.maximum(t_cosmic * gamma_t, 0.001)


def correct_age_ratio(age_ratio, gamma_t):
    """
    Apply TEP correction to observed age ratio.
    
    age_ratio_corrected = age_ratio / Gamma_t
    
    Parameters
    ----------
    age_ratio : float or array
        Observed age ratio (t_stellar / t_cosmic)
    gamma_t : float or array
        Chronological enhancement factor
        
    Returns
    -------
    float or array
        TEP-corrected age ratio
    """
    return age_ratio / gamma_t


def correct_stellar_mass(log_Mstar, gamma_t, n=0.7):
    """
    Apply TEP correction to observed stellar mass.
    
    log_Mstar_true = log_Mstar - n * log10(Gamma_t)
    
    Parameters
    ----------
    log_Mstar : float or array
        Observed log stellar mass
    gamma_t : float or array
        Chronological enhancement factor
    n : float
        M/L power-law index (default: 0.7)
        
    Returns
    -------
    float or array
        TEP-corrected log stellar mass
    """
    ml_bias = isochrony_mass_bias(gamma_t, n=n)
    return log_Mstar - np.log10(ml_bias)
