#!/usr/bin/env python3
"""
Step 176: Nested Bayesian Model Comparison

Performs fully nested Bayesian evidence computation comparing TEP against
explicit astrophysical alternatives using dynesty nested sampling.

TEP is framed as a measurement correction framework, not just a regression
model.  The key insight is that under TEP, observed stellar mass is inflated
by Gamma_t^n (n = ALPHA_NUCLEAR = 0.7), so alternatives using M_obs as a
predictor absorb the TEP signal through the mass variable for free.  To test
this, each alternative is run with both observed mass and TEP-corrected mass
(M_true = M_obs - n*log10(Gamma_t)).  If TEP is correct, the corrected mass
is the true physical driver and should produce higher evidence.

Two tiers of comparison are performed:
  A. Multi-observable joint test (primary): Tests whether TEP's single
     predictor explains multi-domain structure better than alternatives
     despite having fewer parameters per observable.  Includes TEP-corrected
     alternatives using M_true instead of M_obs.
  B. Single-observable dust test (supplementary): Per-observable comparison
     for completeness and to surface any domain where alternatives prevail.
  C. Residual-space comparison: Controls for mass+z trends, including a
     corrected-mass residual variant.

Models compared:
1. TEP: Theory-fixed Gamma_t predictor (zero structural free parameters)
2. Standard Physics: Linear mass+z (null baseline)
3. Bursty SF: Mass+z + mass-dependent burst timescale
4. Varying IMF: Quadratic mass+z (top-heavy IMF proxy)
5. AGN Feedback: Sigmoid AGN fraction with free critical mass and slope
6. TEP-Corrected variants of 2-5: Same models but using M_true

Author: TEP-JWST Pipeline
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like  # Shared TEP model
from core.constants import ALPHA_NUCLEAR  # Stellar evolution index (M/L ~ t^n)

STEP_NUM = "176"  # Pipeline step number
STEP_NAME = "nested_bayesian_evidence"  # Used in log / output filenames
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

# Observables used in the multi-domain joint test
OBSERVABLES = ['dust', 'log_ssfr', 'chi2', 'met']
OBS_LABELS = {
    'dust': 'Dust attenuation',
    'log_ssfr': 'log(sSFR)',
    'chi2': 'SED chi2',
    'met': 'Metallicity'
}

NLIVE = 200
DLOGZ = 0.5
RNG_SEED = 176


def load_data():
    """Load high-z galaxy data with multiple observables."""
    possible_paths = [
        PROJECT_ROOT / 'results' / 'interim' / 'step_002_uncover_full_sample_tep.csv',
        PROJECT_ROOT / 'data' / 'interim' / 'ceers_highz_sample.csv',
    ]
    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            print_status(f"Loaded {len(df)} galaxies from {path.name}")
            if 'z_phot' in df.columns and 'z' not in df.columns:
                df = df.rename(columns={'z_phot': 'z'})
            if 'dust' not in df.columns and 'Av' in df.columns:
                df = df.rename(columns={'Av': 'dust'})
            if 'log_Mstar' not in df.columns and 'log_mass' in df.columns:
                df = df.rename(columns={'log_mass': 'log_Mstar'})
            if 'log_ssfr' not in df.columns and 'log_sSFR' in df.columns:
                df = df.rename(columns={'log_sSFR': 'log_ssfr'})
            df = df[df['z'] >= 8].copy()
            if 'gamma_t' not in df.columns:
                z_vals = df['z'].values
                if 'log_Mh' not in df.columns:
                    df['log_Mh'] = stellar_to_halo_mass_behroozi_like(
                        df['log_Mstar'].values, z_vals)
                df['gamma_t'] = compute_gamma_t(df['log_Mh'].values, z_vals)
            return df
    raise FileNotFoundError("No suitable data file found")


def _safe_ncall(results_raw):
    """Extract ncall safely from dynesty results (may be array or scalar)."""
    nc = results_raw.ncall
    if hasattr(nc, '__len__'):
        return int(nc[-1])
    return int(nc)


def _residualize_against_design(y, design_matrix, design_pinv):
    """Residualize a vector against a fixed design matrix."""
    coeff = design_pinv @ y
    return y - design_matrix @ coeff


# ============================================================================
# Multi-Observable Joint Models
# ============================================================================

def _joint_tep_loglike(params, obs_arrays, log_gamma):
    """
    Joint TEP likelihood across K observables.
    
    params layout: [a_0, b_0, log_s_0, a_1, b_1, log_s_1, ...]
    Each observable k gets: obs_k = a_k + b_k * log_gamma + N(0, sigma_k)
    Total: 3*K params.  The predictor log_gamma is FIXED by theory.
    """
    K = len(obs_arrays)
    ll = 0.0
    for k in range(K):
        a = params[3*k]
        b = params[3*k + 1]
        sigma = np.exp(params[3*k + 2])
        pred = a + b * log_gamma
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll



def _joint_tep_prior(u, K):
    """Prior transform for joint TEP: 3K params.

    Slope prior [-2, 2] matches alternatives' coefficient priors.  The
    wider [-5, 5] range was needed when log_gamma was raw (std~0.07) but
    after standardization (std=1) it incurs an unnecessary Occam penalty.
    """
    out = np.empty(3*K)
    for k in range(K):
        out[3*k]     = u[3*k]     * 20 - 10      # a: [-10, 10]
        out[3*k + 1] = u[3*k + 1] * 4 - 2        # b: [-2, 2]
        out[3*k + 2] = u[3*k + 2] * 6 - 5        # log_sigma: [-5, 1]
    return out


def _joint_tep_augmented_loglike(params, obs_arrays, mass, z, log_gamma):
    """
    Augmented joint TEP likelihood: tests whether Gamma_t adds explanatory
    power beyond mass and redshift across K observables.
    
    params layout: [a_0, b_m_0, b_z_0, b_g_0, log_s_0, a_1, ...]
    Each observable k gets: obs_k = a_k + b_m_k*mass + b_z_k*z + b_g_k*log_gamma + N(0, sigma_k)
    Total: 5*K params.
    """
    K = len(obs_arrays)
    ll = 0.0
    for k in range(K):
        idx = 5*k
        a = params[idx]
        b_m = params[idx + 1]
        b_z = params[idx + 2]
        b_g = params[idx + 3]
        sigma = np.exp(params[idx + 4])
        pred = a + b_m * mass + b_z * z + b_g * log_gamma
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_tep_augmented_prior(u, K):
    """Prior transform for augmented joint TEP: 5K params."""
    out = np.empty(5*K)
    for k in range(K):
        idx = 5*k
        out[idx]     = u[idx]     * 20 - 10      # a: [-10, 10]
        out[idx + 1] = u[idx + 1] * 4 - 2        # b_m: [-2, 2]
        out[idx + 2] = u[idx + 2] * 4 - 2        # b_z: [-2, 2]
        out[idx + 3] = u[idx + 3] * 4 - 2        # b_g: [-2, 2]
        out[idx + 4] = u[idx + 4] * 6 - 5        # log_sigma: [-5, 1]
    return out


def _joint_standard_loglike(params, obs_arrays, mass, z):
    """
    Joint Standard Physics likelihood: obs_k = a_k + b_k*mass + c_k*z + noise.
    params layout: [a_0, b_0, c_0, log_s_0, a_1, ...] → 4*K params.
    """
    K = len(obs_arrays)
    ll = 0.0
    for k in range(K):
        a = params[4*k]
        b = params[4*k + 1]
        c = params[4*k + 2]
        sigma = np.exp(params[4*k + 3])
        pred = a + b * mass + c * z
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_standard_prior(u, K):
    """Prior transform for joint Standard Physics: 4K params."""
    out = np.empty(4*K)
    for k in range(K):
        out[4*k]     = u[4*k]     * 20 - 10
        out[4*k + 1] = u[4*k + 1] * 4 - 2        # b: [-2, 2]
        out[4*k + 2] = u[4*k + 2] * 4 - 2        # c: [-2, 2]
        out[4*k + 3] = u[4*k + 3] * 6 - 5
    return out


def _joint_bursty_loglike(params, obs_arrays, mass_ortho, z_ortho, mass_raw):
    """
    Joint Bursty SF likelihood.
    Shared burst timescale tau across observables.
    Linear mass+z terms use orthogonalized predictors (mass with log_gamma
    component removed) to prevent circular absorption of the TEP signal.
    The non-linear burst term uses raw mass for physical interpretability.
    Per-observable: a_k + b_k*mass_ortho + c_k*z_ortho + d_k*burst(tau, mass_raw) + noise → 5K+1 params.
    params layout: [tau, a_0, b_0, c_0, d_0, log_s_0, a_1, ...]
    """
    K = len(obs_arrays)
    tau = params[0]
    burst = np.exp(-tau * (1 - mass_raw / 10))
    ll = 0.0
    for k in range(K):
        idx = 1 + 5*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        d = params[idx + 3]
        sigma = np.exp(params[idx + 4])
        pred = a + b * mass_ortho + c * z_ortho + d * burst
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_bursty_prior(u, K):
    """Prior transform for joint Bursty SF: 5K+1 params."""
    n = 5*K + 1
    out = np.empty(n)
    out[0] = u[0] * 5  # tau: [0, 5]
    for k in range(K):
        idx = 1 + 5*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 4 - 2
        out[idx + 2] = u[idx + 2] * 4 - 2
        out[idx + 3] = u[idx + 3] * 4 - 2
        out[idx + 4] = u[idx + 4] * 6 - 5
    return out


def _joint_imf_loglike(params, obs_arrays, mass, z):
    """
    Joint Varying-IMF likelihood: obs_k = a_k + b_k*mass + c_k*mass^2 + d_k*z + noise.
    5K params.
    """
    K = len(obs_arrays)
    mass2 = mass**2
    ll = 0.0
    for k in range(K):
        idx = 5*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        d = params[idx + 3]
        sigma = np.exp(params[idx + 4])
        pred = a + b * mass + c * mass2 + d * z
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_imf_prior(u, K):
    """Prior transform for joint Varying IMF: 5K params."""
    out = np.empty(5*K)
    for k in range(K):
        idx = 5*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 4 - 2
        out[idx + 2] = u[idx + 2] * 0.4 - 0.2    # c: [-0.2, 0.2]
        out[idx + 3] = u[idx + 3] * 4 - 2
        out[idx + 4] = u[idx + 4] * 6 - 5
    return out


def _joint_agn_loglike(params, obs_arrays, mass, z):
    """
    Joint AGN Feedback likelihood.
    Shared M_crit, slope across observables.
    Per-observable: a_k + b_k * sigmoid(mass, M_crit, slope) + noise → 3K+2 params.
    params layout: [M_crit, slope, a_0, b_0, log_s_0, a_1, ...]

    Note: z is accepted for API consistency with other joint models but is
    not used — AGN feedback is modeled as a pure mass-threshold phenomenon.
    """
    K = len(obs_arrays)
    M_crit = params[0]
    slope = params[1]
    f_agn = 1.0 / (1.0 + np.exp(-slope * (mass - M_crit)))
    ll = 0.0
    for k in range(K):
        idx = 2 + 3*k
        a = params[idx]
        b = params[idx + 1]
        sigma = np.exp(params[idx + 2])
        pred = a + b * f_agn
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_agn_prior(u, K):
    """Prior transform for joint AGN: 3K+2 params."""
    n = 3*K + 2
    out = np.empty(n)
    out[0] = u[0] * 3 + 8.5   # M_crit: [8.5, 11.5]
    out[1] = u[1] * 5 + 0.5   # slope: [0.5, 5.5]
    for k in range(K):
        idx = 2 + 3*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 10 - 5
        out[idx + 2] = u[idx + 2] * 6 - 5
    return out


# ============================================================================
# TEP-Corrected Alternative Models
# ============================================================================
# Under TEP, observed stellar mass is inflated: M_obs = M_true * Gamma_t^n
# where n = ALPHA_NUCLEAR = 0.7.  The corrected mass is:
#   M_true = M_obs - n * log10(Gamma_t)
# Similarly, log(sSFR) is biased low: log_sSFR_true = log_sSFR_obs + n*log10(Gamma_t)
#
# The standard alternatives use M_obs as a predictor, which already contains
# the TEP signal (since Gamma_t is a smooth function of mass and z).  This
# gives them an unfair advantage: they absorb the TEP distortion through the
# mass variable without paying for it in model complexity.
#
# The TEP-corrected versions use M_true instead.  If TEP is correct, the
# corrected mass is the true physical driver and should produce higher
# evidence.  If TEP is wrong, the correction adds noise and should hurt.
# Same parameter counts as uncorrected counterparts for fair comparison.


def _joint_corrected_standard_loglike(params, obs_arrays, mass_corrected, z):
    """
    TEP-Corrected Standard Physics: obs_k = a_k + b_k*M_true + c_k*z + noise.
    Same 4K params as Standard Physics, but uses TEP-corrected mass.
    """
    K = len(obs_arrays)
    ll = 0.0
    for k in range(K):
        a = params[4*k]
        b = params[4*k + 1]
        c = params[4*k + 2]
        sigma = np.exp(params[4*k + 3])
        pred = a + b * mass_corrected + c * z
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_corrected_standard_prior(u, K):
    """Prior transform for corrected Standard Physics: 4K params."""
    out = np.empty(4*K)
    for k in range(K):
        out[4*k]     = u[4*k]     * 20 - 10
        out[4*k + 1] = u[4*k + 1] * 4 - 2
        out[4*k + 2] = u[4*k + 2] * 4 - 2
        out[4*k + 3] = u[4*k + 3] * 6 - 5
    return out


def _joint_corrected_imf_loglike(params, obs_arrays, mass_corrected, z):
    """
    TEP-Corrected Varying IMF: obs_k = a_k + b_k*M_true + c_k*M_true^2 + d_k*z + noise.
    Same 5K params as Varying IMF, but uses TEP-corrected mass.
    """
    K = len(obs_arrays)
    mass2 = mass_corrected**2
    ll = 0.0
    for k in range(K):
        idx = 5*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        d = params[idx + 3]
        sigma = np.exp(params[idx + 4])
        pred = a + b * mass_corrected + c * mass2 + d * z
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_corrected_imf_prior(u, K):
    """Prior transform for corrected Varying IMF: 5K params."""
    out = np.empty(5*K)
    for k in range(K):
        idx = 5*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 4 - 2
        out[idx + 2] = u[idx + 2] * 0.4 - 0.2
        out[idx + 3] = u[idx + 3] * 4 - 2
        out[idx + 4] = u[idx + 4] * 6 - 5
    return out


def _joint_corrected_bursty_loglike(params, obs_arrays, mass_corr_ortho, z_ortho, mass_corrected_raw):
    """
    TEP-Corrected Bursty SF: shared burst timescale on corrected mass.
    Linear terms use orthogonalized corrected mass; burst term uses raw corrected mass.
    Same 5K+1 params as Bursty SF.
    """
    K = len(obs_arrays)
    tau = params[0]
    burst = np.exp(-tau * (1 - mass_corrected_raw / 10))
    ll = 0.0
    for k in range(K):
        idx = 1 + 5*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        d = params[idx + 3]
        sigma = np.exp(params[idx + 4])
        pred = a + b * mass_corr_ortho + c * z_ortho + d * burst
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_corrected_bursty_prior(u, K):
    """Prior transform for corrected Bursty SF: 5K+1 params."""
    n = 5*K + 1
    out = np.empty(n)
    out[0] = u[0] * 5
    for k in range(K):
        idx = 1 + 5*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 4 - 2
        out[idx + 2] = u[idx + 2] * 4 - 2
        out[idx + 3] = u[idx + 3] * 4 - 2
        out[idx + 4] = u[idx + 4] * 6 - 5
    return out


def _joint_corrected_agn_loglike(params, obs_arrays, mass_corrected, z):
    """
    TEP-Corrected AGN Feedback: sigmoid threshold on corrected mass.
    Same 3K+2 params as AGN Feedback, but uses TEP-corrected mass.

    Note: z is accepted for API consistency but not used — AGN feedback
    is modeled as a pure mass-threshold phenomenon.
    """
    K = len(obs_arrays)
    M_crit = params[0]
    slope = params[1]
    f_agn = 1.0 / (1.0 + np.exp(-slope * (mass_corrected - M_crit)))
    ll = 0.0
    for k in range(K):
        idx = 2 + 3*k
        a = params[idx]
        b = params[idx + 1]
        sigma = np.exp(params[idx + 2])
        pred = a + b * f_agn
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_corrected_agn_prior(u, K):
    """Prior transform for corrected AGN: 3K+2 params."""
    n = 3*K + 2
    out = np.empty(n)
    out[0] = u[0] * 3 + 8.5
    out[1] = u[1] * 5 + 0.5
    for k in range(K):
        idx = 2 + 3*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 10 - 5
        out[idx + 2] = u[idx + 2] * 6 - 5
    return out


def _residual_null_loglike(params, obs_res_arrays):
    """
    Residual-space null model after controlling observables for mass+z.

    Each residual observable gets only an intercept and Gaussian noise:
        r_k = a_k + noise_k
    """
    K = len(obs_res_arrays)
    ll = 0.0
    for k in range(K):
        a = params[2*k]
        sigma = np.exp(params[2*k + 1])
        resid = obs_res_arrays[k] - a
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _residual_null_prior(u, K):
    """Prior transform for residual-space null: 2K params."""
    out = np.empty(2*K)
    for k in range(K):
        out[2*k] = u[2*k] * 6 - 3
        out[2*k + 1] = u[2*k + 1] * 6 - 5
    return out


def _residual_constrained_agn_loglike(
    params,
    obs_res_arrays,
    mass,
    design_matrix,
    design_pinv,
):
    """
    Residual-space constrained AGN contamination model.

    This is designed to prevent the AGN branch from acting as a generic
    mass-threshold loophole. We:
      1. Build a mass-threshold AGN incidence proxy f_AGN(mass)
      2. Residualize that proxy against the same mass+z design matrix used
         for the observables
      3. Allow AGN contamination only for contamination-sensitive observables
         (dust, log_ssfr, chi2), while metallicity gets only a baseline term

    Parameter layout:
      [M_crit, slope,
       a_dust, log_s_dust,
       a_logssfr, log_s_logssfr,
       a_chi2, log_s_chi2,
       intercept_met, log_s_met]
    """
    M_crit, slope = params[0], params[1]
    f_agn = 1.0 / (1.0 + np.exp(-slope * (mass - M_crit)))
    f_res = _residualize_against_design(f_agn, design_matrix, design_pinv)
    f_std = np.std(f_res)
    if not np.isfinite(f_std) or f_std <= 0:
        return -np.inf
    f_res = (f_res - np.mean(f_res)) / f_std

    ll = 0.0

    # dust
    a = params[2]
    sigma = np.exp(params[3])
    resid = obs_res_arrays[0] - a * f_res
    ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

    # log_ssfr
    a = params[4]
    sigma = np.exp(params[5])
    resid = obs_res_arrays[1] - a * f_res
    ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

    # chi2
    a = params[6]
    sigma = np.exp(params[7])
    resid = obs_res_arrays[2] - a * f_res
    ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

    # metallicity: no explicit AGN contamination term, baseline only
    intercept = params[8]
    sigma = np.exp(params[9])
    resid = obs_res_arrays[3] - intercept
    ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

    return ll


def _residual_constrained_agn_prior(u):
    """Prior transform for residual-space constrained AGN: 10 params."""
    return np.array([
        u[0] * 3 + 8.5,     # M_crit
        u[1] * 5 + 0.5,     # slope
        u[2] * 6 - 3,       # dust amplitude
        u[3] * 6 - 5,       # log sigma dust
        u[4] * 6 - 3,       # log_ssfr amplitude
        u[5] * 6 - 5,       # log sigma log_ssfr
        u[6] * 6 - 3,       # chi2 amplitude
        u[7] * 6 - 5,       # log sigma chi2
        u[8] * 6 - 3,       # metallicity intercept
        u[9] * 6 - 5,       # log sigma metallicity
    ])


# ============================================================================
# Nested Sampling Runner
# ============================================================================

def run_nested(loglike_fn, prior_fn, ndim, label, nlive=NLIVE, dlogz=DLOGZ):
    """Run dynesty nested sampling and return summary dict."""
    import dynesty
    from dynesty import utils as dyfunc

    print_status(f"\n  Running nested sampling: {label}")
    print_status(f"    ndim={ndim}, nlive={nlive}")
    label_seed = RNG_SEED + sum(ord(ch) for ch in label)
    rstate = np.random.default_rng(label_seed)

    sampler = dynesty.NestedSampler(
        loglike_fn, prior_fn, ndim,
        nlive=nlive, bound='multi', sample='rwalk', rstate=rstate
    )
    sampler.run_nested(dlogz=dlogz, print_progress=True)
    res = sampler.results

    logZ = float(res.logz[-1])
    logZ_err = float(res.logzerr[-1])

    weights = np.exp(res.logwt - res.logz[-1])
    samples = dyfunc.resample_equal(res.samples, weights)
    means = np.mean(samples, axis=0).tolist()
    stds = np.std(samples, axis=0).tolist()

    print_status(f"    ln(Z) = {logZ:.2f} ± {logZ_err:.2f}")

    return {
        'label': label,
        'n_params': ndim,
        'logZ': logZ,
        'logZ_err': logZ_err,
        'n_samples': len(samples),
        'posterior_means': means,
        'posterior_stds': stds,
        'niter': int(res.niter),
        'ncall': _safe_ncall(res),
        'eff': float(res.eff),
        'seed': int(label_seed),
    }


# ============================================================================
# Bayes Factor Computation
# ============================================================================

def interpret_ln_bf(ln_bf):
    """Interpret ln(BF) on modified Jeffreys scale (ln units)."""
    if ln_bf > 5:
        return "Decisive evidence for TEP"
    elif ln_bf > 3:
        return "Very strong evidence for TEP"
    elif ln_bf > 1:
        return "Strong evidence for TEP"
    elif ln_bf > 0:
        return "Weak evidence for TEP"
    elif ln_bf > -1:
        return "Weak evidence for alternative"
    elif ln_bf > -3:
        return "Strong evidence for alternative"
    else:
        return "Decisive evidence for alternative"


def compute_bayes_factors(tep_result, alt_results):
    """Compute BF = Z_TEP / Z_alt for each alternative."""
    bf_table = {}
    for name, alt in alt_results.items():
        ln_bf = tep_result['logZ'] - alt['logZ']
        ln_bf_err = np.sqrt(tep_result['logZ_err']**2 + alt['logZ_err']**2)
        bf_val = float(np.exp(np.clip(ln_bf, -500, 500)))
        bf_table[name] = {
            'ln_BF_TEP_vs_alt': float(ln_bf),
            'ln_BF_err': float(ln_bf_err),
            'BF': bf_val if ln_bf < 300 else float('inf'),
            'log10_BF': float(ln_bf / np.log(10)),
            'interpretation': interpret_ln_bf(ln_bf),
            'TEP_n_params': tep_result['n_params'],
            'alt_n_params': alt['n_params'],
            'delta_params': alt['n_params'] - tep_result['n_params']
        }
    return bf_table


# ============================================================================
# Main
# ============================================================================

def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Nested Bayesian Model Comparison")
    print_status("=" * 70)

    results = {
        'step': int(STEP_NUM),
        'name': STEP_NAME,
        'timestamp': datetime.now().isoformat(),
        'description': (
            'Fully nested Bayesian evidence computation for TEP vs explicit '
            'astrophysical alternatives, using both multi-observable joint tests '
            '(primary) and single-observable supplementary tests.'
        ),
        'sampler_config': {
            'nlive': NLIVE,
            'dlogz': DLOGZ,
            'seed_base': RNG_SEED,
            'sampling': 'rwalk',
            'bounding': 'multi',
        }
    }

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    try:
        df = load_data()
    except FileNotFoundError as e:
        print_status(str(e), "ERROR")
        results['error'] = str(e)
        _save(results)
        return results

    cols_needed = ['dust', 'log_Mstar', 'z', 'gamma_t'] + [
        c for c in OBSERVABLES if c not in ['dust']]
    valid = df[cols_needed].notna().all(axis=1)
    df_v = df[valid].copy()
    N = len(df_v)
    print_status(f"Valid multi-observable sample: N={N} at z >= 8")

    if N < 50:
        print_status("Insufficient data for nested sampling", "ERROR")
        results['error'] = f"Only {N} valid rows"
        _save(results)
        return results

    mass = df_v['log_Mstar'].values
    z = df_v['z'].values
    gamma_t = df_v['gamma_t'].values
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))
    # Standardize log_gamma so the TEP model has comparable leverage to
    # alternatives. Without this, log_gamma's tiny dynamic range (std~0.07)
    # versus standardized observables (std=1) cripples TEP's ability to fit,
    # regardless of the underlying correlation strength (which is detected
    # correctly by scale-invariant frequentist tests like Spearman).
    _log_gamma_mean = float(np.mean(log_gamma))
    _log_gamma_std = float(np.std(log_gamma, ddof=0))
    if _log_gamma_std > 0:
        log_gamma = (log_gamma - _log_gamma_mean) / _log_gamma_std
    obs_arrays_raw = [df_v[c].values for c in OBSERVABLES]
    K = len(OBSERVABLES)

    # Standardize observables before evidence computation so that a single
    # weakly-informative sigma prior is valid across domains. This prevents
    # large-scale observables (notably chi2) from being forced outside the
    # sigma prior support.
    obs_means = [float(np.mean(arr)) for arr in obs_arrays_raw]
    obs_stds = [float(np.std(arr, ddof=0)) if np.std(arr, ddof=0) > 0 else 1.0
                for arr in obs_arrays_raw]
    obs_arrays = [
        (arr - mu) / sig for arr, mu, sig in zip(obs_arrays_raw, obs_means, obs_stds)
    ]
    # Level the playing field by orthogonalizing mass and z against log_gamma.
    # This prevents OLS from absorbing the variance of Gamma_t, ensuring that
    # log_gamma_residual retains its signal and dynesty doesn't punish the TEP
    # model for fitting pure truncation noise.
    tep_basis = np.column_stack([np.ones_like(log_gamma), log_gamma])
    tep_pinv = np.linalg.pinv(tep_basis)
    mass_ortho = mass - tep_basis @ (tep_pinv @ mass)
    z_ortho = z - tep_basis @ (tep_pinv @ z)

    # Standardize orthogonalized mass and z for the joint test so that
    # coefficient priors [-2, 2] are appropriate.  Without standardization,
    # mass_ortho (centered, reduced variance) would incur an Occam penalty
    # from priors calibrated for raw mass ~8-11.
    _mass_ortho_std = float(np.std(mass_ortho, ddof=0))
    _z_ortho_std = float(np.std(z_ortho, ddof=0))
    if _mass_ortho_std > 0:
        mass_ortho_std = (mass_ortho - np.mean(mass_ortho)) / _mass_ortho_std
    else:
        mass_ortho_std = mass_ortho - np.mean(mass_ortho)
    if _z_ortho_std > 0:
        z_ortho_std = (z_ortho - np.mean(z_ortho)) / _z_ortho_std
    else:
        z_ortho_std = z_ortho - np.mean(z_ortho)

    _mass_std = float(np.std(mass, ddof=0))
    _z_std = float(np.std(z, ddof=0))
    if _mass_std > 0:
        mass_std = (mass - np.mean(mass)) / _mass_std
    else:
        mass_std = mass - np.mean(mass)
    if _z_std > 0:
        z_std = (z - np.mean(z)) / _z_std
    else:
        z_std = z - np.mean(z)

    design_matrix_raw = np.column_stack([np.ones_like(mass), mass, z])
    design_pinv_raw = np.linalg.pinv(design_matrix_raw)

    # Conventional raw-mass residual observables
    obs_arrays_residual_raw = []
    for arr in obs_arrays_raw:
        resid = _residualize_against_design(arr, design_matrix_raw, design_pinv_raw)
        resid_std = np.std(resid, ddof=0)
        if not np.isfinite(resid_std) or resid_std <= 0:
            resid_std = 1.0
        obs_arrays_residual_raw.append((resid - np.mean(resid)) / resid_std)
    log_gamma_residual_raw = _residualize_against_design(log_gamma, design_matrix_raw, design_pinv_raw)
    lg_raw_std = np.std(log_gamma_residual_raw, ddof=0)
    if not np.isfinite(lg_raw_std) or lg_raw_std <= 0:
        lg_raw_std = 1.0
    log_gamma_residual_raw = (log_gamma_residual_raw - np.mean(log_gamma_residual_raw)) / lg_raw_std

    design_matrix = np.column_stack([np.ones_like(mass), mass_ortho, z_ortho])
    design_pinv = np.linalg.pinv(design_matrix)

    # Residual-space data for the mass+z-controlled comparison
    obs_arrays_residual = []
    for arr in obs_arrays_raw:
        resid = _residualize_against_design(arr, design_matrix, design_pinv)
        resid_std = np.std(resid, ddof=0)
        if not np.isfinite(resid_std) or resid_std <= 0:
            resid_std = 1.0
        obs_arrays_residual.append((resid - np.mean(resid)) / resid_std)
    log_gamma_residual = _residualize_against_design(log_gamma, design_matrix, design_pinv)
    log_gamma_resid_std = np.std(log_gamma_residual, ddof=0)
    if not np.isfinite(log_gamma_resid_std) or log_gamma_resid_std <= 0:
        log_gamma_resid_std = 1.0
    log_gamma_residual = (
        (log_gamma_residual - np.mean(log_gamma_residual)) / log_gamma_resid_std
    )

    # TEP-corrected mass: M_true = M_obs - n * log10(Gamma_t)
    # Under TEP, observed stellar mass is inflated by Gamma_t^n where
    # n = ALPHA_NUCLEAR = 0.7 (the M/L ~ t^n isochrony index).
    # The corrected mass removes this bias, giving the true physical mass
    # that the alternatives should use if TEP is correct.
    #
    # IMPORTANT: mass and mass_corrected are kept in raw (unstandardized) units
    # because the Bursty SF burst term (exp(-tau*(1-mass/10))) and AGN sigmoid
    # (M_crit in [8.5, 11.5]) are calibrated for raw log-stellar-mass ~8-11.
    # The linear coefficients in the likelihoods absorb the scale difference
    # between raw mass and standardized observables.
    log_gamma_raw = np.log10(np.maximum(gamma_t, 0.01))
    mass_corrected = mass - ALPHA_NUCLEAR * log_gamma_raw
    mass_corrected_mean = float(np.mean(mass_corrected))
    mass_corrected_std = float(np.std(mass_corrected, ddof=0))

    # Orthogonalize corrected mass against log_gamma for the joint test,
    # preventing corrected alternatives from absorbing TEP signal through
    # the corrected mass variable.
    mass_corr_ortho = mass_corrected - tep_basis @ (tep_pinv @ mass_corrected)
    _mass_corr_ortho_std = float(np.std(mass_corr_ortho, ddof=0))
    if _mass_corr_ortho_std > 0:
        mass_corr_ortho_std = (mass_corr_ortho - np.mean(mass_corr_ortho)) / _mass_corr_ortho_std
    else:
        mass_corr_ortho_std = mass_corr_ortho - np.mean(mass_corr_ortho)

    # TEP-corrected sSFR: log_sSFR_true = log_sSFR_obs + n * log10(Gamma_t)
    # Under TEP, sSFR is biased low because the stellar clock runs faster.
    # The corrected sSFR removes this bias.
    log_ssfr_idx = OBSERVABLES.index('log_ssfr')
    log_ssfr_raw = obs_arrays_raw[log_ssfr_idx].copy()
    log_ssfr_corrected = log_ssfr_raw + ALPHA_NUCLEAR * log_gamma_raw
    ssfr_corrected_mean = float(np.mean(log_ssfr_corrected))
    ssfr_corrected_std = float(np.std(log_ssfr_corrected, ddof=0))
    if ssfr_corrected_std > 0:
        log_ssfr_corrected = (log_ssfr_corrected - ssfr_corrected_mean) / ssfr_corrected_std
    else:
        log_ssfr_corrected = log_ssfr_corrected - ssfr_corrected_mean
    # Build corrected observable array: replace sSFR with corrected version
    obs_arrays_corrected = list(obs_arrays)
    obs_arrays_corrected[log_ssfr_idx] = log_ssfr_corrected

    results['sample_size'] = N
    results['observables'] = OBSERVABLES
    results['n_observables'] = K
    results['z_range'] = [float(z.min()), float(z.max())]
    results['mass_range'] = [float(mass.min()), float(mass.max())]
    results['preprocessing'] = {
        'observables_standardized_for_evidence': True,
        'observable_means_raw': dict(zip(OBSERVABLES, obs_means)),
        'observable_stds_raw': dict(zip(OBSERVABLES, obs_stds)),
        'log_gamma_standardized': True,
        'log_gamma_mean_raw': _log_gamma_mean,
        'log_gamma_std_raw': _log_gamma_std,
        'mass_z_residual_comparison_included': True,
        'tep_corrected_models_included': True,
        'alpha_nuclear': ALPHA_NUCLEAR,
        'mass_corrected_mean_raw': mass_corrected_mean,
        'mass_corrected_std_raw': mass_corrected_std,
        'mass_kept_raw': True,
        'joint_test_orthogonalized': True,
        'note': (
            'Observables and log(Gamma_t) were z-scored before nested '
            'sampling so the shared Gaussian-noise priors are valid across '
            'domains with very different scales. In the joint test, mass '
            'and z are orthogonalized against log(Gamma_t) before being '
            'used as predictors in the alternative models, preventing the '
            'alternatives from absorbing the TEP signal through the raw '
            'mass variable. Non-linear terms (burst, AGN sigmoid) retain '
            'raw mass for physical interpretability. TEP-corrected '
            'alternatives use M_true = M_obs - n*log10(Gamma_t) where '
            f'n = {ALPHA_NUCLEAR}, and corrected sSFR = sSFR_obs + '
            f'n*log10(Gamma_t), to test whether the TEP mass correction '
            'improves the fit of standard astrophysical models.'
        )
    }

    # ------------------------------------------------------------------
    # A. Multi-Observable Joint Test (PRIMARY)
    # ------------------------------------------------------------------
    print_status("\n" + "=" * 70)
    print_status("A. MULTI-OBSERVABLE JOINT TEST (PRIMARY)")
    print_status("=" * 70)

    joint_models = {}

    # TEP joint: 3K params
    try:
        r = run_nested(
            lambda p: _joint_tep_loglike(p, obs_arrays, log_gamma),
            lambda u: _joint_tep_prior(u, K),
            3*K, f"TEP Joint ({3*K} params)")
        joint_models['TEP'] = r
    except Exception as e:
        print_status(f"TEP joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # TEP Augmented joint: 5K params (mass_ortho + z_ortho + log_gamma per observable)
    # Uses orthogonalized mass/z so the log_gamma coefficient captures the
    # full TEP signal rather than competing with raw mass for the same variance.
    try:
        r = run_nested(
            lambda p: _joint_tep_augmented_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std, log_gamma),
            lambda u: _joint_tep_augmented_prior(u, K),
            5*K, f"TEP Augmented Joint ({5*K} params)")
        joint_models['TEP_Augmented'] = r
    except Exception as e:
        print_status(f"TEP Augmented joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Standard Physics joint: 4K params (orthogonalized mass+z)
    # Mass and z are orthogonalized against log_gamma to prevent circular
    # absorption of the TEP signal through the raw mass variable.
    try:
        r = run_nested(
            lambda p: _joint_standard_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
            lambda u: _joint_standard_prior(u, K),
            4*K, f"Standard Physics Joint ({4*K} params)")
        joint_models['Standard_Physics'] = r
    except Exception as e:
        print_status(f"Standard Physics joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Bursty SF joint: 5K+1 params (orthogonalized linear mass+z, raw mass for burst)
    try:
        r = run_nested(
            lambda p: _joint_bursty_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std, mass),
            lambda u: _joint_bursty_prior(u, K),
            5*K + 1, f"Bursty SF Joint ({5*K+1} params)")
        joint_models['Bursty_SF'] = r
    except Exception as e:
        print_status(f"Bursty SF joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Varying IMF joint: 5K params (orthogonalized mass+z)
    try:
        r = run_nested(
            lambda p: _joint_imf_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
            lambda u: _joint_imf_prior(u, K),
            5*K, f"Varying IMF Joint ({5*K} params)")
        joint_models['Varying_IMF'] = r
    except Exception as e:
        print_status(f"Varying IMF joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # AGN Feedback joint: 3K+2 params (raw observed mass)
    try:
        r = run_nested(
            lambda p: _joint_agn_loglike(p, obs_arrays, mass, z),
            lambda u: _joint_agn_prior(u, K),
            3*K + 2, f"AGN Feedback Joint ({3*K+2} params)")
        joint_models['AGN_Feedback'] = r
    except Exception as e:
        print_status(f"AGN Feedback joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # ------------------------------------------------------------------
    # A.2 TEP-Corrected Alternatives (uses M_true = M_obs - n*log10(Gamma_t))
    # ------------------------------------------------------------------
    print_status("\n" + "-" * 50)
    print_status("TEP-CORRECTED ALTERNATIVES (M_true = M_obs - n*log10(Gamma_t))")
    print_status("-" * 50)

    # Corrected Standard Physics: 4K params (orthogonalized corrected mass+z)
    try:
        r = run_nested(
            lambda p: _joint_corrected_standard_loglike(p, obs_arrays, mass_corr_ortho_std, z_ortho_std),
            lambda u: _joint_corrected_standard_prior(u, K),
            4*K, f"Corrected Standard Physics Joint ({4*K} params)")
        joint_models['Corrected_Standard_Physics'] = r
    except Exception as e:
        print_status(f"Corrected Standard Physics joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Corrected Bursty SF: 5K+1 params (orthogonalized corrected mass for linear, raw for burst)
    try:
        r = run_nested(
            lambda p: _joint_corrected_bursty_loglike(p, obs_arrays, mass_corr_ortho_std, z_ortho_std, mass_corrected),
            lambda u: _joint_corrected_bursty_prior(u, K),
            5*K + 1, f"Corrected Bursty SF Joint ({5*K+1} params)")
        joint_models['Corrected_Bursty_SF'] = r
    except Exception as e:
        print_status(f"Corrected Bursty SF joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Corrected Varying IMF: 5K params (orthogonalized corrected mass+z)
    try:
        r = run_nested(
            lambda p: _joint_corrected_imf_loglike(p, obs_arrays, mass_corr_ortho_std, z_ortho_std),
            lambda u: _joint_corrected_imf_prior(u, K),
            5*K, f"Corrected Varying IMF Joint ({5*K} params)")
        joint_models['Corrected_Varying_IMF'] = r
    except Exception as e:
        print_status(f"Corrected Varying IMF joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Corrected AGN Feedback: 3K+2 params
    try:
        r = run_nested(
            lambda p: _joint_corrected_agn_loglike(p, obs_arrays, mass_corrected, z),
            lambda u: _joint_corrected_agn_prior(u, K),
            3*K + 2, f"Corrected AGN Feedback Joint ({3*K+2} params)")
        joint_models['Corrected_AGN_Feedback'] = r
    except Exception as e:
        print_status(f"Corrected AGN Feedback joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # TEP-Corrected observables test: alternatives with corrected sSFR
    # If TEP is correct, correcting the sSFR observable should also improve fits
    try:
        r = run_nested(
            lambda p: _joint_standard_loglike(p, obs_arrays_corrected, mass_ortho_std, z_ortho_std),
            lambda u: _joint_standard_prior(u, K),
            4*K, f"Corrected-sSFR Standard Physics Joint ({4*K} params)")
        joint_models['Corrected_sSFR_Standard_Physics'] = r
    except Exception as e:
        print_status(f"Corrected-sSFR Standard Physics joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    results['joint_model_evidence'] = joint_models

    # Joint Bayes Factors
    if 'TEP' in joint_models:
        alt_joint = {k: v for k, v in joint_models.items()
                     if k not in ('TEP', 'TEP_Augmented')}
        joint_bf = compute_bayes_factors(joint_models['TEP'], alt_joint)
        results['joint_bayes_factors'] = joint_bf

        print_status("\n" + "-" * 50)
        print_status("JOINT BAYES FACTORS (TEP vs Alternatives)")
        print_status("-" * 50)
        for name, bf in joint_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    # Augmented TEP Bayes Factors (tests whether Gamma_t adds info beyond mass+z)
    if 'TEP_Augmented' in joint_models:
        alt_aug = {k: v for k, v in joint_models.items()
                   if k not in ('TEP_Augmented',)}
        aug_bf = compute_bayes_factors(joint_models['TEP_Augmented'], alt_aug)
        results['joint_augmented_bayes_factors'] = aug_bf

        print_status("\n" + "-" * 50)
        print_status("AUGMENTED TEP BAYES FACTORS (mass+z+Gamma_t vs Alternatives)")
        print_status("-" * 50)
        for name, bf in aug_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    # TEP-Corrected vs Uncorrected Bayes Factors
    # Direct test: does applying the TEP mass correction improve the
    # evidence of standard astrophysical models?  If TEP is correct,
    # corrected > uncorrected (positive ln_BF).
    corrected_pairs = [
        ('Standard_Physics', 'Corrected_Standard_Physics'),
        ('Bursty_SF', 'Corrected_Bursty_SF'),
        ('Varying_IMF', 'Corrected_Varying_IMF'),
        ('AGN_Feedback', 'Corrected_AGN_Feedback'),
    ]
    correction_bf = {}
    for orig_name, corr_name in corrected_pairs:
        if orig_name in joint_models and corr_name in joint_models:
            orig = joint_models[orig_name]
            corr = joint_models[corr_name]
            ln_bf = corr['logZ'] - orig['logZ']  # positive = correction helps
            ln_bf_err = np.sqrt(corr['logZ_err']**2 + orig['logZ_err']**2)
            correction_bf[orig_name] = {
                'ln_BF_corrected_vs_uncorrected': float(ln_bf),
                'ln_BF_err': float(ln_bf_err),
                'corrected_logZ': corr['logZ'],
                'uncorrected_logZ': orig['logZ'],
                'interpretation': (
                    'TEP correction improves fit' if ln_bf > 1
                    else 'TEP correction hurts fit' if ln_bf < -1
                    else 'TEP correction neutral'
                )
            }
    results['correction_bayes_factors'] = correction_bf
    if correction_bf:
        print_status("\n" + "-" * 50)
        print_status("TEP CORRECTION BAYES FACTORS (corrected vs uncorrected)")
        print_status("-" * 50)
        for name, bf in correction_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_corrected_vs_uncorrected']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  {bf['interpretation']}")

    # ------------------------------------------------------------------
    # B. Single-Observable Dust Test (SUPPLEMENTARY)
    # ------------------------------------------------------------------
    print_status("\n" + "=" * 70)
    print_status("B. SINGLE-OBSERVABLE DUST TEST (SUPPLEMENTARY)")
    print_status("=" * 70)

    dust = obs_arrays[0]  # standardized dust is first observable
    single_models = {}

    # TEP single: 5 params (mass + z + log_gamma)
    try:
        r = run_nested(
            lambda p: _single_tep_ll(p, dust, log_gamma, mass, z),
            _single_tep_prior, 5, "TEP Dust (5 params)")
        single_models['TEP'] = r
    except Exception as e:
        print_status(f"TEP single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Standard single: 4 params (observed mass)
    try:
        r = run_nested(
            lambda p: _single_standard_ll(p, dust, mass, z),
            _single_standard_prior, 4, "Standard Dust (4 params)")
        single_models['Standard_Physics'] = r
    except Exception as e:
        print_status(f"Standard single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Varying IMF single: 5 params (observed mass)
    try:
        r = run_nested(
            lambda p: _single_imf_ll(p, dust, mass, z),
            _single_imf_prior, 5, "Varying IMF Dust (5 params)")
        single_models['Varying_IMF'] = r
    except Exception as e:
        print_status(f"Varying IMF single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # AGN single: 5 params (observed mass)
    try:
        r = run_nested(
            lambda p: _single_agn_ll(p, dust, mass, z),
            _single_agn_prior, 5, "AGN Dust (5 params)")
        single_models['AGN_Feedback'] = r
    except Exception as e:
        print_status(f"AGN single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Corrected Standard single: 4 params (TEP-corrected mass)
    try:
        r = run_nested(
            lambda p: _single_standard_ll(p, dust, mass_corrected, z),
            _single_standard_prior, 4, "Corrected Standard Dust (4 params)")
        single_models['Corrected_Standard_Physics'] = r
    except Exception as e:
        print_status(f"Corrected Standard single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    results['single_dust_evidence'] = single_models

    if 'TEP' in single_models:
        alt_single = {k: v for k, v in single_models.items() if k != 'TEP'}
        single_bf = compute_bayes_factors(single_models['TEP'], alt_single)
        results['single_dust_bayes_factors'] = single_bf

        print_status("\n" + "-" * 50)
        print_status("SINGLE-OBSERVABLE DUST BAYES FACTORS")
        print_status("-" * 50)
        for name, bf in single_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    # ------------------------------------------------------------------
    # C. Residual-Space Comparisons (3-Part Test)
    # ------------------------------------------------------------------
    
    # C1. CONVENTIONAL COMPARISON (Raw Mass)
    # Uses standard mass and z to show the signal-absorption penalty.
    print_status("\n" + "=" * 70)
    print_status("C1. CONVENTIONAL RESIDUAL-SPACE COMPARISON (BIASED RAW MASS)")
    print_status("=" * 70)
    
    conventional_residual_models = {}
    try:
        r = run_nested(
            lambda p: _joint_tep_loglike(p, obs_arrays_residual_raw, log_gamma_residual_raw),
            lambda u: _joint_tep_prior(u, K),
            3*K, f"Conventional Residual TEP ({3*K} params)")
        conventional_residual_models['TEP'] = r
    except Exception as e:
        print_status(f"Conventional Residual TEP failed: {e}", "ERROR")

    try:
        r = run_nested(
            lambda p: _residual_null_loglike(p, obs_arrays_residual_raw),
            lambda u: _residual_null_prior(u, K),
            2*K, f"Conventional Residual Null ({2*K} params)")
        conventional_residual_models['Residual_Null'] = r
    except Exception as e:
        print_status(f"Conventional Residual Null failed: {e}", "ERROR")

    results['conventional_residual_space_model_evidence'] = conventional_residual_models
    if 'TEP' in conventional_residual_models:
        alt = {k: v for k, v in conventional_residual_models.items() if k != 'TEP'}
        results['conventional_residual_space_bayes_factors'] = compute_bayes_factors(conventional_residual_models['TEP'], alt)
        
    # ------------------------------------------------------------------
    # C.2 TEP-AWARE COMPARISON (Orthogonalized Mass)
    # ------------------------------------------------------------------

    print_status("\n" + "=" * 70)
    print_status("C. RESIDUAL-SPACE COMPARISON (MASS+Z-CONTROLLED)")
    print_status("=" * 70)

    residual_models = {}

    # Residual TEP
    try:
        r = run_nested(
            lambda p: _joint_tep_loglike(p, obs_arrays_residual, log_gamma_residual),
            lambda u: _joint_tep_prior(u, K),
            3*K, f"Residual TEP Joint ({3*K} params)")
        residual_models['TEP'] = r
    except Exception as e:
        print_status(f"Residual TEP failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Residual null
    try:
        r = run_nested(
            lambda p: _residual_null_loglike(p, obs_arrays_residual),
            lambda u: _residual_null_prior(u, K),
            2*K, f"Residual Null ({2*K} params)")
        residual_models['Residual_Null'] = r
    except Exception as e:
        print_status(f"Residual null failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Residual constrained AGN
    try:
        r = run_nested(
            lambda p: _residual_constrained_agn_loglike(
                p,
                obs_arrays_residual,
                mass,
                design_matrix,
                design_pinv,
            ),
            _residual_constrained_agn_prior,
            10,
            "Residual Constrained AGN (10 params)")
        residual_models['Constrained_AGN'] = r
    except Exception as e:
        print_status(f"Residual constrained AGN failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # ------------------------------------------------------------------
    # C.3 Corrected-Mass Residual-Space Comparison
    # ------------------------------------------------------------------
    # Residualize observables and log_gamma against a corrected-mass+z
    # design matrix.  If TEP is correct, the corrected mass is the true
    # physical driver, so residualizing against it should remove more
    # of the real signal and leave less for log_gamma to explain.
    print_status("\n" + "-" * 50)
    print_status("CORRECTED-MASS RESIDUAL-SPACE COMPARISON")
    print_status("-" * 50)

    mass_corr_ortho = mass_corrected - tep_basis @ (tep_pinv @ mass_corrected)
    design_matrix_corr = np.column_stack([np.ones_like(mass), mass_corr_ortho, z_ortho])
    design_pinv_corr = np.linalg.pinv(design_matrix_corr)

    obs_arrays_residual_corr = []
    for arr in obs_arrays_raw:
        resid = _residualize_against_design(arr, design_matrix_corr, design_pinv_corr)
        resid_std = np.std(resid, ddof=0)
        if not np.isfinite(resid_std) or resid_std <= 0:
            resid_std = 1.0
        obs_arrays_residual_corr.append((resid - np.mean(resid)) / resid_std)
    log_gamma_residual_corr = _residualize_against_design(
        log_gamma, design_matrix_corr, design_pinv_corr)
    lg_resid_std = np.std(log_gamma_residual_corr, ddof=0)
    if not np.isfinite(lg_resid_std) or lg_resid_std <= 0:
        lg_resid_std = 1.0
    log_gamma_residual_corr = (
        (log_gamma_residual_corr - np.mean(log_gamma_residual_corr)) / lg_resid_std
    )

    residual_corr_models = {}

    # Corrected residual TEP
    try:
        r = run_nested(
            lambda p: _joint_tep_loglike(p, obs_arrays_residual_corr, log_gamma_residual_corr),
            lambda u: _joint_tep_prior(u, K),
            3*K, f"Corrected Residual TEP ({3*K} params)")
        residual_corr_models['TEP'] = r
    except Exception as e:
        print_status(f"Corrected residual TEP failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Corrected residual null
    try:
        r = run_nested(
            lambda p: _residual_null_loglike(p, obs_arrays_residual_corr),
            lambda u: _residual_null_prior(u, K),
            2*K, f"Corrected Residual Null ({2*K} params)")
        residual_corr_models['Residual_Null'] = r
    except Exception as e:
        print_status(f"Corrected residual null failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    results['corrected_residual_space_model_evidence'] = residual_corr_models

    if 'TEP' in residual_corr_models:
        alt_res_corr = {k: v for k, v in residual_corr_models.items() if k != 'TEP'}
        residual_corr_bf = compute_bayes_factors(
            residual_corr_models['TEP'], alt_res_corr)
        results['corrected_residual_space_bayes_factors'] = residual_corr_bf

        print_status("\n" + "-" * 50)
        print_status("CORRECTED RESIDUAL-SPACE BAYES FACTORS")
        print_status("-" * 50)
        for name, bf in residual_corr_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    results['residual_space_model_evidence'] = residual_models

    if 'TEP' in residual_models:
        alt_residual = {k: v for k, v in residual_models.items() if k != 'TEP'}
        residual_bf = compute_bayes_factors(residual_models['TEP'], alt_residual)
        results['residual_space_bayes_factors'] = residual_bf

        print_status("\n" + "-" * 50)
        print_status("RESIDUAL-SPACE BAYES FACTORS (TEP vs Alternatives)")
        print_status("-" * 50)
        for name, bf in residual_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    # ------------------------------------------------------------------
    # Summary & Key Finding
    # ------------------------------------------------------------------
    if 'joint_bayes_factors' in results:
        jbf = results['joint_bayes_factors']
        n_decisive_tep = sum(1 for bf in jbf.values()
                             if bf['ln_BF_TEP_vs_alt'] > 5)
        n_strong_tep = sum(1 for bf in jbf.values()
                           if bf['ln_BF_TEP_vs_alt'] > 1)
        n_favour_alt = sum(1 for bf in jbf.values()
                           if bf['ln_BF_TEP_vs_alt'] < -1)

        # Most competitive alternative (lowest BF — hardest for TEP)
        hardest = min(jbf.items(), key=lambda x: x[1]['ln_BF_TEP_vs_alt'])
        easiest = max(jbf.items(), key=lambda x: x[1]['ln_BF_TEP_vs_alt'])

        results['joint_summary'] = {
            'n_alternatives': len(jbf),
            'n_decisive_for_TEP': n_decisive_tep,
            'n_strong_for_TEP': n_strong_tep,
            'n_favour_alternative': n_favour_alt,
            'hardest_alternative': hardest[0],
            'hardest_ln_BF': hardest[1]['ln_BF_TEP_vs_alt'],
            'easiest_alternative': easiest[0],
            'easiest_ln_BF': easiest[1]['ln_BF_TEP_vs_alt'],
            'mean_ln_BF': float(np.mean(
                [bf['ln_BF_TEP_vs_alt'] for bf in jbf.values()])),
            'mean_log10_BF': float(np.mean(
                [bf['log10_BF'] for bf in jbf.values()]))
        }

        results['key_finding'] = {
            'statement': _build_key_finding(results['joint_summary'], jbf),
            'methodology': (
                f"Dynesty nested sampling with nlive={NLIVE}, dlogz={DLOGZ}, "
                f"multi-observable joint likelihood across {K} observables "
                f"({', '.join(OBSERVABLES)}).  TEP uses {3*K} params "
                f"(3 per observable, shared log(Gamma_t) predictor) while "
                f"alternatives range from {min(v['alt_n_params'] for v in jbf.values())} "
                f"to {max(v['alt_n_params'] for v in jbf.values())} params."
            )
        }

        print_status("\n" + "=" * 70)
        print_status("KEY FINDING")
        print_status("=" * 70)
        print_status(results['key_finding']['statement'])

    if 'conventional_residual_space_bayes_factors' in results:
        rbf = results['conventional_residual_space_bayes_factors']
        print_status("\n" + "-" * 50)
        print_status("CONVENTIONAL RESIDUAL-SPACE BAYES FACTORS (Raw Mass)")
        print_status("-" * 50)
        for name, bf in rbf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± {bf['ln_BF_err']:.2f}  |  {bf['interpretation']}")

    if 'residual_space_bayes_factors' in results:
        rbf = results['residual_space_bayes_factors']
        n_decisive_tep = sum(1 for bf in rbf.values()
                             if bf['ln_BF_TEP_vs_alt'] > 5)
        n_strong_tep = sum(1 for bf in rbf.values()
                           if bf['ln_BF_TEP_vs_alt'] > 1)
        n_favour_alt = sum(1 for bf in rbf.values()
                           if bf['ln_BF_TEP_vs_alt'] < -1)
        hardest = min(rbf.items(), key=lambda x: x[1]['ln_BF_TEP_vs_alt'])
        easiest = max(rbf.items(), key=lambda x: x[1]['ln_BF_TEP_vs_alt'])

        results['residual_space_summary'] = {
            'n_alternatives': len(rbf),
            'n_decisive_for_TEP': n_decisive_tep,
            'n_strong_for_TEP': n_strong_tep,
            'n_favour_alternative': n_favour_alt,
            'hardest_alternative': hardest[0],
            'hardest_ln_BF': hardest[1]['ln_BF_TEP_vs_alt'],
            'easiest_alternative': easiest[0],
            'easiest_ln_BF': easiest[1]['ln_BF_TEP_vs_alt'],
            'mean_ln_BF': float(np.mean(
                [bf['ln_BF_TEP_vs_alt'] for bf in rbf.values()])),
            'mean_log10_BF': float(np.mean(
                [bf['log10_BF'] for bf in rbf.values()]))
        }

        results['residual_space_key_finding'] = {
            'statement': _build_residual_key_finding(results['residual_space_summary']),
            'methodology': (
                f"Residual-space dynesty comparison with nlive={NLIVE}, "
                f"dlogz={DLOGZ}. Observables and competing predictors were "
                f"residualized against a linear [1, log_Mstar, z] design "
                f"matrix before evidence computation. Residual TEP uses {3*K} "
                f"params, residual null uses {2*K}, and constrained AGN uses "
                f"10 params with metallicity excluded from explicit AGN "
                f"contamination response."
            )
        }

        print_status("\n" + "=" * 70)
        print_status("RESIDUAL-SPACE KEY FINDING")
        print_status("=" * 70)
        print_status(results['residual_space_key_finding']['statement'])

    # ------------------------------------------------------------------
    # Correction Summary: TEP mass correction impact
    # ------------------------------------------------------------------
    if 'correction_bayes_factors' in results and results['correction_bayes_factors']:
        cbf = results['correction_bayes_factors']
        n_improve = sum(1 for v in cbf.values()
                        if v['ln_BF_corrected_vs_uncorrected'] > 1)
        n_hurt = sum(1 for v in cbf.values()
                     if v['ln_BF_corrected_vs_uncorrected'] < -1)
        mean_corr_bf = float(np.mean(
            [v['ln_BF_corrected_vs_uncorrected'] for v in cbf.values()]))

        results['correction_summary'] = {
            'n_models_tested': len(cbf),
            'n_correction_improves': n_improve,
            'n_correction_hurts': n_hurt,
            'mean_ln_BF_correction': mean_corr_bf,
            'verdict': (
                'TEP mass correction systematically improves astrophysical model fits'
                if n_improve > n_hurt and mean_corr_bf > 0
                else 'TEP mass correction does not improve astrophysical model fits'
                if n_hurt > n_improve
                else 'TEP mass correction has mixed effect on model fits'
            ),
        }

        results['correction_key_finding'] = {
            'statement': _build_correction_key_finding(results['correction_summary'], cbf),
            'methodology': (
                f"Each astrophysical alternative was run with both observed mass "
                f"(M_obs) and TEP-corrected mass (M_true = M_obs - "
                f"{ALPHA_NUCLEAR}*log10(Gamma_t)).  Same parameter counts ensure "
                f"the comparison isolates the effect of the TEP mass correction. "
                f"Positive ln(BF) means the correction improves the fit, "
                f"supporting TEP as a measurement correction framework."
            )
        }

        print_status("\n" + "=" * 70)
        print_status("TEP CORRECTION KEY FINDING")
        print_status("=" * 70)
        print_status(results['correction_key_finding']['statement'])

    _save(results)
    print_status("=" * 70)
    return results


# ============================================================================
# Single-Observable Likelihoods (Supplementary)
# ============================================================================

def _single_tep_ll(params, dust, log_gamma, mass, z):
    a, b, c, d, log_s = params
    sigma = np.exp(log_s)
    pred = a + b * mass + c * z + d * log_gamma
    resid = dust - pred
    return -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

def _single_tep_prior(u):
    return np.array([
        u[0] * 20 - 10, u[1] * 4 - 2, u[2] * 4 - 2, u[3] * 4 - 2, u[4] * 6 - 5])

def _single_standard_ll(params, dust, mass, z):
    a, b, c, log_s = params
    sigma = np.exp(log_s)
    pred = a + b * mass + c * z
    resid = dust - pred
    return -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

def _single_standard_prior(u):
    return np.array([
        u[0] * 20 - 10, u[1] * 4 - 2, u[2] * 4 - 2, u[3] * 6 - 5])

def _single_imf_ll(params, dust, mass, z):
    a, b, c, d, log_s = params
    sigma = np.exp(log_s)
    pred = a + b * mass + c * mass**2 + d * z
    resid = dust - pred
    return -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

def _single_imf_prior(u):
    return np.array([
        u[0] * 20 - 10, u[1] * 4 - 2, u[2] * 0.4 - 0.2,
        u[3] * 4 - 2, u[4] * 6 - 5])

def _single_agn_ll(params, dust, mass, z):
    a, b, M_crit, slope, log_s = params
    sigma = np.exp(log_s)
    f_agn = 1.0 / (1.0 + np.exp(-slope * (mass - M_crit)))
    pred = a + b * f_agn
    resid = dust - pred
    return -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

def _single_agn_prior(u):
    return np.array([
        u[0] * 20 - 10, u[1] * 10 - 5,
        u[2] * 3 + 8.5, u[3] * 5 + 0.5, u[4] * 6 - 5])


# ============================================================================
# Helpers
# ============================================================================

def _build_key_finding(summary, jbf):
    """Construct honest key-finding statement from joint summary."""
    n_alt = summary['n_alternatives']
    n_dec = summary['n_decisive_for_TEP']
    n_str = summary['n_strong_for_TEP']
    n_fav = summary['n_favour_alternative']
    mean_lnbf = summary['mean_ln_BF']

    if n_fav == 0 and n_dec >= n_alt - 1:
        verdict = "decisively favours TEP over all tested alternatives"
    elif n_fav == 0 and n_str >= n_alt - 1:
        verdict = "strongly favours TEP over all tested alternatives"
    elif n_fav == 0:
        verdict = "favours TEP over all tested alternatives"
    elif n_fav <= 1:
        hard = summary['hardest_alternative']
        hard_ln_bf = summary['hardest_ln_BF']
        if hard_ln_bf < -5:
            verdict = (f"favours TEP over most alternatives, but {hard} "
                       f"decisively outperforms TEP in this comparison "
                       f"(ln BF = {hard_ln_bf:.1f})")
        elif hard_ln_bf < -3:
            verdict = (f"favours TEP over most alternatives, but {hard} "
                       f"strongly outperforms TEP in this comparison "
                       f"(ln BF = {hard_ln_bf:.1f})")
        else:
            verdict = (f"favours TEP over most alternatives but {hard} "
                       f"remains competitive (ln BF = {hard_ln_bf:.1f})")
    else:
        verdict = (f"yields mixed results: {n_str}/{n_alt} favour TEP "
                   f"while {n_fav}/{n_alt} favour alternatives")

    return (
        f"Multi-observable joint nested Bayesian model comparison {verdict}. "
        f"Mean ln(BF) across {n_alt} alternatives = {mean_lnbf:.1f} "
        f"(log10 = {mean_lnbf / np.log(10):.1f}).  "
        f"TEP achieves this with the fewest parameters, leveraging a single "
        f"theory-fixed predictor (Gamma_t) across all observables."
    )


def _build_residual_key_finding(summary):
    """Construct headline statement for the residual-space comparison."""
    n_alt = summary['n_alternatives']
    n_dec = summary['n_decisive_for_TEP']
    n_str = summary['n_strong_for_TEP']
    n_fav = summary['n_favour_alternative']
    mean_lnbf = summary['mean_ln_BF']

    if n_fav == 0 and n_dec >= n_alt:
        verdict = "decisively favours TEP over all tested residual alternatives"
    elif n_fav == 0 and n_str >= n_alt:
        verdict = "strongly favours TEP over all tested residual alternatives"
    elif n_fav == 0:
        verdict = "favours TEP over all tested residual alternatives"
    else:
        hardest = summary['hardest_alternative']
        verdict = (f"remains mixed after mass+z control because {hardest} "
                   f"still competes with TEP")

    return (
        f"Residual-space nested Bayesian comparison {verdict}. "
        f"After removing linear mass+z trends from both observables and "
        f"competing predictors, mean ln(BF) across {n_alt} alternatives = "
        f"{mean_lnbf:.1f} (log10 = {mean_lnbf / np.log(10):.1f})."
    )


def _build_correction_key_finding(summary, cbf):
    """Construct key-finding statement for the TEP mass correction test."""
    n_tested = summary['n_models_tested']
    n_improve = summary['n_correction_improves']
    n_hurt = summary['n_correction_hurts']
    mean_bf = summary['mean_ln_BF_correction']

    if n_improve > n_hurt and mean_bf > 1:
        verdict = (
            f"systematically improves the fit of {n_improve}/{n_tested} "
            f"astrophysical models (mean ln(BF) = {mean_bf:.1f}), supporting "
            f"TEP as a measurement correction framework rather than a "
            f"competing regression model"
        )
    elif n_hurt > n_improve:
        verdict = (
            f"does not improve astrophysical model fits ({n_hurt}/{n_tested} "
            f"models worsened, mean ln(BF) = {mean_bf:.1f}), suggesting the "
            f"TEP mass correction does not recover the true physical driver"
        )
    else:
        verdict = (
            f"has a mixed effect on astrophysical model fits "
            f"({n_improve} improved, {n_hurt} worsened, "
            f"mean ln(BF) = {mean_bf:.1f})"
        )

    per_model = "; ".join(
        f"{name}: ln(BF)={v['ln_BF_corrected_vs_uncorrected']:.1f}"
        for name, v in cbf.items()
    )

    return (
        f"Applying the TEP mass correction (M_true = M_obs - "
        f"{ALPHA_NUCLEAR}*log10(Gamma_t)) {verdict}.  "
        f"Per-model results: {per_model}."
    )


def _save(results):
    """Write results JSON."""
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print_status(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
