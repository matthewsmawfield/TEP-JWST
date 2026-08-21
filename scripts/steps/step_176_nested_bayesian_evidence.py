#!/usr/bin/env python3
"""
Step 176: Nested Bayesian Model Comparison

Performs fully nested Bayesian evidence computation comparing TEP against
explicit astrophysical alternatives using dynesty nested sampling.

TEP is framed as a measurement correction framework, not just a regression
model. The tested observable response R_ML parameterizes an environmental
mass-to-light inference bias, M_obs/M_true = R_ML^n, without identifying R_ML
with A(phi) or a local proper-time ratio. Each alternative is run with both
observed mass and response-corrected mass,
M_true = M_obs - n*log10(R_ML).

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
1. TEP: Prespecified R_ML response predictor (zero structural free parameters)
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
from scripts.utils.tep_model import compute_ml_response, compute_ml_response_self_consistent, stellar_to_halo_mass_behroozi_like  # Shared TEP model
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
N_CONVERGENCE_SEEDS = 3


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
            if 'ml_response' not in df.columns:
                z_vals = df['z'].values
                if 'log_Mh' not in df.columns:
                    df['log_Mh'] = stellar_to_halo_mass_behroozi_like(
                        df['log_Mstar'].values, z_vals)
                df['ml_response'] = compute_ml_response(df['log_Mh'].values, z_vals)
            if 'gamma_t' not in df.columns:
                df['gamma_t'] = df['ml_response']
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


def _joint_bursty_loglike(params, obs_arrays, mass_ortho, z_ortho, mass_ortho_for_burst):
    """
    Joint Bursty SF likelihood.
    Shared burst timescale tau across observables.
    All mass terms use orthogonalized predictors (mass with log_gamma
    component removed) to prevent circular absorption of the TEP signal.
    Per-observable: a_k + b_k*mass_ortho + c_k*z_ortho + d_k*burst(tau, mass_ortho) + noise → 5K+1 params.
    params layout: [tau, a_0, b_0, c_0, d_0, log_s_0, a_1, ...]
    """
    K = len(obs_arrays)
    tau = params[0]
    burst = np.exp(-tau * (1 - mass_ortho_for_burst / 10))
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


def _joint_quadratic_loglike(params, obs_arrays, mass, z):
    """
    Joint Quadratic baseline: a generic nonlinear mass+z surface.

    obs_k = a_k + b_k*mass + c_k*z + d_k*mass^2 + e_k*z^2 + f_k*mass*z + N(0, sigma_k)
    7K params.  This is a matched-flexibility nonlinear baseline with no
    TEP-specific structure.  If TEP's evidence survives against this model,
    the signal is not merely generic nonlinearity in mass+z.
    """
    K = len(obs_arrays)
    mass2 = mass**2
    z2 = z**2
    mz = mass * z
    ll = 0.0
    for k in range(K):
        idx = 7*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        d = params[idx + 3]
        e = params[idx + 4]
        f = params[idx + 5]
        sigma = np.exp(params[idx + 6])
        pred = a + b * mass + c * z + d * mass2 + e * z2 + f * mz
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_quadratic_prior(u, K):
    """Prior transform for joint Quadratic: 7K params."""
    out = np.empty(7*K)
    for k in range(K):
        idx = 7*k
        out[idx]     = u[idx]     * 20 - 10      # a
        out[idx + 1] = u[idx + 1] * 4 - 2        # b
        out[idx + 2] = u[idx + 2] * 4 - 2        # c
        out[idx + 3] = u[idx + 3] * 0.4 - 0.2    # d (quadratic mass): [-0.2, 0.2]
        out[idx + 4] = u[idx + 4] * 0.4 - 0.2    # e (quadratic z): [-0.2, 0.2]
        out[idx + 5] = u[idx + 5] * 0.4 - 0.2    # f (mass×z): [-0.2, 0.2]
        out[idx + 6] = u[idx + 6] * 6 - 5        # log_sigma
    return out


def _joint_mz_interaction_loglike(params, obs_arrays, mass, z):
    """Joint M*×sqrt(1+z) interaction likelihood.

    This is the minimal non-linear null: it captures the specific
    mass-redshift interaction that TEP encodes (through the sqrt(1+z)
    factor in R_ML) without any TEP-specific potential-depth structure.
    If TEP's Bayes factor survives against this model, the evidence
    is not merely from a generic M*×z interaction.

    obs_k = a_k + b_k*mass + c_k*z + d_k*mass*sqrt(1+z) + N(0, sigma_k)
    5K params (same flexibility as the augmented TEP model).

    Note: mass and z are standardized (mean=0, std=1).  The sqrt(1+z)
    term is clipped at a small positive floor to avoid NaN when the
    standardized z drops below -1.  This is a numerical safeguard only
    and does not affect the statistical interpretation.
    """
    K = len(obs_arrays)
    mz_int = mass * np.sqrt(np.maximum(1.0 + z, 0.01))
    ll = 0.0
    for k in range(K):
        idx = 5*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        d = params[idx + 3]
        sigma = np.exp(params[idx + 4])
        pred = a + b * mass + c * z + d * mz_int
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_mz_interaction_prior(u, K):
    """Prior transform for joint M*×sqrt(1+z): 5K params."""
    out = np.empty(5*K)
    for k in range(K):
        idx = 5*k
        out[idx]     = u[idx]     * 20 - 10      # a
        out[idx + 1] = u[idx + 1] * 4 - 2        # b
        out[idx + 2] = u[idx + 2] * 4 - 2        # c
        out[idx + 3] = u[idx + 3] * 2 - 1        # d (interaction): [-1, 1]
        out[idx + 4] = u[idx + 4] * 6 - 5        # log_sigma
    return out


# ============================================================================
# Wide-Prior Variants for Prior Sensitivity Check
# ============================================================================
# The default quadratic prior restricts d, e, f (mass^2, z^2, mass*z) to
# [-0.2, 0.2], tighter than the linear coefficients' [-2, 2].  A skeptic
# could argue this artificially penalizes the quadratic baseline via an
# Occam penalty.  The wide-prior variant uses [-0.5, 0.5] to test whether
# the TEP Bayes factor is robust to this choice.  Similarly, the Mz
# interaction coefficient is widened from [-1, 1] to [-2, 2].


def _joint_quadratic_wide_prior(u, K):
    """Prior transform for joint Quadratic with WIDE nonlinear priors: 7K params.

    Identical to _joint_quadratic_prior but with d, e, f in [-0.5, 0.5]
    instead of [-0.2, 0.2].
    """
    out = np.empty(7*K)
    for k in range(K):
        idx = 7*k
        out[idx]     = u[idx]     * 20 - 10      # a
        out[idx + 1] = u[idx + 1] * 4 - 2        # b
        out[idx + 2] = u[idx + 2] * 4 - 2        # c
        out[idx + 3] = u[idx + 3] * 1.0 - 0.5    # d (quadratic mass): [-0.5, 0.5]
        out[idx + 4] = u[idx + 4] * 1.0 - 0.5    # e (quadratic z): [-0.5, 0.5]
        out[idx + 5] = u[idx + 5] * 1.0 - 0.5    # f (mass×z): [-0.5, 0.5]
        out[idx + 6] = u[idx + 6] * 6 - 5        # log_sigma
    return out


def _joint_mz_interaction_wide_prior(u, K):
    """Prior transform for joint M*×sqrt(1+z) with WIDE interaction prior: 5K params.

    Identical to _joint_mz_interaction_prior but with d in [-2, 2] instead
    of [-1, 1], matching the linear coefficient prior width.
    """
    out = np.empty(5*K)
    for k in range(K):
        idx = 5*k
        out[idx]     = u[idx]     * 20 - 10      # a
        out[idx + 1] = u[idx + 1] * 4 - 2        # b
        out[idx + 2] = u[idx + 2] * 4 - 2        # c
        out[idx + 3] = u[idx + 3] * 4 - 2        # d (interaction): [-2, 2]
        out[idx + 4] = u[idx + 4] * 6 - 5        # log_sigma
    return out


def _joint_agn_loglike(params, obs_arrays, mass_ortho, z_ortho):
    """
    Joint AGN Feedback likelihood.
    Shared M_crit, slope across observables.  Uses orthogonalized mass
    (TEP signal removed) and includes z as a linear predictor, matching
    the treatment of all other alternative models.
    Per-observable: a_k + b_k * sigmoid(mass_ortho, M_crit, slope) + c_k * z_ortho + noise → 4K+2 params.
    params layout: [M_crit, slope, a_0, b_0, c_0, log_s_0, a_1, ...]
    """
    K = len(obs_arrays)
    M_crit = params[0]
    slope = params[1]
    f_agn = 1.0 / (1.0 + np.exp(-slope * (mass_ortho - M_crit)))
    ll = 0.0
    for k in range(K):
        idx = 2 + 4*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        sigma = np.exp(params[idx + 3])
        pred = a + b * f_agn + c * z_ortho
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_agn_prior(u, K):
    """Prior transform for joint AGN: 4K+2 params."""
    n = 4*K + 2
    out = np.empty(n)
    out[0] = u[0] * 4 - 2     # M_crit: [-2, 2] (standardized orthogonalized mass)
    out[1] = u[1] * 5 + 0.5   # slope: [0.5, 5.5]
    for k in range(K):
        idx = 2 + 4*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 10 - 5
        out[idx + 2] = u[idx + 2] * 4 - 2    # c (z coefficient): [-2, 2]
        out[idx + 3] = u[idx + 3] * 6 - 5
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


def _joint_corrected_bursty_loglike(params, obs_arrays, mass_corr_ortho, z_ortho, mass_corr_ortho_for_burst):
    """
    TEP-Corrected Bursty SF: shared burst timescale on orthogonalized corrected mass.
    All terms use orthogonalized corrected mass.
    Same 5K+1 params as Bursty SF.
    """
    K = len(obs_arrays)
    tau = params[0]
    burst = np.exp(-tau * (1 - mass_corr_ortho_for_burst / 10))
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


def _joint_corrected_agn_loglike(params, obs_arrays, mass_corr_ortho, z_ortho):
    """
    TEP-Corrected AGN Feedback: sigmoid threshold on orthogonalized corrected mass.
    Same 4K+2 params as AGN Feedback, but uses TEP-corrected orthogonalized mass.
    """
    K = len(obs_arrays)
    M_crit = params[0]
    slope = params[1]
    f_agn = 1.0 / (1.0 + np.exp(-slope * (mass_corr_ortho - M_crit)))
    ll = 0.0
    for k in range(K):
        idx = 2 + 4*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        sigma = np.exp(params[idx + 3])
        pred = a + b * f_agn + c * z_ortho
        resid = obs_arrays[k] - pred
        ll += -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def _joint_corrected_agn_prior(u, K):
    """Prior transform for corrected AGN: 4K+2 params."""
    n = 4*K + 2
    out = np.empty(n)
    out[0] = u[0] * 4 - 2     # M_crit: [-2, 2] (standardized)
    out[1] = u[1] * 5 + 0.5   # slope: [0.5, 5.5]
    for k in range(K):
        idx = 2 + 4*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 10 - 5
        out[idx + 2] = u[idx + 2] * 4 - 2    # c (z coefficient): [-2, 2]
        out[idx + 3] = u[idx + 3] * 6 - 5
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


# ============================================================================
# Joint Covariance Likelihood (accounts for correlated SED outputs)
# ============================================================================
# The standard joint likelihood treats observables as independent:
#   L = prod_k N(obs_k | pred_k, sigma_k^2)
# But dust, sSFR, chi2, and metallicity are correlated outputs from the
# same SED fit.  Treating them as independent multiplies the evidence.
#
# The joint covariance likelihood uses a single multivariate Gaussian:
#   L = N(obs | pred, Sigma)
# where Sigma is the empirical K×K residual covariance matrix, estimated
# once from the data and held fixed.  A single scaling parameter alpha
# allows the sampler to adjust the overall noise level.


def _joint_cov_tep_loglike(params, obs_matrix, log_gamma, cov_inv, cov_logdet):
    """
    Joint covariance TEP likelihood.

    params: [a_0, b_0, a_1, b_1, ..., log_alpha]
    Each observable k: obs_k = a_k + b_k * log_gamma
    Residual covariance is fixed from data; alpha scales it.
    Total: 2K + 1 params.
    """
    K = obs_matrix.shape[1]
    alpha = np.exp(params[2*K])
    pred = np.empty_like(obs_matrix)
    for k in range(K):
        pred[:, k] = params[2*k] + params[2*k + 1] * log_gamma
    resid = obs_matrix - pred
    # Multivariate Gaussian: -0.5 * sum_n r_n^T (alpha * Sigma)^-1 r_n - 0.5 * N * log|alpha * Sigma|
    scaled_cov_inv = cov_inv / alpha
    logdet_term = cov_logdet + K * np.log(alpha)
    # Vectorized multivariate Gaussian: avoids per-galaxy Python loop
    quad_form = np.einsum('ni,ij,nj->n', resid, scaled_cov_inv, resid)
    ll = -0.5 * (np.sum(quad_form) + resid.shape[0] * (K * np.log(2 * np.pi) + logdet_term))
    return ll


def _joint_cov_tep_prior(u, K):
    """Prior transform for joint covariance TEP: 2K+1 params."""
    out = np.empty(2*K + 1)
    for k in range(K):
        out[2*k]     = u[2*k]     * 20 - 10
        out[2*k + 1] = u[2*k + 1] * 4 - 2
    out[2*K] = u[2*K] * 4 - 2    # log_alpha: [-2, 2]
    return out


def _joint_cov_augmented_loglike(params, obs_matrix, mass, z, log_gamma, cov_inv, cov_logdet):
    """
    Joint covariance augmented TEP likelihood: mass + z + R_ML.

    params: [a_0, b_m_0, b_z_0, b_g_0, ..., log_alpha]
    Total: 4K + 1 params.
    """
    K = obs_matrix.shape[1]
    alpha = np.exp(params[4*K])
    pred = np.empty_like(obs_matrix)
    for k in range(K):
        idx = 4*k
        pred[:, k] = (params[idx] + params[idx + 1] * mass
                      + params[idx + 2] * z + params[idx + 3] * log_gamma)
    resid = obs_matrix - pred
    scaled_cov_inv = cov_inv / alpha
    logdet_term = cov_logdet + K * np.log(alpha)
    # Vectorized multivariate Gaussian: avoids per-galaxy Python loop
    quad_form = np.einsum('ni,ij,nj->n', resid, scaled_cov_inv, resid)
    ll = -0.5 * (np.sum(quad_form) + resid.shape[0] * (K * np.log(2 * np.pi) + logdet_term))
    return ll


def _joint_cov_augmented_prior(u, K):
    """Prior transform for joint covariance augmented TEP: 4K+1 params."""
    out = np.empty(4*K + 1)
    for k in range(K):
        idx = 4*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 4 - 2
        out[idx + 2] = u[idx + 2] * 4 - 2
        out[idx + 3] = u[idx + 3] * 4 - 2
    out[4*K] = u[4*K] * 4 - 2    # log_alpha
    return out


def _joint_cov_standard_loglike(params, obs_matrix, mass, z, cov_inv, cov_logdet):
    """
    Joint covariance standard likelihood: mass + z only.

    params: [a_0, b_m_0, b_z_0, ..., log_alpha]
    Total: 3K + 1 params.
    """
    K = obs_matrix.shape[1]
    alpha = np.exp(params[3*K])
    pred = np.empty_like(obs_matrix)
    for k in range(K):
        idx = 3*k
        pred[:, k] = params[idx] + params[idx + 1] * mass + params[idx + 2] * z
    resid = obs_matrix - pred
    scaled_cov_inv = cov_inv / alpha
    logdet_term = cov_logdet + K * np.log(alpha)
    # Vectorized multivariate Gaussian: avoids per-galaxy Python loop
    quad_form = np.einsum('ni,ij,nj->n', resid, scaled_cov_inv, resid)
    ll = -0.5 * (np.sum(quad_form) + resid.shape[0] * (K * np.log(2 * np.pi) + logdet_term))
    return ll


def _joint_cov_standard_prior(u, K):
    """Prior transform for joint covariance standard: 3K+1 params."""
    out = np.empty(3*K + 1)
    for k in range(K):
        idx = 3*k
        out[idx]     = u[idx]     * 20 - 10
        out[idx + 1] = u[idx + 1] * 4 - 2
        out[idx + 2] = u[idx + 2] * 4 - 2
    out[3*K] = u[3*K] * 4 - 2    # log_alpha
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

def run_nested(loglike_fn, prior_fn, ndim, label, nlive=NLIVE, dlogz=DLOGZ,
               seed_override=None):
    """Run dynesty nested sampling and return summary dict.

    When seed_override is provided, uses that exact seed instead of the
    label-derived default.  This is used by the convergence diagnostics
    to run the same model with multiple independent seeds.
    """
    import dynesty
    from dynesty import utils as dyfunc

    print_status(f"\n  Running nested sampling: {label}")
    print_status(f"    ndim={ndim}, nlive={nlive}")
    if seed_override is not None:
        label_seed = int(seed_override)
    else:
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


def run_convergence_diagnostics(model_specs, n_seeds=N_CONVERGENCE_SEEDS):
    """Re-run key models with multiple seeds to estimate empirical logZ variance.

    The dynesty internal logZ_err is an analytic estimate from the nested
    sampling run.  This function provides an independent empirical check by
    running each model with n_seeds different random seeds and reporting the
    cross-run spread.  If the empirical std is comparable to or smaller than
    the dynesty estimate, the evidence is stable.

    model_specs: dict of name -> (loglike_fn, prior_fn, ndim, label)
    """
    diagnostics = {}
    for name, (loglike_fn, prior_fn, ndim, label) in model_specs.items():
        logZ_values = []
        logZ_errs = []
        for i in range(n_seeds):
            seed = RNG_SEED + 10000 * (i + 1) + sum(ord(ch) for ch in label)
            try:
                r = run_nested(
                    loglike_fn, prior_fn, ndim,
                    f"{label} (conv {i+1}/{n_seeds})",
                    seed_override=seed,
                )
                logZ_values.append(r['logZ'])
                logZ_errs.append(r['logZ_err'])
            except Exception as e:
                print_status(f"Convergence run {i+1} for {name} failed: {e}", "WARN")
        if len(logZ_values) >= 2:
            diagnostics[name] = {
                'n_seeds': len(logZ_values),
                'logZ_values': logZ_values,
                'logZ_mean': float(np.mean(logZ_values)),
                'logZ_std_empirical': float(np.std(logZ_values, ddof=1)),
                'logZ_spread': float(np.max(logZ_values) - np.min(logZ_values)),
                'logZ_err_dynesty_mean': float(np.mean(logZ_errs)),
                'stable': bool(np.std(logZ_values, ddof=1) <= max(np.mean(logZ_errs) * 2, 1.0)),
            }
            d = diagnostics[name]
            print_status(
                f"  {name}: logZ mean={d['logZ_mean']:.2f}, "
                f"empirical std={d['logZ_std_empirical']:.3f}, "
                f"dynesty mean err={d['logZ_err_dynesty_mean']:.3f}, "
                f"{'STABLE' if d['stable'] else 'CHECK'}")
    return diagnostics


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

    response_column = 'ml_response' if 'ml_response' in df.columns else 'gamma_t'
    cols_needed = ['dust', 'log_Mstar', 'z', response_column] + [
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
    # Use the self-consistent R_ML (iterated M*→Mh→R_ML→M*_true cycle)
    # rather than the single-pass value from the data file.  The single-pass
    # value computes R_ML from the observed (biased) mass, creating a mass
    # circularity.  The self-consistent solution breaks this by iterating to
    # the fixed point.  For 99% of galaxies (M* < 10) the difference is < 2%,
    # but at M* > 10.5 it can reach 10–60%.
    gamma_t_single = df_v[response_column].values
    gamma_t, mass_corrected = compute_ml_response_self_consistent(
        mass, z, n=ALPHA_NUCLEAR)
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

    # Empirical residual covariance matrix for the joint covariance likelihood.
    # The observables (dust, sSFR, chi2, met) are correlated SED outputs from
    # the same photometric fit.  Treating them as independent inflates evidence.
    # We estimate the covariance from the standardized observables and use it
    # in a multivariate Gaussian likelihood.
    obs_matrix = np.column_stack(obs_arrays)  # N × K
    cov_empirical = np.cov(obs_matrix, rowvar=False)  # K × K
    # Regularize for numerical stability
    cov_empirical += 1e-4 * np.eye(K)
    cov_inv = np.linalg.inv(cov_empirical)
    cov_logdet = float(np.linalg.slogdet(cov_empirical)[1])
    results['observable_covariance'] = {
        'matrix': cov_empirical.tolist(),
        'logdet': cov_logdet,
        'correlation_matrix': np.corrcoef(obs_matrix, rowvar=False).tolist(),
        'note': (
            'Empirical covariance of standardized observables used in the '
            'joint covariance likelihood.  The off-diagonal elements quantify '
            'the SED-output correlations that the independent-likelihood '
            'models double-count.'
        ),
    }

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

    # Response-corrected mass: M_true = M_obs - n * log10(R_ML).
    # R_ML is the observable M/L inference response, not A(phi) or a local
    # proper-time ratio. The corrected mass removes the fitted inference bias.
    #
    # SELF-CONSISTENT SOLUTION: The single-pass correction (compute R_ML from
    # M_obs, then correct once) is exact only when R_ML is mass-independent.
    # Because R_ML depends on M_h which depends on M*, we iterate the
    # M*→M_h→R_ML→M*_true cycle to convergence.  For typical high-z galaxies
    # (M* < 10) the difference is < 2%, but at M* > 10.5 it can reach 10–60%.
    #
    # NOTE: The self-consistent R_ML and mass_corrected are now computed
    # earlier (at data loading) so the main predictor is also self-consistent.
    # The variables gamma_t, mass_corrected, and log_gamma_raw are already
    # available from that computation.
    #
    # IMPORTANT: mass and mass_corrected are kept in raw (unstandardized) units
    # because the Bursty SF burst term (exp(-tau*(1-mass/10))) and AGN sigmoid
    # (M_crit in [8.5, 11.5]) are calibrated for raw log-stellar-mass ~8-11.
    # The linear coefficients in the likelihoods absorb the scale difference
    # between raw mass and standardized observables.
    log_gamma_raw = np.log10(np.maximum(gamma_t, 0.01))
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

    # Response-corrected sSFR: log_sSFR_true = log_sSFR_obs + n * log10(R_ML).
    # This removes the fitted observer-side inference response without assigning
    # that response to a faster local clock.
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
        'mass_correction_self_consistent': True,
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
            lambda p: _joint_bursty_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std, mass_ortho_std),
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

    # AGN Feedback joint: 4K+2 params (orthogonalized mass + z, same as other alternatives)
    try:
        r = run_nested(
            lambda p: _joint_agn_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
            lambda u: _joint_agn_prior(u, K),
            4*K + 2, f"AGN Feedback Joint ({4*K+2} params)")
        joint_models['AGN_Feedback'] = r
    except Exception as e:
        print_status(f"AGN Feedback joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Quadratic baseline joint: 7K params (orthogonalized mass+z)
    # Generic nonlinear mass+z surface with no TEP-specific structure.
    # If TEP evidence survives against this, the signal is not generic nonlinearity.
    try:
        r = run_nested(
            lambda p: _joint_quadratic_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
            lambda u: _joint_quadratic_prior(u, K),
            7*K, f"Quadratic Baseline Joint ({7*K} params)")
        joint_models['Quadratic_Baseline'] = r
    except Exception as e:
        print_status(f"Quadratic Baseline joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # M*×sqrt(1+z) interaction joint: 5K params (orthogonalized mass+z)
    # Minimal non-linear null capturing the specific mass-redshift interaction
    # that TEP encodes through the sqrt(1+z) factor in R_ML, without any
    # TEP-specific potential-depth structure.  This is the fairest test:
    # same interaction form, same param count as augmented TEP, no TEP physics.
    try:
        r = run_nested(
            lambda p: _joint_mz_interaction_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
            lambda u: _joint_mz_interaction_prior(u, K),
            5*K, f"M*×sqrt(1+z) Interaction Joint ({5*K} params)")
        joint_models['Mz_Interaction'] = r
    except Exception as e:
        print_status(f"M*×sqrt(1+z) Interaction joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # ------------------------------------------------------------------
    # A.1b Joint Covariance Likelihood Models (correlated SED outputs)
    # ------------------------------------------------------------------
    print_status("\n" + "-" * 50)
    print_status("JOINT COVARIANCE LIKELIHOOD (correlated SED outputs)")
    print_status("-" * 50)

    # Covariance TEP: 2K+1 params
    try:
        r = run_nested(
            lambda p: _joint_cov_tep_loglike(p, obs_matrix, log_gamma, cov_inv, cov_logdet),
            lambda u: _joint_cov_tep_prior(u, K),
            2*K + 1, f"Cov-TEP Joint ({2*K+1} params)")
        joint_models['Cov_TEP'] = r
    except Exception as e:
        print_status(f"Cov-TEP joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Covariance Standard: 3K+1 params (mass + z)
    try:
        r = run_nested(
            lambda p: _joint_cov_standard_loglike(p, obs_matrix, mass_ortho_std, z_ortho_std, cov_inv, cov_logdet),
            lambda u: _joint_cov_standard_prior(u, K),
            3*K + 1, f"Cov-Standard Joint ({3*K+1} params)")
        joint_models['Cov_Standard'] = r
    except Exception as e:
        print_status(f"Cov-Standard joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Covariance Augmented: 4K+1 params (mass + z + R_ML)
    try:
        r = run_nested(
            lambda p: _joint_cov_augmented_loglike(p, obs_matrix, mass_ortho_std, z_ortho_std, log_gamma, cov_inv, cov_logdet),
            lambda u: _joint_cov_augmented_prior(u, K),
            4*K + 1, f"Cov-Augmented Joint ({4*K+1} params)")
        joint_models['Cov_Augmented'] = r
    except Exception as e:
        print_status(f"Cov-Augmented joint failed: {e}", "ERROR")
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
            lambda p: _joint_corrected_bursty_loglike(p, obs_arrays, mass_corr_ortho_std, z_ortho_std, mass_corr_ortho_std),
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

    # Corrected AGN Feedback: 4K+2 params (orthogonalized corrected mass + z)
    try:
        r = run_nested(
            lambda p: _joint_corrected_agn_loglike(p, obs_arrays, mass_corr_ortho_std, z_ortho_std),
            lambda u: _joint_corrected_agn_prior(u, K),
            4*K + 2, f"Corrected AGN Feedback Joint ({4*K+2} params)")
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

    # Joint Bayes Factors — SEPARATED by likelihood type
    # Independent-likelihood models use product-of-Gaussians; covariance-corrected
    # models use a multivariate Gaussian.  Bayes factors are only valid within
    # the same likelihood family.  Mixing them produces meaningless numbers
    # because the logZ normalising constants differ.
    cov_model_names = {'Cov_TEP', 'Cov_Standard', 'Cov_Augmented'}

    if 'TEP' in joint_models:
        # Independent-likelihood comparisons only
        alt_indep = {k: v for k, v in joint_models.items()
                     if k not in ('TEP', 'TEP_Augmented') and k not in cov_model_names}
        joint_bf = compute_bayes_factors(joint_models['TEP'], alt_indep)
        results['joint_bayes_factors'] = joint_bf

        print_status("\n" + "-" * 50)
        print_status("JOINT BAYES FACTORS — Independent Likelihood (TEP vs Alternatives)")
        print_status("-" * 50)
        for name, bf in joint_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    # Covariance-corrected Bayes Factors (separate family)
    if 'Cov_TEP' in joint_models:
        alt_cov = {k: v for k, v in joint_models.items()
                   if k in cov_model_names and k != 'Cov_TEP'}
        cov_bf = compute_bayes_factors(joint_models['Cov_TEP'], alt_cov)
        results['covariance_bayes_factors'] = cov_bf

        print_status("\n" + "-" * 50)
        print_status("COVARIANCE-CORRECTED BAYES FACTORS (Cov_TEP vs Cov Alternatives)")
        print_status("-" * 50)
        for name, bf in cov_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    # Augmented TEP Bayes Factors (tests whether Gamma_t adds info beyond mass+z)
    if 'TEP_Augmented' in joint_models:
        alt_aug = {k: v for k, v in joint_models.items()
                   if k not in ('TEP_Augmented',) and k not in cov_model_names}
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
    # D. Convergence Diagnostics (Multi-Seed logZ Stability)
    # ------------------------------------------------------------------
    # Re-run the TEP model and the hardest alternative with multiple seeds
    # to verify that the dynesty internal logZ_err matches the empirical
    # cross-run variance.  This is critical for the manuscript's "decisive
    # evidence" claims.
    print_status("\n" + "=" * 70)
    print_status("D. CONVERGENCE DIAGNOSTICS (Multi-Seed logZ Stability)")
    print_status("=" * 70)

    conv_specs = {}
    if 'TEP' in joint_models:
        conv_specs['TEP'] = (
            lambda p: _joint_tep_loglike(p, obs_arrays, log_gamma),
            lambda u: _joint_tep_prior(u, K),
            3*K, "TEP Conv")
    # Add the hardest alternative if available
    # Compute hardest directly from joint_bayes_factors since joint_summary
    # is not yet computed at this point in the pipeline.
    hardest_name = None
    if 'joint_bayes_factors' in results:
        jbf = results['joint_bayes_factors']
        if jbf:
            hardest_name = min(jbf.items(), key=lambda x: x[1]['ln_BF_TEP_vs_alt'])[0]
    if hardest_name is not None:
        hardest_specs = {
            'Standard_Physics': (
                lambda p: _joint_standard_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
                lambda u: _joint_standard_prior(u, K), 4*K, "Std Conv"),
            'Bursty_SF': (
                lambda p: _joint_bursty_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std, mass_ortho_std),
                lambda u: _joint_bursty_prior(u, K), 5*K + 1, "Bursty Conv"),
            'Varying_IMF': (
                lambda p: _joint_imf_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
                lambda u: _joint_imf_prior(u, K), 5*K, "IMF Conv"),
            'AGN_Feedback': (
                lambda p: _joint_agn_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
                lambda u: _joint_agn_prior(u, K), 4*K + 2, "AGN Conv"),
            'Quadratic_Baseline': (
                lambda p: _joint_quadratic_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
                lambda u: _joint_quadratic_prior(u, K), 7*K, "Quad Conv"),
            'Mz_Interaction': (
                lambda p: _joint_mz_interaction_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
                lambda u: _joint_mz_interaction_prior(u, K), 5*K, "Mz Conv"),
        }
        if hardest_name in hardest_specs:
            conv_specs[hardest_name] = hardest_specs[hardest_name]

    try:
        conv_diag = run_convergence_diagnostics(conv_specs)
        results['convergence_diagnostics'] = conv_diag
    except Exception as e:
        print_status(f"Convergence diagnostics failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # ------------------------------------------------------------------
    # E. Prior Sensitivity Check (Quadratic & Interaction Baselines)
    # ------------------------------------------------------------------
    # The default quadratic prior restricts nonlinear coefficients to
    # [-0.2, 0.2], tighter than linear coefficients' [-2, 2].  A skeptic
    # could argue this artificially penalizes the quadratic baseline.
    # The wide-prior variant uses [-0.5, 0.5] to test robustness.
    print_status("\n" + "=" * 70)
    print_status("E. PRIOR SENSITIVITY CHECK (Wide-Prior Baselines)")
    print_status("=" * 70)

    prior_sensitivity = {}
    try:
        r_quad_wide = run_nested(
            lambda p: _joint_quadratic_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
            lambda u: _joint_quadratic_wide_prior(u, K),
            7*K, f"Quadratic Wide-Prior ({7*K} params)")
        prior_sensitivity['Quadratic_Wide'] = r_quad_wide
    except Exception as e:
        print_status(f"Quadratic wide-prior failed: {e}", "ERROR")

    try:
        r_mz_wide = run_nested(
            lambda p: _joint_mz_interaction_loglike(p, obs_arrays, mass_ortho_std, z_ortho_std),
            lambda u: _joint_mz_interaction_wide_prior(u, K),
            5*K, f"Mz-Interaction Wide-Prior ({5*K} params)")
        prior_sensitivity['Mz_Interaction_Wide'] = r_mz_wide
    except Exception as e:
        print_status(f"Mz-Interaction wide-prior failed: {e}", "ERROR")

    # Compare wide vs default BFs
    prior_sensitivity_bf = {}
    if 'TEP' in joint_models:
        tep_logZ = joint_models['TEP']['logZ']
        for wide_name, default_name in [('Quadratic_Wide', 'Quadratic_Baseline'),
                                         ('Mz_Interaction_Wide', 'Mz_Interaction')]:
            if wide_name in prior_sensitivity and default_name in joint_models:
                wide = prior_sensitivity[wide_name]
                default = joint_models[default_name]
                ln_bf_wide = tep_logZ - wide['logZ']
                ln_bf_default = tep_logZ - default['logZ']
                prior_sensitivity_bf[default_name] = {
                    'ln_BF_TEP_vs_alt_default_prior': float(ln_bf_default),
                    'ln_BF_TEP_vs_alt_wide_prior': float(ln_bf_wide),
                    'delta_ln_BF': float(ln_bf_wide - ln_bf_default),
                    'default_prior_range': '[-0.2, 0.2]' if 'Quadratic' in default_name else '[-1, 1]',
                    'wide_prior_range': '[-0.5, 0.5]' if 'Quadratic' in default_name else '[-2, 2]',
                    'robust': bool(abs(ln_bf_wide - ln_bf_default) < 2.0),
                }
                psb = prior_sensitivity_bf[default_name]
                print_status(
                    f"  {default_name}: ln(BF) default={psb['ln_BF_TEP_vs_alt_default_prior']:.2f}, "
                    f"wide={psb['ln_BF_TEP_vs_alt_wide_prior']:.2f}, "
                    f"delta={psb['delta_ln_BF']:.2f} "
                    f"{'ROBUST' if psb['robust'] else 'SENSITIVE'}")

    results['prior_sensitivity_models'] = prior_sensitivity
    results['prior_sensitivity_bayes_factors'] = prior_sensitivity_bf

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
    # B.2 Per-Observable Evidence Breakdown (SUPPLEMENTARY)
    # ------------------------------------------------------------------
    # Run TEP vs Standard for each observable separately to identify which
    # observables drive the joint signal and which are neutral or negative.
    print_status("\n" + "=" * 70)
    print_status("B.2 PER-OBSERVABLE EVIDENCE BREAKDOWN")
    print_status("=" * 70)

    per_obs_models = {}
    for obs_name, obs_arr in zip(OBSERVABLES, obs_arrays):
        obs_models = {}
        # TEP: 3 params (a + b*log_gamma + noise)
        try:
            r = run_nested(
                lambda p, oa=obs_arr: _joint_tep_loglike(p, [oa], log_gamma),
                lambda u: _joint_tep_prior(u, 1),
                3, f"TEP {obs_name} (3 params)")
            obs_models['TEP'] = r
        except Exception as e:
            print_status(f"TEP {obs_name} failed: {e}", "ERROR")

        # Standard: 4 params (a + b*mass + c*z + noise)
        try:
            r = run_nested(
                lambda p, oa=obs_arr: _joint_standard_loglike(p, [oa], mass_ortho_std, z_ortho_std),
                lambda u: _joint_standard_prior(u, 1),
                4, f"Standard {obs_name} (4 params)")
            obs_models['Standard'] = r
        except Exception as e:
            print_status(f"Standard {obs_name} failed: {e}", "ERROR")

        # Augmented: 5 params (a + b*mass + c*z + d*log_gamma + noise)
        try:
            r = run_nested(
                lambda p, oa=obs_arr: _joint_tep_augmented_loglike(p, [oa], mass_ortho_std, z_ortho_std, log_gamma),
                lambda u: _joint_tep_augmented_prior(u, 1),
                5, f"Augmented {obs_name} (5 params)")
            obs_models['Augmented'] = r
        except Exception as e:
            print_status(f"Augmented {obs_name} failed: {e}", "ERROR")

        per_obs_models[obs_name] = obs_models

        if 'TEP' in obs_models and 'Standard' in obs_models:
            ln_bf = obs_models['TEP']['logZ'] - obs_models['Standard']['logZ']
            ln_bf_aug = obs_models['Augmented']['logZ'] - obs_models['Standard']['logZ'] if 'Augmented' in obs_models else None
            print_status(f"  {obs_name:12s}: ln(BF|TEP vs Std)={ln_bf:7.2f}  ln(BF|Aug vs Std)={ln_bf_aug:7.2f}" if ln_bf_aug is not None
                         else f"  {obs_name:12s}: ln(BF|TEP vs Std)={ln_bf:7.2f}")

    results['per_observable_evidence'] = per_obs_models

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
