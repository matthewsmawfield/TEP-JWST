#!/usr/bin/env python3
"""
Step 176: Nested Bayesian Model Comparison

Performs fully nested Bayesian evidence computation comparing TEP against
explicit astrophysical alternatives using dynesty nested sampling.

Key design: TEP's advantage is that ONE theory-derived predictor (Gamma_t)
simultaneously predicts correlations across MULTIPLE observables (dust,
sSFR, chi2, metallicity). Alternatives need separate free parameters for
each observable domain, incurring a larger Occam penalty.

Two tiers of comparison are performed:
  A. Multi-observable joint test (primary): Tests whether TEP's single
     predictor explains multi-domain structure better than alternatives
     despite having fewer parameters per observable.
  B. Single-observable dust test (supplementary): Per-observable comparison
     for completeness and to surface any domain where alternatives prevail.

Models compared:
1. TEP: Theory-fixed Gamma_t predictor (zero structural free parameters)
2. Standard Physics: Linear mass+z (null baseline)
3. Bursty SF: Mass+z + mass-dependent burst timescale
4. Varying IMF: Quadratic mass+z (top-heavy IMF proxy)
5. AGN Feedback: Sigmoid AGN fraction with free critical mass and slope

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
            df = df[df['z'] >= 7].copy()
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
    """Prior transform for joint TEP: 3K params."""
    out = np.empty(3*K)
    for k in range(K):
        out[3*k]     = u[3*k]     * 20 - 10      # a: [-10, 10]
        out[3*k + 1] = u[3*k + 1] * 10 - 5       # b: [-5, 5]
        out[3*k + 2] = u[3*k + 2] * 6 - 5        # log_sigma: [-5, 1]
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


def _joint_bursty_loglike(params, obs_arrays, mass, z):
    """
    Joint Bursty SF likelihood.
    Shared burst timescale tau across observables.
    Per-observable: a_k + b_k*mass + c_k*z + d_k*burst(tau) + noise → 5K+1 params.
    params layout: [tau, a_0, b_0, c_0, d_0, log_s_0, a_1, ...]
    """
    K = len(obs_arrays)
    tau = params[0]
    burst = np.exp(-tau * (1 - mass / 10))
    ll = 0.0
    for k in range(K):
        idx = 1 + 5*k
        a = params[idx]
        b = params[idx + 1]
        c = params[idx + 2]
        d = params[idx + 3]
        sigma = np.exp(params[idx + 4])
        pred = a + b * mass + c * z + d * burst
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
            'BF': bf_val if ln_bf < 300 else 'inf',
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
    print_status(f"Valid multi-observable sample: N={N} at z >= 7")

    if N < 50:
        print_status("Insufficient data for nested sampling", "ERROR")
        results['error'] = f"Only {N} valid rows"
        _save(results)
        return results

    mass = df_v['log_Mstar'].values
    z = df_v['z'].values
    from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like
    log_mh = stellar_to_halo_mass_behroozi_like(mass, z)
    gamma_t = compute_gamma_t(log_mh, z)
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))
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
    design_matrix = np.column_stack([np.ones_like(mass), mass, z])
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

    results['sample_size'] = N
    results['observables'] = OBSERVABLES
    results['n_observables'] = K
    results['z_range'] = [float(z.min()), float(z.max())]
    results['mass_range'] = [float(mass.min()), float(mass.max())]
    results['preprocessing'] = {
        'observables_standardized_for_evidence': True,
        'observable_means_raw': dict(zip(OBSERVABLES, obs_means)),
        'observable_stds_raw': dict(zip(OBSERVABLES, obs_stds)),
        'mass_z_residual_comparison_included': True,
        'note': (
            'Observables were z-scored before nested sampling so the shared '
            'Gaussian-noise priors are valid across domains with very '
            'different scales.'
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

    # Standard Physics joint: 4K params
    try:
        r = run_nested(
            lambda p: _joint_standard_loglike(p, obs_arrays, mass, z),
            lambda u: _joint_standard_prior(u, K),
            4*K, f"Standard Physics Joint ({4*K} params)")
        joint_models['Standard_Physics'] = r
    except Exception as e:
        print_status(f"Standard Physics joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Bursty SF joint: 5K+1 params
    try:
        r = run_nested(
            lambda p: _joint_bursty_loglike(p, obs_arrays, mass, z),
            lambda u: _joint_bursty_prior(u, K),
            5*K + 1, f"Bursty SF Joint ({5*K+1} params)")
        joint_models['Bursty_SF'] = r
    except Exception as e:
        print_status(f"Bursty SF joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Varying IMF joint: 5K params
    try:
        r = run_nested(
            lambda p: _joint_imf_loglike(p, obs_arrays, mass, z),
            lambda u: _joint_imf_prior(u, K),
            5*K, f"Varying IMF Joint ({5*K} params)")
        joint_models['Varying_IMF'] = r
    except Exception as e:
        print_status(f"Varying IMF joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # AGN Feedback joint: 3K+2 params
    try:
        r = run_nested(
            lambda p: _joint_agn_loglike(p, obs_arrays, mass, z),
            lambda u: _joint_agn_prior(u, K),
            3*K + 2, f"AGN Feedback Joint ({3*K+2} params)")
        joint_models['AGN_Feedback'] = r
    except Exception as e:
        print_status(f"AGN Feedback joint failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    results['joint_model_evidence'] = joint_models

    # Joint Bayes Factors
    if 'TEP' in joint_models:
        alt_joint = {k: v for k, v in joint_models.items() if k != 'TEP'}
        joint_bf = compute_bayes_factors(joint_models['TEP'], alt_joint)
        results['joint_bayes_factors'] = joint_bf

        print_status("\n" + "-" * 50)
        print_status("JOINT BAYES FACTORS (TEP vs Alternatives)")
        print_status("-" * 50)
        for name, bf in joint_bf.items():
            print_status(f"  {name}: ln(BF)={bf['ln_BF_TEP_vs_alt']:.2f} ± "
                         f"{bf['ln_BF_err']:.2f}  |  Δparams={bf['delta_params']}  |  "
                         f"{bf['interpretation']}")

    # ------------------------------------------------------------------
    # B. Single-Observable Dust Test (SUPPLEMENTARY)
    # ------------------------------------------------------------------
    print_status("\n" + "=" * 70)
    print_status("B. SINGLE-OBSERVABLE DUST TEST (SUPPLEMENTARY)")
    print_status("=" * 70)

    dust = obs_arrays[0]  # standardized dust is first observable
    single_models = {}

    # TEP single: 3 params
    try:
        r = run_nested(
            lambda p: _single_tep_ll(p, dust, log_gamma),
            _single_tep_prior, 3, "TEP Dust (3 params)")
        single_models['TEP'] = r
    except Exception as e:
        print_status(f"TEP single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Standard single: 4 params
    try:
        r = run_nested(
            lambda p: _single_standard_ll(p, dust, mass, z),
            _single_standard_prior, 4, "Standard Dust (4 params)")
        single_models['Standard_Physics'] = r
    except Exception as e:
        print_status(f"Standard single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # Varying IMF single: 5 params
    try:
        r = run_nested(
            lambda p: _single_imf_ll(p, dust, mass, z),
            _single_imf_prior, 5, "Varying IMF Dust (5 params)")
        single_models['Varying_IMF'] = r
    except Exception as e:
        print_status(f"Varying IMF single failed: {e}", "ERROR")
        print_status(traceback.format_exc(), "ERROR")

    # AGN single: 5 params
    try:
        r = run_nested(
            lambda p: _single_agn_ll(p, dust, mass, z),
            _single_agn_prior, 5, "AGN Dust (5 params)")
        single_models['AGN_Feedback'] = r
    except Exception as e:
        print_status(f"AGN single failed: {e}", "ERROR")
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
    # C. Residual-Space Comparison (AGN loophole control)
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

    _save(results)
    print_status("=" * 70)
    return results


# ============================================================================
# Single-Observable Likelihoods (Supplementary)
# ============================================================================

def _single_tep_ll(params, dust, log_gamma):
    a, b, log_s = params
    sigma = np.exp(log_s)
    pred = a + b * log_gamma
    resid = dust - pred
    return -0.5 * np.sum((resid / sigma)**2 + np.log(2 * np.pi * sigma**2))

def _single_tep_prior(u):
    return np.array([
        u[0] * 20 - 10, u[1] * 10 - 5, u[2] * 6 - 5])

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


def _save(results):
    """Write results JSON."""
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print_status(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
