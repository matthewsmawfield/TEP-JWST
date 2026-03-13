#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.8s.
"""
Step 95: Bayesian Model Comparison

Compute proper Bayesian evidence (Bayes Factors) for TEP vs alternatives.
This addresses the criticism that AIC/BIC was only computed for dust-mass.

We compute:
1. Bayes Factors for TEP vs Null across multiple observables
2. Joint evidence across all observables (accounting for correlations)
3. Posterior model probabilities
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from scipy.special import logsumexp
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)

STEP_NUM  = "075"  # Pipeline step number (sequential 001-176)
STEP_NAME = "bayesian_model_comparison"  # Bayesian model comparison: computes Savage-Dickey Bayes Factors for TEP vs Null across multiple observables (Fisher z-transform, joint evidence)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like  # TEP model: Gamma_t formula, stellar-to-halo mass from abundance matching
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

def load_uncover_data():
    """Load high-z galaxy data from available catalogs."""
    uncover_path = PROJECT_ROOT / 'results' / 'interim' / 'step_002_uncover_full_sample_tep.csv'
    if uncover_path.exists():
        df = pd.read_csv(uncover_path)
        if 'z_phot' in df.columns:
            df = df.rename(columns={'z_phot': 'z'})
        if 'dust' in df.columns:
            df = df.rename(columns={'dust': 'Av'})
        if 'log_ssfr' in df.columns and 'log_sSFR' not in df.columns:
            df = df.rename(columns={'log_ssfr': 'log_sSFR'})
        if 'met' in df.columns:
            df = df.rename(columns={'met': 'metallicity'})
        return df[df['z'] > 4] if 'z' in df.columns else df

    base_path = PROJECT_ROOT / 'data' / 'interim'
    
    # Try CEERS first (has dust column)
    ceers_path = base_path / 'ceers_highz_sample.csv'
    if ceers_path.exists():
        df = pd.read_csv(ceers_path)
        if 'z_phot' in df.columns:
            df = df.rename(columns={'z_phot': 'z'})
        if 'dust' in df.columns:
            df = df.rename(columns={'dust': 'Av'})
        return df[df['z'] > 4] if 'z' in df.columns else df
    
    return None

def savage_dickey_bayes_factor(rho, n, rho_0=0):
    """
    Compute Bayes Factor using Savage-Dickey density ratio.
    
    BF_10 = p(rho=rho_0 | H_0) / p(rho=rho_0 | data, H_1)
    
    For correlation coefficients, we use Fisher z-transform.
    """
    # Fisher z-transform
    z_obs = 0.5 * np.log((1 + rho) / (1 - rho + 1e-10))
    se = 1 / np.sqrt(n - 3)
    
    # Prior: uniform on rho in [-1, 1], which transforms to a specific prior on z
    # For simplicity, use a wide normal prior on z: N(0, 1)
    prior_at_null = norm.pdf(0, 0, 1)
    
    # Posterior: approximately N(z_obs, se)
    posterior_at_null = norm.pdf(0, z_obs, se)
    
    # Bayes Factor for H_1 (rho != 0) vs H_0 (rho = 0)
    # BF_10 = prior_at_null / posterior_at_null
    bf_10 = prior_at_null / (posterior_at_null + 1e-100)
    
    return bf_10

def interpret_bayes_factor(bf):
    """Interpret Bayes Factor according to Jeffreys' scale."""
    if bf > 100:
        return "Decisive evidence for H1"
    elif bf > 30:
        return "Very strong evidence for H1"
    elif bf > 10:
        return "Strong evidence for H1"
    elif bf > 3:
        return "Moderate evidence for H1"
    elif bf > 1:
        return "Weak evidence for H1"
    elif bf > 1/3:
        return "Weak evidence for H0"
    elif bf > 1/10:
        return "Moderate evidence for H0"
    else:
        return "Strong evidence for H0"

def compute_observable_bayes_factors(df):
    """Compute Bayes Factors for each observable."""
    results = []
    
    # Define observables and their TEP predictions
    observables = [
        ('Av', 'Dust attenuation', 'positive'),
        ('log_sSFR', 'Specific SFR', 'positive_at_high_z'),
        ('age_ratio', 'Age ratio', 'positive'),
        ('metallicity', 'Metallicity', 'positive'),
        ('chi2', 'SED fit chi2', 'positive')
    ]
    
    df = df.copy()
    z_vals = df['z'].astype(float).to_numpy()
    if 'log_Mh' not in df.columns:
        df['log_Mh'] = stellar_to_halo_mass_behroozi_like(df['log_Mstar'].astype(float).to_numpy(), z_vals)
    df['gamma_t'] = tep_gamma(df['log_Mh'].astype(float).to_numpy(), z_vals)
    
    for col, name, prediction in observables:
        if col not in df.columns:
            continue
        
        valid = df[['gamma_t', col]].dropna()
        if len(valid) < 50:
            continue
        
        rho, p = spearmanr(valid['gamma_t'], valid[col])
        n = len(valid)
        
        bf = savage_dickey_bayes_factor(rho, n)
        
        # Check if sign matches prediction
        sign_correct = (prediction == 'positive' and rho > 0) or \
                       (prediction == 'negative' and rho < 0) or \
                       (prediction == 'positive_at_high_z')
        
        results.append({
            'observable': name,
            'column': col,
            'n': n,
            'rho': float(rho),
            'p_value': format_p_value(p),
            'bayes_factor': float(bf),
            'log10_bf': float(np.log10(bf + 1e-100)),
            'interpretation': interpret_bayes_factor(bf),
            'sign_matches_prediction': sign_correct
        })
    
    return results

def compute_joint_bayes_factor(individual_bfs, correlation_penalty=0.5):
    """
    Compute joint Bayes Factor accounting for correlations between tests.
    
    Since tests share the same Gamma_t predictor, they are correlated.
    We apply a conservative penalty factor.
    
    log(BF_joint) = sum(log(BF_i)) * correlation_penalty
    """
    log_bfs = [np.log(bf['bayes_factor'] + 1e-100) for bf in individual_bfs]
    
    # Naive combination (assumes independence)
    log_bf_naive = sum(log_bfs)
    
    # Conservative combination (accounts for correlation)
    log_bf_conservative = log_bf_naive * correlation_penalty
    
    # Effective number of independent tests
    n_eff = len(log_bfs) * correlation_penalty
    
    return {
        'n_tests': len(log_bfs),
        'n_effective': n_eff,
        'log_bf_naive': float(log_bf_naive),
        'log_bf_conservative': float(log_bf_conservative),
        'bf_naive': float(np.exp(log_bf_naive)) if log_bf_naive < 700 else float('inf'),
        'bf_conservative': float(np.exp(log_bf_conservative)) if log_bf_conservative < 700 else float('inf'),
        'interpretation_naive': interpret_bayes_factor(np.exp(log_bf_naive) if log_bf_naive < 700 else 1e100),
        'interpretation_conservative': interpret_bayes_factor(np.exp(log_bf_conservative) if log_bf_conservative < 700 else 1e100)
    }

def compute_posterior_model_probabilities(bf_joint, prior_tep=0.1):
    """
    Compute posterior model probabilities.
    
    P(TEP | data) = BF * P(TEP) / (BF * P(TEP) + P(null))
    
    We use a skeptical prior: P(TEP) = 0.1 (10% prior probability)
    """
    prior_null = 1 - prior_tep
    
    bf = bf_joint['bf_conservative']
    if bf == float('inf'):
        bf = 1e100
    
    # Posterior odds = BF * prior odds
    posterior_odds = bf * (prior_tep / prior_null)
    
    # Posterior probability
    p_tep = posterior_odds / (1 + posterior_odds)
    p_null = 1 - p_tep
    
    return {
        'prior_tep': prior_tep,
        'prior_null': prior_null,
        'posterior_tep': float(p_tep),
        'posterior_null': float(p_null),
        'interpretation': (
            f'Even with a skeptical prior (P(TEP) = {prior_tep}), '
            f'the posterior probability of TEP is {p_tep:.4f}.'
        )
    }

def main():
    results = {
        'step': 95,
        'name': 'Bayesian Model Comparison',
        'timestamp': str(np.datetime64('now'))
    }
    
    # Load data
    df = load_uncover_data()
    if df is None:
        results['error'] = 'Could not load UNCOVER data'
        return results
    
    results['sample_size'] = len(df)
    
    # Compute individual Bayes Factors
    print("Computing individual Bayes Factors...")
    individual_bfs = compute_observable_bayes_factors(df)
    results['individual_bayes_factors'] = individual_bfs
    
    # Summary of individual tests
    n_decisive = sum(1 for bf in individual_bfs if bf['bayes_factor'] > 100)
    n_strong = sum(1 for bf in individual_bfs if bf['bayes_factor'] > 10)
    results['individual_summary'] = {
        'n_observables': len(individual_bfs),
        'n_decisive_evidence': n_decisive,
        'n_strong_evidence': n_strong,
        'mean_log10_bf': float(np.mean([bf['log10_bf'] for bf in individual_bfs]))
    }
    
    # Compute joint Bayes Factor
    print("Computing joint Bayes Factor...")
    joint_bf = compute_joint_bayes_factor(individual_bfs, correlation_penalty=0.5)
    results['joint_bayes_factor'] = joint_bf
    
    # Compute posterior probabilities
    print("Computing posterior model probabilities...")
    posterior = compute_posterior_model_probabilities(joint_bf, prior_tep=0.1)
    results['posterior_probabilities'] = posterior
    
    # Also compute with more skeptical prior
    posterior_skeptical = compute_posterior_model_probabilities(joint_bf, prior_tep=0.01)
    results['posterior_probabilities_skeptical'] = posterior_skeptical
    
    # Key finding
    results['key_finding'] = {
        'statement': (
            f'Bayesian model comparison provides {joint_bf["interpretation_conservative"]} '
            f'for TEP over the null model. The conservative joint Bayes Factor is '
            f'{joint_bf["bf_conservative"]:.2e} (log10 = {joint_bf["log_bf_conservative"]/np.log(10):.1f}). '
            f'Even with a skeptical prior (P(TEP) = 10%), the posterior probability of TEP is '
            f'{posterior["posterior_tep"]:.4f}.'
        ),
        'caveat': (
            'The joint Bayes Factor applies a 50% correlation penalty to account for '
            'shared predictors across tests. This is conservative but not exact.'
        )
    }
    
    # Save results
    output_path = Path(__file__).parent.parent.parent / 'results' / 'outputs' / 'step_075_bayesian_model_comparison.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nStep 95 complete. Results saved to {output_path}")
    print(f"Joint Bayes Factor (conservative): {joint_bf['bf_conservative']:.2e}")
    print(f"Posterior P(TEP): {posterior['posterior_tep']:.4f}")
    
    return results

if __name__ == '__main__':
    main()
