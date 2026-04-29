#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.1s.
"""
Step 120: Alternative Model Comparison

Compares TEP against specific alternative explanations for high-z anomalies:
1. Top-heavy IMF
2. Stochastic star formation
3. AGN contamination
4. Standard physics (null model)

Computes AIC/BIC for each model and provides quantitative model selection.

Author: TEP-JWST Pipeline
"""

import sys

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))
import json
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"  # Results root directory
OUTPUTS_DIR = RESULTS_DIR / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_DIR = RESULTS_DIR / "figures"  # Publication figures directory (PNG/PDF for manuscript)
INTERIM_DIR = RESULTS_DIR / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)

STEP_NUM = "098"  # Pipeline step number (sequential 001-176)
STEP_NAME = "alternative_model_comparison"  # Alternative model comparison: AIC/BIC model selection comparing TEP vs top-heavy IMF, stochastic SF, AGN contamination, standard physics (null)

LOGS_DIR = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


def compute_aic_bic(log_likelihood, n_params, n_data):
    """Compute AIC and BIC from log-likelihood."""
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_data) - 2 * log_likelihood
    return aic, bic


def fit_null_model(dust, mass, z):
    """
    Null model: dust depends only on mass and redshift (no TEP).
    
    dust = a + b*mass + c*z + noise
    """
    from scipy.optimize import minimize
    
    def neg_log_likelihood(params):
        a, b, c, log_sigma = params
        sigma = np.exp(log_sigma)
        pred = a + b * mass + c * z
        residuals = dust - pred
        ll = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
        return -ll
    
    # Initial guess
    x0 = [0, 0.1, 0, np.log(0.5)]
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
    
    n_params = 4  # a, b, c, sigma
    log_likelihood = -result.fun
    
    return log_likelihood, n_params, result.x


def fit_tep_model(dust, mass, z, gamma_t):
    """
    TEP model: dust depends on Gamma_t (which encodes mass and z).
    
    dust = a + b*log(Gamma_t) + noise
    """
    from scipy.optimize import minimize
    
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))
    
    def neg_log_likelihood(params):
        a, b, log_sigma = params
        sigma = np.exp(log_sigma)
        pred = a + b * log_gamma
        residuals = dust - pred
        ll = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
        return -ll
    
    x0 = [0.5, 0.3, np.log(0.4)]
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
    
    n_params = 3  # a, b, sigma
    log_likelihood = -result.fun
    
    return log_likelihood, n_params, result.x


def fit_topheavy_imf_model(dust, mass, z):
    """
    Top-heavy IMF model: massive galaxies have more dust due to IMF variation.
    
    Assumes IMF slope varies with mass: alpha_IMF = kappa_gal + delta * (mass - 10)
    This produces more massive stars -> more dust per unit stellar mass.
    
    dust = a + b*mass + c*mass^2 + d*z + noise
    """
    from scipy.optimize import minimize
    
    def neg_log_likelihood(params):
        a, b, c, d, log_sigma = params
        sigma = np.exp(log_sigma)
        pred = a + b * mass + c * mass**2 + d * z
        residuals = dust - pred
        ll = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
        return -ll
    
    x0 = [0, 0.1, 0.01, 0, np.log(0.5)]
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
    
    n_params = 5  # a, b, c, d, sigma
    log_likelihood = -result.fun
    
    return log_likelihood, n_params, result.x


def fit_stochastic_sf_model(dust, mass, z, ssfr=None):
    """
    Stochastic SF model: dust correlates with recent star formation bursts.
    
    If sSFR available, use it; otherwise use mass as proxy.
    
    dust = a + b*mass + c*z + d*ssfr + noise
    """
    from scipy.optimize import minimize
    
    if ssfr is None:
        # Use mass as proxy for burstiness (lower mass = more stochastic)
        ssfr = -0.5 * (mass - 9)  # Proxy
    
    def neg_log_likelihood(params):
        a, b, c, d, log_sigma = params
        sigma = np.exp(log_sigma)
        pred = a + b * mass + c * z + d * ssfr
        residuals = dust - pred
        ll = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
        return -ll
    
    x0 = [0, 0.1, 0, 0.1, np.log(0.5)]
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
    
    n_params = 5  # a, b, c, d, sigma
    log_likelihood = -result.fun
    
    return log_likelihood, n_params, result.x


def fit_agn_contamination_model(dust, mass, z):
    """
    AGN contamination model: apparent dust is actually AGN reddening.
    
    AGN fraction increases with mass, creating apparent dust-mass correlation.
    
    dust = a + b*f_AGN(mass) + c*z + noise
    where f_AGN = sigmoid(mass - 10)
    """
    from scipy.optimize import minimize
    
    # AGN fraction model: sigmoid centered at log M* = 10
    f_agn = 1 / (1 + np.exp(-(mass - 10)))
    
    def neg_log_likelihood(params):
        a, b, c, log_sigma = params
        sigma = np.exp(log_sigma)
        pred = a + b * f_agn + c * z
        residuals = dust - pred
        ll = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
        return -ll
    
    x0 = [0, 0.5, 0, np.log(0.5)]
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
    
    n_params = 4  # a, b, c, sigma
    log_likelihood = -result.fun
    
    return log_likelihood, n_params, result.x


def compute_bayes_factor(aic1, aic2):
    """Approximate Bayes factor from AIC difference."""
    # BF ≈ exp(-0.5 * delta_AIC)
    delta_aic = aic1 - aic2
    bf = np.exp(-0.5 * delta_aic)
    return bf


def interpret_bayes_factor(bf):
    """Interpret Bayes factor according to Kass & Raftery (1995)."""
    if bf > 150:
        return "Very strong evidence for Model 2"
    elif bf > 20:
        return "Strong evidence for Model 2"
    elif bf > 3:
        return "Positive evidence for Model 2"
    elif bf > 1:
        return "Weak evidence for Model 2"
    elif bf > 1/3:
        return "Weak evidence for Model 1"
    elif bf > 1/20:
        return "Positive evidence for Model 1"
    elif bf > 1/150:
        return "Strong evidence for Model 1"
    else:
        return "Very strong evidence for Model 1"


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Alternative Model Comparison")
    print_status("=" * 70)
    
    # Load data
    possible_paths = [
        INTERIM_DIR / "step_002_uncover_full_sample_tep.csv",
        INTERIM_DIR / "step_002_uncover_multi_property_sample_tep.csv",
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    
    if data_path is None:
        print_status("No TEP data file found", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded {len(df)} galaxies")
    
    # Filter to z > 8 where TEP effects are strongest
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    df_highz = df[df[z_col] > 8].copy()
    
    if len(df_highz) < 50:
        print_status(f"Only {len(df_highz)} galaxies at z > 8, using z > 6")
        df_highz = df[df[z_col] > 6].copy()
    
    print_status(f"High-z sample: {len(df_highz)} galaxies")
    
    # Check required columns
    required = ['gamma_t', 'dust']
    mass_col = 'log_Mstar' if 'log_Mstar' in df_highz.columns else 'log_mass'
    for col in required:
        if col not in df_highz.columns:
            print_status(f"Missing column: {col}", "ERROR")
            return
    
    # Extract data
    dust = df_highz['dust'].values
    mass = df_highz[mass_col].values
    z = df_highz[z_col].values
    gamma_t = df_highz['gamma_t'].values
    
    # Handle missing values
    valid = ~(np.isnan(dust) | np.isnan(mass) | np.isnan(z) | np.isnan(gamma_t))
    dust = dust[valid]
    mass = mass[valid]
    z = z[valid]
    gamma_t = gamma_t[valid]
    
    n_data = len(dust)
    print_status(f"Valid data points: {n_data}")
    
    if n_data < 30:
        print_status("Insufficient data for model comparison", "ERROR")
        return
    
    # Fit all models
    print_status("\nFitting models...")
    
    models = {}
    
    # 1. Null model (standard physics)
    ll_null, k_null, params_null = fit_null_model(dust, mass, z)
    aic_null, bic_null = compute_aic_bic(ll_null, k_null, n_data)
    models['null'] = {
        'name': 'Standard Physics (Null)',
        'description': 'Dust depends on mass and redshift only',
        'log_likelihood': ll_null,
        'n_params': k_null,
        'aic': aic_null,
        'bic': bic_null,
    }
    print_status(f"  Null: LL={ll_null:.1f}, AIC={aic_null:.1f}, BIC={bic_null:.1f}")
    
    # 2. TEP model
    ll_tep, k_tep, params_tep = fit_tep_model(dust, mass, z, gamma_t)
    aic_tep, bic_tep = compute_aic_bic(ll_tep, k_tep, n_data)
    models['tep'] = {
        'name': 'TEP (Temporal Equivalence)',
        'description': 'Dust depends on Gamma_t (temporal enhancement factor)',
        'log_likelihood': ll_tep,
        'n_params': k_tep,
        'aic': aic_tep,
        'bic': bic_tep,
    }
    print_status(f"  TEP: LL={ll_tep:.1f}, AIC={aic_tep:.1f}, BIC={bic_tep:.1f}")
    
    # 3. Top-heavy IMF model
    ll_imf, k_imf, params_imf = fit_topheavy_imf_model(dust, mass, z)
    aic_imf, bic_imf = compute_aic_bic(ll_imf, k_imf, n_data)
    models['topheavy_imf'] = {
        'name': 'Top-Heavy IMF',
        'description': 'IMF varies with mass, producing more dust in massive galaxies',
        'log_likelihood': ll_imf,
        'n_params': k_imf,
        'aic': aic_imf,
        'bic': bic_imf,
    }
    print_status(f"  Top-heavy IMF: LL={ll_imf:.1f}, AIC={aic_imf:.1f}, BIC={bic_imf:.1f}")
    
    # 4. Stochastic SF model
    ssfr = df_highz['log_ssfr'].values[valid] if 'log_ssfr' in df_highz.columns else None
    ll_stoch, k_stoch, params_stoch = fit_stochastic_sf_model(dust, mass, z, ssfr)
    aic_stoch, bic_stoch = compute_aic_bic(ll_stoch, k_stoch, n_data)
    models['stochastic_sf'] = {
        'name': 'Stochastic Star Formation',
        'description': 'Dust correlates with recent star formation bursts',
        'log_likelihood': ll_stoch,
        'n_params': k_stoch,
        'aic': aic_stoch,
        'bic': bic_stoch,
    }
    print_status(f"  Stochastic SF: LL={ll_stoch:.1f}, AIC={aic_stoch:.1f}, BIC={bic_stoch:.1f}")
    
    # 5. AGN contamination model
    ll_agn, k_agn, params_agn = fit_agn_contamination_model(dust, mass, z)
    aic_agn, bic_agn = compute_aic_bic(ll_agn, k_agn, n_data)
    models['agn'] = {
        'name': 'AGN Contamination',
        'description': 'Apparent dust is AGN reddening, fraction increases with mass',
        'log_likelihood': ll_agn,
        'n_params': k_agn,
        'aic': aic_agn,
        'bic': bic_agn,
    }
    print_status(f"  AGN: LL={ll_agn:.1f}, AIC={aic_agn:.1f}, BIC={bic_agn:.1f}")
    
    # Compute model rankings
    print_status("\n" + "=" * 70)
    print_status("MODEL COMPARISON RESULTS")
    print_status("=" * 70)
    
    # Rank by AIC
    model_list = [(name, m['aic'], m['bic']) for name, m in models.items()]
    model_list.sort(key=lambda x: x[1])  # Sort by AIC
    
    best_model = model_list[0][0]
    best_aic = model_list[0][1]
    
    print_status("\nRanking by AIC (lower is better):")
    for i, (name, aic, bic) in enumerate(model_list):
        delta_aic = aic - best_aic
        print_status(f"  {i+1}. {models[name]['name']}: AIC={aic:.1f} (ΔAIC={delta_aic:.1f})")
    
    # Compute Bayes factors vs TEP
    print_status("\nBayes Factors (vs TEP):")
    comparisons = {}
    for name, m in models.items():
        if name == 'tep':
            continue
        bf = compute_bayes_factor(m['aic'], aic_tep)
        interpretation = interpret_bayes_factor(bf)
        comparisons[name] = {
            'bayes_factor': bf,
            'interpretation': interpretation,
            'delta_aic': m['aic'] - aic_tep,
            'delta_bic': m['bic'] - bic_tep,
        }
        print_status(f"  TEP vs {models[name]['name']}: BF={1/bf:.1f} ({interpretation})")
    
    # Akaike weights
    print_status("\nAkaike Weights (probability of being best model):")
    aics = np.array([m['aic'] for m in models.values()])
    delta_aics = aics - np.min(aics)
    weights = np.exp(-0.5 * delta_aics) / np.sum(np.exp(-0.5 * delta_aics))
    
    for i, (name, m) in enumerate(models.items()):
        m['akaike_weight'] = weights[i]
        print_status(f"  {m['name']}: w={weights[i]:.3f} ({100*weights[i]:.1f}%)")
    
    # Summary
    print_status("\n" + "=" * 70)
    print_status("SUMMARY")
    print_status("=" * 70)
    
    tep_weight = models['tep']['akaike_weight']
    tep_rank = [i+1 for i, (name, _, _) in enumerate(model_list) if name == 'tep'][0]
    
    if tep_rank == 1:
        conclusion = f"TEP is the BEST model (Akaike weight = {tep_weight:.1%})"
    else:
        conclusion = f"TEP ranks #{tep_rank} (Akaike weight = {tep_weight:.1%})"
    
    print_status(f"\n{conclusion}")
    print_status(f"Best alternative: {models[best_model]['name']}" if best_model != 'tep' else "No alternative beats TEP")
    
    # Key discriminating features
    print_status("\nKey discriminating features:")
    print_status("  - TEP uses fewer parameters (3) than alternatives (4-5)")
    print_status("  - TEP's Gamma_t encodes physics (potential depth) rather than ad-hoc correlations")
    print_status("  - TEP makes testable predictions for screening and spectroscopy")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: Alternative Model Comparison',
        'n_data': n_data,
        'z_range': [float(z.min()), float(z.max())],
        'models': {name: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in m.items()} 
                  for name, m in models.items()},
        'ranking_by_aic': [{'rank': i+1, 'model': name, 'aic': aic, 'delta_aic': aic - best_aic}
                          for i, (name, aic, _) in enumerate(model_list)],
        'comparisons_vs_tep': comparisons,
        'conclusion': {
            'best_model': best_model,
            'tep_rank': tep_rank,
            'tep_akaike_weight': float(tep_weight),
            'interpretation': conclusion,
        }
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_alternative_model_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: AIC comparison
        ax1 = axes[0]
        model_names = [models[name]['name'].replace(' ', '\n') for name, _, _ in model_list]
        aics_sorted = [aic for _, aic, _ in model_list]
        colors = ['green' if model_list[i][0] == 'tep' else 'steelblue' for i in range(len(model_list))]
        
        bars = ax1.barh(range(len(model_list)), aics_sorted, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(model_list)))
        ax1.set_yticklabels(model_names, fontsize=9)
        ax1.set_xlabel('AIC (lower is better)', fontsize=12)
        ax1.set_title('Model Comparison by AIC', fontsize=14)
        ax1.invert_yaxis()
        
        # Add delta AIC labels
        for i, (bar, (name, aic, _)) in enumerate(zip(bars, model_list)):
            delta = aic - best_aic
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'Δ={delta:.0f}', va='center', fontsize=9)
        
        # Panel 2: Akaike weights
        ax2 = axes[1]
        weights_sorted = [models[name]['akaike_weight'] for name, _, _ in model_list]
        colors2 = ['green' if model_list[i][0] == 'tep' else 'steelblue' for i in range(len(model_list))]
        
        ax2.bar(range(len(model_list)), weights_sorted, color=colors2, edgecolor='black')
        ax2.set_xticks(range(len(model_list)))
        ax2.set_xticklabels([m.split('\n')[0][:10] for m in model_names], rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Akaike Weight', fontsize=12)
        ax2.set_title('Model Probability', fontsize=14)
        ax2.set_ylim(0, 1)
        
        # Add percentage labels
        for i, w in enumerate(weights_sorted):
            ax2.text(i, w + 0.02, f'{100*w:.0f}%', ha='center', fontsize=9)
        
        # Panel 3: Parameter count vs fit quality
        ax3 = axes[2]
        for name, m in models.items():
            color = 'green' if name == 'tep' else 'steelblue'
            marker = '*' if name == 'tep' else 'o'
            size = 200 if name == 'tep' else 100
            ax3.scatter(m['n_params'], -m['log_likelihood'], 
                       c=color, s=size, marker=marker, edgecolor='black',
                       label=m['name'].split('(')[0].strip())
        
        ax3.set_xlabel('Number of Parameters', fontsize=12)
        ax3.set_ylabel('Negative Log-Likelihood', fontsize=12)
        ax3.set_title('Parsimony vs Fit Quality', fontsize=14)
        ax3.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_alternative_model_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
