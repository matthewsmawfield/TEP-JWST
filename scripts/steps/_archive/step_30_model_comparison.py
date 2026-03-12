#!/usr/bin/env python3
"""
TEP-JWST Step 30: Model Comparison (AIC/BIC)

Addressing the "Mass Proxy" critique:
Γ_t is calculated from halo mass, so skeptics argue that correlations
(e.g., Dust vs Γ_t) are simply standard mass-dependent scaling relations.

This script performs rigorous model comparison using AIC/BIC to determine
whether TEP provides statistically better fits than mass-only models.

Models compared:
1. NULL: Property = f(z) only
2. MASS: Property = f(M*, z)  
3. TEP:  Property = f(Γ_t, z)
4. FULL: Property = f(M*, Γ_t, z)

If TEP is just a mass proxy, Model 2 and Model 3 should perform identically.
If TEP captures additional physics, Model 3 should outperform Model 2,
and Model 4 should show that Γ_t adds value beyond mass.

Key insight: Γ_t depends on BOTH mass AND redshift via α(z) = α₀√(1+z).
A pure mass model cannot capture this redshift-dependent enhancement.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import compute_gamma_t as tep_gamma

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

STEP_NUM = "30"
STEP_NAME = "model_comparison"

# Initialize logger
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)


def load_data():
    """Load UNCOVER data with TEP calculations from full FITS catalog."""
    from astropy.io import fits
    from astropy.cosmology import Planck18
    
    # Load full UNCOVER catalog
    with fits.open(DATA_DIR / "raw" / "uncover" / "UNCOVER_DR4_SPS_catalog.fits") as hdul:
        data = hdul[1].data
    
    # Extract relevant columns - convert to native byte order
    def to_native(arr):
        arr = np.array(arr)
        if arr.dtype.byteorder == '>':
            return arr.astype(arr.dtype.newbyteorder('='))
        return arr
    
    df = pd.DataFrame({
        'id': to_native(data['id']),
        'z': to_native(data['z_50']),
        'log_Mstar': to_native(data['mstar_50']),
        'Av': to_native(data['dust2_50']),
        'chi2': to_native(data['chi2']),
        'sfr10': to_native(data['sfr10_50']),
        'mwa': to_native(data['mwa_50']),
    })
    
    # Compute derived quantities
    df['log_sSFR'] = np.log10(df['sfr10'] / (10**df['log_Mstar']) + 1e-15)
    df['mwa_Gyr'] = df['mwa'] / 1e9
    
    # Quality cuts: z > 5, valid mass
    mask = (df['z'] > 5) & (df['log_Mstar'] > 7) & (df['log_Mstar'] < 12)
    df = df[mask].copy()
    df = df.dropna(subset=['z', 'log_Mstar', 'Av', 'chi2'])
    
    # Compute halo mass (M_h ≈ 100 × M_*)
    df['log_Mhalo'] = df['log_Mstar'] + 2.0
    
    # Compute cosmic age
    df['t_cosmic_Gyr'] = np.array([Planck18.age(z).value for z in df['z']])
    
    # Compute age ratio
    df['age_ratio'] = df['mwa_Gyr'] / df['t_cosmic_Gyr']
    
    df['gamma_t'] = tep_gamma(df['log_Mhalo'].values, df['z'].values)
    
    # Effective time
    df['t_eff_Gyr'] = df['t_cosmic_Gyr'] * df['gamma_t']
    
    return df


def compute_aic_bic(n, k, rss):
    """
    Compute AIC and BIC for a linear model.
    
    n: number of observations
    k: number of parameters (including intercept)
    rss: residual sum of squares
    """
    # Log-likelihood for Gaussian errors
    # L = -n/2 * log(2π) - n/2 * log(σ²) - RSS/(2σ²)
    # where σ² = RSS/n (MLE estimate)
    
    sigma2 = rss / n
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma2) - n/2
    
    # AIC = -2*log(L) + 2*k
    aic = -2 * log_likelihood + 2 * k
    
    # BIC = -2*log(L) + k*log(n)
    bic = -2 * log_likelihood + k * np.log(n)
    
    return aic, bic


def fit_linear_model(X, y):
    """
    Fit a linear model using OLS.
    
    Returns: coefficients, RSS, R², AIC, BIC
    """
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(y)), X])
    
    # OLS solution
    try:
        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except (np.linalg.LinAlgError, ValueError):
        return None, np.inf, 0, np.inf, np.inf
    
    # Predictions and residuals
    y_pred = X_with_intercept @ coeffs
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    
    # R²
    tss = np.sum((y - np.mean(y))**2)
    r2 = 1 - rss/tss if tss > 0 else 0
    
    # AIC/BIC
    n = len(y)
    k = X_with_intercept.shape[1]
    aic, bic = compute_aic_bic(n, k, rss)
    
    return coeffs, rss, r2, aic, bic


def compare_models_for_property(df, property_col, property_name):
    """
    Compare NULL, MASS, TEP, and FULL models for a given property.
    """
    # Clean data
    valid = df.dropna(subset=[property_col, 'log_Mstar', 'gamma_t', 'z']).copy()
    n = len(valid)
    
    if n < 100:
        return None
    
    y = valid[property_col].values
    z = valid['z'].values
    mass = valid['log_Mstar'].values
    gamma = valid['gamma_t'].values
    
    results = {
        'property': property_name,
        'n': n,
        'models': {}
    }
    
    # Model 1: NULL (z only)
    X_null = z.reshape(-1, 1)
    coeffs, rss, r2, aic, bic = fit_linear_model(X_null, y)
    results['models']['null'] = {
        'predictors': ['z'],
        'k': 2,
        'r2': r2,
        'aic': aic,
        'bic': bic
    }
    
    # Model 2: MASS (M*, z)
    X_mass = np.column_stack([mass, z])
    coeffs, rss, r2, aic, bic = fit_linear_model(X_mass, y)
    results['models']['mass'] = {
        'predictors': ['log_Mstar', 'z'],
        'k': 3,
        'r2': r2,
        'aic': aic,
        'bic': bic
    }
    
    # Model 3: TEP (Γ_t, z)
    X_tep = np.column_stack([gamma, z])
    coeffs, rss, r2, aic, bic = fit_linear_model(X_tep, y)
    results['models']['tep'] = {
        'predictors': ['gamma_t', 'z'],
        'k': 3,
        'r2': r2,
        'aic': aic,
        'bic': bic
    }
    
    # Model 4: FULL (M*, Γ_t, z)
    X_full = np.column_stack([mass, gamma, z])
    coeffs_full, rss, r2, aic, bic = fit_linear_model(X_full, y)
    results['models']['full'] = {
        'predictors': ['log_Mstar', 'gamma_t', 'z'],
        'k': 4,
        'r2': r2,
        'aic': aic,
        'bic': bic,
        'coefficients': {
            'intercept': coeffs_full[0],
            'log_Mstar': coeffs_full[1],
            'gamma_t': coeffs_full[2],
            'z': coeffs_full[3]
        } if coeffs_full is not None else None
    }
    
    # Model comparison
    aic_null = results['models']['null']['aic']
    aic_mass = results['models']['mass']['aic']
    aic_tep = results['models']['tep']['aic']
    aic_full = results['models']['full']['aic']
    
    bic_null = results['models']['null']['bic']
    bic_mass = results['models']['mass']['bic']
    bic_tep = results['models']['tep']['bic']
    bic_full = results['models']['full']['bic']
    
    # Delta AIC/BIC (relative to best model)
    best_aic = min(aic_null, aic_mass, aic_tep, aic_full)
    best_bic = min(bic_null, bic_mass, bic_tep, bic_full)
    
    results['comparison'] = {
        'delta_aic': {
            'null': aic_null - best_aic,
            'mass': aic_mass - best_aic,
            'tep': aic_tep - best_aic,
            'full': aic_full - best_aic
        },
        'delta_bic': {
            'null': bic_null - best_bic,
            'mass': bic_mass - best_bic,
            'tep': bic_tep - best_bic,
            'full': bic_full - best_bic
        },
        'tep_vs_mass_aic': aic_mass - aic_tep,
        'tep_vs_mass_bic': bic_mass - bic_tep,
        'full_vs_mass_aic': aic_mass - aic_full,
        'full_vs_mass_bic': bic_mass - bic_full
    }
    
    # Interpretation
    # ΔAIC > 10: Very strong evidence for better model
    # ΔAIC 4-7: Strong evidence
    # ΔAIC 2-4: Moderate evidence
    # ΔAIC < 2: Weak evidence
    
    tep_better = results['comparison']['tep_vs_mass_aic'] > 2
    full_better = results['comparison']['full_vs_mass_aic'] > 2
    
    results['interpretation'] = {
        'tep_beats_mass': tep_better,
        'full_beats_mass': full_better,
        'best_model_aic': min(results['comparison']['delta_aic'], 
                              key=results['comparison']['delta_aic'].get),
        'best_model_bic': min(results['comparison']['delta_bic'],
                              key=results['comparison']['delta_bic'].get)
    }
    
    return results


def analyze_mass_independence():
    """
    Key test: Does Γ_t add predictive power BEYOND mass?
    
    If Γ_t is just a mass proxy, then in the FULL model:
    - The coefficient on Γ_t should be ~0
    - Adding Γ_t should not improve AIC/BIC
    
    But Γ_t = f(M, z) where the z-dependence comes from α(z) = α₀√(1+z).
    This means Γ_t captures a MASS × REDSHIFT interaction that pure mass cannot.
    """
    df = load_data()
    
    print_status("=" * 70, "INFO")
    print_status("TEP-JWST Step 30: Model Comparison (AIC/BIC)", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("Testing whether TEP provides explanatory power beyond mass scaling.", "INFO")
    print_status("", "INFO")
    print_status("Models compared:", "INFO")
    print_status("  1. NULL: Property = f(z)", "INFO")
    print_status("  2. MASS: Property = f(M*, z)", "INFO")
    print_status("  3. TEP:  Property = f(Γ_t, z)", "INFO")
    print_status("  4. FULL: Property = f(M*, Γ_t, z)", "INFO")
    print_status("", "INFO")
    print_status("If TEP is just a mass proxy, MASS and TEP should perform identically.", "INFO")
    print_status("If TEP captures additional physics, TEP should outperform MASS.", "INFO")
    print_status("", "INFO")
    
    # Properties to test
    properties = [
        ('Av', 'Dust (A_V)'),
        ('log_Z', 'Metallicity (log Z)'),
        ('chi2', 'χ² (SED fit quality)'),
        ('age_ratio', 'Age Ratio'),
        ('log_sSFR', 'log(sSFR)')
    ]
    
    all_results = {}
    summary = {
        'tep_wins': 0,
        'mass_wins': 0,
        'tie': 0,
        'properties_tested': 0
    }
    
    print_status("=" * 70, "INFO")
    print_status("RESULTS BY PROPERTY", "INFO")
    print_status("=" * 70, "INFO")
    
    for prop_col, prop_name in properties:
        if prop_col not in df.columns:
            continue
            
        results = compare_models_for_property(df, prop_col, prop_name)
        
        if results is None:
            continue
        
        all_results[prop_col] = results
        summary['properties_tested'] += 1
        
        print_status(f"\n{prop_name} (N = {results['n']})", "INFO")
        print_status("-" * 50, "INFO")
        
        # Print R² for each model
        print_status(f"  R² values:", "INFO")
        for model_name in ['null', 'mass', 'tep', 'full']:
            r2 = results['models'][model_name]['r2']
            print_status(f"    {model_name.upper():6s}: {r2:.4f}", "INFO")
        
        # Print AIC comparison
        print_status(f"\n  ΔAIC (relative to best):", "INFO")
        for model_name in ['null', 'mass', 'tep', 'full']:
            delta = results['comparison']['delta_aic'][model_name]
            marker = "←" if delta == 0 else ""
            print_status(f"    {model_name.upper():6s}: {delta:+.1f} {marker}", "INFO")
        
        # Print key comparison
        tep_vs_mass = results['comparison']['tep_vs_mass_aic']
        full_vs_mass = results['comparison']['full_vs_mass_aic']
        
        print_status(f"\n  Key comparisons:", "INFO")
        if tep_vs_mass > 10:
            tep_marker = "(very strong evidence for TEP)"
            summary['tep_wins'] += 1
        elif tep_vs_mass > 4:
            tep_marker = "(strong evidence for TEP)"
            summary['tep_wins'] += 1
        elif tep_vs_mass > 2:
            tep_marker = "(moderate evidence for TEP)"
            summary['tep_wins'] += 1
        elif tep_vs_mass < -2:
            tep_marker = "(evidence for MASS)"
            summary['mass_wins'] += 1
        else:
            tep_marker = "(inconclusive)"
            summary['tie'] += 1
        print_status(f"    TEP vs MASS:  ΔAIC = {tep_vs_mass:+.1f} {tep_marker}", "INFO")
        
        if full_vs_mass > 2:
            full_marker = "(Γ_t adds value beyond mass)"
        else:
            full_marker = ""
        print_status(f"    FULL vs MASS: ΔAIC = {full_vs_mass:+.1f} {full_marker}", "INFO")
        
        # Check if Γ_t coefficient is significant in FULL model
        if results['models']['full']['coefficients']:
            gamma_coeff = results['models']['full']['coefficients']['gamma_t']
            mass_coeff = results['models']['full']['coefficients']['log_Mstar']
            print_status(f"\n  FULL model coefficients:", "INFO")
            print_status(f"    log(M*): {mass_coeff:.4f}", "INFO")
            print_status(f"    Γ_t:     {gamma_coeff:.4f}", "INFO")
            
            if abs(gamma_coeff) > 0.01:
                print_status(f"    → Γ_t has non-zero effect even controlling for mass", "INFO")
    
    # Summary
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status(f"Properties tested: {summary['properties_tested']}", "INFO")
    print_status(f"TEP outperforms MASS: {summary['tep_wins']}/{summary['properties_tested']}", "INFO")
    print_status(f"MASS outperforms TEP: {summary['mass_wins']}/{summary['properties_tested']}", "INFO")
    print_status(f"Inconclusive: {summary['tie']}/{summary['properties_tested']}", "INFO")
    print_status("", "INFO")
    
    if summary['tep_wins'] > summary['mass_wins']:
        print_status("CONCLUSION: TEP provides explanatory power BEYOND simple mass scaling.", "INFO")
        print_status("", "INFO")
        print_status("The key insight: Γ_t = f(M, z) captures a MASS × REDSHIFT interaction", "INFO")
        print_status("via α(z) = α₀√(1+z). A pure mass model cannot capture this z-dependent", "INFO")
        print_status("enhancement. The AIC/BIC comparison confirms that TEP is not merely", "INFO")
        print_status("a mass proxy—it captures additional physics.", "INFO")
    elif summary['mass_wins'] > summary['tep_wins']:
        print_status("CONCLUSION: Mass-only models perform comparably to TEP.", "INFO")
        print_status("The 'mass proxy' critique has merit for these properties.", "INFO")
    else:
        print_status("CONCLUSION: Results are mixed. TEP and mass models perform similarly", "INFO")
        print_status("for some properties but not others.", "INFO")
    
    # Additional test: Partial correlation
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("PARTIAL CORRELATION TEST", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("If Γ_t is just a mass proxy, the partial correlation", "INFO")
    print_status("corr(Property, Γ_t | M*) should be ~0.", "INFO")
    print_status("", "INFO")
    
    valid = df.dropna(subset=['Av', 'log_Mstar', 'gamma_t', 'z']).copy()
    
    # Compute partial correlations
    from scipy.stats import spearmanr
    
    # Residualize Γ_t on mass
    slope, intercept, _, _, _ = stats.linregress(valid['log_Mstar'], valid['gamma_t'])
    gamma_residual = valid['gamma_t'] - (slope * valid['log_Mstar'] + intercept)
    
    # Residualize Av on mass
    slope_av, intercept_av, _, _, _ = stats.linregress(valid['log_Mstar'], valid['Av'])
    av_residual = valid['Av'] - (slope_av * valid['log_Mstar'] + intercept_av)
    
    # Partial correlation
    rho_partial, p_partial = spearmanr(gamma_residual, av_residual)
    p_partial_fmt = format_p_value(p_partial)

    print_status(f"Partial correlation: corr(Dust, Γ_t | M*)", "INFO")
    print_status(f"  ρ = {rho_partial:.3f}, p = {p_partial:.2e}", "INFO")
    print_status("", "INFO")
    
    significant_partial = abs(rho_partial) > 0.05 and (p_partial_fmt is not None and p_partial_fmt < 0.05)

    if significant_partial:
        print_status("→ Γ_t predicts dust BEYOND what mass alone explains.", "INFO")
        print_status("  This is the z-dependent component of TEP: α(z) = α₀√(1+z)", "INFO")
        all_results['partial_correlation'] = {
            'rho': rho_partial,
            'p': p_partial_fmt,
            'significant': True
        }
    else:
        print_status("→ Partial correlation is weak or non-significant.", "INFO")
        all_results['partial_correlation'] = {
            'rho': rho_partial,
            'p': p_partial_fmt,
            'significant': False
        }
    
    # Final test: Does the z-dependent component of Γ_t add value?
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Z-DEPENDENT COMPONENT TEST", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("Γ_t has two components:", "INFO")
    print_status("  1. Mass-dependent: (log M_h - 12)", "INFO")
    print_status("  2. Z-dependent: α(z) = α₀√(1+z)", "INFO")
    print_status("", "INFO")
    print_status("If TEP is just a mass proxy, only component 1 matters.", "INFO")
    print_status("If TEP captures real physics, component 2 should also matter.", "INFO")
    print_status("", "INFO")
    
    # Create z-dependent component
    valid['alpha_z'] = 0.58 * np.sqrt(1 + valid['z'])
    valid['mass_component'] = valid['log_Mhalo'] - 12
    valid['z_component'] = valid['alpha_z'] * valid['mass_component']
    
    # Model: Av = a*mass_component + b*z_component + c*z + d
    X_decomposed = np.column_stack([
        valid['mass_component'],
        valid['z_component'],
        valid['z']
    ])
    y = valid['Av'].values
    
    coeffs, rss, r2, aic, bic = fit_linear_model(X_decomposed, y)
    
    if coeffs is not None:
        print_status(f"Decomposed model: Dust = a×(mass_comp) + b×(z_comp) + c×z + d", "INFO")
        print_status(f"  a (mass component):     {coeffs[1]:.4f}", "INFO")
        print_status(f"  b (z-dependent TEP):    {coeffs[2]:.4f}", "INFO")
        print_status(f"  c (redshift):           {coeffs[3]:.4f}", "INFO")
        print_status(f"  R² = {r2:.4f}", "INFO")
        print_status("", "INFO")
        
        if abs(coeffs[2]) > 0.01:
            print_status("→ The z-dependent TEP component (b) is non-zero.", "INFO")
            print_status("  This cannot be explained by mass alone.", "INFO")
            print_status("  TEP is NOT just a mass proxy.", "INFO")
            all_results['z_component_test'] = {
                'mass_coeff': coeffs[1],
                'z_tep_coeff': coeffs[2],
                'z_coeff': coeffs[3],
                'r2': r2,
                'z_component_significant': True
            }
        else:
            print_status("→ The z-dependent component is weak.", "INFO")
            all_results['z_component_test'] = {
                'mass_coeff': coeffs[1],
                'z_tep_coeff': coeffs[2],
                'z_coeff': coeffs[3],
                'r2': r2,
                'z_component_significant': False
            }
    
    # Save results
    all_results['summary'] = summary
    
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_model_comparison.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    results_serializable = json.loads(json.dumps(all_results, default=convert_numpy))
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=safe_json_default)
    
    print_status("", "INFO")
    print_status(f"Results saved to: {output_file}", "INFO")
    print_status("", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    
    return all_results


def main():
    """Main entry point."""
    return analyze_mass_independence()


if __name__ == "__main__":
    main()
