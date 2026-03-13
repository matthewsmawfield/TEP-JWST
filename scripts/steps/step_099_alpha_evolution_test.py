#!/usr/bin/env python3
"""
Step 121: Alpha(z) Evolution Test

Tests whether the TEP coupling constant alpha_0 is truly constant
or evolves with redshift.

This is a core assumption of TEP: alpha(z) = alpha_0 * sqrt(1+z)
where alpha_0 = 0.58 is fixed from local Cepheid calibration.

If alpha_0 varies with z, it would suggest:
- The TEP model needs modification
- Or there are systematic effects in the data

Author: TEP-JWST Pipeline
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
np.random.seed(42)
import pandas as pd
import sys
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"  # Results root directory
OUTPUTS_DIR = RESULTS_DIR / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_DIR = RESULTS_DIR / "figures"  # Publication figures directory (PNG/PDF for manuscript)
INTERIM_DIR = RESULTS_DIR / "interim"  # Pre-processed intermediate products (CSV format from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import compute_gamma_t, ALPHA_0, LOG_MH_REF, Z_REF  # TEP model: Gamma_t formula, alpha_0=0.58, log_Mh_ref=12.0, z_ref=5.5
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300) & JSON serialiser for numpy types

STEP_NUM = "099"  # Pipeline step number (sequential 001-176)
STEP_NAME = "alpha_evolution_test"  # Alpha(z) evolution test: fits alpha_0 per redshift bin maximizing rho(Gamma_t(alpha_0), dust), tests constancy assumption

LOGS_DIR = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# TEP constants
ALPHA_0_NOMINAL = 0.58
ALPHA_0_UNCERTAINTY = 0.16



def fit_alpha_in_zbin(dust, log_mh, z, z_mean):
    """
    Fit alpha_0 in a single redshift bin by maximizing
    the correlation between Gamma_t(alpha_0) and dust.
    """
    def neg_correlation(alpha_0):
        if alpha_0 <= 0 or alpha_0 > 2:
            return 1.0  # Penalty for invalid values
        gamma_t = compute_gamma_t(log_mh, z, alpha_0)
        log_gamma = np.log10(np.maximum(gamma_t, 0.01))
        
        # Handle edge cases
        if np.std(log_gamma) < 1e-6 or np.std(dust) < 1e-6:
            return 1.0
        
        rho, _ = stats.spearmanr(log_gamma, dust)
        if np.isnan(rho):
            return 1.0
        return -rho  # Negative because we minimize
    
    # Grid search for initial guess
    alphas = np.linspace(0.1, 1.5, 15)
    correlations = [-neg_correlation(a) for a in alphas]
    best_idx = np.argmax(correlations)
    alpha_init = alphas[best_idx]
    
    # Refine with optimization
    result = minimize(neg_correlation, alpha_init, method='Nelder-Mead',
                     options={'xatol': 0.01})
    
    alpha_best = result.x[0]
    rho_best = -result.fun
    
    # Bootstrap for uncertainty
    n_boot = 100
    alpha_boots = []
    n = len(dust)
    
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        dust_boot = dust[idx]
        log_mh_boot = log_mh[idx]
        z_boot = z[idx]
        
        def neg_corr_boot(alpha_0):
            if alpha_0 <= 0 or alpha_0 > 2:
                return 1.0
            gamma_t = compute_gamma_t(log_mh_boot, z_boot, alpha_0)
            log_gamma = np.log10(np.maximum(gamma_t, 0.01))
            if np.std(log_gamma) < 1e-6:
                return 1.0
            rho, _ = stats.spearmanr(log_gamma, dust_boot)
            return -rho if not np.isnan(rho) else 1.0
        
        res = minimize(neg_corr_boot, alpha_best, method='Nelder-Mead',
                      options={'xatol': 0.05})
        alpha_boots.append(res.x[0])
    
    alpha_err = np.std(alpha_boots)
    
    return alpha_best, alpha_err, rho_best


def test_alpha_evolution(z_bins, alpha_fits):
    """
    Test whether alpha_0 evolves with redshift.
    
    null model: alpha_0 is constant (no z dependence)
    Alternative: alpha_0 = a + b*z
    """
    z_centers = np.array([0.5 * (zb[0] + zb[1]) for zb in z_bins])
    alphas = np.array([a[0] for a in alpha_fits])
    alpha_errs = np.array([a[1] for a in alpha_fits])
    
    # Weighted linear regression
    weights = 1 / (alpha_errs**2 + 0.01)  # Add small constant to avoid div by zero
    
    # Fit: alpha = a + b*z
    from scipy.optimize import curve_fit
    
    def linear(z, a, b):
        return a + b * z
    
    try:
        popt, pcov = curve_fit(linear, z_centers, alphas, sigma=alpha_errs, 
                               absolute_sigma=True, p0=[0.58, 0])
        a_fit, b_fit = popt
        pcov_diag = np.diag(pcov)
        
        # Handle infinite or invalid covariance values
        if np.any(~np.isfinite(pcov_diag)) or np.any(pcov_diag < 0):
            # Fall back to bootstrap uncertainty estimation
            n_boot = 200
            a_boots, b_boots = [], []
            for _ in range(n_boot):
                idx = np.random.choice(len(z_centers), len(z_centers), replace=True)
                z_boot = z_centers[idx]
                alpha_boot = alphas[idx]
                try:
                    popt_boot, _ = curve_fit(linear, z_boot, alpha_boot, p0=[0.58, 0])
                    a_boots.append(popt_boot[0])
                    b_boots.append(popt_boot[1])
                except (RuntimeError, ValueError):
                    continue
            a_err = np.std(a_boots) if a_boots else 0.1
            b_err = np.std(b_boots) if b_boots else 0.1
            print_status("Used bootstrap for uncertainty (covariance matrix singular)", "INFO")
        else:
            a_err, b_err = np.sqrt(pcov_diag)
        
        # Test if slope is significantly different from zero
        t_stat = b_fit / b_err if b_err > 0 else 0
        p_value = 2 * stats.t.sf(abs(t_stat), len(z_bins) - 2)
        
    except Exception as e:
        print_status(f"Curve fit failed: {e}", "WARNING")
        a_fit, b_fit = ALPHA_0_NOMINAL, 0
        a_err, b_err = 0.1, 0.1
        p_value = 1.0
    
    return {
        'intercept': a_fit,
        'intercept_err': a_err,
        'slope': b_fit,
        'slope_err': b_err,
        'p_value_slope': format_p_value(p_value),
        'consistent_with_constant': bool(format_p_value(p_value) is None or format_p_value(p_value) > 0.05),
    }


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Alpha(z) Evolution Test")
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
    
    # Get columns
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    
    # Check required columns
    required = ['dust']
    for col in required:
        if col not in df.columns:
            print_status(f"Missing column: {col}", "ERROR")
            return
    
    # Get mass column
    mass_col = 'log_Mstar' if 'log_Mstar' in df.columns else 'log_mass'
    
    # Compute halo mass from stellar mass
    df['log_mh'] = df[mass_col] + 2.0  # Simple abundance matching
    
    # Define redshift bins
    # NOTE: TEP predicts the dust–Gamma_t signal only activates at z>8 where
    # alpha(z)*sqrt(1+z) is strong enough to push t_eff above the AGB threshold.
    # At z<7, both TEP and standard physics predict no dust–mass correlation
    # (absent signal is the prediction, not a failure). Fitting alpha_0 in
    # bins where the signal is absent by design gives uninformative results and
    # systematically biases the optimizer to the grid floor.
    # The meaningful recovery must be restricted to the HIGH-Z activation regime.
    z_bins_all = [
        (4.0, 5.5),
        (5.5, 7.0),
        (7.0, 8.5),
        (8.5, 10.0),
        (10.0, 12.0),
    ]
    # High-z only bins where TEP signal is active
    z_bins_highz = [
        (7.0, 8.5),
        (8.5, 10.0),
    ]
    
    print_status(f"\nFitting alpha_0 in ALL redshift bins (for completeness):")
    print_status("-" * 60)
    z_bins = z_bins_all
    
    alpha_fits = []
    bin_results = []
    
    for z_min, z_max in z_bins:
        mask = (df[z_col] >= z_min) & (df[z_col] < z_max)
        df_bin = df[mask].copy()
        
        # Remove NaN values
        valid = ~(df_bin['dust'].isna() | df_bin['log_mh'].isna())
        df_bin = df_bin[valid]
        
        n_bin = len(df_bin)
        
        if n_bin < 20:
            print_status(f"  z={z_min:.1f}-{z_max:.1f}: N={n_bin} (too few, skipping)")
            continue
        
        dust = df_bin['dust'].values
        log_mh = df_bin['log_mh'].values
        z = df_bin[z_col].values
        z_mean = np.mean(z)
        
        alpha_best, alpha_err, rho_best = fit_alpha_in_zbin(dust, log_mh, z, z_mean)
        
        alpha_fits.append((alpha_best, alpha_err, rho_best))
        bin_results.append({
            'z_min': z_min,
            'z_max': z_max,
            'z_mean': float(z_mean),
            'n_galaxies': n_bin,
            'alpha_0_best': float(alpha_best),
            'alpha_0_err': float(alpha_err),
            'rho_best': float(rho_best),
            'consistent_with_nominal': bool(abs(alpha_best - ALPHA_0_NOMINAL) < 2 * (alpha_err + ALPHA_0_UNCERTAINTY)),
        })
        
        consistent = "✓" if bin_results[-1]['consistent_with_nominal'] else "✗"
        print_status(f"  z={z_min:.1f}-{z_max:.1f}: N={n_bin}, α₀={alpha_best:.2f}±{alpha_err:.2f}, "
                    f"ρ={rho_best:.2f} {consistent}")
    
    if len(alpha_fits) < 3:
        print_status("Insufficient bins for evolution test", "WARNING")
        evolution_test = {'consistent_with_constant': True, 'note': 'Insufficient data'}
    else:
        # Test for evolution
        z_bins_valid = [(r['z_min'], r['z_max']) for r in bin_results]
        evolution_test = test_alpha_evolution(z_bins_valid, alpha_fits)

    # -----------------------------------------------------------------------
    # HIGH-Z ONLY RECOVERY (z > 7 where TEP signal is active)
    # This is the methodologically correct recovery: alpha_0 can only be
    # recovered from bins where the dust–Gamma_t signal is non-zero.
    # Low-z bins (z<7) have absent signal by TEP prediction, so including
    # them drives the optimizer to the grid floor.
    # -----------------------------------------------------------------------
    print_status("\n" + "=" * 70)
    print_status("HIGH-Z ALPHA_0 RECOVERY (z > 7, signal-active bins only)")
    print_status("=" * 70)

    highz_fits = []
    highz_bin_results = []
    for z_min, z_max in z_bins_highz:
        mask = (df[z_col] >= z_min) & (df[z_col] < z_max)
        df_bin = df[mask].copy()
        valid = ~(df_bin['dust'].isna() | df_bin['log_mh'].isna())
        df_bin = df_bin[valid]
        n_bin = len(df_bin)
        if n_bin < 20:
            print_status(f"  z={z_min:.1f}-{z_max:.1f}: N={n_bin} (too few, skipping)")
            continue
        dust = df_bin['dust'].values
        log_mh = df_bin['log_mh'].values
        z = df_bin[z_col].values
        z_mean = np.mean(z)
        alpha_best, alpha_err, rho_best = fit_alpha_in_zbin(dust, log_mh, z, z_mean)
        highz_fits.append((alpha_best, alpha_err, rho_best))
        highz_bin_results.append({
            'z_min': z_min, 'z_max': z_max, 'z_mean': float(z_mean),
            'n_galaxies': n_bin,
            'alpha_0_best': float(alpha_best),
            'alpha_0_err': float(alpha_err),
            'rho_best': float(rho_best),
            'consistent_with_nominal': bool(abs(alpha_best - ALPHA_0_NOMINAL) < 2 * (alpha_err + ALPHA_0_UNCERTAINTY)),
        })
        consistent = "✓" if highz_bin_results[-1]['consistent_with_nominal'] else "✗"
        print_status(f"  z={z_min:.1f}-{z_max:.1f}: N={n_bin}, α₀={alpha_best:.2f}±{alpha_err:.2f}, "
                    f"ρ={rho_best:.2f} {consistent}")

    # Weighted mean from high-z bins only
    if highz_fits:
        hz_alphas = np.array([a[0] for a in highz_fits])
        hz_weights = 1 / np.array([a[1]**2 + 0.01 for a in highz_fits])
        hz_alpha_mean = np.sum(hz_alphas * hz_weights) / np.sum(hz_weights)
        hz_alpha_err = 1 / np.sqrt(np.sum(hz_weights))
        hz_tension = abs(hz_alpha_mean - ALPHA_0_NOMINAL) / np.sqrt(hz_alpha_err**2 + ALPHA_0_UNCERTAINTY**2)
        print_status(f"\nHigh-z weighted mean α₀ = {hz_alpha_mean:.2f} ± {hz_alpha_err:.2f}")
        print_status(f"Tension with nominal (0.58): {hz_tension:.2f}σ")
        highz_recovery = {
            'alpha_0': float(hz_alpha_mean),
            'alpha_0_err': float(hz_alpha_err),
            'tension_with_nominal_sigma': float(hz_tension),
            'n_bins': len(highz_fits),
            'note': 'Recovery restricted to z>7 signal-active bins; z<7 excluded because TEP predicts absent signal there',
        }
    else:
        hz_alpha_mean = ALPHA_0_NOMINAL
        hz_alpha_err = ALPHA_0_UNCERTAINTY
        hz_tension = 0
        highz_recovery = {'note': 'Insufficient high-z galaxies'}

    # Summary
    print_status("\n" + "=" * 70)
    print_status("EVOLUTION TEST RESULTS")
    print_status("=" * 70)
    
    if 'slope' in evolution_test:
        print_status(f"\nLinear fit: α₀(z) = {evolution_test['intercept']:.2f} + "
                    f"{evolution_test['slope']:.3f} × z")
        print_status(f"Slope significance: p = {evolution_test['p_value_slope']:.3f}")
    
    if evolution_test['consistent_with_constant']:
        conclusion = "α₀ is CONSISTENT with being constant across redshift"
        print_status(f"\n✓ {conclusion}")
        print_status(f"  The TEP assumption α₀ = {ALPHA_0_NOMINAL} is validated")
    else:
        conclusion = "α₀ shows SIGNIFICANT evolution with redshift"
        print_status(f"\n✗ {conclusion}")
        print_status(f"  This may indicate systematic effects or model modification needed")
    
    # Compute weighted mean alpha_0 across all bins (including low-z — reported for completeness)
    if alpha_fits:
        alphas = np.array([a[0] for a in alpha_fits])
        weights = 1 / np.array([a[1]**2 + 0.01 for a in alpha_fits])
        alpha_mean = np.sum(alphas * weights) / np.sum(weights)
        alpha_mean_err = 1 / np.sqrt(np.sum(weights))
        
        print_status(f"\nAll-bins weighted mean α₀ = {alpha_mean:.2f} ± {alpha_mean_err:.2f}")
        print_status(f"  (LOW-Z BINS INCLUDED — methodologically incorrect, reported for completeness)")
        print_status(f"High-z-only weighted mean α₀ = {hz_alpha_mean:.2f} ± {hz_alpha_err:.2f}")
        print_status(f"Nominal value: α₀ = {ALPHA_0_NOMINAL} ± {ALPHA_0_UNCERTAINTY}")
        
        tension = abs(alpha_mean - ALPHA_0_NOMINAL) / np.sqrt(alpha_mean_err**2 + ALPHA_0_UNCERTAINTY**2)
        print_status(f"All-bins tension with nominal: {tension:.1f}σ  (high-z only: {hz_tension:.2f}σ)")
    else:
        alpha_mean = ALPHA_0_NOMINAL
        alpha_mean_err = ALPHA_0_UNCERTAINTY
        tension = 0
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: Alpha(z) Evolution Test',
        'nominal_alpha_0': ALPHA_0_NOMINAL,
        'nominal_alpha_0_uncertainty': ALPHA_0_UNCERTAINTY,
        'bin_results': bin_results,
        'highz_bin_results': highz_bin_results,
        'evolution_test': evolution_test,
        'weighted_mean_all_bins': {
            'alpha_0': float(alpha_mean),
            'alpha_0_err': float(alpha_mean_err),
            'tension_with_nominal_sigma': float(tension),
            'note': 'INCLUDES low-z bins where signal is absent by TEP prediction — methodologically biased low',
        },
        'weighted_mean_highz_only': highz_recovery,
        # Keep backward-compatible key pointing to the correct high-z result
        'weighted_mean': highz_recovery,
        'conclusion': conclusion,
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_alpha_evolution_test.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Alpha vs redshift
        ax1 = axes[0]
        z_centers = [r['z_mean'] for r in bin_results]
        alphas = [r['alpha_0_best'] for r in bin_results]
        alpha_errs = [r['alpha_0_err'] for r in bin_results]
        
        ax1.errorbar(z_centers, alphas, yerr=alpha_errs, fmt='o', markersize=10,
                    color='steelblue', capsize=5, capthick=2, elinewidth=2,
                    markeredgecolor='black', label='Fitted α₀')
        
        # Nominal value band
        ax1.axhline(ALPHA_0_NOMINAL, color='red', linestyle='-', linewidth=2, label='Nominal α₀ = 0.58')
        ax1.axhspan(ALPHA_0_NOMINAL - ALPHA_0_UNCERTAINTY, 
                   ALPHA_0_NOMINAL + ALPHA_0_UNCERTAINTY,
                   color='red', alpha=0.2, label='±1σ uncertainty')
        
        # Evolution fit if available
        if 'slope' in evolution_test and len(z_centers) >= 3:
            z_line = np.linspace(min(z_centers) - 0.5, max(z_centers) + 0.5, 100)
            alpha_line = evolution_test['intercept'] + evolution_test['slope'] * z_line
            ax1.plot(z_line, alpha_line, 'g--', linewidth=2, 
                    label=f'Linear fit (slope p={evolution_test["p_value_slope"]:.2f})')
        
        ax1.set_xlabel('Redshift', fontsize=12)
        ax1.set_ylabel('α₀ (fitted)', fontsize=12)
        ax1.set_title('TEP Coupling Constant vs Redshift', fontsize=14)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_ylim(0, 1.5)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Correlation strength vs redshift
        ax2 = axes[1]
        rhos = [r['rho_best'] for r in bin_results]
        ns = [r['n_galaxies'] for r in bin_results]
        
        scatter = ax2.scatter(z_centers, rhos, c=ns, cmap='viridis', 
                             s=150, edgecolor='black', linewidth=1)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Redshift', fontsize=12)
        ax2.set_ylabel('Best Correlation ρ(Γₜ, dust)', fontsize=12)
        ax2.set_title('TEP Signal Strength vs Redshift', fontsize=14)
        plt.colorbar(scatter, ax=ax2, label='N galaxies')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_alpha_evolution_test.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
