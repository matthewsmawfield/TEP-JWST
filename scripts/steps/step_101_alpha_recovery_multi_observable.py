#!/usr/bin/env python3
"""
Step 101: Multi-Observable κ_gal Recovery Test

Addresses the weakness that fitting κ_gal to dust alone gives values
in tension with the nominal κ_gal = (9.6 ± 4.0) × 10⁵ mag from
the Paper 11 Cepheid analysis.

This step jointly fits κ_gal across multiple observables to see if
the combined constraint converges to the Cepheid-calibrated value.

Key insight: Different observables may have different sensitivities
to κ_gal. The joint constraint should be more robust than any single one.

Author: TEP-JWST Pipeline
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
np.random.seed(42)  # Fixed seed for reproducible optimisation
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize, minimize_scalar  # Multi-D and 1-D optimisation for joint kappa fit
import sys
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.tep_model import compute_gamma_t, KAPPA_GAL, KAPPA_GAL_UNCERTAINTY, LOG_MH_REF, Z_REF  # TEP model: Gamma_t formula, KAPPA_GAL=9.6e5 mag, log_Mh_ref=12.0, z_ref=5.5
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types (handles NaN, inf, float32)
RESULTS_DIR = PROJECT_ROOT / "results"  # Results root directory
OUTPUTS_DIR = RESULTS_DIR / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_DIR = RESULTS_DIR / "figures"  # Publication figures directory (PNG/PDF for manuscript)
INTERIM_DIR = RESULTS_DIR / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)

STEP_NUM = "101"  # Pipeline step number (sequential 001-176)
STEP_NAME = "alpha_recovery_multi_observable"  # Multi-observable KAPPA_GAL recovery: joint fit across dust, sSFR, M/L, ages to converge to Cepheid-calibrated KAPPA_GAL=9.6e5 mag

LOGS_DIR = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# TEP coupling constant and its external prior uncertainty (Cepheid-calibrated)
KAPPA_GAL_NOMINAL = KAPPA_GAL
KAPPA_GAL_SIGMA = KAPPA_GAL_UNCERTAINTY
KAPPA_FIT_BOUNDS = (1.0e5, 2.0e6)



def compute_observable_correlation(observable, log_mh, z, kappa_gal):
    """Compute Pearson correlation between observable and log(Gamma_t(kappa_gal)).
    Pearson is used here (not Spearman) because Pearson is sensitive to the
    calibrated scale of log_Gamma_t, which changes with kappa_gal."""
    gamma_t = compute_gamma_t(log_mh, z, kappa_gal)
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))
    
    valid = ~(np.isnan(observable) | np.isnan(log_gamma))
    if np.sum(valid) < 10:
        return 0.0
    
    r, _ = stats.pearsonr(log_gamma[valid], observable[valid])
    return r if not np.isnan(r) else 0.0


def fit_alpha_single_observable(observable, log_mh, z, obs_name, expected_sign=1):
    """
    Fit kappa_gal by maximising Pearson R² between log(Gamma_t(kappa_gal)) and observable.

    KEY FIX: The previous implementation normalised both observable and log_gamma to
    [0,1] before computing RSS. This made the objective RANK-INVARIANT — identical
    for all kappa_gal > 0 because min/max normalisation of a monotonic function of
    (log_Mh, z) depends only on the extremes of the sample, not on kappa_gal.
    Confirmed: RSS is nearly invariant under bounded rank-only objectives, so
    this fit uses mag-response units and Pearson-scale objectives instead.

    The fix: use Pearson R² directly, WITHOUT normalisation. Pearson R² is
    sensitive to the calibrated variance of log_Gamma_t, which scales with kappa_gal.
    A larger kappa_gal spreads the log_Gamma_t distribution more widely, changing
    both the regression slope and the residual variance. This breaks rank-invariance.

    Args:
        observable: The observable values
        log_mh: Log halo mass
        z: Redshift
        obs_name: Name of observable
        expected_sign: +1 if positive correlation expected, -1 if negative

    Returns:
        kappa_best, rho_best, uncertainty
    """
    valid = ~np.isnan(observable)
    obs_valid = observable[valid]
    log_mh_valid = log_mh[valid]
    z_valid = z[valid]

    if len(obs_valid) < 20:
        return KAPPA_GAL_NOMINAL, 0.0, KAPPA_GAL_SIGMA

    def neg_r2(kappa_gal):
        """Negative Pearson R² between log(Gamma_t) and observable.
        Pearson R² is NOT rank-invariant: varying kappa_gal changes the spread of
        log_Gamma_t and thus the regression fit quality."""
        if kappa_gal <= KAPPA_FIT_BOUNDS[0] or kappa_gal > KAPPA_FIT_BOUNDS[1]:
            return 0.0  # worst R²
        gamma_t = compute_gamma_t(log_mh_valid, z_valid, kappa_gal)
        log_gamma = np.log10(np.maximum(gamma_t, 0.01))
        if np.std(log_gamma) < 1e-8:
            return 0.0
        r, _ = stats.pearsonr(log_gamma, obs_valid)
        if np.isnan(r):
            return 0.0
        # Penalise wrong sign
        if expected_sign > 0 and r < 0:
            return 0.0
        if expected_sign < 0 and r > 0:
            return 0.0
        return -(r ** 2)  # maximise R²

    # Grid search
    kappas = np.linspace(KAPPA_FIT_BOUNDS[0], KAPPA_FIT_BOUNDS[1], 40)
    r2_grid = [-neg_r2(k) for k in kappas]
    best_idx = np.argmax(r2_grid)
    kappa_init = kappas[best_idx]

    # Refine
    result = minimize_scalar(neg_r2, bounds=KAPPA_FIT_BOUNDS, method='bounded')
    kappa_best = result.x

    # Pearson r at best kappa
    gamma_t_best = compute_gamma_t(log_mh_valid, z_valid, kappa_best)
    log_gamma_best = np.log10(np.maximum(gamma_t_best, 0.01))
    rho_best, _ = stats.pearsonr(log_gamma_best, obs_valid)
    rho_best = float(rho_best) if not np.isnan(rho_best) else 0.0

    # Bootstrap CI
    n = len(obs_valid)
    n_boot = 200
    kappa_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        obs_b = obs_valid[idx]
        lmh_b = log_mh_valid[idx]
        z_b = z_valid[idx]

        def neg_r2_boot(kappa_candidate):
            if kappa_candidate <= KAPPA_FIT_BOUNDS[0] or kappa_candidate > KAPPA_FIT_BOUNDS[1]:
                return 0.0
            gt = compute_gamma_t(lmh_b, z_b, kappa_candidate)
            lg = np.log10(np.maximum(gt, 0.01))
            if np.std(lg) < 1e-8:
                return 0.0
            r, _ = stats.pearsonr(lg, obs_b)
            if np.isnan(r):
                return 0.0
            if expected_sign > 0 and r < 0:
                return 0.0
            if expected_sign < 0 and r > 0:
                return 0.0
            return -(r ** 2)

        res = minimize_scalar(neg_r2_boot, bounds=KAPPA_FIT_BOUNDS, method='bounded')
        kappa_boots.append(res.x)

    kappa_err = np.std(kappa_boots)
    if kappa_err < 0.05 * KAPPA_GAL_NOMINAL:
        kappa_err = 0.05 * KAPPA_GAL_NOMINAL

    return kappa_best, rho_best, kappa_err


def fit_alpha_joint(observables_dict, log_mh, z):
    """
    Jointly fit kappa_gal across multiple observables by maximising weighted sum of
    Pearson R² values.  Uses Pearson (not Spearman, not normalised RSS) to avoid
    rank-invariance.
    """
    valid = np.ones(len(log_mh), dtype=bool)
    for obs_name, (observable, expected_sign, weight) in observables_dict.items():
        valid &= ~np.isnan(observable)

    log_mh_valid = log_mh[valid]
    z_valid = z[valid]

    if np.sum(valid) < 50:
        return KAPPA_GAL_NOMINAL, KAPPA_GAL_SIGMA

    obs_arrays = {}
    for obs_name, (observable, expected_sign, weight) in observables_dict.items():
        obs_arrays[obs_name] = (observable[valid], expected_sign, weight)

    def neg_total_r2(kappa_gal):
        if kappa_gal <= KAPPA_FIT_BOUNDS[0] or kappa_gal > KAPPA_FIT_BOUNDS[1]:
            return 0.0
        gamma_t = compute_gamma_t(log_mh_valid, z_valid, kappa_gal)
        log_gamma = np.log10(np.maximum(gamma_t, 0.01))
        if np.std(log_gamma) < 1e-8:
            return 0.0
        total = 0.0
        for obs_name, (obs_vals, expected_sign, weight) in obs_arrays.items():
            r, _ = stats.pearsonr(log_gamma, obs_vals)
            if np.isnan(r):
                continue
            # Only count if sign matches prediction
            signed_r2 = (r ** 2) if (r * expected_sign >= 0) else 0.0
            total += weight * signed_r2
        return -total

    kappas = np.linspace(KAPPA_FIT_BOUNDS[0], KAPPA_FIT_BOUNDS[1], 40)
    r2_grid = [-neg_total_r2(k) for k in kappas]
    result = minimize_scalar(neg_total_r2, bounds=KAPPA_FIT_BOUNDS, method='bounded')
    kappa_best = result.x

    n = np.sum(valid)
    n_boot = 200
    kappa_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        lmh_b = log_mh_valid[idx]
        z_b = z_valid[idx]
        obs_b = {k: (v[idx], s, w) for k, (v, s, w) in obs_arrays.items()}

        def neg_r2_boot(kappa_candidate):
            if kappa_candidate <= KAPPA_FIT_BOUNDS[0] or kappa_candidate > KAPPA_FIT_BOUNDS[1]:
                return 0.0
            gt = compute_gamma_t(lmh_b, z_b, kappa_candidate)
            lg = np.log10(np.maximum(gt, 0.01))
            if np.std(lg) < 1e-8:
                return 0.0
            total = 0.0
            for obs_name, (ov, es, w) in obs_b.items():
                r, _ = stats.pearsonr(lg, ov)
                if np.isnan(r):
                    continue
                total += w * ((r ** 2) if r * es >= 0 else 0.0)
            return -total

        res = minimize_scalar(neg_r2_boot, bounds=KAPPA_FIT_BOUNDS, method='bounded')
        kappa_boots.append(res.x)

    kappa_err = np.std(kappa_boots)
    if kappa_err < 0.10 * KAPPA_GAL_NOMINAL:
        kappa_err = 0.10 * KAPPA_GAL_NOMINAL

    return kappa_best, kappa_err


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Multi-Observable κ_gal Recovery Test")
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
    mass_col = 'log_Mstar' if 'log_Mstar' in df.columns else 'log_mass'
    
    # Compute halo mass
    df['log_mh'] = df[mass_col] + 2.0
    
    # Filter to high-z where TEP effects are strongest
    df_highz = df[df[z_col] > 6].copy()
    print_status(f"High-z sample (z > 6): {len(df_highz)} galaxies")
    
    # Extract arrays
    log_mh = df_highz['log_mh'].values
    z = df_highz[z_col].values
    
    # Define observables with expected signs
    # Positive sign: higher Gamma_t → higher observable
    # Negative sign: higher Gamma_t → lower observable
    
    observables = {}
    
    # 1. Dust (positive correlation expected)
    if 'dust' in df_highz.columns:
        dust = df_highz['dust'].values
        valid = ~np.isnan(dust)
        if np.sum(valid) > 50:
            observables['dust'] = (dust, +1, 1.0)
            print_status(f"  Added: dust (N={np.sum(valid)})")
    
    # 2. Age ratio (positive correlation expected - older effective ages)
    if 'age_ratio' in df_highz.columns:
        age_ratio = df_highz['age_ratio'].values
        valid = ~np.isnan(age_ratio)
        if np.sum(valid) > 50:
            observables['age_ratio'] = (age_ratio, +1, 1.0)
            print_status(f"  Added: age_ratio (N={np.sum(valid)})")
    
    # 3. Metallicity (positive correlation expected - more enrichment time)
    if 'met' in df_highz.columns:
        met = df_highz['met'].values
        valid = ~np.isnan(met)
        if np.sum(valid) > 50:
            observables['metallicity'] = (met, +1, 0.5)  # Lower weight due to degeneracies
            print_status(f"  Added: metallicity (N={np.sum(valid)})")
    
    # 4. sSFR (negative correlation expected at high-z due to quenching)
    if 'log_ssfr' in df_highz.columns:
        ssfr = df_highz['log_ssfr'].values
        valid = ~np.isnan(ssfr)
        if np.sum(valid) > 50:
            observables['ssfr'] = (ssfr, -1, 0.5)  # Lower weight
            print_status(f"  Added: log_ssfr (N={np.sum(valid)})")
    
    # 5. U-V color (positive correlation expected - redder)
    # Check for color columns
    for color_col in ['uv_color', 'U-V', 'rest_UV']:
        if color_col in df_highz.columns:
            color = df_highz[color_col].values
            valid = ~np.isnan(color)
            if np.sum(valid) > 50:
                observables['color'] = (color, +1, 0.5)
                print_status(f"  Added: {color_col} (N={np.sum(valid)})")
                break
    
    if len(observables) < 2:
        print_status("Insufficient observables for joint fit", "WARNING")
        # Add synthetic observables based on available data
        if 'dust' in observables:
            print_status("Using dust-only fit")
    
    print_status(f"\nFitting κ_gal to {len(observables)} observables:")
    
    # Fit each observable individually
    individual_fits = {}
    for obs_name, (observable, expected_sign, weight) in observables.items():
        kappa_best, rho_best, kappa_err = fit_alpha_single_observable(
            observable, log_mh, z, obs_name, expected_sign
        )
        individual_fits[obs_name] = {
            'kappa_gal': float(kappa_best),
            'kappa_gal_err': float(kappa_err),
            'rho_best': float(rho_best),
            'expected_sign': expected_sign,
            'weight': weight,
        }
        
        consistent = "✓" if abs(kappa_best - KAPPA_GAL_NOMINAL) < 2 * np.sqrt(kappa_err**2 + KAPPA_GAL_SIGMA**2) else "✗"
        print_status(f"  {obs_name}: κ_gal = {kappa_best:.3e} ± {kappa_err:.3e} mag, ρ = {rho_best:.2f} {consistent}")
    
    # Joint fit across all observables
    print_status("\nJoint fit across all observables:")
    kappa_joint, kappa_joint_err = fit_alpha_joint(observables, log_mh, z)
    
    tension_joint = abs(kappa_joint - KAPPA_GAL_NOMINAL) / np.sqrt(kappa_joint_err**2 + KAPPA_GAL_SIGMA**2)
    consistent_joint = "✓" if tension_joint < 2 else "✗"
    
    print_status(f"  Joint κ_gal = {kappa_joint:.3e} ± {kappa_joint_err:.3e} mag")
    print_status(f"  Nominal κ_gal = {KAPPA_GAL_NOMINAL:.3e} ± {KAPPA_GAL_SIGMA:.3e} mag")
    print_status(f"  Tension: {tension_joint:.1f}σ {consistent_joint}")
    
    # Weighted mean of individual fits
    kappas = np.array([f['kappa_gal'] for f in individual_fits.values()])
    weights = np.array([1/(f['kappa_gal_err']**2 + (0.05 * KAPPA_GAL_NOMINAL)**2) for f in individual_fits.values()])
    kappa_weighted = np.sum(kappas * weights) / np.sum(weights)
    kappa_weighted_err = 1 / np.sqrt(np.sum(weights))
    
    tension_weighted = abs(kappa_weighted - KAPPA_GAL_NOMINAL) / np.sqrt(kappa_weighted_err**2 + KAPPA_GAL_SIGMA**2)
    
    print_status(f"\nWeighted mean of individual fits:")
    print_status(f"  κ_gal = {kappa_weighted:.3e} ± {kappa_weighted_err:.3e} mag")
    print_status(f"  Tension with nominal: {tension_weighted:.1f}σ")
    
    # Summary
    print_status("\n" + "=" * 70)
    print_status("SUMMARY")
    print_status("=" * 70)
    
    # Check if joint/weighted is closer to nominal than individual dust fit
    dust_kappa = individual_fits.get('dust', {}).get('kappa_gal', KAPPA_GAL_NOMINAL)
    
    improvement = abs(dust_kappa - KAPPA_GAL_NOMINAL) - abs(kappa_joint - KAPPA_GAL_NOMINAL)
    
    if improvement > 0:
        print_status(f"\n✓ Joint fit IMPROVES agreement with nominal κ_gal")
        print_status(f"  Dust-only: κ_gal = {dust_kappa:.3e} (Δ = {abs(dust_kappa - KAPPA_GAL_NOMINAL):.3e})")
        print_status(f"  Joint: κ_gal = {kappa_joint:.3e} (Δ = {abs(kappa_joint - KAPPA_GAL_NOMINAL):.3e})")
        print_status(f"  Improvement: {improvement:.2f}")
    else:
        print_status(f"\n✗ Joint fit does not improve agreement")
        print_status(f"  This may indicate systematic effects in the data")
    
    # Interpretation
    if tension_joint < 2:
        conclusion = "Joint κ_gal is CONSISTENT with Cepheid calibration within 2σ"
    elif tension_joint < 3:
        conclusion = "Joint κ_gal shows MILD TENSION with Cepheid calibration (2-3σ)"
    else:
        conclusion = "Joint κ_gal shows SIGNIFICANT TENSION with Cepheid calibration (>3σ)"
    
    print_status(f"\n{conclusion}")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: Multi-Observable κ_gal Recovery Test',
        'nominal_kappa_gal': KAPPA_GAL_NOMINAL,
        'nominal_kappa_gal_uncertainty': KAPPA_GAL_SIGMA,
        'n_observables': len(observables),
        'individual_fits': individual_fits,
        'joint_fit': {
            'kappa_gal': float(kappa_joint),
            'kappa_gal_err': float(kappa_joint_err),
            'tension_sigma': float(tension_joint),
            'consistent_2sigma': bool(tension_joint < 2),
        },
        'weighted_mean': {
            'kappa_gal': float(kappa_weighted),
            'kappa_gal_err': float(kappa_weighted_err),
            'tension_sigma': float(tension_weighted),
        },
        'improvement_over_dust_only': float(improvement) if 'dust' in individual_fits else None,
        'conclusion': conclusion,
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_alpha_recovery_multi_observable.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Individual fits
        ax1 = axes[0]
        obs_names = list(individual_fits.keys())
        alphas_ind = [individual_fits[n]['kappa_gal'] for n in obs_names]
        errs_ind = [individual_fits[n]['kappa_gal_err'] for n in obs_names]
        
        y_pos = range(len(obs_names))
        ax1.errorbar(alphas_ind, y_pos, xerr=errs_ind, fmt='o', markersize=10,
                    color='steelblue', capsize=5, capthick=2, elinewidth=2)
        
        # Add joint and weighted
        ax1.errorbar([kappa_joint], [len(obs_names)], xerr=[kappa_joint_err], 
                    fmt='s', markersize=12, color='green', capsize=5, capthick=2,
                    label='Joint fit')
        ax1.errorbar([kappa_weighted], [len(obs_names) + 1], xerr=[kappa_weighted_err],
                    fmt='^', markersize=12, color='orange', capsize=5, capthick=2,
                    label='Weighted mean')
        
        # Nominal value
        ax1.axvline(KAPPA_GAL_NOMINAL, color='red', linestyle='-', linewidth=2, label='Cepheid κ_gal')
        ax1.axvspan(KAPPA_GAL_NOMINAL - KAPPA_GAL_UNCERTAINTY, 
                   KAPPA_GAL_NOMINAL + KAPPA_GAL_UNCERTAINTY,
                   color='red', alpha=0.2)
        
        ax1.set_yticks(list(y_pos) + [len(obs_names), len(obs_names) + 1])
        ax1.set_yticklabels(obs_names + ['Joint', 'Weighted'])
        ax1.set_xlabel('κ_gal (mag)', fontsize=12)
        ax1.set_title('κ_gal Recovery by Observable', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.set_xlim(0, 20e5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Panel 2: Correlation strength at nominal vs fitted κ_gal
        ax2 = axes[1]
        
        rhos_nominal = []
        rhos_fitted = []
        for obs_name, (observable, expected_sign, weight) in observables.items():
            rho_nom = compute_observable_correlation(observable, log_mh, z, KAPPA_GAL_NOMINAL)
            rho_fit = individual_fits[obs_name]['rho_best']
            rhos_nominal.append(expected_sign * rho_nom)
            rhos_fitted.append(expected_sign * rho_fit)
        
        x = np.arange(len(obs_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, rhos_nominal, width, label=f'κ_gal = {KAPPA_GAL_NOMINAL:.2e} (Cepheid)',
                       color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, rhos_fitted, width, label='κ_gal = fitted',
                       color='steelblue', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(obs_names, rotation=45, ha='right')
        ax2.set_ylabel('Signed Correlation (ρ × expected_sign)', fontsize=12)
        ax2.set_title('Correlation Strength: Nominal vs Fitted κ_gal', fontsize=14)
        ax2.legend()
        ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_alpha_recovery.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
