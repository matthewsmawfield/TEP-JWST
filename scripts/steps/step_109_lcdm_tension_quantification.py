#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.4s.
"""
Step 109: Lambda-CDM Tension Quantification Analysis

This module evaluates the quantitative discrepancy between the observed high-redshift
stellar mass density and the predictions of the standard Lambda Cold Dark Matter (ΛCDM) 
cosmological model.

Mathematical Framework:
1. ΛCDM Prediction: We approximate the expected comoving stellar mass density for 
   massive galaxies (M* > 10^10 M_sun) as a function of redshift based on extrapolations 
   from standard abundance matching and halo mass functions (e.g., Behroozi+19).
2. Observed Density: Calculated by integrating the observed mass function within the 
   survey comoving volume.
3. TEP Correction: The Temporal Enhancement of Potentials model predicts that observed 
   stellar masses are inflated by the isochrony bias. The true mass is lower by a factor
   of Gamma_t^0.7.
   
We quantify the tension in terms of statistical significance (sigma), comparing the
uncorrected (observed) mass density against ΛCDM, and then evaluating how much this
tension is mitigated when the TEP isochrony correction is applied.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
import sys
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import KAPPA_GAL, LOG_MH_REF, Z_REF, compute_gamma_t  # TEP model: KAPPA_GAL=9.6e5 mag, log_Mh_ref=12.0, z_ref=5.5, Gamma_t formula

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_DIR = RESULTS_DIR / "figures"  # Publication figures directory (PNG/PDF for manuscript)
INTERIM_DIR = RESULTS_DIR / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)

STEP_NUM = 109  # Pipeline step number (sequential 001-176)
STEP_NAME = "lcdm_tension_quantification"  # ΛCDM tension quantification: evaluates stellar mass density discrepancy vs ΛCDM predictions, tests TEP isochrony correction (factor Gamma_t^0.7)

LOGS_DIR = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


# TEP constants and compute_gamma_t imported from scripts.utils.tep_model


def lcdm_stellar_mass_density(z):
    """
    Evaluate the predicted comoving stellar mass density (log rho_*) for massive 
    galaxies (M* > 10^10 M_sun) under standard ΛCDM physics.
    Uses parameterized extrapolations derived from standard halo mass function integrations.
    """
    log_rho_0 = 7.0  # z = 0 value for massive galaxies
    
    if z < 2:
        log_rho = log_rho_0 - 0.5 * z
    else:
        log_rho = log_rho_0 - 1.0 - 0.8 * (z - 2)
    
    return log_rho


def observed_stellar_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3, mass_thresh=10.0):
    """
    Calculate the observed comoving stellar mass density (log rho_*) for galaxies 
    above the mass threshold within a specific redshift bin.
    
    Mathematical logic:
    rho_* = (Sum of 10^M_star for all galaxies in bin) / Comoving Volume
    """
    mask = (df[z_col] >= z_min) & (df[z_col] < z_max) & (df[mass_col] >= mass_thresh)
    masses = 10**df.loc[mask, mass_col].values
    total_mass = np.sum(masses)
    rho = total_mass / volume_mpc3
    return np.log10(rho) if rho > 0 else -np.inf


def tep_corrected_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3, mass_thresh=10.0):
    """
    Calculate the TEP-corrected comoving stellar mass density for massive galaxies.
    
    Mathematical logic:
    1. The standard SED-fitting process overestimates stellar masses for galaxies in deep 
       gravitational potentials because it incorrectly assumes universal isochrony.
    2. TEP calculates the true stellar mass by scaling the observed mass down by the 
       temporal enhancement factor: M_true = M_obs / Gamma_t^0.7
    3. We apply this correction to ALL galaxies in the redshift bin, and then integrate 
       the mass density only for those galaxies whose TRUE mass remains above the 
       specified mass threshold.
    """
    from scripts.utils.tep_model import isochrony_mass_bias, correct_stellar_mass, stellar_to_halo_mass_behroozi_like
    
    mask_all = (df[z_col] >= z_min) & (df[z_col] < z_max)
    df_bin = df.loc[mask_all].copy()
    
    if len(df_bin) == 0:
        return -np.inf, 1.0
    
    # Compute Gamma_t for structural potential depth correction
    log_mh = stellar_to_halo_mass_behroozi_like(df_bin[mass_col].values, df_bin[z_col].values)
    z = df_bin[z_col].values
    gamma_t = compute_gamma_t(log_mh, z)
    
    # Apply structural TEP isochrony correction
    log_masses_corrected = correct_stellar_mass(df_bin[mass_col].values, gamma_t)
    
    # Filter strictly by true corrected mass threshold
    mask_true = log_masses_corrected >= mass_thresh
    masses_corrected = 10**log_masses_corrected[mask_true]
    
    total_mass = np.sum(masses_corrected)
    rho = total_mass / volume_mpc3
    
    # Extract median Gamma_t of the observed massive sub-population for contextual logging
    mask_obs_massive = df_bin[mass_col].values >= mass_thresh
    med_gamma = np.median(gamma_t[mask_obs_massive]) if np.sum(mask_obs_massive) > 0 else 1.0
    
    return np.log10(rho) if rho > 0 else -np.inf, med_gamma


def z_from_t(t_gyr):
    from astropy.cosmology import Planck18 as cosmo
    from scipy.optimize import minimize_scalar
    def diff(z_guess): return (cosmo.age(z_guess).value - t_gyr)**2
    res = minimize_scalar(diff, bounds=(0, 20), method='bounded')
    return res.x


def compute_tension_sigma(observed, predicted, uncertainty=0.3):
    """
    Compute tension in sigma between observed and predicted.
    If observed <= predicted (or observed is -inf), tension is 0 (no tension).
    """
    if observed == -np.inf or observed <= predicted:
        return 0.0
    diff = observed - predicted
    sigma = abs(diff) / uncertainty
    return sigma


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: ΛCDM Tension Quantification")
    print_status("=" * 70)
    
    # Load data
    data_path = INTERIM_DIR / "step_002_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    mass_col = 'log_Mstar' if 'log_Mstar' in df.columns else 'log_mass'
    
    print_status(f"Loaded {len(df)} galaxies")
    
    # Survey volume (approximate for UNCOVER)
    # UNCOVER covers ~45 arcmin^2, depth z ~ 5-12
    # At z ~ 8, 1 arcmin ~ 0.5 Mpc (comoving)
    area_mpc2 = 45 * (0.5)**2  # ~11 Mpc^2
    
    # Redshift bins
    z_bins = [(7, 8), (8, 9), (9, 10), (10, 12)]
    
    results_by_bin = []
    
    print_status("\n--- Stellar Mass Density Comparison ---")
    print_status(f"{'z bin':<10} {'ΛCDM':<12} {'Observed':<12} {'TEP-corr':<12} {'Tension (obs)':<15} {'Tension (TEP)':<15}")
    print_status("-" * 80)
    
    from astropy.cosmology import Planck18 as cosmo
    
    for z_min, z_max in z_bins:
        # Calculate volume accurately using astropy
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u
        
        # UNCOVER area = 28.8 arcmin^2
        area_sr = (28.8 * (u.arcmin)**2).to(u.sr).value
        vol_min = cosmo.comoving_volume(z_min).value * (area_sr / (4*np.pi))
        vol_max = cosmo.comoving_volume(z_max).value * (area_sr / (4*np.pi))
        volume_mpc3 = vol_max - vol_min
        
        # ΛCDM prediction at observed redshift
        z_mid = (z_min + z_max) / 2
        lcdm_pred = lcdm_stellar_mass_density(z_mid)
        
        # Observed mass density
        obs = observed_stellar_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3)
        
        # TEP-corrected mass density (true mass is smaller by Gamma_t^0.7)
        tep_corr, med_gamma = tep_corrected_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3)
        
        # Compute tensions (sigma difference)
        tension_obs = compute_tension_sigma(obs, lcdm_pred)
        tension_tep = compute_tension_sigma(tep_corr, lcdm_pred)
        
        # Handle infinite cases for JSON compliance
        mass_red_dex = float(obs - tep_corr) if (np.isfinite(obs) and np.isfinite(tep_corr)) else None
        
        if tension_obs > 0:
            tension_reduction = (tension_obs - tension_tep) / tension_obs * 100 if tension_obs > 0 else 0
            
            print_status(f"Bin z={z_min}-{z_max} (Gamma={med_gamma:.2f}):")
            print_status(f"  ΛCDM pred : {lcdm_pred:.2f} (at z={z_mid:.2f})")
            print_status(f"  Observed  : {obs:.2f} -> Tension: {tension_obs:.1f}σ")
            print_status(f"  TEP corr  : {tep_corr if np.isfinite(tep_corr) else '< LCDM'} -> Tension: {tension_tep:.1f}σ")
            
            results_by_bin.append({
                'z_min': z_min,
                'z_max': z_max,
                'z_mid': z_mid,
                'median_gamma_t': float(med_gamma),
                'lcdm_prediction': float(lcdm_pred),
                'observed': float(obs) if np.isfinite(obs) else None,
                'tep_corrected': float(tep_corr) if np.isfinite(tep_corr) else None,
                'tension_observed_sigma': float(tension_obs),
                'tension_tep_sigma': float(tension_tep),
                'tension_reduction_pct': float(tension_reduction),
                'mass_reduction_dex': mass_red_dex,
            })
    
    # Summary statistics (exclude bins with no data)
    valid_bins = [r for r in results_by_bin if r['tension_observed_sigma'] is not None]
    
    if valid_bins:
        mean_tension_obs = np.mean([r['tension_observed_sigma'] for r in valid_bins])
        mean_tension_tep = np.mean([r['tension_tep_sigma'] for r in valid_bins])
        mean_reduction = np.mean([r['tension_reduction_pct'] for r in valid_bins])
        
        valid_mass = [r['mass_reduction_dex'] for r in valid_bins if r['mass_reduction_dex'] is not None]
        mean_mass_reduction = np.mean(valid_mass) if valid_mass else None
    else:
        mean_tension_obs = mean_tension_tep = mean_reduction = 0
        mean_mass_reduction = None
    
    print_status("\n" + "=" * 70)
    print_status("SUMMARY")
    print_status("=" * 70)
    
    print_status(f"\nMean tension (observed vs ΛCDM): {mean_tension_obs:.1f}σ")
    print_status(f"Mean tension (TEP-corrected vs ΛCDM): {mean_tension_tep:.1f}σ")
    print_status(f"Mean tension reduction: {mean_reduction:.0f}%")
    if mean_mass_reduction is not None:
        print_status(f"Mean mass reduction: {mean_mass_reduction:.2f} dex ({10**mean_mass_reduction:.1f}×)")
    else:
        print_status(f"Mean mass reduction: fully clears threshold")
    
    # Key finding
    if mean_tension_tep < 2:
        conclusion = f"TEP reduces ΛCDM tension from {mean_tension_obs:.1f}σ to {mean_tension_tep:.1f}σ (< 2σ)"
        status = "SUPPORTED"
    elif mean_tension_tep < mean_tension_obs:
        conclusion = f"TEP reduces ΛCDM tension from {mean_tension_obs:.1f}σ to {mean_tension_tep:.1f}σ"
        status = "PARTIALLY SUPPORTED"
    else:
        conclusion = "-> TEP does not measurably reduce ΛCDM tension"
        status = "NOT SUPPORTED"
    
    print_status(f"\n✓ {conclusion}")
    print_status(f"  Status: {status}")
    
    # Implications
    print_status("\n--- Implications ---")
    if mean_mass_reduction is not None:
        print_status(f"  1. Observed stellar mass density at z > 7 is {10**mean_mass_reduction:.1f}× higher than ΛCDM")
    else:
        print_status(f"  1. Observed massive galaxies (M* > 10^10) at z > 7 exceed ΛCDM bounds entirely.")
    print_status(f"  2. TEP correction reduces this excess by {mean_reduction:.0f}%")
    print_status(f"  3. Remaining tension ({mean_tension_tep:.1f}σ) may be genuine early structure formation")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: ΛCDM Tension Quantification',
        'results_by_redshift': results_by_bin,
        'summary': {
            'mean_tension_observed_sigma': float(mean_tension_obs),
            'mean_tension_tep_sigma': float(mean_tension_tep),
            'mean_tension_reduction_pct': float(mean_reduction),
            'mean_mass_reduction_dex': float(mean_mass_reduction) if mean_mass_reduction is not None else None,
            'status': status,
        },
        'conclusion': conclusion,
        'implications': [
            f'Observed massive galaxies at z > 7 exceed ΛCDM' if mean_mass_reduction is None else f'Observed stellar mass density at z > 7 is {10**mean_mass_reduction:.1f}× higher than ΛCDM',
            f'TEP correction reduces this excess by {mean_reduction:.0f}%',
            f'Remaining tension ({mean_tension_tep:.1f}σ) may be genuine early structure formation',
        ],
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_lcdm_tension_quantification.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: Stellar mass density vs redshift
        ax1 = axes[0]
        
        z_mids = [r['z_mid'] for r in results_by_bin]
        lcdm = [r['lcdm_prediction'] for r in results_by_bin]
        obs = [r['observed'] for r in results_by_bin]
        tep = [r['tep_corrected'] for r in results_by_bin]
        
        ax1.plot(z_mids, lcdm, 'k--', linewidth=2, label='ΛCDM prediction')
        ax1.scatter(z_mids, obs, s=100, c='red', marker='o', label='Observed', zorder=5)
        ax1.scatter(z_mids, tep, s=100, c='blue', marker='s', label='TEP-corrected', zorder=5)
        
        # Error bars (approximate)
        for i in range(len(z_mids)):
            if obs[i] is not None:
                ax1.errorbar(z_mids[i], obs[i], yerr=0.3, fmt='none', c='red', capsize=5)
            if tep[i] is not None:
                ax1.errorbar(z_mids[i], tep[i], yerr=0.3, fmt='none', c='blue', capsize=5)
        
        ax1.set_xlabel('Redshift', fontsize=12)
        ax1.set_ylabel('log ρ* [M☉/Mpc³]', fontsize=12)
        ax1.set_title('Stellar Mass Density: ΛCDM vs Observations', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Tension comparison (use results_by_bin which excludes invalid bins)
        ax2 = axes[1]
        
        tension_obs = [r['tension_observed_sigma'] for r in results_by_bin]
        tension_tep = [r['tension_tep_sigma'] for r in results_by_bin]
        z_labels = [f"{r['z_min']}-{r['z_max']}" for r in results_by_bin]
        
        x = np.arange(len(results_by_bin))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, tension_obs, width, label='Observed', color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, tension_tep, width, label='TEP-corrected', color='blue', alpha=0.7)
        
        ax2.axhline(2, color='gray', linestyle='--', label='2σ threshold')
        ax2.axhline(5, color='gray', linestyle=':', label='5σ threshold')
        
        ax2.set_xlabel('Redshift Bin', fontsize=12)
        ax2.set_ylabel('Tension with ΛCDM (σ)', fontsize=12)
        ax2.set_title('ΛCDM Tension: Before and After TEP Correction', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(z_labels)
        ax2.legend(fontsize=10)
        
        # Add reduction labels
        for i, (t_obs, t_tep) in enumerate(zip(tension_obs, tension_tep)):
            reduction = (t_obs - t_tep) / t_obs * 100 if t_obs > 0 else 0
            ax2.text(i, max(t_obs, t_tep) + 0.5, f'-{reduction:.0f}%', 
                    ha='center', fontsize=9, color='green')
        
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_lcdm_tension.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
