#!/usr/bin/env python3
"""
Step 132: ΛCDM Tension Quantification

Quantifies how much TEP reduces the "too massive, too early" tension
with ΛCDM predictions.

Compares:
- Observed stellar mass density at z > 7
- ΛCDM predicted stellar mass density
- TEP-corrected stellar mass density

Author: TEP-JWST Pipeline
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
FIGURES_DIR = RESULTS_DIR / "figures"
INTERIM_DIR = RESULTS_DIR / "interim"

STEP_NUM = 132

# TEP constants
ALPHA_0 = 0.58
LOG_MH_REF = 12.0
Z_REF = 5.5


def print_status(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def compute_gamma_t(log_Mh, z, alpha_0=ALPHA_0):
    """Compute TEP Gamma_t."""
    alpha_z = alpha_0 * np.sqrt(1 + z)
    log_mh_ref_z = LOG_MH_REF - 1.5 * np.log10(1 + z)
    delta_log_Mh = log_Mh - log_mh_ref_z
    z_factor = (1 + z) / (1 + Z_REF)
    argument = alpha_z * (2/3) * delta_log_Mh * z_factor
    return np.exp(argument)


def lcdm_stellar_mass_density(z):
    """
    ΛCDM predicted stellar mass density at redshift z.
    
    Based on Behroozi et al. (2019) cosmic star formation history
    integrated to give stellar mass density.
    
    Returns log10(rho_*) in M_sun / Mpc^3
    """
    # Approximate ΛCDM prediction from simulations
    # At z = 0: log(rho_*) ~ 8.5
    # Declines roughly as (1+z)^(-2.5) at high z
    
    log_rho_0 = 8.5  # z = 0 value
    
    if z < 2:
        log_rho = log_rho_0 - 0.3 * z
    else:
        log_rho = log_rho_0 - 0.6 - 0.5 * (z - 2)
    
    return log_rho


def observed_stellar_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3):
    """
    Compute observed stellar mass density in a redshift bin.
    """
    mask = (df[z_col] >= z_min) & (df[z_col] < z_max)
    masses = 10**df.loc[mask, mass_col].values
    total_mass = np.sum(masses)
    rho = total_mass / volume_mpc3
    return np.log10(rho) if rho > 0 else -np.inf


def tep_corrected_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3):
    """
    Compute TEP-corrected stellar mass density.
    
    TEP correction: M*_true = M*_obs / Gamma_t^n
    where n ~ 0.5 is the M/L scaling exponent
    """
    n = 0.5  # M/L scaling exponent
    
    mask = (df[z_col] >= z_min) & (df[z_col] < z_max)
    df_bin = df.loc[mask].copy()
    
    if len(df_bin) == 0:
        return -np.inf
    
    # Compute Gamma_t
    log_mh = df_bin[mass_col].values + 2.0
    z = df_bin[z_col].values
    gamma_t = compute_gamma_t(log_mh, z)
    
    # Apply TEP correction
    masses_obs = 10**df_bin[mass_col].values
    masses_corrected = masses_obs / (gamma_t**n)
    
    total_mass = np.sum(masses_corrected)
    rho = total_mass / volume_mpc3
    return np.log10(rho) if rho > 0 else -np.inf


def compute_tension_sigma(observed, predicted, uncertainty=0.3):
    """
    Compute tension in sigma between observed and predicted.
    """
    diff = observed - predicted
    sigma = abs(diff) / uncertainty
    return sigma


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: ΛCDM Tension Quantification")
    print_status("=" * 70)
    
    # Load data
    data_path = INTERIM_DIR / "step_02_uncover_full_sample_tep.csv"
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
    
    for z_min, z_max in z_bins:
        # Compute volume for this bin
        # Approximate: depth ~ 200 Mpc per unit z at z ~ 8
        depth_mpc = (z_max - z_min) * 200
        volume_mpc3 = area_mpc2 * depth_mpc
        
        # ΛCDM prediction
        z_mid = (z_min + z_max) / 2
        lcdm_pred = lcdm_stellar_mass_density(z_mid)
        
        # Observed
        obs = observed_stellar_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3)
        
        # TEP-corrected
        tep_corr = tep_corrected_mass_density(df, z_col, mass_col, z_min, z_max, volume_mpc3)
        
        # Compute tensions
        tension_obs = compute_tension_sigma(obs, lcdm_pred)
        tension_tep = compute_tension_sigma(tep_corr, lcdm_pred)
        
        # Tension reduction
        tension_reduction = (tension_obs - tension_tep) / tension_obs * 100 if tension_obs > 0 else 0
        
        print_status(f"{z_min}-{z_max:<7} {lcdm_pred:<12.2f} {obs:<12.2f} {tep_corr:<12.2f} {tension_obs:<15.1f}σ {tension_tep:<15.1f}σ")
        
        # Only include bins with valid data (no Infinity/NaN)
        if np.isfinite(obs) and np.isfinite(tep_corr):
            results_by_bin.append({
                'z_min': z_min,
                'z_max': z_max,
                'z_mid': z_mid,
                'lcdm_prediction': float(lcdm_pred),
                'observed': float(obs),
                'tep_corrected': float(tep_corr),
                'tension_observed_sigma': float(tension_obs),
                'tension_tep_sigma': float(tension_tep),
                'tension_reduction_pct': float(tension_reduction),
                'mass_reduction_dex': float(obs - tep_corr),
            })
    
    # Summary statistics (exclude bins with no data)
    valid_bins = [r for r in results_by_bin if np.isfinite(r['tension_observed_sigma'])]
    
    if valid_bins:
        mean_tension_obs = np.mean([r['tension_observed_sigma'] for r in valid_bins])
        mean_tension_tep = np.mean([r['tension_tep_sigma'] for r in valid_bins])
        mean_reduction = np.mean([r['tension_reduction_pct'] for r in valid_bins])
        mean_mass_reduction = np.mean([r['mass_reduction_dex'] for r in valid_bins])
    else:
        mean_tension_obs = mean_tension_tep = mean_reduction = mean_mass_reduction = 0
    
    print_status("\n" + "=" * 70)
    print_status("SUMMARY")
    print_status("=" * 70)
    
    print_status(f"\nMean tension (observed vs ΛCDM): {mean_tension_obs:.1f}σ")
    print_status(f"Mean tension (TEP-corrected vs ΛCDM): {mean_tension_tep:.1f}σ")
    print_status(f"Mean tension reduction: {mean_reduction:.0f}%")
    print_status(f"Mean mass reduction: {mean_mass_reduction:.2f} dex ({10**mean_mass_reduction:.1f}×)")
    
    # Key finding
    if mean_tension_tep < 2:
        conclusion = f"TEP reduces ΛCDM tension from {mean_tension_obs:.1f}σ to {mean_tension_tep:.1f}σ (< 2σ)"
        status = "RESOLVED"
    elif mean_tension_tep < mean_tension_obs:
        conclusion = f"TEP reduces ΛCDM tension from {mean_tension_obs:.1f}σ to {mean_tension_tep:.1f}σ"
        status = "PARTIALLY RESOLVED"
    else:
        conclusion = "TEP does not reduce ΛCDM tension"
        status = "NOT RESOLVED"
    
    print_status(f"\n✓ {conclusion}")
    print_status(f"  Status: {status}")
    
    # Implications
    print_status("\n--- Implications ---")
    print_status(f"  1. Observed stellar mass density at z > 7 is {10**mean_mass_reduction:.1f}× higher than ΛCDM")
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
            'mean_mass_reduction_dex': float(mean_mass_reduction),
            'status': status,
        },
        'conclusion': conclusion,
        'implications': [
            f'Observed stellar mass density at z > 7 is {10**mean_mass_reduction:.1f}× higher than ΛCDM',
            f'TEP correction reduces this excess by {mean_reduction:.0f}%',
            f'Remaining tension ({mean_tension_tep:.1f}σ) may be genuine early structure formation',
        ],
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_lcdm_tension.json"
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
        ax1.errorbar(z_mids, obs, yerr=0.3, fmt='none', c='red', capsize=5)
        ax1.errorbar(z_mids, tep, yerr=0.3, fmt='none', c='blue', capsize=5)
        
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
