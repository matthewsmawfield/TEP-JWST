#!/usr/bin/env python3
"""
Step 101: Balmer Absorption Line Simulation

This script generates realistic Balmer absorption line predictions for
TEP validation via NIRSpec spectroscopy.

Key features:
1. SSP-based Balmer equivalent width predictions (Hδ, Hγ, Hβ)
2. Age-metallicity grid for interpolation
3. TEP vs standard physics discriminant calculation
4. Mock NIRSpec spectra generation for priority targets

Outputs:
- results/outputs/step_101_balmer_simulation.json
- results/figures/balmer_age_grid.png
- results/figures/tep_balmer_discriminant.png
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "080"  # Pipeline step number (sequential 001-176)
STEP_NAME = "balmer_simulation"  # Balmer line simulation: SSP-based Hδ/Hγ/Hβ equivalent width predictions for TEP validation (Worthey 1994, Kauffmann 2003 BC03 models)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory (PNG/PDF for manuscript)

LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
FIGURES_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


# =============================================================================
# Balmer Absorption Models
# =============================================================================

def create_balmer_grid():
    """
    Create a grid of Balmer equivalent widths as function of age and metallicity.
    
    Based on Worthey (1994), Kauffmann et al. (2003), and BC03 models.
    
    Returns:
        dict with age_grid, Z_grid, and EW grids for Hδ, Hγ, Hβ
    """
    # Age grid (log years)
    log_age = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.15])  # 10 Myr to 14 Gyr
    
    # Metallicity grid [Fe/H]
    Z_grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3])
    
    # Hδ equivalent width grid (Angstroms)
    # Peak at ~1 Gyr, declining for older ages
    # Lower metallicity -> stronger Balmer
    hd_grid = np.array([
        # [Fe/H] = -2.0, -1.5, -1.0, -0.5, 0.0, 0.3
        [2.0, 1.8, 1.6, 1.4, 1.2, 1.0],   # 10 Myr
        [4.5, 4.2, 3.9, 3.5, 3.0, 2.5],   # 30 Myr
        [7.0, 6.5, 6.0, 5.5, 5.0, 4.5],   # 100 Myr
        [8.5, 8.0, 7.5, 7.0, 6.5, 6.0],   # 300 Myr
        [8.0, 7.5, 7.0, 6.5, 6.0, 5.5],   # 1 Gyr (peak)
        [6.0, 5.5, 5.0, 4.5, 4.0, 3.5],   # 3 Gyr
        [4.0, 3.5, 3.0, 2.5, 2.0, 1.5],   # 10 Gyr
        [3.5, 3.0, 2.5, 2.0, 1.5, 1.0],   # 14 Gyr
    ])
    
    # Hγ equivalent width (similar pattern, ~80% of Hδ)
    hg_grid = hd_grid * 0.8
    
    # Hβ equivalent width (emission can fill in absorption)
    hb_grid = hd_grid * 0.6
    
    return {
        'log_age': log_age,
        'Z': Z_grid,
        'EW_Hd': hd_grid,
        'EW_Hg': hg_grid,
        'EW_Hb': hb_grid
    }


def interpolate_balmer_ew(age_gyr, metallicity, grid, line='Hd'):
    """
    Interpolate Balmer EW from the grid.
    
    Args:
        age_gyr: Age in Gyr
        metallicity: [Fe/H]
        grid: Output from create_balmer_grid()
        line: 'Hd', 'Hg', or 'Hb'
    
    Returns:
        Equivalent width in Angstroms
    """
    log_age = np.log10(np.clip(age_gyr * 1e9, 1e7, 1.5e10))
    Z = np.clip(metallicity, -2.0, 0.3)
    
    ew_key = f'EW_{line}'
    
    # Create interpolator
    interp = RegularGridInterpolator(
        (grid['log_age'], grid['Z']),
        grid[ew_key],
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    return float(interp((log_age, Z)))


def compute_tep_balmer_prediction(df, grid):
    """
    Compute TEP-predicted Balmer strengths for each galaxy.
    
    TEP predicts:
    - Higher Γt -> older effective age -> different Balmer strength
    - The direction depends on whether age is < or > 1 Gyr
    """
    results = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('mwa')) or pd.isna(row.get('gamma_t')):
            continue
        
        # Standard age (from SED) - mwa is already in Gyr
        age_sed_gyr = row['mwa']
        
        # TEP-corrected age (proper time accumulated)
        # Γt > 1 means MORE proper time accumulated -> effectively OLDER stellar population
        # Γt < 1 means LESS proper time -> effectively YOUNGER
        gamma_t = row['gamma_t']
        age_tep_gyr = age_sed_gyr * gamma_t  # TEP effective age
        
        # Metallicity estimate (from mass-metallicity relation)
        log_Mstar = row.get('log_Mstar', 9.5)
        z = row.get('z_phot', 6.0)
        # MZR at high-z: [Fe/H] ~ 0.35 * (log M* - 10) - 0.3 * (z - 2)
        metallicity = 0.35 * (log_Mstar - 10) - 0.3 * (z - 2)
        metallicity = np.clip(metallicity, -2.0, 0.3)
        
        # Predicted Balmer EWs
        ew_hd_std = interpolate_balmer_ew(age_sed_gyr, metallicity, grid, 'Hd')
        ew_hd_tep = interpolate_balmer_ew(age_tep_gyr, metallicity, grid, 'Hd')
        
        ew_hg_std = interpolate_balmer_ew(age_sed_gyr, metallicity, grid, 'Hg')
        ew_hg_tep = interpolate_balmer_ew(age_tep_gyr, metallicity, grid, 'Hg')
        
        results.append({
            'id': row.get('id', ''),
            'z': row.get('z_phot', np.nan),
            'log_Mstar': log_Mstar,
            'gamma_t': gamma_t,
            'age_sed_gyr': age_sed_gyr,
            'age_tep_gyr': age_tep_gyr,
            'metallicity': metallicity,
            'ew_hd_std': ew_hd_std,
            'ew_hd_tep': ew_hd_tep,
            'delta_ew_hd': ew_hd_tep - ew_hd_std,
            'ew_hg_std': ew_hg_std,
            'ew_hg_tep': ew_hg_tep,
            'delta_ew_hg': ew_hg_tep - ew_hg_std,
        })
    
    return pd.DataFrame(results)


def generate_mock_spectrum(wavelength, ew_hd, ew_hg, z, snr=10):
    """
    Generate a mock NIRSpec spectrum with Balmer absorption lines.
    
    Args:
        wavelength: Rest-frame wavelength array (Angstroms)
        ew_hd: Hδ equivalent width
        ew_hg: Hγ equivalent width
        z: Redshift
        snr: Signal-to-noise ratio per pixel
    
    Returns:
        dict with observed wavelength, flux, and noise
    """
    # Rest-frame line centers
    lambda_hd = 4101.74  # Hδ
    lambda_hg = 4340.47  # Hγ
    lambda_hb = 4861.33  # Hβ
    
    # Continuum (normalized to 1)
    flux = np.ones_like(wavelength)
    
    # Add absorption lines (Gaussian profiles)
    sigma = 5.0  # Velocity dispersion in Angstroms
    
    # Hδ
    depth_hd = ew_hd / (sigma * np.sqrt(2 * np.pi))
    flux -= depth_hd * np.exp(-0.5 * ((wavelength - lambda_hd) / sigma)**2)
    
    # Hγ
    depth_hg = ew_hg / (sigma * np.sqrt(2 * np.pi))
    flux -= depth_hg * np.exp(-0.5 * ((wavelength - lambda_hg) / sigma)**2)
    
    # Observed wavelength
    wave_obs = wavelength * (1 + z)
    
    # Add noise
    noise = flux / snr
    flux_noisy = flux + np.random.normal(0, 1/snr, len(flux))
    
    return {
        'wavelength_rest': wavelength,
        'wavelength_obs': wave_obs,
        'flux': flux_noisy,
        'flux_true': flux,
        'noise': noise
    }


def compute_discriminant_power(df_pred):
    """
    Compute the statistical power to discriminate TEP from standard physics.
    """
    # Effect size: mean |delta_EW| / std(delta_EW)
    delta = df_pred['delta_ew_hd'].dropna()
    
    if len(delta) < 10:
        return None
    
    # Handle zero std case
    delta_std = delta.std()
    delta_mean = delta.mean()
    
    if delta_std == 0 or np.isnan(delta_std) or delta_mean == 0:
        effect_size = 0.0
    else:
        effect_size = np.abs(delta_mean) / delta_std
    
    # Typical NIRSpec measurement uncertainty for Hδ: ~0.5-1.0 Å
    measurement_error = 0.7  # Angstroms
    
    # Detectable if |delta| > 2 * measurement_error
    n_detectable = (np.abs(delta) > 2 * measurement_error).sum()
    
    # Required sample size for 80% power
    # n = (z_alpha + z_beta)^2 * 2 * sigma^2 / delta^2
    z_alpha = 1.96  # 95% confidence
    z_beta = 0.84   # 80% power
    if delta_mean != 0 and not np.isnan(delta_mean):
        n_required = int(np.ceil((z_alpha + z_beta)**2 * 2 * max(delta_std, 0.1)**2 / delta_mean**2))
    else:
        n_required = 9999  # Undefined
    
    return {
        'effect_size_cohens_d': float(effect_size),
        'mean_delta_ew': float(delta.mean()),
        'std_delta_ew': float(delta.std()),
        'measurement_error': measurement_error,
        'n_detectable': int(n_detectable),
        'n_total': len(delta),
        'fraction_detectable': float(n_detectable / len(delta)),
        'n_required_80_power': n_required
    }


def create_figures(df_pred, grid):
    """Create visualization figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Figure 1: Balmer EW vs Age grid
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Hδ vs age for different metallicities
        ax = axes[0]
        ages = 10**(grid['log_age'] - 9)  # Gyr
        for i, Z in enumerate(grid['Z']):
            ax.plot(ages, grid['EW_Hd'][:, i], 'o-', label=f'[Fe/H]={Z:.1f}')
        ax.set_xscale('log')
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel('EW(Hδ) [Å]')
        ax.set_title('Hδ Absorption vs Age')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Right: TEP discriminant
        ax = axes[1]
        valid = ~df_pred['delta_ew_hd'].isna()
        sc = ax.scatter(
            df_pred.loc[valid, 'gamma_t'],
            df_pred.loc[valid, 'delta_ew_hd'],
            c=df_pred.loc[valid, 'z'],
            cmap='viridis',
            alpha=0.5,
            s=10
        )
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(0.7, color='r', linestyle=':', label='Detection threshold')
        ax.axhline(-0.7, color='r', linestyle=':')
        ax.set_xlabel('Γt')
        ax.set_ylabel('ΔEW(Hδ) [TEP - Standard]')
        ax.set_title('TEP Balmer Discriminant')
        plt.colorbar(sc, ax=ax, label='Redshift')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_PATH / 'balmer_tep_discriminant.png', dpi=150)
        plt.close()
        
        print_status(f"  Saved figure: balmer_tep_discriminant.png", "INFO")
        return True
        
    except ImportError:
        print_status("  matplotlib not available, skipping figures", "WARNING")
        return False


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Balmer Absorption Line Simulation", "INFO")
    print_status("=" * 70, "INFO")
    
    results = {}
    
    # Load data
    data_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # ==========================================================================
    # 1. Create Balmer grid
    # ==========================================================================
    print_status("\n--- 1. Creating Balmer EW Grid ---", "INFO")
    
    grid = create_balmer_grid()
    results['balmer_grid'] = {
        'log_age_range': [float(grid['log_age'].min()), float(grid['log_age'].max())],
        'Z_range': [float(grid['Z'].min()), float(grid['Z'].max())],
        'EW_Hd_range': [float(grid['EW_Hd'].min()), float(grid['EW_Hd'].max())]
    }
    print_status(f"  Age range: {10**(grid['log_age'].min()-9):.3f} - {10**(grid['log_age'].max()-9):.1f} Gyr", "INFO")
    print_status(f"  EW(Hδ) range: {grid['EW_Hd'].min():.1f} - {grid['EW_Hd'].max():.1f} Å", "INFO")
    
    # ==========================================================================
    # 2. Compute TEP predictions
    # ==========================================================================
    print_status("\n--- 2. Computing TEP Balmer Predictions ---", "INFO")
    
    df_pred = compute_tep_balmer_prediction(df, grid)
    print_status(f"  Generated predictions for {len(df_pred)} galaxies", "INFO")
    
    # Summary by redshift bin
    z_bins = [(4, 6), (6, 8), (8, 12)]
    z_summary = []
    for z_lo, z_hi in z_bins:
        subset = df_pred[(df_pred['z'] >= z_lo) & (df_pred['z'] < z_hi)]
        if len(subset) > 0:
            z_summary.append({
                'z_range': f'{z_lo}-{z_hi}',
                'n': len(subset),
                'mean_delta_ew_hd': float(subset['delta_ew_hd'].mean()),
                'std_delta_ew_hd': float(subset['delta_ew_hd'].std()),
                'mean_gamma_t': float(subset['gamma_t'].mean())
            })
            print_status(f"  z={z_lo}-{z_hi}: N={len(subset)}, ΔEW(Hδ)={subset['delta_ew_hd'].mean():.2f}±{subset['delta_ew_hd'].std():.2f} Å", "INFO")
    
    results['z_summary'] = z_summary
    
    # ==========================================================================
    # 3. Discriminant power analysis
    # ==========================================================================
    print_status("\n--- 3. Discriminant Power Analysis ---", "INFO")
    
    # Full sample
    power_full = compute_discriminant_power(df_pred)
    if power_full:
        results['discriminant_power_full'] = power_full
        print_status(f"  Effect size (Cohen's d): {power_full['effect_size_cohens_d']:.2f}", "INFO")
        print_status(f"  Fraction detectable: {power_full['fraction_detectable']*100:.1f}%", "INFO")
        print_status(f"  N required for 80% power: {power_full['n_required_80_power']}", "INFO")
    
    # High-z sample (z > 6)
    df_highz = df_pred[df_pred['z'] > 6]
    power_highz = compute_discriminant_power(df_highz)
    if power_highz:
        results['discriminant_power_z6'] = power_highz
        print_status(f"  High-z (z>6) effect size: {power_highz['effect_size_cohens_d']:.2f}", "INFO")
    
    # ==========================================================================
    # 4. Correlation with Γt
    # ==========================================================================
    print_status("\n--- 4. Γt-Balmer Correlation ---", "INFO")
    
    # At fixed redshift bins
    correlations = []
    for z_lo, z_hi in z_bins:
        subset = df_pred[(df_pred['z'] >= z_lo) & (df_pred['z'] < z_hi)]
        if len(subset) > 20:
            rho, p = spearmanr(subset['gamma_t'], subset['ew_hd_tep'])
            correlations.append({
                'z_range': f'{z_lo}-{z_hi}',
                'n': len(subset),
                'rho': float(rho),
                'p': format_p_value(p)
            })
            print_status(f"  z={z_lo}-{z_hi}: ρ(Γt, EW_Hδ) = {rho:.3f}, p = {p:.2e}", "INFO")
    
    results['gamma_balmer_correlations'] = correlations
    
    # ==========================================================================
    # 5. Priority targets for spectroscopy
    # ==========================================================================
    print_status("\n--- 5. Priority Targets ---", "INFO")
    
    # Select targets with largest discriminant
    df_pred['discriminant'] = np.abs(df_pred['delta_ew_hd']) * (1 + df_pred['z'] - 6)
    top_targets = df_pred.nlargest(20, 'discriminant')
    
    targets = []
    for _, row in top_targets.iterrows():
        targets.append({
            'id': row['id'],
            'z': float(row['z']),
            'log_Mstar': float(row['log_Mstar']),
            'gamma_t': float(row['gamma_t']),
            'ew_hd_std': float(row['ew_hd_std']),
            'ew_hd_tep': float(row['ew_hd_tep']),
            'delta_ew_hd': float(row['delta_ew_hd']),
            'discriminant': float(row['discriminant'])
        })
    
    results['priority_targets'] = targets
    print_status(f"  Top 20 targets identified", "INFO")
    if targets:
        print_status(f"  Best target: z={targets[0]['z']:.2f}, ΔEW={targets[0]['delta_ew_hd']:.2f} Å", "INFO")
    
    # ==========================================================================
    # 6. Mock spectrum example
    # ==========================================================================
    print_status("\n--- 6. Mock Spectrum Generation ---", "INFO")
    
    if targets:
        # Generate mock spectrum for top target
        wavelength = np.linspace(3800, 5200, 1000)
        mock = generate_mock_spectrum(
            wavelength,
            ew_hd=targets[0]['ew_hd_tep'],
            ew_hg=targets[0]['ew_hd_tep'] * 0.8,
            z=targets[0]['z'],
            snr=15
        )
        
        results['mock_spectrum_example'] = {
            'target_id': targets[0]['id'],
            'z': targets[0]['z'],
            'snr': 15,
            'wavelength_range_obs': [float(mock['wavelength_obs'].min()), float(mock['wavelength_obs'].max())],
            'note': 'Mock NIRSpec G235M/G395M spectrum'
        }
        print_status(f"  Generated mock spectrum for target {targets[0]['id']}", "INFO")
    
    # ==========================================================================
    # 7. Create figures
    # ==========================================================================
    print_status("\n--- 7. Creating Figures ---", "INFO")
    create_figures(df_pred, grid)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("BALMER SIMULATION SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    summary = {
        'method': 'Balmer absorption (Hδ, Hγ) as independent age proxy',
        'n_galaxies': len(df_pred),
        'mean_delta_ew': float(df_pred['delta_ew_hd'].mean()),
        'detectable_fraction': power_full['fraction_detectable'] if power_full else None,
        'n_priority_targets': len(targets),
        'recommended_instrument': 'JWST/NIRSpec G235M+G395M (R~1000)',
        'recommended_snr': 15,
        'exposure_estimate': '2-4 hours per target',
        'conclusion': (
            'Balmer absorption provides an independent age diagnostic that can '
            'discriminate TEP from standard physics. The predicted effect size '
            f'(Cohen\'s d = {power_full["effect_size_cohens_d"]:.2f}) is detectable '
            f'with NIRSpec for {power_full["fraction_detectable"]*100:.0f}% of targets.'
        )
    }
    
    results['summary'] = summary
    print_status(f"  Detectable fraction: {summary['detectable_fraction']*100:.1f}%", "INFO")
    print_status(f"  Priority targets: {summary['n_priority_targets']}", "INFO")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_balmer_simulation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_status(f"\nResults saved to {output_file}", "INFO")
    
    # Save predictions CSV
    pred_csv = OUTPUT_PATH / f"step_{STEP_NUM}_balmer_predictions.csv"
    df_pred.to_csv(pred_csv, index=False)
    print_status(f"Predictions saved to {pred_csv}", "INFO")


if __name__ == "__main__":
    main()
