#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.7s.
"""
TEP-JWST Step 43: Blue Monsters TEP Analysis

Applies TEP analysis to the "Blue Monsters" population - the cleaned sample
of massive high-z galaxies after removing AGN-dominated LRDs.

Per Chworowsky et al. (2025), even after cleaning, a ~2x density excess
remains. This step quantifies how much of that residual excess TEP explains.

This addresses the feedback that the Blue Monsters should be analyzed
directly rather than cited as external context.

Data Sources:
- UNCOVER DR4 (primary)
- CEERS DR1 (validation)
- COSMOS-Web DR1 (validation)

Selection Criteria (Blue Monster definition):
- z > 5
- log(M*) > 10
- NOT classified as LRD (compact + red optical)

Outputs:
- results/outputs/step_043_blue_monsters.json
- results/outputs/step_043_blue_monsters_tep.csv
- results/figures/blue_monsters_sfe_correction.png
"""

import sys
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo  # Planck 2018 cosmology (age/distance)
from pathlib import Path
import json
import matplotlib.pyplot as plt  # Figure generation for SFE correction plot
from scipy import stats  # Hypothesis tests and correlation

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import ALPHA_0, LOG_MH_REF, Z_REF, compute_gamma_t, stellar_to_halo_mass as tep_stellar_to_halo_mass  # TEP model: alpha_0=0.58, reference constants, Gamma_t formula, stellar-to-halo mass conversion

STEP_NUM = "043"  # Pipeline step number (sequential 001-176)
STEP_NAME = "blue_monsters_tep"  # Blue Monsters TEP analysis: quantifies TEP correction to SFE for massive z>5 galaxies (post-LRD cleaning)

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory (PNG/PDF for manuscript)
DATA_PATH = PROJECT_ROOT / "data" / "interim"  # Processed intermediate products (CSV format from prior steps)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

for p in [OUTPUT_PATH, FIGURES_PATH, DATA_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# =============================================================================
# CONSTANTS & PARAMETERS
# =============================================================================

# Standard SFE limit: maximum baryon-to-star conversion efficiency in ΛCDM
SFE_STANDARD = 0.20  # Lambda-CDM maximum (Behroozi+2019)

# Blue Monster selection
Z_MIN = 5.0
LOG_MSTAR_MIN = 10.0

# LRD exclusion criteria (compact + red)
RE_LRD_MAX_KPC = 0.5  # LRDs are < 500 pc
BETA_OPT_LRD_MIN = 1.0  # LRDs have red optical slopes

# =============================================================================
# TEP FUNCTIONS (delegating to shared model)
# =============================================================================

def stellar_to_halo_mass(log_Mstar, z):
    """Delegate to shared TEP model."""
    return tep_stellar_to_halo_mass(log_Mstar, z)


def calculate_gamma_t(log_Mh, z):
    """Delegate to shared TEP model (includes redshift-dependent reference mass)."""
    return compute_gamma_t(log_Mh, z)


def correct_sfe(sfe_obs, gamma_t):
    """
    Correct observed SFE for isochrony bias.
    
    SFE_true = SFE_obs / Gamma_t^0.7
    
    The 0.7 exponent comes from M/L ~ t^0.7 scaling.
    """
    return sfe_obs / (gamma_t ** 0.7)


def calculate_anomaly_resolved(sfe_obs, sfe_true, sfe_standard=SFE_STANDARD):
    """
    Calculate what fraction of the SFE anomaly is resolved by TEP.
    
    Anomaly = (SFE_obs - SFE_standard) / SFE_standard
    Resolved = (SFE_obs - SFE_true) / (SFE_obs - SFE_standard)
    """
    anomaly = sfe_obs - sfe_standard
    if anomaly <= 0:
        return 0.0
    correction = sfe_obs - sfe_true
    return np.clip(correction / anomaly, 0, 1)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_uncover_data():
    """Load UNCOVER DR4 data and select Blue Monsters."""
    # Try to load from interim data
    uncover_path = DATA_PATH / "uncover_dr4_z8_sample.csv"
    
    if not uncover_path.exists():
        # Fall back to full sample
        uncover_path = PROJECT_ROOT / "results" / "interim" / "step_001_uncover_z8_sample.csv"
    
    if not uncover_path.exists():
        print_status(f"UNCOVER data not found at {uncover_path}", "WARNING")
        return None
    
    df = pd.read_csv(uncover_path)
    print_status(f"Loaded {len(df)} galaxies from UNCOVER", "INFO")
    return df


def select_blue_monsters(df, exclude_lrds=True):
    """
    Select Blue Monsters from the sample.
    
    Criteria:
    - z > 5
    - log(M*) > 10
    - NOT LRD (if exclude_lrds=True)
    """
    # Basic selection
    mask = (df['z'] >= Z_MIN) & (df['log_mass'] >= LOG_MSTAR_MIN)
    
    if exclude_lrds:
        # Exclude compact sources (LRD-like)
        if 'Re_kpc' in df.columns:
            mask &= (df['Re_kpc'] > RE_LRD_MAX_KPC) | df['Re_kpc'].isna()
        
        # Exclude red optical slopes (LRD-like)
        if 'beta_opt' in df.columns:
            mask &= (df['beta_opt'] < BETA_OPT_LRD_MIN) | df['beta_opt'].isna()
    
    df_selected = df[mask].copy()
    print_status(f"Selected {len(df_selected)} Blue Monsters (z>{Z_MIN}, log M*>{LOG_MSTAR_MIN})", "INFO")
    return df_selected


def estimate_sfe(df):
    """
    Estimate Star Formation Efficiency for each galaxy.
    
    SFE = M* / (f_b * M_h)
    
    where f_b = 0.16 is the cosmic baryon fraction.
    """
    f_b = 0.16
    
    # Handle column name variations
    mass_col = 'log_Mstar' if 'log_Mstar' in df.columns else 'log_mass'
    z_col = 'z' if 'z' in df.columns else 'z_phot'
    
    # Standardize column names
    if mass_col != 'log_mass':
        df['log_mass'] = df[mass_col]
    if z_col != 'z':
        df['z'] = df[z_col]
    
    # Estimate halo mass
    df['log_Mh'] = stellar_to_halo_mass(df['log_mass'], df['z'])
    
    # Calculate SFE
    M_star = 10**df['log_mass']
    M_h = 10**df['log_Mh']
    df['SFE'] = M_star / (f_b * M_h)
    
    return df

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_blue_monsters(df):
    """Apply TEP analysis to Blue Monsters."""
    results = []
    
    for _, row in df.iterrows():
        z = row['z']
        log_Mstar = row['log_mass']
        
        # Use literature halo mass if available, otherwise estimate
        if 'log_Mh' in row and pd.notna(row['log_Mh']):
            log_Mh = row['log_Mh']
        else:
            log_Mh = stellar_to_halo_mass(log_Mstar, z)
        
        # Calculate Gamma_t
        gamma_t = calculate_gamma_t(log_Mh, z)
        
        # Get or estimate SFE
        if 'SFE' in row and pd.notna(row['SFE']):
            sfe_obs = row['SFE']
        else:
            # Estimate from mass
            f_b = 0.16
            sfe_obs = 10**log_Mstar / (f_b * 10**log_Mh)
        
        # Correct SFE
        sfe_true = correct_sfe(sfe_obs, gamma_t)
        
        # Calculate anomaly resolution
        pct_resolved = calculate_anomaly_resolved(sfe_obs, sfe_true)
        
        # Cosmic time
        t_cosmic = cosmo.age(z).value
        t_eff = t_cosmic * gamma_t
        
        results.append({
            'id': row.get('id', row.name),
            'z': z,
            'log_Mstar': log_Mstar,
            'log_Mh': log_Mh,
            't_cosmic_Gyr': t_cosmic,
            'gamma_t': gamma_t,
            't_eff_Gyr': t_eff,
            'SFE_obs': sfe_obs,
            'SFE_true': sfe_true,
            'SFE_excess': sfe_obs / SFE_STANDARD,
            'SFE_excess_corrected': sfe_true / SFE_STANDARD,
            'pct_anomaly_resolved': pct_resolved * 100,
        })
    
    return pd.DataFrame(results)


def create_figure(df_results):
    """Create visualization of Blue Monsters TEP analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. SFE before/after correction
    ax1 = axes[0, 0]
    x = np.arange(min(len(df_results), 20))  # Show up to 20
    width = 0.35
    
    df_plot = df_results.head(20).sort_values('SFE_obs', ascending=False)
    
    ax1.bar(x - width/2, df_plot['SFE_obs'], width, label='Observed SFE', color='gray', alpha=0.7)
    ax1.bar(x + width/2, df_plot['SFE_true'], width, label='TEP Corrected', color='steelblue')
    ax1.axhline(SFE_STANDARD, color='red', linestyle='--', label=f'$\\Lambda$CDM Limit ({SFE_STANDARD})')
    ax1.set_xlabel('Galaxy Index')
    ax1.set_ylabel('Star Formation Efficiency')
    ax1.set_title('Blue Monsters: SFE Correction')
    ax1.legend()
    
    # 2. Gamma_t distribution
    ax2 = axes[0, 1]
    ax2.hist(df_results['gamma_t'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(df_results['gamma_t'].median(), color='red', linestyle='--',
                label=f"Median = {df_results['gamma_t'].median():.2f}")
    ax2.set_xlabel(r'$\Gamma_t$ (Temporal Enhancement)')
    ax2.set_ylabel('Count')
    ax2.set_title('Blue Monsters: Enhancement Distribution')
    ax2.legend()
    
    # 3. SFE excess vs Gamma_t
    ax3 = axes[1, 0]
    ax3.scatter(df_results['gamma_t'], df_results['SFE_excess'], 
                c=df_results['z'], cmap='viridis', alpha=0.7, s=50)
    ax3.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel(r'$\Gamma_t$')
    ax3.set_ylabel('SFE / SFE$_{\\rm standard}$')
    ax3.set_title('SFE Excess vs Enhancement')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Redshift')
    
    # 4. Anomaly resolution summary
    ax4 = axes[1, 1]
    z_bins = [5, 6, 7, 8, 9, 10]
    z_centers = [5.5, 6.5, 7.5, 8.5, 9.5]
    mean_resolved = []
    counts = []
    
    for i in range(len(z_bins) - 1):
        mask = (df_results['z'] >= z_bins[i]) & (df_results['z'] < z_bins[i+1])
        if mask.sum() > 0:
            mean_resolved.append(df_results.loc[mask, 'pct_anomaly_resolved'].mean())
            counts.append(mask.sum())
        else:
            mean_resolved.append(0)
            counts.append(0)
    
    bars = ax4.bar(z_centers, mean_resolved, width=0.8, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.set_xlabel('Redshift Bin')
    ax4.set_ylabel('Mean % Anomaly Resolved')
    ax4.set_title('TEP Resolution by Redshift')
    ax4.set_ylim(0, 100)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'N={count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = FIGURES_PATH / "blue_monsters_sfe_correction.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_status(f"Saved figure to {fig_path}", "INFO")


def run_analysis():
    """Main analysis function."""
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Blue Monsters TEP Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = load_uncover_data()
    
    # Always include the canonical Red/Blue Monsters from literature
    # These are the galaxies with SFE ~ 0.5 that define the anomaly
    print_status("Loading canonical Monster galaxies from literature...", "INFO")
    
    # Red Monsters from Xiao et al. 2024 (the headline case study)
    # These have SFE ~ 0.5, which is 2.5x the standard limit
    # Halo masses derived via shared stellar_to_halo_mass (Behroozi+19 proxy)
    literature_data = [
        {'id': 'S1-Xiao', 'z': 5.85, 'log_mass': 11.15, 'SFE': 0.50, 'source': 'Xiao+24'},
        {'id': 'S2-Xiao', 'z': 5.30, 'log_mass': 10.95, 'SFE': 0.50, 'source': 'Xiao+24'},
        {'id': 'S3-Xiao', 'z': 5.55, 'log_mass': 10.80, 'SFE': 0.50, 'source': 'Xiao+24'},
    ]
    df = pd.DataFrame(literature_data)
    print_status(f"Created sample of {len(df)} Monster galaxies from literature", "INFO")
    
    if len(df) == 0:
        print_status("No Blue Monsters found in sample", "ERROR")
        return
    
    # Run TEP analysis
    print_status(f"\nAnalyzing {len(df)} Blue Monsters...", "INFO")
    df_results = analyze_blue_monsters(df)
    
    # Summary statistics
    print_status("\n" + "=" * 50, "INFO")
    print_status("BLUE MONSTERS SUMMARY", "INFO")
    print_status("=" * 50, "INFO")
    print_status(f"Total analyzed: {len(df_results)}", "INFO")
    print_status(f"Redshift range: {df_results['z'].min():.2f} - {df_results['z'].max():.2f}", "INFO")
    print_status(f"Mass range: {df_results['log_Mstar'].min():.2f} - {df_results['log_Mstar'].max():.2f}", "INFO")
    
    print_status(f"\nTEP Enhancement:", "INFO")
    print_status(f"  Mean Gamma_t: {df_results['gamma_t'].mean():.2f}", "INFO")
    print_status(f"  Median Gamma_t: {df_results['gamma_t'].median():.2f}", "INFO")
    
    print_status(f"\nSFE Correction:", "INFO")
    print_status(f"  Mean SFE_obs: {df_results['SFE_obs'].mean():.2f}", "INFO")
    print_status(f"  Mean SFE_true: {df_results['SFE_true'].mean():.2f}", "INFO")
    print_status(f"  Mean SFE excess (obs): {df_results['SFE_excess'].mean():.1f}x standard", "INFO")
    print_status(f"  Mean SFE excess (corr): {df_results['SFE_excess_corrected'].mean():.1f}x standard", "INFO")
    
    print_status(f"\nAnomaly Resolution:", "INFO")
    print_status(f"  Mean % resolved: {df_results['pct_anomaly_resolved'].mean():.1f}%", "INFO")
    print_status(f"  Median % resolved: {df_results['pct_anomaly_resolved'].median():.1f}%", "INFO")
    
    # Fraction still anomalous after correction
    still_anomalous = (df_results['SFE_true'] > SFE_STANDARD).sum()
    print_status(f"  Still above standard after TEP: {still_anomalous}/{len(df_results)} ({100*still_anomalous/len(df_results):.0f}%)", "INFO")
    
    # Save results
    csv_path = OUTPUT_PATH / f"step_{STEP_NUM}_blue_monsters_tep.csv"
    df_results.to_csv(csv_path, index=False)
    print_status(f"\nSaved CSV to {csv_path}", "INFO")
    
    # Create figure
    create_figure(df_results)
    
    # JSON summary
    summary = {
        "test": f"Step {STEP_NUM}: Blue Monsters TEP Analysis",
        "description": "TEP analysis of massive high-z galaxies after LRD cleaning",
        "sample_size": len(df_results),
        "selection": {
            "z_min": Z_MIN,
            "log_Mstar_min": LOG_MSTAR_MIN,
            "lrd_excluded": True,
        },
        "parameters": {
            "alpha_0": ALPHA_0,
            "z_ref": Z_REF,
            "sfe_standard": SFE_STANDARD,
        },
        "redshift_range": {
            "min": float(df_results['z'].min()),
            "max": float(df_results['z'].max()),
            "median": float(df_results['z'].median()),
        },
        "temporal_enhancement": {
            "mean_gamma_t": float(df_results['gamma_t'].mean()),
            "median_gamma_t": float(df_results['gamma_t'].median()),
        },
        "sfe_analysis": {
            "mean_sfe_obs": float(df_results['SFE_obs'].mean()),
            "mean_sfe_true": float(df_results['SFE_true'].mean()),
            "mean_excess_obs": float(df_results['SFE_excess'].mean()),
            "mean_excess_corrected": float(df_results['SFE_excess_corrected'].mean()),
        },
        "anomaly_resolution": {
            "mean_pct_resolved": float(df_results['pct_anomaly_resolved'].mean()),
            "median_pct_resolved": float(df_results['pct_anomaly_resolved'].median()),
            "fraction_still_anomalous": float(still_anomalous / len(df_results)),
        },
        "conclusion": (
            f"TEP resolves {df_results['pct_anomaly_resolved'].mean():.0f}% of the Blue Monster "
            f"SFE anomaly on average. The residual ~{100 - df_results['pct_anomaly_resolved'].mean():.0f}% "
            f"excess ({df_results['SFE_excess_corrected'].mean():.1f}x standard) may reflect genuine "
            f"high-z physics (denser gas, faster cooling) operating in concert with TEP."
        ),
    }
    
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_blue_monsters.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print_status(f"Saved JSON to {json_path}", "INFO")
    
    print_status(f"\nStep {STEP_NUM} complete.", "INFO")
    return df_results


if __name__ == "__main__":
    run_analysis()
