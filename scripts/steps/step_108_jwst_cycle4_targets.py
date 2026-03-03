#!/usr/bin/env python3
"""
Step 108: JWST Cycle 4 Observing Proposal Targets

Generates a prioritized target list for JWST Cycle 4 proposals to
test TEP predictions with maximum discriminating power.

Includes:
- NIRSpec targets for Balmer absorption ages
- NIRCam targets for morphology-TEP correlation
- MIRI targets for dust confirmation
- IFU targets for resolved screening

Author: TEP-JWST Pipeline
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
import sys

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.tep_model import compute_gamma_t, ALPHA_0, LOG_MH_REF, Z_REF
from scripts.utils.p_value_utils import safe_json_default
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
FIGURES_DIR = RESULTS_DIR / "figures"
INTERIM_DIR = RESULTS_DIR / "interim"

STEP_NUM = 131

# TEP constants
def print_status(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")




def compute_discriminating_power(gamma_t, z):
    """
    Compute discriminating power between TEP and standard physics.
    
    Higher values = more useful for testing TEP.
    """
    # TEP effect is strongest at high Gamma_t and high z
    # But also need detectable signal (not too faint)
    
    # Discriminating power ~ Gamma_t * sqrt(1+z) / (1 + z)^2
    # Balances TEP signal strength against observability
    
    dp = gamma_t * np.sqrt(1 + z) / (1 + z)**1.5
    return dp


def generate_nirspec_targets(df, z_col, mass_col, n_targets=20):
    """
    Generate NIRSpec targets for Balmer absorption age test.
    
    Priority: High Gamma_t, z > 6, detectable in ~5 hours
    """
    # Filter to z > 6
    df_highz = df[df[z_col] > 6].copy()
    
    # Compute Gamma_t
    log_mh = df_highz[mass_col].values + 2.0
    z = df_highz[z_col].values
    gamma_t = compute_gamma_t(log_mh, z)
    df_highz['gamma_t'] = gamma_t
    
    # Compute discriminating power
    df_highz['disc_power'] = compute_discriminating_power(gamma_t, z)
    
    # Filter to enhanced regime (Gamma_t > 1)
    df_enhanced = df_highz[df_highz['gamma_t'] > 1].copy()
    
    # Sort by discriminating power
    df_enhanced = df_enhanced.sort_values('disc_power', ascending=False)
    
    # Select top targets
    targets = df_enhanced.head(n_targets)
    
    return targets[[z_col, mass_col, 'gamma_t', 'disc_power']].to_dict('records')


def generate_observing_programs():
    """
    Generate recommended observing programs for JWST Cycle 4.
    """
    programs = [
        {
            'program_id': 'TEP-BALMER',
            'title': 'Balmer Absorption Ages in High-Γₜ Galaxies',
            'instrument': 'NIRSpec MSA',
            'mode': 'G235M/F170LP + G395M/F290LP',
            'targets': '20 galaxies at z > 6 with Γₜ > 1.5',
            'integration': '5-10 hours per target',
            'total_time': '100-200 hours',
            'science_goal': 'Test TEP prediction that stellar ages correlate with Γₜ at fixed z',
            'tep_prediction': 'EW(Hδ) correlates with Γₜ (ρ > 0.5)',
            'standard_prediction': 'EW(Hδ) correlates with z only',
            'falsification': 'No mass trend in Balmer ages at fixed z',
            'priority': 'HIGH',
        },
        {
            'program_id': 'TEP-MORPH',
            'title': 'Morphology-TEP Correlation at z > 5',
            'instrument': 'NIRCam imaging',
            'mode': 'F277W + F444W (rest-frame optical)',
            'targets': '100 galaxies at z > 5 with existing spectroscopy',
            'integration': '2-3 hours per pointing',
            'total_time': '50-75 hours',
            'science_goal': 'Test if compact galaxies have higher Γₜ',
            'tep_prediction': 'Sérsic n correlates with Γₜ (ρ > 0.3)',
            'standard_prediction': 'No Γₜ dependence',
            'falsification': 'No correlation between compactness and Γₜ',
            'priority': 'MEDIUM',
        },
        {
            'program_id': 'TEP-IFU',
            'title': 'Resolved Screening in Massive z > 4 Galaxies',
            'instrument': 'NIRSpec IFU',
            'mode': 'G235H/F170LP',
            'targets': '5 massive (log M* > 10.5) galaxies at z = 4-6',
            'integration': '15-20 hours per target',
            'total_time': '75-100 hours',
            'science_goal': 'Test radial age gradients from core screening',
            'tep_prediction': 'Cores appear younger than outskirts (screened)',
            'standard_prediction': 'Cores appear older (inside-out growth)',
            'falsification': 'Standard inside-out gradients in all targets',
            'priority': 'HIGH',
        },
        {
            'program_id': 'TEP-MIRI',
            'title': 'MIRI Confirmation of Dust-Γₜ Correlation',
            'instrument': 'MIRI imaging',
            'mode': 'F770W + F1000W + F1500W',
            'targets': '30 galaxies at z > 7 spanning Γₜ range',
            'integration': '3-5 hours per pointing',
            'total_time': '90-150 hours',
            'science_goal': 'Confirm dust-Γₜ correlation with rest-frame MIR',
            'tep_prediction': 'Dust emission correlates with Γₜ (ρ > 0.4)',
            'standard_prediction': 'Dust correlates with mass only',
            'falsification': 'No Γₜ dependence after controlling for mass',
            'priority': 'MEDIUM',
        },
        {
            'program_id': 'TEP-CLUSTER',
            'title': 'Environmental Screening in z > 4 Protoclusters',
            'instrument': 'NIRSpec MSA',
            'mode': 'PRISM/CLEAR',
            'targets': '50 galaxies in 3 protoclusters + 50 field controls',
            'integration': '2-3 hours per pointing',
            'total_time': '50-75 hours',
            'science_goal': 'Test environmental screening prediction',
            'tep_prediction': 'Cluster galaxies show weaker TEP signatures (screened)',
            'standard_prediction': 'No environmental dependence',
            'falsification': 'Cluster = Field TEP signatures',
            'priority': 'HIGH',
        },
    ]
    
    return programs


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: JWST Cycle 4 Observing Proposal Targets")
    print_status("=" * 70)
    
    # Load data
    # Check for both naming conventions
    data_path_v1 = INTERIM_DIR / "step_02_uncover_full_sample_tep.csv"
    data_path_v2 = INTERIM_DIR / "step_002_uncover_full_sample_tep.csv"
    
    if data_path_v1.exists():
        data_path = data_path_v1
    elif data_path_v2.exists():
        data_path = data_path_v2
    else:
        print_status(f"Data file not found: {data_path_v1} or {data_path_v2}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    mass_col = 'log_Mstar' if 'log_Mstar' in df.columns else 'log_mass'
    
    print_status(f"Loaded {len(df)} galaxies")
    
    # Generate NIRSpec targets
    print_status("\n--- NIRSpec Balmer Targets ---")
    nirspec_targets = generate_nirspec_targets(df, z_col, mass_col, n_targets=20)
    
    print_status(f"Generated {len(nirspec_targets)} priority targets")
    for i, t in enumerate(nirspec_targets[:5], 1):
        print_status(f"  {i}. z={t[z_col]:.2f}, log M*={t[mass_col]:.2f}, Γₜ={t['gamma_t']:.2f}")
    
    # Generate observing programs
    print_status("\n--- Recommended Observing Programs ---")
    programs = generate_observing_programs()
    
    for prog in programs:
        print_status(f"\n{prog['program_id']}: {prog['title']}")
        print_status(f"  Instrument: {prog['instrument']}")
        print_status(f"  Time: {prog['total_time']}")
        print_status(f"  Priority: {prog['priority']}")
    
    # Summary statistics
    total_time_min = sum(int(p['total_time'].split('-')[0]) for p in programs)
    total_time_max = sum(int(p['total_time'].split('-')[1].split()[0]) for p in programs)
    
    print_status("\n" + "=" * 70)
    print_status("SUMMARY")
    print_status("=" * 70)
    
    print_status(f"\nTotal programs: {len(programs)}")
    print_status(f"Total time requested: {total_time_min}-{total_time_max} hours")
    print_status(f"High priority programs: {sum(1 for p in programs if p['priority'] == 'HIGH')}")
    
    print_status("\nKey science goals:")
    print_status("  1. Balmer absorption ages (direct age measurement)")
    print_status("  2. Morphology-TEP correlation (compactness test)")
    print_status("  3. Resolved screening (IFU radial gradients)")
    print_status("  4. MIRI dust confirmation (rest-frame MIR)")
    print_status("  5. Environmental screening (cluster vs field)")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: JWST Cycle 4 Observing Proposal Targets',
        'nirspec_targets': nirspec_targets,
        'observing_programs': programs,
        'summary': {
            'n_programs': len(programs),
            'total_time_hours': f'{total_time_min}-{total_time_max}',
            'n_high_priority': sum(1 for p in programs if p['priority'] == 'HIGH'),
        },
        'key_tests': [
            'Balmer absorption ages vs Γₜ',
            'Morphology (Sérsic n) vs Γₜ',
            'Resolved radial age gradients',
            'MIRI dust-Γₜ correlation',
            'Environmental screening in protoclusters',
        ],
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_jwst_cycle4_targets.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: Target distribution in z-mass space
        ax1 = axes[0]
        
        df_highz = df[df[z_col] > 6].copy()
        log_mh = df_highz[mass_col].values + 2.0
        z = df_highz[z_col].values
        gamma_t = compute_gamma_t(log_mh, z)
        
        scatter = ax1.scatter(z, df_highz[mass_col].values, c=np.log10(gamma_t), 
                             cmap='RdYlBu_r', alpha=0.5, s=20)
        plt.colorbar(scatter, ax=ax1, label='log Γₜ')
        
        # Mark priority targets
        for t in nirspec_targets[:10]:
            ax1.scatter(t[z_col], t[mass_col], s=100, marker='*', 
                       c='red', edgecolor='black', zorder=5)
        
        ax1.set_xlabel('Redshift', fontsize=12)
        ax1.set_ylabel('log M* [M☉]', fontsize=12)
        ax1.set_title('JWST Cycle 4 Target Selection\n(★ = Priority NIRSpec targets)', fontsize=12)
        
        # Panel 2: Program time allocation
        ax2 = axes[1]
        
        prog_names = [p['program_id'].replace('TEP-', '') for p in programs]
        times_min = [int(p['total_time'].split('-')[0]) for p in programs]
        times_max = [int(p['total_time'].split('-')[1].split()[0]) for p in programs]
        times_mid = [(t1 + t2) / 2 for t1, t2 in zip(times_min, times_max)]
        times_err = [(t2 - t1) / 2 for t1, t2 in zip(times_min, times_max)]
        
        colors = ['green' if p['priority'] == 'HIGH' else 'orange' for p in programs]
        
        bars = ax2.barh(prog_names, times_mid, xerr=times_err, color=colors, 
                       alpha=0.7, edgecolor='black', capsize=5)
        ax2.set_xlabel('Observing Time (hours)', fontsize=12)
        ax2.set_title('Proposed Observing Programs\n(Green = High Priority)', fontsize=12)
        
        # Add labels
        for i, (bar, prog) in enumerate(zip(bars, programs)):
            ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    prog['instrument'].split()[0], va='center', fontsize=9)
        
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_jwst_cycle4.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
