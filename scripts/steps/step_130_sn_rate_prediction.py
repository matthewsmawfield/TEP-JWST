#!/usr/bin/env python3
"""
Step 130: Time-Domain SN Rate Prediction

Predicts supernova rate enhancement in high-Gamma_t galaxies.

TEP Prediction: Galaxies with higher Gamma_t should show enhanced
SN rates because their stellar populations have experienced more
effective time for stellar evolution.

Author: TEP-JWST Pipeline
"""

import json
import sys
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

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.p_value_utils import format_p_value

STEP_NUM = 130

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


def predict_sn_rates():
    """
    Generate predictions for SN rates in different Gamma_t regimes.
    
    Key physics:
    - Type Ia SNe: Delay time distribution peaks at ~1 Gyr
    - Core-collapse SNe: Immediate (< 50 Myr after star formation)
    - TEP effect: High-Gamma_t galaxies have more effective time
    """
    
    predictions = {
        'type_ia': {
            'name': 'Type Ia Supernovae',
            'delay_time': '~1 Gyr (peak of DTD)',
            'tep_prediction': 'Enhanced rate in high-Gamma_t galaxies',
            'mechanism': 'More effective time for WD binary evolution',
            'expected_enhancement': 'Rate ~ Gamma_t for t_eff > 1 Gyr',
            'observable': 'SN Ia rate per unit stellar mass',
            'falsification': 'No rate difference between high/low Gamma_t at fixed mass',
        },
        'core_collapse': {
            'name': 'Core-Collapse Supernovae',
            'delay_time': '< 50 Myr',
            'tep_prediction': 'Weak or no enhancement',
            'mechanism': 'CC SNe track recent SFR, not integrated history',
            'expected_enhancement': 'Rate ~ SFR (minimal Gamma_t dependence)',
            'observable': 'CC SN rate per unit SFR',
            'falsification': 'Strong Gamma_t dependence would contradict TEP',
        },
        'ratio_ia_cc': {
            'name': 'Type Ia / Core-Collapse Ratio',
            'tep_prediction': 'Higher ratio in high-Gamma_t galaxies',
            'mechanism': 'Ia rate enhanced, CC rate unchanged',
            'expected_enhancement': 'Ratio ~ Gamma_t^0.5 for enhanced regime',
            'observable': 'N(Ia) / N(CC) per galaxy',
            'falsification': 'Constant ratio across Gamma_t would contradict TEP',
        },
    }
    
    return predictions


def simulate_sn_rates(n_galaxies=1000):
    """
    Simulate SN rates in a mock galaxy population.
    
    Model:
    - Type Ia rate ~ M_* * f(t_eff) where f increases with effective age
    - CC rate ~ SFR (independent of Gamma_t)
    """
    np.random.seed(42)
    
    # Generate mock galaxy population
    z = np.random.uniform(0.5, 2.0, n_galaxies)  # Lower z for SN detection
    log_mass = np.random.normal(10.0, 0.5, n_galaxies)
    log_mass = np.clip(log_mass, 9, 11.5)
    
    # SFR (log-normal)
    log_sfr = np.random.normal(0.5, 0.5, n_galaxies)
    sfr = 10**log_sfr
    
    # Compute Gamma_t
    log_mh = log_mass + 2.0
    gamma_t = compute_gamma_t(log_mh, z)
    
    # Effective time (Gyr)
    from astropy.cosmology import Planck18 as cosmo
    t_cosmic = np.array([cosmo.age(zi).value for zi in z])
    t_eff = gamma_t * t_cosmic
    
    # Type Ia rate model
    # Rate ~ M_* * DTD(t_eff)
    # DTD peaks at ~1 Gyr, so rate increases with t_eff up to ~few Gyr
    mass = 10**log_mass
    dtd_factor = np.minimum(t_eff / 1.0, 3.0)  # Saturates at ~3 Gyr
    rate_ia = mass * dtd_factor * 1e-13  # Normalize to ~1e-13 per M_sun per year
    
    # Add noise
    rate_ia *= np.random.lognormal(0, 0.3, n_galaxies)
    
    # Core-collapse rate model
    # Rate ~ SFR (no Gamma_t dependence)
    rate_cc = sfr * 0.01  # ~1 CC SN per 100 M_sun of SF
    rate_cc *= np.random.lognormal(0, 0.3, n_galaxies)
    
    # Compute correlations
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))
    
    # Ia rate vs Gamma_t (controlling for mass)
    # Partial correlation
    from scipy.stats import spearmanr
    
    # Simple correlation
    rho_ia_gamma, p_ia = spearmanr(log_gamma, np.log10(rate_ia))
    rho_cc_gamma, p_cc = spearmanr(log_gamma, np.log10(rate_cc))
    
    # Ratio
    ratio_ia_cc = rate_ia / (rate_cc + 1e-10)
    rho_ratio_gamma, p_ratio = spearmanr(log_gamma, np.log10(ratio_ia_cc))
    
    # Split by Gamma_t
    high_gamma = gamma_t > np.median(gamma_t)
    low_gamma = ~high_gamma
    
    mean_ia_high = np.mean(rate_ia[high_gamma])
    mean_ia_low = np.mean(rate_ia[low_gamma])
    ia_enhancement = mean_ia_high / mean_ia_low
    
    mean_cc_high = np.mean(rate_cc[high_gamma])
    mean_cc_low = np.mean(rate_cc[low_gamma])
    cc_enhancement = mean_cc_high / mean_cc_low
    
    return {
        'n_galaxies': n_galaxies,
        'type_ia': {
            'rho_vs_gamma_t': float(rho_ia_gamma),
            'p_value': format_p_value(p_ia),
            'enhancement_high_vs_low': float(ia_enhancement),
        },
        'core_collapse': {
            'rho_vs_gamma_t': float(rho_cc_gamma),
            'p_value': format_p_value(p_cc),
            'enhancement_high_vs_low': float(cc_enhancement),
        },
        'ratio_ia_cc': {
            'rho_vs_gamma_t': float(rho_ratio_gamma),
            'p_value': format_p_value(p_ratio),
        },
        'interpretation': {
            'ia_enhanced': bool(ia_enhancement > 1.2),
            'cc_not_enhanced': bool(cc_enhancement < 1.2),
            'consistent_with_tep': bool((ia_enhancement > 1.2) and (cc_enhancement < 1.5)),
        },
    }


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Time-Domain SN Rate Prediction")
    print_status("=" * 70)
    
    # Generate predictions
    predictions = predict_sn_rates()
    
    print_status("\n--- TEP Predictions for SN Rates ---")
    for key, pred in predictions.items():
        print_status(f"\n{pred['name']}:")
        print_status(f"  Prediction: {pred['tep_prediction']}")
        print_status(f"  Mechanism: {pred['mechanism']}")
        print_status(f"  Observable: {pred['observable']}")
        print_status(f"  Falsification: {pred['falsification']}")
    
    # Run simulation
    print_status("\n--- Simulated SN Rate Test ---")
    sim_results = simulate_sn_rates(n_galaxies=1000)
    
    print_status(f"\nSimulation (N = {sim_results['n_galaxies']} galaxies):")
    print_status(f"\nType Ia SNe:")
    print_status(f"  ρ(rate, Γₜ) = {sim_results['type_ia']['rho_vs_gamma_t']:.3f}")
    print_status(f"  Enhancement (high/low Γₜ) = {sim_results['type_ia']['enhancement_high_vs_low']:.2f}×")
    
    print_status(f"\nCore-Collapse SNe:")
    print_status(f"  ρ(rate, Γₜ) = {sim_results['core_collapse']['rho_vs_gamma_t']:.3f}")
    print_status(f"  Enhancement (high/low Γₜ) = {sim_results['core_collapse']['enhancement_high_vs_low']:.2f}×")
    
    print_status(f"\nIa/CC Ratio:")
    print_status(f"  ρ(ratio, Γₜ) = {sim_results['ratio_ia_cc']['rho_vs_gamma_t']:.3f}")
    
    # Key discriminant
    print_status("\n" + "=" * 70)
    print_status("KEY DISCRIMINANT")
    print_status("=" * 70)
    
    print_status("\nTEP predicts DIFFERENTIAL SN rate response:")
    print_status("  • Type Ia: ENHANCED in high-Γₜ galaxies (more time for WD evolution)")
    print_status("  • Core-Collapse: NO enhancement (tracks recent SFR)")
    print_status("  • Ia/CC Ratio: INCREASES with Γₜ")
    
    print_status("\nStandard physics predicts:")
    print_status("  • Both Ia and CC rates scale with mass/SFR")
    print_status("  • No differential based on Γₜ at fixed mass")
    
    # Observing requirements
    print_status("\n--- Observing Requirements ---")
    print_status("  Survey: Roman Space Telescope High-Latitude Time Domain Survey")
    print_status("  Sample: ~1000 SNe at z < 2 with host galaxy properties")
    print_status("  Key measurement: SN rate per unit mass in bins of host Γₜ")
    print_status("  Timeline: Roman launch 2027, first results ~2029")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: Time-Domain SN Rate Prediction',
        'predictions': predictions,
        'simulation': sim_results,
        'key_discriminant': {
            'type_ia': 'Enhanced rate in high-Gamma_t galaxies',
            'core_collapse': 'No enhancement (tracks SFR)',
            'ratio': 'Ia/CC ratio increases with Gamma_t',
            'falsification': 'No differential response would falsify TEP',
        },
        'observing_requirements': {
            'survey': 'Roman High-Latitude Time Domain Survey',
            'sample_size': '~1000 SNe at z < 2',
            'key_measurement': 'SN rate per unit mass vs host Gamma_t',
            'timeline': 'First results ~2029',
        },
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_sn_rate_prediction.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Regenerate simulation data for plotting
        np.random.seed(42)
        n_galaxies = 1000
        z = np.random.uniform(0.5, 2.0, n_galaxies)
        log_mass = np.random.normal(10.0, 0.5, n_galaxies)
        log_mass = np.clip(log_mass, 9, 11.5)
        log_sfr = np.random.normal(0.5, 0.5, n_galaxies)
        sfr = 10**log_sfr
        
        log_mh = log_mass + 2.0
        gamma_t = compute_gamma_t(log_mh, z)
        log_gamma = np.log10(np.maximum(gamma_t, 0.01))
        
        from astropy.cosmology import Planck18 as cosmo
        t_cosmic = np.array([cosmo.age(zi).value for zi in z])
        t_eff = gamma_t * t_cosmic
        
        mass = 10**log_mass
        dtd_factor = np.minimum(t_eff / 1.0, 3.0)
        rate_ia = mass * dtd_factor * 1e-13
        rate_ia *= np.random.lognormal(0, 0.3, n_galaxies)
        
        rate_cc = sfr * 0.01
        rate_cc *= np.random.lognormal(0, 0.3, n_galaxies)
        
        # Panel 1: Type Ia rate vs Gamma_t
        ax1 = axes[0]
        ax1.scatter(log_gamma, np.log10(rate_ia), alpha=0.3, s=20, c='red')
        z_fit = np.polyfit(log_gamma, np.log10(rate_ia), 1)
        ax1.plot(np.sort(log_gamma), np.poly1d(z_fit)(np.sort(log_gamma)), 'r-', linewidth=2)
        ax1.set_xlabel('log Γₜ', fontsize=12)
        ax1.set_ylabel('log(Type Ia Rate)', fontsize=12)
        ax1.set_title(f'Type Ia Rate vs Γₜ\nρ = {sim_results["type_ia"]["rho_vs_gamma_t"]:.3f}', fontsize=12)
        
        # Panel 2: CC rate vs Gamma_t
        ax2 = axes[1]
        ax2.scatter(log_gamma, np.log10(rate_cc), alpha=0.3, s=20, c='blue')
        z_fit = np.polyfit(log_gamma, np.log10(rate_cc), 1)
        ax2.plot(np.sort(log_gamma), np.poly1d(z_fit)(np.sort(log_gamma)), 'b-', linewidth=2)
        ax2.set_xlabel('log Γₜ', fontsize=12)
        ax2.set_ylabel('log(CC Rate)', fontsize=12)
        ax2.set_title(f'Core-Collapse Rate vs Γₜ\nρ = {sim_results["core_collapse"]["rho_vs_gamma_t"]:.3f}', fontsize=12)
        
        # Panel 3: Ia/CC ratio vs Gamma_t
        ax3 = axes[2]
        ratio = rate_ia / (rate_cc + 1e-10)
        ax3.scatter(log_gamma, np.log10(ratio), alpha=0.3, s=20, c='green')
        z_fit = np.polyfit(log_gamma, np.log10(ratio), 1)
        ax3.plot(np.sort(log_gamma), np.poly1d(z_fit)(np.sort(log_gamma)), 'g-', linewidth=2)
        ax3.set_xlabel('log Γₜ', fontsize=12)
        ax3.set_ylabel('log(Ia/CC Ratio)', fontsize=12)
        ax3.set_title(f'Ia/CC Ratio vs Γₜ\nρ = {sim_results["ratio_ia_cc"]["rho_vs_gamma_t"]:.3f}', fontsize=12)
        
        plt.suptitle('TEP Prediction: Type Ia rate enhanced, CC rate unchanged', fontsize=14, y=1.02)
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_sn_rate.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
