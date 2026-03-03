#!/usr/bin/env python3
"""
Step 129: Emission Line Diagnostic Predictions

Predicts [OIII]/Hβ vs Gamma_t correlation for gas-phase metallicity.

TEP Prediction: Gas-phase metallicity (from emission lines) should show
WEAKER correlation with Gamma_t than stellar metallicity, because gas
metallicity reflects recent enrichment while stellar metallicity integrates
over the full star formation history.

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

STEP_NUM = 129

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


def predict_emission_line_ratios():
    """
    Generate predictions for emission line diagnostics.
    
    Key predictions:
    1. [OIII]/Hβ vs Gamma_t: Weak or no correlation (gas-phase)
    2. Stellar metallicity vs Gamma_t: Strong positive correlation
    3. Gas metallicity vs stellar metallicity: Offset in high-Gamma_t systems
    """
    
    predictions = {
        'oiii_hbeta_vs_gamma_t': {
            'observable': '[OIII]5007/Hβ ratio',
            'tep_prediction': 'Weak or no correlation',
            'physical_reason': 'Gas-phase metallicity reflects recent enrichment, not integrated history',
            'expected_rho': 0.0,
            'expected_rho_range': [-0.1, 0.1],
            'falsification': 'Strong positive correlation (ρ > 0.3) would contradict TEP',
        },
        'stellar_met_vs_gamma_t': {
            'observable': 'Stellar metallicity [Z/H]',
            'tep_prediction': 'Positive correlation',
            'physical_reason': 'Stellar metallicity integrates over effective time, enhanced by Gamma_t',
            'expected_rho': 0.3,
            'expected_rho_range': [0.2, 0.5],
            'falsification': 'No correlation (ρ < 0.1) would contradict TEP',
        },
        'gas_stellar_offset': {
            'observable': 'Gas metallicity - Stellar metallicity',
            'tep_prediction': 'Negative offset in high-Gamma_t systems',
            'physical_reason': 'High-Gamma_t systems have older stellar populations but similar gas',
            'expected_offset': -0.2,  # dex
            'expected_offset_range': [-0.4, 0.0],
            'falsification': 'Positive offset in high-Gamma_t systems would contradict TEP',
        },
        'nii_ha_vs_gamma_t': {
            'observable': '[NII]6583/Hα ratio',
            'tep_prediction': 'Weak correlation (similar to [OIII]/Hβ)',
            'physical_reason': 'Nitrogen abundance is gas-phase, not integrated',
            'expected_rho': 0.05,
            'expected_rho_range': [-0.1, 0.15],
            'falsification': 'Strong correlation (ρ > 0.3) would contradict TEP',
        },
    }
    
    return predictions


def simulate_emission_line_test(n_sim=500):
    """
    Simulate what we expect to see in emission line data.
    
    Model:
    - Stellar metallicity correlates with Gamma_t (TEP effect)
    - Gas metallicity has weaker correlation (recent enrichment)
    - [OIII]/Hβ traces gas metallicity
    """
    np.random.seed(42)
    
    # Generate mock sample
    z = np.random.uniform(6, 10, n_sim)
    log_mass = np.random.normal(9.0, 0.8, n_sim)
    log_mass = np.clip(log_mass, 7.5, 11)
    
    # Compute Gamma_t
    log_mh = log_mass + 2.0
    gamma_t = compute_gamma_t(log_mh, z)
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))
    
    # Stellar metallicity: correlates with Gamma_t
    # Z_stellar ~ 0.3 * log(Gamma_t) + noise
    z_stellar = 0.3 * log_gamma + np.random.normal(0, 0.2, n_sim)
    z_stellar = np.clip(z_stellar, -2, 0.5)
    
    # Gas metallicity: weaker correlation (recent enrichment)
    # Z_gas ~ 0.05 * log(Gamma_t) + noise
    z_gas = 0.05 * log_gamma + np.random.normal(0, 0.25, n_sim)
    z_gas = np.clip(z_gas, -2, 0.5)
    
    # [OIII]/Hβ: anti-correlates with gas metallicity (standard diagnostic)
    # log([OIII]/Hβ) ~ -0.5 * Z_gas + noise
    log_o3hb = -0.5 * z_gas + np.random.normal(0, 0.2, n_sim)
    
    # Compute correlations
    rho_stellar, p_stellar = stats.spearmanr(log_gamma, z_stellar)
    rho_gas, p_gas = stats.spearmanr(log_gamma, z_gas)
    rho_o3hb, p_o3hb = stats.spearmanr(log_gamma, log_o3hb)
    
    return {
        'n_sim': n_sim,
        'stellar_metallicity': {
            'rho_vs_gamma_t': float(rho_stellar),
            'p_value': format_p_value(p_stellar),
            'interpretation': 'Strong correlation as predicted by TEP',
        },
        'gas_metallicity': {
            'rho_vs_gamma_t': float(rho_gas),
            'p_value': format_p_value(p_gas),
            'interpretation': 'Weak correlation as predicted by TEP',
        },
        'oiii_hbeta': {
            'rho_vs_gamma_t': float(rho_o3hb),
            'p_value': format_p_value(p_o3hb),
            'interpretation': 'Weak correlation (traces gas, not stellar)',
        },
        'stellar_gas_difference': {
            'mean_offset': float(np.mean(z_stellar - z_gas)),
            'interpretation': 'Stellar > Gas in high-Gamma_t systems',
        },
    }


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Emission Line Diagnostic Predictions")
    print_status("=" * 70)
    
    # Generate predictions
    predictions = predict_emission_line_ratios()
    
    print_status("\n--- TEP Predictions for Emission Lines ---")
    for key, pred in predictions.items():
        print_status(f"\n{pred['observable']}:")
        print_status(f"  Prediction: {pred['tep_prediction']}")
        print_status(f"  Reason: {pred['physical_reason']}")
        print_status(f"  Falsification: {pred['falsification']}")
    
    # Run simulation
    print_status("\n--- Simulated Emission Line Test ---")
    sim_results = simulate_emission_line_test(n_sim=500)
    
    print_status(f"\nSimulation (N = {sim_results['n_sim']}):")
    print_status(f"  Stellar Z vs Γₜ: ρ = {sim_results['stellar_metallicity']['rho_vs_gamma_t']:.3f}")
    print_status(f"  Gas Z vs Γₜ: ρ = {sim_results['gas_metallicity']['rho_vs_gamma_t']:.3f}")
    print_status(f"  [OIII]/Hβ vs Γₜ: ρ = {sim_results['oiii_hbeta']['rho_vs_gamma_t']:.3f}")
    print_status(f"  Mean (Z_stellar - Z_gas): {sim_results['stellar_gas_difference']['mean_offset']:.2f} dex")
    
    # Key discriminant
    print_status("\n" + "=" * 70)
    print_status("KEY DISCRIMINANT")
    print_status("=" * 70)
    
    print_status("\nTEP predicts a DIFFERENTIAL response:")
    print_status("  • Stellar metallicity: STRONG correlation with Γₜ (integrated history)")
    print_status("  • Gas metallicity: WEAK correlation with Γₜ (recent enrichment)")
    print_status("  • [OIII]/Hβ: WEAK correlation (traces gas, not stellar)")
    
    print_status("\nStandard physics predicts:")
    print_status("  • Both stellar and gas metallicity should correlate similarly with mass")
    print_status("  • No differential based on Γₜ")
    
    print_status("\nFalsification test:")
    print_status("  If [OIII]/Hβ shows STRONG correlation with Γₜ (ρ > 0.3),")
    print_status("  TEP is falsified because gas-phase should not track integrated time.")
    
    # Observing requirements
    print_status("\n--- Observing Requirements ---")
    print_status("  Instrument: JWST NIRSpec (R ~ 1000-2700)")
    print_status("  Lines needed: [OIII]5007, Hβ, [NII]6583, Hα")
    print_status("  Sample: N ≥ 30 galaxies at z > 6 with both emission and absorption")
    print_status("  Integration: ~5-10 hours per target for SNR > 5 on weak lines")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: Emission Line Diagnostic Predictions',
        'predictions': predictions,
        'simulation': sim_results,
        'key_discriminant': {
            'stellar_vs_gas': 'Stellar metallicity should correlate more strongly with Gamma_t than gas metallicity',
            'expected_ratio': 'ρ(stellar)/ρ(gas) > 3',
            'falsification': '[OIII]/Hβ showing ρ > 0.3 with Gamma_t would falsify TEP',
        },
        'observing_requirements': {
            'instrument': 'JWST NIRSpec',
            'resolution': 'R ~ 1000-2700',
            'sample_size': 'N ≥ 30 at z > 6',
            'integration_time': '5-10 hours per target',
        },
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_emission_line_diagnostic.json"
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
        n_sim = 500
        z = np.random.uniform(6, 10, n_sim)
        log_mass = np.random.normal(9.0, 0.8, n_sim)
        log_mass = np.clip(log_mass, 7.5, 11)
        log_mh = log_mass + 2.0
        gamma_t = compute_gamma_t(log_mh, z)
        log_gamma = np.log10(np.maximum(gamma_t, 0.01))
        
        z_stellar = 0.3 * log_gamma + np.random.normal(0, 0.2, n_sim)
        z_gas = 0.05 * log_gamma + np.random.normal(0, 0.25, n_sim)
        log_o3hb = -0.5 * z_gas + np.random.normal(0, 0.2, n_sim)
        
        # Panel 1: Stellar metallicity vs Gamma_t
        ax1 = axes[0]
        ax1.scatter(log_gamma, z_stellar, alpha=0.3, s=20, c='red', label='Stellar Z')
        z_fit = np.polyfit(log_gamma, z_stellar, 1)
        ax1.plot(np.sort(log_gamma), np.poly1d(z_fit)(np.sort(log_gamma)), 'r-', linewidth=2)
        ax1.set_xlabel('log Γₜ', fontsize=12)
        ax1.set_ylabel('[Z/H]', fontsize=12)
        ax1.set_title(f'Stellar Metallicity vs Γₜ\nρ = {sim_results["stellar_metallicity"]["rho_vs_gamma_t"]:.3f}', fontsize=12)
        ax1.legend()
        
        # Panel 2: Gas metallicity vs Gamma_t
        ax2 = axes[1]
        ax2.scatter(log_gamma, z_gas, alpha=0.3, s=20, c='blue', label='Gas Z')
        z_fit = np.polyfit(log_gamma, z_gas, 1)
        ax2.plot(np.sort(log_gamma), np.poly1d(z_fit)(np.sort(log_gamma)), 'b-', linewidth=2)
        ax2.set_xlabel('log Γₜ', fontsize=12)
        ax2.set_ylabel('[Z/H]', fontsize=12)
        ax2.set_title(f'Gas Metallicity vs Γₜ\nρ = {sim_results["gas_metallicity"]["rho_vs_gamma_t"]:.3f}', fontsize=12)
        ax2.legend()
        
        # Panel 3: [OIII]/Hβ vs Gamma_t
        ax3 = axes[2]
        ax3.scatter(log_gamma, log_o3hb, alpha=0.3, s=20, c='green', label='[OIII]/Hβ')
        z_fit = np.polyfit(log_gamma, log_o3hb, 1)
        ax3.plot(np.sort(log_gamma), np.poly1d(z_fit)(np.sort(log_gamma)), 'g-', linewidth=2)
        ax3.set_xlabel('log Γₜ', fontsize=12)
        ax3.set_ylabel('log([OIII]/Hβ)', fontsize=12)
        ax3.set_title(f'[OIII]/Hβ vs Γₜ\nρ = {sim_results["oiii_hbeta"]["rho_vs_gamma_t"]:.3f}', fontsize=12)
        ax3.legend()
        
        plt.suptitle('TEP Prediction: Stellar Z correlates with Γₜ, Gas Z does not', fontsize=14, y=1.02)
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_emission_line.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
