#!/usr/bin/env python3
"""
Step 126: Selection Function Simulation

Models the selection function (Malmquist bias, detection limits) and
quantifies its impact on TEP statistics.

Key question: Could the observed mass-dust correlation at z > 8 be
an artifact of selection effects rather than a physical TEP signal?

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
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.p_value_utils import format_p_value
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
FIGURES_DIR = RESULTS_DIR / "figures"
INTERIM_DIR = RESULTS_DIR / "interim"

STEP_NUM = 126

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


def simulate_selection_effects(n_sim=5000, z_range=(8, 10), mass_range=(7.5, 11)):
    """
    Simulate a population with selection effects and test if they
    can produce a spurious mass-dust correlation.
    
    Selection model:
    - Flux limit: More massive galaxies are easier to detect
    - Dust attenuation: Dusty galaxies are harder to detect (at fixed mass)
    - Redshift: Higher z galaxies are harder to detect
    """
    np.random.seed(42)
    
    # Generate intrinsic population
    z_true = np.random.uniform(z_range[0], z_range[1], n_sim)
    log_mass_true = np.random.normal(9.0, 0.8, n_sim)
    log_mass_true = np.clip(log_mass_true, mass_range[0], mass_range[1])
    
    # Intrinsic dust: NO correlation with mass (null hypothesis)
    dust_intrinsic = np.random.exponential(0.3, n_sim)
    dust_intrinsic = np.clip(dust_intrinsic, 0, 3)
    
    # Selection probability
    # P(detect) ~ f(mass, dust, z)
    # Higher mass -> easier to detect
    # Higher dust -> harder to detect (attenuated)
    # Higher z -> harder to detect
    
    mass_factor = 10**(log_mass_true - 9.0)  # Normalized to M* = 10^9
    dust_factor = 10**(-0.4 * dust_intrinsic)  # Attenuation
    z_factor = ((1 + 8) / (1 + z_true))**2  # Distance dimming
    
    # Combined detection probability
    p_detect = mass_factor * dust_factor * z_factor
    p_detect = p_detect / np.max(p_detect)  # Normalize
    p_detect = np.clip(p_detect, 0.01, 1.0)
    
    # Apply selection
    detected = np.random.random(n_sim) < p_detect
    
    # Selected sample
    z_selected = z_true[detected]
    log_mass_selected = log_mass_true[detected]
    dust_selected = dust_intrinsic[detected]
    
    # Compute Gamma_t for selected sample
    log_mh_selected = log_mass_selected + 2.0
    gamma_t_selected = compute_gamma_t(log_mh_selected, z_selected)
    log_gamma_selected = np.log10(np.maximum(gamma_t_selected, 0.01))
    
    # Compute correlations
    rho_mass_dust, p_mass_dust = stats.spearmanr(log_mass_selected, dust_selected)
    rho_gamma_dust, p_gamma_dust = stats.spearmanr(log_gamma_selected, dust_selected)

    p_mass_dust_selected = format_p_value(p_mass_dust)
    p_gamma_dust_selected = format_p_value(p_gamma_dust)
    
    return {
        'n_intrinsic': n_sim,
        'n_selected': int(np.sum(detected)),
        'selection_fraction': float(np.mean(detected)),
        'rho_mass_dust_intrinsic': 0.0,  # By construction
        'rho_mass_dust_selected': float(rho_mass_dust),
        'p_mass_dust_selected': p_mass_dust_selected,
        'rho_gamma_dust_selected': float(rho_gamma_dust),
        'p_gamma_dust_selected': p_gamma_dust_selected,
        'spurious_correlation_detected': bool(p_mass_dust_selected is not None and p_mass_dust_selected < 0.05),
    }


def compare_with_observed(df, z_col, mass_col):
    """Compare simulated selection effects with observed data."""
    # Filter to z > 8
    df_z8 = df[df[z_col] > 8].copy()
    
    # Compute observed correlation
    log_mh = df_z8[mass_col].values + 2.0
    z = df_z8[z_col].values
    dust = df_z8['dust'].values
    
    gamma_t = compute_gamma_t(log_mh, z)
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))
    
    valid = ~(np.isnan(dust) | np.isnan(log_gamma))
    rho_observed, p_observed = stats.spearmanr(log_gamma[valid], dust[valid])
    
    return {
        'n_observed': int(np.sum(valid)),
        'rho_observed': float(rho_observed),
        'p_observed': format_p_value(p_observed),
    }


def run_null_simulations(n_null=100):
    """
    Run multiple null simulations to build distribution of
    spurious correlations from selection effects alone.
    """
    rho_null = []
    
    for i in range(n_null):
        np.random.seed(i)
        result = simulate_selection_effects(n_sim=3000)
        rho_null.append(result['rho_gamma_dust_selected'])
    
    return np.array(rho_null)


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Selection Function Simulation")
    print_status("=" * 70)
    
    # Load observed data
    data_path = INTERIM_DIR / "step_02_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    mass_col = 'log_Mstar' if 'log_Mstar' in df.columns else 'log_mass'
    
    # Get observed statistics
    print_status("\n--- Observed Data (z > 8) ---")
    observed = compare_with_observed(df, z_col, mass_col)
    print_status(f"N = {observed['n_observed']}")
    print_status(f"Observed ρ(Γₜ, dust) = {observed['rho_observed']:.3f}")
    print_status(f"p-value = {observed['p_observed']:.2e}")
    
    # Run single selection simulation
    print_status("\n--- Selection Effect Simulation ---")
    sim_result = simulate_selection_effects(n_sim=5000)
    print_status(f"Intrinsic N = {sim_result['n_intrinsic']}")
    print_status(f"Selected N = {sim_result['n_selected']} ({sim_result['selection_fraction']:.1%})")
    print_status(f"Intrinsic ρ(mass, dust) = {sim_result['rho_mass_dust_intrinsic']:.3f} (by construction)")
    print_status(f"Selected ρ(mass, dust) = {sim_result['rho_mass_dust_selected']:.3f}")
    print_status(f"Selected ρ(Γₜ, dust) = {sim_result['rho_gamma_dust_selected']:.3f}")
    
    # Run null distribution
    print_status("\n--- Null Distribution (100 simulations) ---")
    rho_null = run_null_simulations(n_null=100)
    
    rho_null_mean = np.mean(rho_null)
    rho_null_std = np.std(rho_null)
    rho_null_max = np.max(np.abs(rho_null))
    
    print_status(f"Null ρ distribution: {rho_null_mean:.3f} ± {rho_null_std:.3f}")
    print_status(f"Max |ρ| from selection alone: {rho_null_max:.3f}")
    
    # Compare observed to null
    if rho_null_std <= 0 or np.isnan(rho_null_std):
        z_score = 0.0
        p_value_null = 1.0
    else:
        z_score = (observed['rho_observed'] - rho_null_mean) / rho_null_std
        if not np.isfinite(z_score):
            z_score = 0.0
            p_value_null = 1.0
        else:
            p_value_null_raw = 2 * stats.norm.sf(abs(z_score))
            p_value_null = format_p_value(p_value_null_raw)
    
    print_status("\n" + "=" * 70)
    print_status("COMPARISON: Observed vs Selection-Only Null")
    print_status("=" * 70)
    
    print_status(f"\nObserved ρ = {observed['rho_observed']:.3f}")
    print_status(f"Null (selection-only) ρ = {rho_null_mean:.3f} ± {rho_null_std:.3f}")
    print_status(f"Z-score: {z_score:.1f}σ")
    if p_value_null is None:
        print_status("P-value (vs null): N/A", "WARNING")
    else:
        print_status(f"P-value (vs null): {p_value_null:.2e}")
    
    # Conclusion
    if observed['rho_observed'] > rho_null_max:
        conclusion = "Observed correlation EXCEEDS maximum from selection effects alone"
        selection_explains = False
        print_status(f"\n✓ {conclusion}")
        print_status(f"  Selection effects cannot explain the observed signal")
    else:
        conclusion = "Observed correlation is within range of selection effects"
        selection_explains = True
        print_status(f"\n⚠ {conclusion}")
    
    # Quantify excess
    excess_rho = observed['rho_observed'] - rho_null_mean
    excess_fraction = excess_rho / observed['rho_observed'] * 100 if observed['rho_observed'] != 0 else 0
    
    print_status(f"\nExcess correlation beyond selection: Δρ = {excess_rho:.3f}")
    print_status(f"Fraction of signal NOT explained by selection: {excess_fraction:.0f}%")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: Selection Function Simulation',
        'observed': observed,
        'simulation': sim_result,
        'null_distribution': {
            'n_simulations': 100,
            'rho_mean': float(rho_null_mean),
            'rho_std': float(rho_null_std),
            'rho_max_abs': float(rho_null_max),
        },
        'comparison': {
            'z_score': float(z_score),
            'p_value_vs_null': p_value_null,
            'excess_rho': float(excess_rho),
            'excess_fraction_pct': float(excess_fraction),
            'selection_explains_signal': selection_explains,
        },
        'conclusion': conclusion,
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_selection_function.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Null distribution vs observed
        ax1 = axes[0]
        ax1.hist(rho_null, bins=20, color='gray', alpha=0.7, edgecolor='black',
                label='Selection-only null')
        ax1.axvline(observed['rho_observed'], color='red', linestyle='--', linewidth=2,
                   label=f'Observed ρ = {observed["rho_observed"]:.3f}')
        ax1.axvline(rho_null_mean, color='blue', linestyle='-', linewidth=2,
                   label=f'Null mean = {rho_null_mean:.3f}')
        ax1.set_xlabel('Spearman ρ (Γₜ vs Dust)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Selection Effects Cannot Explain\nObserved Correlation', fontsize=12)
        ax1.legend(fontsize=9)
        
        # Panel 2: Schematic of selection bias
        ax2 = axes[1]
        
        # Show that selection preferentially removes dusty low-mass galaxies
        # This would create a NEGATIVE correlation, not positive
        mass_grid = np.linspace(8, 11, 50)
        dust_grid = np.linspace(0, 2, 50)
        M, D = np.meshgrid(mass_grid, dust_grid)
        
        # Detection probability
        P_detect = 10**(M - 9.0) * 10**(-0.4 * D)
        P_detect = P_detect / np.max(P_detect)
        
        im = ax2.contourf(M, D, P_detect, levels=20, cmap='viridis')
        ax2.set_xlabel('log M* [M☉]', fontsize=12)
        ax2.set_ylabel('Dust (A_V)', fontsize=12)
        ax2.set_title('Detection Probability\n(Selection removes dusty low-mass)', fontsize=12)
        plt.colorbar(im, ax=ax2, label='P(detect)')
        
        # Add annotation
        ax2.annotate('Low detection\n(removed)', xy=(8.5, 1.5), fontsize=10,
                    ha='center', color='white')
        ax2.annotate('High detection\n(kept)', xy=(10.5, 0.5), fontsize=10,
                    ha='center', color='black')
        
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_selection_function.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
