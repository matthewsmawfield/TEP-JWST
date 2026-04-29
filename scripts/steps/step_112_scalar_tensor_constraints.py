#!/usr/bin/env python3
"""
Step 112: Scalar-Tensor Sector-Dictionary Constraint Check

Keeps the TEP observable response coefficient κ_gal distinct from the bare
dimensionless scalar couplings constrained by PPN, binary-pulsar, and
cosmological tests. This step intentionally does not convert κ_gal into a
Brans-Dicke omega value: that conversion is only valid for a microscopic
dimensionless coupling such as α0 or β.

Author: TEP-JWST Pipeline
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

import json
import sys
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300) & JSON serialiser for numpy types
from scripts.utils.tep_model import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY as KAPPA_GAL_ERR  # TEP model: KAPPA_GAL=9.6e5 ± 4.0e5 mag
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_DIR = RESULTS_DIR / "figures"  # Publication figures directory (PNG/PDF for manuscript)

STEP_NUM = "112"  # Pipeline step number (sequential 001-176)
STEP_NAME = "scalar_tensor_constraints"  # Sector-dictionary check: KAPPA_GAL is an observable response coefficient, not a bare PPN coupling

LOGS_DIR = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log



def alpha_to_omega_bd(alpha):
    """Convert TEP alpha to Brans-Dicke omega."""
    if alpha == 0:
        return np.inf
    return (1 - alpha**2) / (2 * alpha**2)


def omega_bd_to_alpha(omega):
    """Convert Brans-Dicke omega to TEP alpha."""
    if omega == np.inf:
        return 0
    return 1 / np.sqrt(2 * omega + 3)


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Scalar-Tensor Sector-Dictionary Check")
    print_status("=" * 70)

    print_status(f"\nObservable response coefficient: κ_gal = {KAPPA_GAL:.3e} ± {KAPPA_GAL_ERR:.3e} mag")
    print_status("Bare PPN/photon-sector benchmark: α0, β ≲ 3e-3 (dimensionless)")

    # Existing constraints on dimensionless bare/effective scalar couplings.
    constraints = [
        {
            'name': 'Cassini (Shapiro delay)',
            'omega_bd_limit': 40000,
            'alpha_limit': omega_bd_to_alpha(40000),
            'reference': 'Bertotti et al. 2003',
            'environment': 'Solar System',
            'screened': True,
        },
        {
            'name': 'Lunar Laser Ranging',
            'omega_bd_limit': 1000,
            'alpha_limit': omega_bd_to_alpha(1000),
            'reference': 'Williams et al. 2004',
            'environment': 'Earth-Moon',
            'screened': True,
        },
        {
            'name': 'Binary Pulsar (Hulse-Taylor)',
            'omega_bd_limit': 100,
            'alpha_limit': omega_bd_to_alpha(100),
            'reference': 'Damour & Taylor 1992',
            'environment': 'NS binary',
            'screened': True,
        },
        {
            'name': 'Binary Pulsar (Double Pulsar)',
            'omega_bd_limit': 200,
            'alpha_limit': omega_bd_to_alpha(200),
            'reference': 'Kramer et al. 2021',
            'environment': 'NS-NS binary',
            'screened': True,
        },
        {
            'name': 'BBN (light element abundances)',
            'omega_bd_limit': 500,
            'alpha_limit': omega_bd_to_alpha(500),
            'reference': 'Coc et al. 2006',
            'environment': 'Early universe',
            'screened': False,  # But different epoch
        },
        {
            'name': 'CMB (Planck)',
            'omega_bd_limit': 1000,
            'alpha_limit': omega_bd_to_alpha(1000),
            'reference': 'Planck 2018',
            'environment': 'z ~ 1100',
            'screened': False,  # But different epoch
        },
        {
            'name': 'Cepheid P-L (TEP-H0)',
            'omega_bd_limit': None,
            'alpha_limit': None,
            'reference': 'TEP-H0 (Paper 11)',
            'environment': 'Galactic halos',
            'screened': False,
            'quantity': 'observable_response_kappa_gal_mag',
            'kappa_gal': KAPPA_GAL,
            'kappa_gal_err': KAPPA_GAL_ERR,
        },
    ]

    print_status("\n--- Dimensionless Bare/Eff Coupling Constraints ---")
    print_status(f"{'Constraint':<30} {'ω_BD limit':>12} {'α limit':>10} {'Screened':>10}")
    print_status("-" * 70)

    for c in constraints:
        if c.get("alpha_limit") is None:
            continue
        screened_str = "Yes" if c['screened'] else "No"
        print_status(f"{c['name']:<30} {c['omega_bd_limit']:>12.0f} {c['alpha_limit']:>10.4f} {screened_str:>10}")

    sector_dictionary = {
        "kappa_gal": {
            "value": KAPPA_GAL,
            "uncertainty": KAPPA_GAL_ERR,
            "units": "mag",
            "sector": "observable magnitude/stellar-population response",
            "not_equivalent_to": ["bare beta", "PPN alpha0", "Brans-Dicke coupling"],
        },
        "bare_scalar_bound": {
            "value": 0.003,
            "units": "dimensionless",
            "sector": "PPN/photon/fifth-force",
            "source": "Cassini-style scalar-tensor mapping",
        },
    }

    print_status("\n--- Sector Dictionary ---")
    print_status("  κ_gal is not converted to ω_BD.")
    print_status("  Compatibility is a transfer-function/screening requirement, not a direct κ_gal < α0 test.")
    
    # Screening threshold
    rho_c = 20  # g/cm^3
    
    environments = [
        {'name': 'Solar core', 'rho': 150, 'screened': True},
        {'name': 'Earth core', 'rho': 13, 'screened': False},  # Marginal
        {'name': 'NS interior', 'rho': 1e15, 'screened': True},
        {'name': 'Milky Way disk', 'rho': 1e-24, 'screened': False},
        {'name': 'Galaxy cluster', 'rho': 1e-27, 'screened': False},
        {'name': 'High-z halo', 'rho': 1e-26, 'screened': False},
    ]
    
    print_status(f"\nScreening threshold: ρ_c ≈ {rho_c} g/cm³")
    print_status(f"{'Environment':<20} {'ρ (g/cm³)':>15} {'Screened':>10}")
    print_status("-" * 50)
    
    for env in environments:
        screened_str = "Yes" if env['rho'] > rho_c else "No"
        print_status(f"{env['name']:<20} {env['rho']:>15.2e} {screened_str:>10}")
    
    # Summary
    print_status("\n" + "=" * 70)
    print_status("SUMMARY")
    print_status("=" * 70)
    
    print_status("\nKey findings:")
    print_status("  1. κ_gal is an observable response coefficient in mag, not α0.")
    print_status("  2. PPN/Brans-Dicke constraints apply to dimensionless bare/effective couplings.")
    print_status("  3. A microscopic transfer calculation is required to map κ_gal to α0/β.")
    print_status("  4. Screening statements must be phrased as sector compatibility, not direct equality.")
    
    # Compile results
    results = {
        'step': f'Step {STEP_NUM}: Scalar-Tensor Sector-Dictionary Constraint Check',
        'tep_coupling': {
            'kappa_gal': KAPPA_GAL,
            'kappa_gal_err': KAPPA_GAL_ERR,
            'omega_bd_equivalent': None,
            'note': 'κ_gal is not a dimensionless bare scalar coupling; no Brans-Dicke conversion is performed.',
        },
        'sector_dictionary': sector_dictionary,
        'constraints': constraints,
        'screening': {
            'mechanism': 'Chameleon',
            'rho_c_g_cm3': rho_c,
            'environments': environments,
        },
        'compatibility': {
            'solar_system': 'Requires suppressed effective bare coupling in screened local regime',
            'binary_pulsars': 'Requires compact-object screening/transfer calculation',
            'bbn': 'Requires cosmological transfer calculation',
            'cmb': 'Requires cosmological transfer calculation',
            'jwst_galaxies': 'Uses κ_gal only as observable response prior',
        },
        'summary': {
            'direct_kappa_vs_ppn_comparison_valid': False,
            'brans_dicke_conversion_performed': False,
            'tep_compatible_with_all_constraints': None,
            'screening_required': True,
            'key_insight': 'Do not conflate observable response coefficients with bare scalar-tensor couplings.',
        },
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_scalar_tensor_constraints.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {output_path}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot constraints
        plotted = [c for c in constraints if c.get('alpha_limit') is not None]
        names = [c['name'] for c in plotted]
        alphas = [c['alpha_limit'] for c in plotted]
        colors = ['gray' if c['screened'] else 'blue' for c in plotted]
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, alphas, color=colors, alpha=0.7, edgecolor='black')
        
        ax.axvline(0.003, color='red', linestyle='--', linewidth=2, label='Bare α0 benchmark = 0.003')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Dimensionless bare/effective coupling α', fontsize=12)
        ax.set_title('Scalar-Tensor Constraints\n(κ_gal is not plotted on this axis)', fontsize=12)
        ax.set_xscale('log')
        ax.set_xlim(1e-4, 1)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        fig_path = FIGURES_DIR / f"figure_{STEP_NUM}_scalar_tensor.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
