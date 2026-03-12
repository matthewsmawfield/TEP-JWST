#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.4s.
"""
Step 89: Big Bang Nucleosynthesis (BBN) Compatibility Analysis

This script validates that the TEP scalar-tensor modification preserves
standard BBN yields within 1% by analyzing the modified expansion history
during the radiation-dominated era.

Key Physics:
- During radiation domination, the scalar field is frozen (φ ≈ const)
- The TEP coupling only becomes active when matter dominates
- BBN occurs at T ~ 0.1-1 MeV (z ~ 10^8-10^9), deep in radiation era
- The expansion rate H(z) determines freeze-out and nuclear reaction rates

Output:
- results/outputs/step_071_bbn_analysis.json
- Validates the "1% preservation" claim in the manuscript
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM  = "071"
STEP_NAME = "bbn_analysis"
LOGS_PATH = PROJECT_ROOT / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)
from scripts.utils.p_value_utils import safe_json_default

# =============================================================================
# CONSTANTS & PARAMETERS
# =============================================================================

# Physical constants
G_N = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8  # m/s
M_Pl = 1.221e19  # GeV (reduced Planck mass)
k_B = 8.617e-5  # eV/K

# Cosmological parameters (Planck 2018)
H0 = 67.4  # km/s/Mpc
Omega_r = 9.24e-5  # Radiation density today
Omega_m = 0.315  # Matter density today
Omega_Lambda = 0.685  # Dark energy density today
T_CMB = 2.725  # K (CMB temperature today)

# BBN parameters
T_BBN_start = 1.0  # MeV (BBN begins)
T_BBN_end = 0.01  # MeV (BBN ends)
z_BBN_start = T_BBN_start * 1e6 / (k_B * T_CMB)  # ~ 4e9
z_BBN_end = T_BBN_end * 1e6 / (k_B * T_CMB)  # ~ 4e7

# TEP parameters
from scripts.utils.tep_model import ALPHA_0, RHO_CRIT_G_CM3

alpha_0 = ALPHA_0  # Coupling constant from Cepheids
rho_c_screening = RHO_CRIT_G_CM3  # g/cm^3 (critical screening density)

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def hubble_standard(z):
    """Standard ΛCDM Hubble parameter."""
    return H0 * np.sqrt(Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + Omega_Lambda)

def hubble_tep(z, alpha=alpha_0):
    """
    TEP-modified Hubble parameter.
    
    During radiation domination (z >> z_eq ~ 3400), the scalar field
    is frozen because the trace of the stress-energy tensor vanishes
    for radiation (T^μ_μ = 0 for relativistic matter).
    
    The modification only becomes significant when matter dominates.
    
    Physics: G_eff only modifies the matter gravitational coupling.
    H^2 = (8πG/3)(ρ_r + (G_eff/G) × ρ_m + ρ_Λ)
    Radiation is NOT affected by the scalar coupling.
    """
    # Scalar field contribution (frozen during radiation era)
    # The effective coupling scales as (matter fraction)^2
    # matter_fraction ~ 0 at z ~ 10^9
    rho_r = Omega_r * (1 + z)**4
    rho_m = Omega_m * (1 + z)**3
    matter_fraction = rho_m / (rho_r + rho_m)
    
    # TEP modification to G_eff (only active when matter dominates)
    # In screened scalar-tensor theories, G_eff = G * (1 + 2*alpha^2) in unscreened limit
    # But during radiation era, the scalar is frozen, so G_eff ≈ G
    delta_G = 2 * alpha**2 * matter_fraction**2  # Quadratic suppression in radiation era
    
    G_eff_ratio = 1 + delta_G
    
    # G_eff only modifies the matter term, not radiation or Lambda
    return H0 * np.sqrt(
        rho_r + G_eff_ratio * rho_m + Omega_Lambda
    )

def compute_bbn_deviation():
    """
    Compute the fractional deviation in expansion rate during BBN.
    
    BBN yields are sensitive to the expansion rate through:
    - Neutron freeze-out temperature (n/p ratio)
    - Nuclear reaction rates (D, He-3, He-4, Li-7 production)
    
    A 1% change in H(z) during BBN leads to ~1% change in He-4 mass fraction.
    """
    # Sample redshifts during BBN epoch
    z_bbn = np.logspace(np.log10(z_BBN_end), np.log10(z_BBN_start), 1000)
    
    H_std = hubble_standard(z_bbn)
    H_tep = hubble_tep(z_bbn)
    
    # Fractional deviation
    # Avoid division by zero if H_std is 0 (physically anomalous here but good practice)
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_H = (H_tep - H_std) / H_std
    
    return {
        'z_range': [float(z_BBN_end), float(z_BBN_start)],
        'max_deviation': float(np.max(np.abs(delta_H))),
        'mean_deviation': float(np.mean(np.abs(delta_H))),
        'rms_deviation': float(np.sqrt(np.mean(delta_H**2))),
        'deviation_at_freeze_out': float(delta_H[np.argmin(np.abs(z_bbn - 1e9))]),  # T ~ 1 MeV
    }

def compute_helium_yield_shift(delta_H_mean):
    """
    Estimate the shift in primordial He-4 mass fraction.
    
    The He-4 yield Y_p ≈ 0.247 is sensitive to the expansion rate:
    dY_p/Y_p ≈ 0.4 * dH/H (from BBN theory)
    
    This is because faster expansion leads to earlier freeze-out,
    higher n/p ratio, and more He-4 production.
    """
    Y_p_standard = 0.247  # Planck 2018 + BBN
    sensitivity = 0.4  # dY_p/Y_p per dH/H
    
    delta_Y_p = Y_p_standard * sensitivity * delta_H_mean
    
    return {
        'Y_p_standard': Y_p_standard,
        'Y_p_tep': Y_p_standard + delta_Y_p,
        'delta_Y_p': float(delta_Y_p),
        'fractional_shift': float(delta_Y_p / Y_p_standard),
    }

def compute_deuterium_shift(delta_H_mean):
    """
    Estimate the shift in primordial deuterium abundance.
    
    D/H ≈ 2.5e-5 is more sensitive to expansion rate:
    d(D/H)/(D/H) ≈ -1.6 * dH/H (faster expansion = less D rejection)
    """
    DH_standard = 2.5e-5
    sensitivity = -1.6
    
    delta_DH = DH_standard * sensitivity * delta_H_mean
    
    return {
        'DH_standard': DH_standard,
        'DH_tep': DH_standard + delta_DH,
        'delta_DH': float(delta_DH),
        'fractional_shift': float(delta_DH / DH_standard),
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the BBN compatibility analysis."""
    print("=" * 60)
    print("Step 89: BBN Compatibility Analysis")
    print("=" * 60)
    
    results = {
        'step': 89,
        'name': 'BBN Compatibility Analysis',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'alpha_0': alpha_0,
            'H0': H0,
            'Omega_r': Omega_r,
            'Omega_m': Omega_m,
            'T_BBN_start_MeV': T_BBN_start,
            'T_BBN_end_MeV': T_BBN_end,
        }
    }
    
    # Compute expansion rate deviation
    print("\n1. Computing expansion rate deviation during BBN...")
    deviation = compute_bbn_deviation()
    results['expansion_deviation'] = deviation
    
    print(f"   Redshift range: z = {deviation['z_range'][0]:.2e} to {deviation['z_range'][1]:.2e}")
    print(f"   Max |ΔH/H|: {deviation['max_deviation']:.2e}")
    print(f"   Mean |ΔH/H|: {deviation['mean_deviation']:.2e}")
    print(f"   RMS ΔH/H: {deviation['rms_deviation']:.2e}")
    
    # Compute He-4 yield shift
    print("\n2. Computing He-4 yield shift...")
    helium = compute_helium_yield_shift(deviation['mean_deviation'])
    results['helium_4'] = helium
    
    print(f"   Y_p (standard): {helium['Y_p_standard']:.4f}")
    print(f"   Y_p (TEP): {helium['Y_p_tep']:.6f}")
    print(f"   Fractional shift: {helium['fractional_shift']*100:.4f}%")
    
    # Compute D/H shift
    print("\n3. Computing D/H shift...")
    deuterium = compute_deuterium_shift(deviation['mean_deviation'])
    results['deuterium'] = deuterium
    
    print(f"   D/H (standard): {deuterium['DH_standard']:.2e}")
    print(f"   D/H (TEP): {deuterium['DH_tep']:.2e}")
    print(f"   Fractional shift: {deuterium['fractional_shift']*100:.4f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: BBN Compatibility")
    print("=" * 60)
    
    max_yield_shift = max(
        abs(helium['fractional_shift']),
        abs(deuterium['fractional_shift'])
    )
    
    results['summary'] = {
        'max_yield_shift_percent': float(max_yield_shift * 100),
        'within_1_percent': max_yield_shift < 0.01,
        'mechanism': 'Scalar field frozen during radiation domination (T^μ_μ = 0)',
        'conclusion': 'TEP preserves BBN yields within 1%' if max_yield_shift < 0.01 else 'BBN constraint violated'
    }
    
    print(f"\nMaximum yield shift: {max_yield_shift*100:.4f}%")
    print(f"Within 1% threshold: {'YES ✓' if max_yield_shift < 0.01 else 'NO ✗'}")
    print(f"\nPhysical mechanism:")
    print("   During radiation domination, the trace of the stress-energy tensor")
    print("   vanishes (T^μ_μ = 0 for relativistic matter). This freezes the scalar")
    print("   field, preventing TEP modifications to the expansion rate during BBN.")
    print("   The coupling only becomes active when matter dominates (z < 3400).")
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'step_071_bbn_analysis.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    main()
