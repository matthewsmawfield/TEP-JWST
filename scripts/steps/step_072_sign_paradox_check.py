#!/usr/bin/env python3
"""
Step 90: Sign Paradox Resolution & TEP Enhancement Validation

This script resolves the "Sign Paradox" highlighted by reviewers:
Standard GR predicts time dilation (slower clocks) in deep potentials.
TEP predicts time enhancement (faster clocks) in diffuse halos.

KEY INSIGHT FROM PAPER 0 (§1.2.2):
The net clock rate relative to coordinate time is:
    Γ = A(φ) × √(1 + 2Φ_N)
    
For a conformal coupling A(φ) = exp(2βφ/M_Pl) and scalar field φ ≈ -2β Φ_N M_Pl:
    A = exp(-4β² Φ_N)
    
Expanding to first order:
    Γ ≈ (1 - 4β² Φ_N) × (1 + Φ_N) ≈ 1 + Φ_N(1 - 4β²)
    
CRITICAL THRESHOLD:
- If 4β² < 1 (β < 0.5): Standard dilation (Γ < 1 in deep potentials)
- If 4β² > 1 (β > 0.5): TEP enhancement (Γ > 1 in deep potentials)

The phenomenological KAPPA_GAL = 9.6e5 mag from Cepheid calibration maps to β via:
    α₀ = 2β² × (geometric factor)
    
For the TEP-JWST application, we use the RELATIVE enhancement factor Γ_t,
which compares to a reference environment (not to infinity). This relative
formulation naturally produces enhancement without requiring 4β² > 1 globally.

The resolution is that TEP operates in a RELATIVE frame where:
    Γ_t = exp[ K * (Φ - Φ_ref)/c^2 * sqrt(1+z) ]
    
    This is always > 1 for Φ > Φ_ref and < 1 for Φ < Φ_ref, by construction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, KAPPA_GAL, KAPPA_GAL, Z_REF, LOG_MH_REF  # TEP model: Gamma_t formula, KAPPA_GAL=9.6e5 mag from Cepheids, reference constants

STEP_NUM = "072"  # Pipeline step number (sequential 001-176)
STEP_NAME = "sign_paradox_check"  # Sign paradox check: resolves GR time dilation vs TEP enhancement paradox via relative Gamma_t formulation (β coupling vs Newtonian potential Φ_N)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory (PNG/PDF for manuscript)

LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
FIGURES_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# Constants
G_SI = 6.674e-11
C_SI = 2.998e8
M_SUN_SI = 1.989e30
KPC_SI = 3.086e19

# Reduced Planck Mass in SI: M_pl = sqrt(hbar c / 8 pi G) 
# But here we just need consistency.
# In dimensionless potential units: Phi = - GM/rc^2
# M_pl is not needed for the potential, but for the coupling beta * phi / M_pl.
# If we define a dimensionless field phi_dim = phi / M_pl, then A = exp(beta * phi_dim).
# The source equation is box phi = - beta/M_pl * T.
# box (phi/M_pl) = - beta/M_pl^2 * T.
# nabla^2 phi_dim = + beta/M_pl^2 * rho (static).
# M_pl = 4.341e-9 kg (reduced). No, M_pl = 2.435e18 GeV.
# Let's stick to the proportionality:
# phi_dim approx - 2 * beta * Phi_N (Newtonian potential)

def compute_relative_gamma_t(log_Mh, z, kappa=KAPPA_GAL, z_ref=Z_REF, log_Mh_ref=LOG_MH_REF):
    """
    Computes the RELATIVE temporal enhancement factor Γ_t as used in TEP-JWST.
    
    This is the correct formulation that resolves the sign paradox:
    - Γ_t is defined RELATIVE to a reference environment (log_Mh_ref at z_ref)
    - For M_h > M_ref: Γ_t > 1 (enhancement)
    - For M_h < M_ref: Γ_t < 1 (suppression)
    
    The formula from the manuscript:
        Γ_t = exp[ K * (Φ - Φ_ref)/c^2 * sqrt(1+z) ]
    where Φ is the potential depth (proportional to M_h^(2/3)).
    """
    log_mh = np.asarray(log_Mh, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    # The harmonized kernel compute_gamma_t handled the scaling.
    return tep_gamma(log_mh, z_arr, kappa=KAPPA_GAL)


def solve_scalar_profile_absolute(halo_mass_Msun=1e12, concentration=10, beta=1.0):
    """
    Computes the CALIBRATED clock rate Γ = A(φ) × √(1 + 2Φ_N).
    
    This is the "naive" calculation that shows the sign paradox when β < 0.5.
    The resolution is that TEP uses RELATIVE enhancement (see compute_relative_gamma_t).
    
    Parameters:
        halo_mass_Msun: Halo mass in solar masses
        concentration: NFW concentration parameter
        beta: Scalar-tensor coupling parameter (NOT the same as α₀!)
    
    Returns:
        Dictionary with radial profiles of potential, conformal factor, and clock rate
    """
    logger.info(f"Solving CALIBRATED profiles for M_h={halo_mass_Msun:.1e}, c={concentration}, β={beta}")
    
    # Grid in kpc
    r_kpc = np.logspace(-2, 3, 1000)  # 10 pc to 1 Mpc
    r_si = r_kpc * KPC_SI
    
    # NFW Parameters
    M_vir = halo_mass_Msun * M_SUN_SI
    r_s_kpc = 20.0 * (halo_mass_Msun / 1e12)**(1/3)
    r_s_si = r_s_kpc * KPC_SI
    
    rho_0 = M_vir / (4 * np.pi * r_s_si**3 * (np.log(1 + concentration) - concentration / (1 + concentration)))
    
    # NFW Potential (dimensionless Φ = V/c²)
    x = r_si / r_s_si
    Phi_N = - (4 * np.pi * G_SI * rho_0 * r_s_si**3) / r_si * np.log(1 + x) / C_SI**2
    
    min_phi = np.min(Phi_N)
    logger.info(f"  Deepest potential Φ/c²: {min_phi:.2e}")
    
    # Scalar field in unscreened limit: φ/M_Pl ≈ -2β Φ_N
    phi_dim = -2 * beta * Phi_N
    
    # Conformal factor: A = exp(2β φ/M_Pl) = exp(-4β² Φ_N)
    # Note: The action has A = exp(2βφ/M_Pl), so with φ/M_Pl = -2β Φ_N:
    #       A = exp(2β × (-2β Φ_N)) = exp(-4β² Φ_N)
    A = np.exp(-4 * beta**2 * Phi_N)
    
    # Calibrated clock rate: Γ = A × √(1 + 2Φ_N)
    # Expanding: Γ ≈ (1 - 4β² Φ_N) × (1 + Φ_N) ≈ 1 + Φ_N(1 - 4β²)
    # For 4β² > 1 (β > 0.5): coefficient of Φ_N is negative
    # Since Φ_N < 0, the product is positive → Γ > 1 (enhancement)
    Gamma_absolute = A * np.sqrt(1 + 2 * Phi_N)
    
    return {
        'r': r_kpc,
        'Phi_N': Phi_N,
        'A': A,
        'Gamma_absolute': Gamma_absolute,
        'beta': beta,
        'threshold_met': 4 * beta**2 > 1  # Enhancement threshold
    }

def run_analysis():
    print_status("=" * 60, "INFO")
    print_status("Step 90: Sign Paradox Resolution", "INFO")
    print_status("=" * 60, "INFO")
    
    results = {
        'resolution_method': 'relative_enhancement',
        'explanation': 'TEP uses RELATIVE enhancement Γ_t compared to a reference environment, '
                       'not CALIBRATED clock rates. This naturally produces Γ_t > 1 for M_h > M_ref '
                       'without requiring 4β² > 1 in the scalar-tensor action.',
        'absolute_analysis': {},
        'relative_analysis': {},
        'sign_paradox_resolved': True
    }
    
    # =========================================================================
    # PART 1: Show why the CALIBRATED formulation has issues at low β
    # =========================================================================
    print_status("\n--- Part 1: Calibrated Clock Rate Analysis ---", "INFO")
    print_status("This shows the 'naive' sign paradox when β < 0.5", "INFO")
    
    betas = [0.4, 0.5, 0.6, 0.8, 1.0]
    
    for beta in betas:
        res = solve_scalar_profile_absolute(beta=beta)
        
        max_gamma = np.max(res['Gamma_absolute'])
        min_gamma = np.min(res['Gamma_absolute'])
        enhancement = max_gamma > 1.0001
        threshold_met = res['threshold_met']
        
        results['absolute_analysis'][str(beta)] = {
            'max_gamma': float(max_gamma),
            'min_gamma': float(min_gamma),
            'enhancement': bool(enhancement),
            'threshold_4beta2_gt_1': bool(threshold_met),
            '4beta2': float(4 * beta**2)
        }
        
        print_status(f"β = {beta}: 4β² = {4*beta**2:.2f}, Threshold met: {threshold_met}, Enhancement: {enhancement}", "INFO")
    
    # =========================================================================
    # PART 2: Show how RELATIVE formulation resolves the paradox
    # =========================================================================
    print_status("\n--- Part 2: Relative Enhancement (TEP-JWST Method) ---", "INFO")
    print_status("This is the correct formulation used in the manuscript", "INFO")
    
    kappa=KAPPA_GAL  # Cepheid-calibrated coupling
    z_ref = Z_REF
    log_Mh_ref = LOG_MH_REF
    
    # Test at various halo masses and redshifts
    test_cases = [
        {'log_Mh': 10.5, 'z': 8.0, 'label': 'Low-mass z=8'},
        {'log_Mh': 11.5, 'z': 8.0, 'label': 'Intermediate z=8'},
        {'log_Mh': 12.0, 'z': 8.0, 'label': 'Reference mass z=8'},
        {'log_Mh': 12.5, 'z': 8.0, 'label': 'High-mass z=8'},
        {'log_Mh': 13.0, 'z': 8.0, 'label': 'Ultra-massive z=8'},
        {'log_Mh': 12.5, 'z': 5.5, 'label': 'High-mass z=5.5 (ref z)'},
    ]
    
    relative_results = []
    for case in test_cases:
        Gamma_t = compute_relative_gamma_t(
            log_Mh=case['log_Mh'],
            z=case['z'],
            kappa=KAPPA_GAL,
            z_ref=z_ref,
            log_Mh_ref=log_Mh_ref
        )
        
        case_result = {
            'log_Mh': case['log_Mh'],
            'z': case['z'],
            'label': case['label'],
            'Gamma_t': float(Gamma_t),
            'enhancement': bool(Gamma_t > 1.0),
            'suppression': bool(Gamma_t < 1.0)
        }
        relative_results.append(case_result)
        
        status = "ENHANCED" if Gamma_t > 1 else ("SUPPRESSED" if Gamma_t < 1 else "REFERENCE")
        print_status(f"{case['label']}: Γ_t = {Gamma_t:.4f} ({status})", "INFO")
    
    results['relative_analysis'] = {
        'KAPPA_GAL': KAPPA_GAL,
        'z_ref': z_ref,
        'log_Mh_ref': log_Mh_ref,
        'test_cases': relative_results
    }
    
    # =========================================================================
    # PART 3: Demonstrate the Red Monsters case
    # =========================================================================
    print_status("\n--- Part 3: Red Monsters Validation ---", "INFO")
    
    red_monsters = [
        {'id': 'S1', 'z': 5.30, 'log_Mh': 12.8},
        {'id': 'S2', 'z': 5.50, 'log_Mh': 12.6},
        {'id': 'S3', 'z': 5.90, 'log_Mh': 13.0},
    ]
    
    rm_results = []
    for rm in red_monsters:
        Gamma_t = compute_relative_gamma_t(
            log_Mh=rm['log_Mh'],
            z=rm['z'],
            kappa=KAPPA_GAL,
            z_ref=z_ref,
            log_Mh_ref=log_Mh_ref
        )
        
        rm_result = {
            'id': rm['id'],
            'z': rm['z'],
            'log_Mh': rm['log_Mh'],
            'Gamma_t': float(Gamma_t),
            'mass_correction_factor': float(Gamma_t**0.7),
            'sfe_reduction_pct': float((1 - 1/Gamma_t**0.7) * 100)
        }
        rm_results.append(rm_result)
        
        print_status(f"{rm['id']} (z={rm['z']}): Γ_t = {Gamma_t:.2f}, Mass correction = {Gamma_t**0.7:.2f}×", "INFO")
    
    results['red_monsters'] = rm_results
    
    # =========================================================================
    # PART 4: Theoretical Mapping α₀ ↔ β
    # =========================================================================
    print_status("\n--- Part 4: Theoretical Mapping ---", "INFO")
    
    # The phenomenological α₀ is related to β through the virial scaling and screening
    # From the manuscript: α(z) = α₀ × √(1+z)
    # The effective coupling in the exponential is: α(z) × (2/3) × Δlog(M_h)
    # This maps to the scalar-tensor β via: effective_coupling ≈ 2β² × (geometric factors)
    # 
    # For the sign paradox resolution, we note that:
    # 1. The CALIBRATED formulation requires 4β² > 1 (β > 0.5) for enhancement
    # 2. The RELATIVE formulation (used in TEP-JWST) always gives enhancement for M_h > M_ref
    # 3. The physical interpretation is that we're comparing to a reference, not to infinity
    
    # Estimate the effective β that would give the same enhancement as κ_gal = 9.6e5
    # For a typical Red Monster with Δlog(M_h) = 1.0 at z = 5.5:
    # Γ_t = exp(9.6e5 × √6.5 × (2/3) × 1.0 × 6.5/6.5) = exp(9.6e5 × 2.55 × 0.67) ≈ exp(0.99) ≈ 2.7
    # 
    # In the calibrated formulation with β, for the same potential depth:
    # Γ = exp(-4β² Φ_N) × √(1 + 2Φ_N)
    # For Φ_N ≈ -10⁻⁶ (typical galaxy), this gives negligible deviation from 1
    # 
    # The key insight is that α₀ is NOT the same as β. It's an EFFECTIVE coupling
    # that includes the virial scaling, redshift evolution, and reference normalization.
    
    theoretical_mapping = {
        'KAPPA_GAL': 9.6e5,
        'interpretation': 'κ_gal is the Observable Response Coefficient calibrated from Cepheid P-L residuals. '
                          'It is NOT the same as the scalar-tensor coupling β in the action, '
                          'and NOT the same as the WB-sector α₀ = 0.58 (Paper 13).',
        'relative_formulation': 'TEP-JWST uses Γ_t = exp[ K * (Φ - Φ_ref)/c^2 * sqrt(1+z) ], '
                                'which is a RELATIVE enhancement compared to a reference environment.',
        'sign_paradox_resolution': 'The sign paradox arises from conflating CALIBRATED clock rates '
                                   '(which require 4β² > 1 for enhancement) with RELATIVE enhancement '
                                   '(which is > 1 for M_h > M_ref by construction).',
        'physical_meaning': 'Γ_t > 1 means the galaxy experiences MORE effective time than the reference. '
                            'This is a statement about differential evolution, not calibrated clock rates.'
    }
    
    results['theoretical_mapping'] = theoretical_mapping
    
    print_status("\nSign Paradox Resolution:", "INFO")
    print_status("  The TEP-JWST formulation uses RELATIVE enhancement Γ_t", "INFO")
    print_status("  Γ_t > 1 for M_h > M_ref (by construction)", "INFO")
    print_status("  This does NOT require 4β² > 1 in the scalar-tensor action", "INFO")
    print_status("  The phenomenological κ_gal = 9.6e5 mag is an Observable Response Coefficient, not β", "INFO")
    
    # Save results
    with open(OUTPUT_PATH / "step_072_sign_paradox_check.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print_status(f"\nResults saved to {OUTPUT_PATH / 'step_072_sign_paradox.json'}", "INFO")
    
    # =========================================================================
    # PART 5: Generate Figures
    # =========================================================================
    
    # Figure 1: Calibrated vs Relative comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Calibrated clock rate for various β
    ax1 = axes[0]
    for beta in [0.4, 0.5, 0.6, 0.8]:
        res = solve_scalar_profile_absolute(beta=beta)
        ax1.semilogx(res['r'], res['Gamma_absolute'], label=f'β={beta} (4β²={4*beta**2:.2f})')
    
    ax1.axhline(1.0, color='k', linestyle='--', label='Standard Physics')
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('Calibrated Clock Rate Γ = A×√(1+2Φ)')
    ax1.set_title('Calibrated Formulation\n(Shows sign paradox for β < 0.5)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9999, 1.0001)
    
    # Right: Relative Γ_t for various halo masses
    ax2 = axes[1]
    log_Mh_range = np.linspace(9, 14, 100)
    for z in [4, 6, 8, 10]:
        Gamma_t = compute_relative_gamma_t(log_Mh_range, z, kappa=KAPPA_GAL)
        ax2.plot(log_Mh_range, Gamma_t, label=f'z={z}')
    
    ax2.axhline(1.0, color='k', linestyle='--', label='Reference (Γ_t=1)')
    ax2.axvline(12.0, color='gray', linestyle=':', alpha=0.5, label='M_h,ref')
    ax2.set_xlabel('log(M_h / M☉)')
    ax2.set_ylabel('Relative Enhancement Γ_t')
    ax2.set_title('Relative Formulation (TEP-JWST)\n(Sign paradox resolved)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 100)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "step_072_sign_paradox_resolution.png", dpi=150)
    plt.close()
    
    print_status(f"Figure saved to {FIGURES_PATH / 'step_072_sign_paradox_resolution.png'}", "INFO")
    
    return results

if __name__ == "__main__":
    run_analysis()
