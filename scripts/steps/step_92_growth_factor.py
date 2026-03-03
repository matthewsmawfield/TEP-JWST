#!/usr/bin/env python3
"""
Step 92: Linear Growth Factor and Sigma8 Prediction

This script calculates the linear growth of structure D(z) and σ8(z)
in the TEP scalar-tensor framework compared to standard ΛCDM.

Physics:
The growth of density perturbations δ = δρ/ρ is governed by:
δ'' + 2Hδ' - 4π G_eff ρ_m δ = 0

In TEP (chameleon/symmetron type):
G_eff = G_N * (1 + 2β^2) in unscreened regime (voids, linear scales)
G_eff = G_N in screened regime (halos, nonlinear scales)

For linear structure formation (large scales), we use an effective coupling
that depends on the background scalar field evolution.
We model this as G_eff(z) = G_N * (1 + 2 * β_eff(z)^2)
where β_eff(z) ~ α(z)^2 ? Or just constant?
In TEP, α(z) = α0 * sqrt(1+z).
Let's assume the enhancement to G corresponds to the TEP coupling strength.
If Γ_t enhancement implies stronger gravity (or modified metric),
we check if G_eff > G.

Actually, TEP claims "Time Enhancement" Γ_t > 1.
This comes from A(φ) > 1 in potentials.
Does this enhance growth?
Yes, fifth forces generally enhance growth (G_eff > G).

We will solve the ODE for δ(z) and compute fσ8(z).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from astropy.cosmology import Planck18
from pathlib import Path
import json
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# Setup logging
STEP_NUM = "92"
STEP_NAME = "growth_factor"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# Constants
Om0 = Planck18.Om0
Ode0 = Planck18.Ode0
H0 = Planck18.H0.value
sigma8_0 = 0.811  # Planck 2018

def hubble_normalized(a):
    """E(a) = H(a)/H0"""
    z = 1.0/a - 1.0
    return np.sqrt(Om0 * (1+z)**3 + Ode0)

def growth_ode(y, a, params):
    """
    dy/da for growth factor D(a) and its derivative.
    y = [D, dD/da]
    
    Equation:
    D'' + (3/a + E'/E) D' - (3/2) (Om(a)/a^2) (G_eff/G) D = 0
    
    Om(a) = Om0 / (a^3 E(a)^2)
    """
    D, dDda = y
    
    z = 1.0/a - 1.0
    E = hubble_normalized(a)
    
    # Derivative of E w.r.t a: dE/da
    # E^2 = Om0 a^-3 + Ode0
    # 2 E dE/da = -3 Om0 a^-4
    # dE/da = -1.5 * Om0 / (a^4 * E)
    dEda = -1.5 * Om0 / (a**4 * E)
    
    Om_a = Om0 * a**-3 / E**2
    
    # Effective G
    # params['alpha0'] corresponds to TEP coupling
    # G_eff/G = 1 + 2 * beta^2
    # We assume beta_eff(z) scales with alpha(z)?
    # Or constant?
    # TEP: alpha(z) = alpha0 * sqrt(1+z)
    # This alpha is in the exponent of A(phi).
    # It relates to beta. Let's assume enhancement scales with alpha^2.
    # G_eff_ratio = 1 + C * alpha(z)^2
    
    # Standard GR
    geff_ratio = 1.0
    
    if params['model'] == 'TEP':
        # Phenomenological enhancement
        # Assume scale-dependent growth is relevant, but for linear theory
        # we take an average effect.
        # If alpha0 = 0.58, and it acts as a coupling:
        # G_eff = G * (1 + 2*beta^2)
        # Let's assume modest enhancement consistent with alpha0.
        # beta approx alpha0.
        # But this would be HUGE (1 + 2*0.58^2 ~ 1.67).
        # However, TEP says this is SCREENED in dense regions.
        # On linear scales (voids/average), is it screened?
        # Large scale structure is mostly unscreened?
        # But observational constraints (CMB/LSS) are tight.
        # The TEP paper claims "modest enhancement".
        # Maybe alpha is suppressed on horizon scales?
        # Let's assume a Yukawa suppression or similar.
        # For this check, let's test a conservative 10% enhancement to G_eff at high z
        # decaying to 0 at z=0 (screening restoration).
        
        # Scaling ansatz: Enhancement proportional to alpha(z)^2 but suppressed by background density?
        # Actually, screening is efficient when density is HIGH.
        # Density increases with z. So screening should be STRONGER at high z?
        # NO, TEP says alpha(z) INCREASES with z.
        # This implies LESS screening at high z (lower density relative to critical?).
        # Wait, density goes as (1+z)^3. Critical density is constant?
        # Paper 7 says rho_c ~ 20 g/cm^3.
        # Cosmic mean density is ~ 10^-30 g/cm^3.
        # So cosmic background is ALWAYS in the low-density (unscreened) regime?
        # If so, G_eff should be large?
        # Why is CMB not broken?
        # Because in radiation era, scalar is frozen.
        # So we turn on enhancement only in matter era.
        
        # Model: G_eff/G = 1 + epsilon * (a / a_eq) ?
        # Let's assume the coupling strength calculated in Step 90:
        # beta ~ 0.58.
        # If fully unscreened on linear scales: G_eff ~ 1.67 G.
        # This would break Sigma8 (too high).
        # We must assume the coupling relevant for structure formation is weaker,
        # or the scalar force is short-range (Yukawa).
        # Let's compute the prediction for the "Unscreened TEP" case to quantify the tension.
        
        geff_ratio = params.get('geff_ratio', None)
        if geff_ratio is None:
            beta = params.get('beta', params.get('alpha0', 0.0))
            geff_ratio = 1.0 + 2.0 * beta**2
    
    # Friction term
    term1 = (3.0/a + dEda/E) * dDda
    
    # Source term
    term2 = 1.5 * Om_a * geff_ratio * D / a**2
    
    return [dDda, term2 - term1]

def run_growth_calculation():
    print_status("=" * 60, "INFO")
    print_status("Step 92: Linear Growth Factor D(z) and σ8(z)", "INFO")
    print_status("=" * 60, "INFO")
    
    # Redshift range: z=100 to z=0
    a_start = 1.0 / 101.0
    a_end = 1.0
    a_grid = np.linspace(a_start, a_end, 1000)
    z_grid = 1.0/a_grid - 1.0
    
    # Initial conditions (Matter dominated: D ~ a)
    y0 = [a_start, 1.0] # D, dD/da
    
    # 1. Standard ΛCDM
    params_lcdm = {'model': 'LCDM'}
    sol_lcdm = odeint(growth_ode, y0, a_grid, args=(params_lcdm,))
    D_lcdm = sol_lcdm[:, 0]
    
    # 2. TEP (Unscreened / Modest)
    # We test a 5% enhancement in G_eff at z=0 scaling as (1+z)^0.5 ?
    # Let's test what 'epsilon' is needed to match high-z overabundance.
    # If JWST suggests 2x mass density, maybe we need faster growth?
    # Let's try epsilon = 0.05 (5% enhancement).
    beta_unscreened = 0.58
    geff_ratio_unscreened = 1.0 + 2.0 * beta_unscreened**2
    params_tep = {'model': 'TEP', 'geff_ratio': geff_ratio_unscreened}
    sol_tep = odeint(growth_ode, y0, a_grid, args=(params_tep,))
    D_tep = sol_tep[:, 0]
    
    # Normalize to match CMB at high z? 
    # Usually we normalize to CMB sigma8 (start at high z with same amplitude).
    # D_lcdm and D_tep start at a_start with same D=a.
    # So we compare their evolution forward.
    
    # Sigma8(z) = sigma8_0 * D(z) / D(0)_lcdm (if normalized to today)
    # But we want to fix high-z amplitude (CMB).
    # So Sigma8(z) = sigma8_cmb * D(z)/D(z_cmb).
    # Let's look at the ratio D_tep(z) / D_lcdm(z).
    
    ratio = D_tep / D_lcdm
    
    # Compute Sigma8(z=0) predicted by TEP if matching LCDM at recombination
    sigma8_tep_0 = sigma8_0 * ratio[-1]

    sigma8_sigma = 0.006
    sigma8_upper_2sigma = sigma8_0 + 2.0 * sigma8_sigma

    beta_eff_max_2sigma = beta_unscreened
    if sigma8_tep_0 > sigma8_upper_2sigma:
        beta_lo = 0.0
        beta_hi = beta_unscreened
        for _ in range(60):
            beta_mid = 0.5 * (beta_lo + beta_hi)
            params_mid = {'model': 'TEP', 'beta': beta_mid}
            sol_mid = odeint(growth_ode, y0, a_grid, args=(params_mid,))
            D_mid = sol_mid[:, 0]
            ratio_mid = D_mid / D_lcdm
            sigma8_mid = sigma8_0 * ratio_mid[-1]
            if sigma8_mid > sigma8_upper_2sigma:
                beta_hi = beta_mid
            else:
                beta_lo = beta_mid
        beta_eff_max_2sigma = beta_lo

    geff_ratio_max_2sigma = 1.0 + 2.0 * beta_eff_max_2sigma**2
    suppression_factor_beta_2sigma = beta_eff_max_2sigma / beta_unscreened if beta_unscreened != 0 else float('nan')

    print_status(f"Growth enhancement at z=0: {ratio[-1]:.4f}", "INFO")
    print_status(f"Predicted σ8(0): {sigma8_tep_0:.4f} (LCDM: {sigma8_0})", "INFO")
    print_status(f"Planck σ8 (2σ upper): {sigma8_upper_2sigma:.4f}", "INFO")
    print_status(f"Max β_eff (2σ): {beta_eff_max_2sigma:.4f}", "INFO")
    
    # Growth at z=10
    idx_z10 = np.argmin(np.abs(z_grid - 10.0))
    enhancement_z10 = ratio[idx_z10]
    print_status(f"Growth enhancement at z=10: {enhancement_z10:.4f}", "INFO")
    
    results = {
        'sigma8_lcdm': sigma8_0,
        'sigma8_tep': float(sigma8_tep_0),
        'enhancement_z0': float(ratio[-1]),
        'enhancement_z10': float(enhancement_z10),
        'beta_unscreened': float(beta_unscreened),
        'geff_ratio_unscreened': float(geff_ratio_unscreened),
        'sigma8_sigma_planck18': float(sigma8_sigma),
        'beta_eff_max_2sigma': float(beta_eff_max_2sigma),
        'geff_ratio_max_2sigma': float(geff_ratio_max_2sigma),
        'suppression_factor_beta_2sigma': float(suppression_factor_beta_2sigma)
    }
    
    with open(OUTPUT_PATH / "step_92_growth_factor.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(z_grid, ratio, label='TEP / ΛCDM')
    plt.axhline(1.0, linestyle='--', color='k')
    plt.xlabel('Redshift z')
    plt.ylabel('Growth Factor Ratio D_TEP(z) / D_LCDM(z)')
    plt.title('Linear Growth Enhancement in TEP')
    plt.xlim(0, 20)
    plt.grid(True)
    plt.legend()
    plt.savefig(FIGURES_PATH / "step_92_growth_ratio.png")
    
    return results

if __name__ == "__main__":
    run_growth_calculation()
