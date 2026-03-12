
"""
TEP-JWST Discovery: Little Red Dots & Overmassive Black Holes
Analyzes whether TEP can explain the 'Little Red Dot' (LRD) and 'Overmassive Black Hole' anomalies.

Hypothesis:
    LRDs are compact (R ~ 150 pc) but massive (M* ~ 10^9 - 10^10).
    This implies extremely deep gravitational potentials (high sigma).
    TEP predicts time runs FASTER in deep potentials (Gamma_t > 1).
    Faster time -> Faster Black Hole growth (exp(t_eff/t_salpeter)).
    
    This could resolve why z ~ 5-9 BHs are 10-100x more massive than expected.

Data References:
    - Matthee et al. 2024 (LRD properties)
    - Greene et al. 2024 (Overmassive BHs)
    - Pacucci et al. 2024 (BH-Galaxy relation evolution)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as c
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_model import compute_gamma_t as tep_gamma, LOG_MH_REF

# =============================================================================
# TEP PARAMETERS (From step_00_first_principles.py)
# =============================================================================

def get_tep_gamma_halo(log_Mh, z):
    """Standard TEP prediction based on Halo Mass (assuming virial scaling)."""
    return tep_gamma(log_Mh, z)

def get_tep_gamma_sigma(sigma_km_s, z):
    """
    TEP prediction based on Velocity Dispersion (Potential Depth).
    Derivation:
        Phi ~ sigma^2
        Gamma_t = exp(alpha * Phi/c^2)
    
    We need to calibrate the coefficient 'k' such that it matches the halo definition
    for a reference halo.
    
    Ref Halo: log Mh = 12.0 at z = 5.5
    Virial velocity v_vir = sqrt(G M / R_vir)
    For M_h = 10^12 at z=5.5:
        H(z) = H0 * sqrt(Om (1+z)^3 + Ol)
        R_vir = (3 M / (4 pi 200 rho_c))^1/3
        rho_c = 3 H^2 / (8 pi G)
        v_vir ~ 300 km/s? (Need to check)
        
    Let's assume the standard relation holds: Gamma_t = exp(C(z) * (sigma/sigma_ref)^2) ?
    No, the original form is exponential in Potential.
    Phi \propto sigma^2.
    Gamma_t = exp( A(z) * sigma^2 )
    
    Let's stick to the Halo Mass proxy for consistency unless we prove LRDs deviate from M-sigma.
    But LRDs are notable for being COMPACT.
    
    If R is small, Sigma is high.
    Sigma^2 ~ G M / R.
    
    Let's calculate inferred Sigma for LRDs and comparable normal galaxies.
    """
    # Placeholder for direct sigma calculation
    pass

# =============================================================================
# DATA: Little Red Dots (Representative)
# =============================================================================
# Sources: Matthee+24, Greene+24, Labbe+23
lrd_data = [
    {"id": "LRD-1", "z": 7.0, "log_Mstar": 9.5, "Re_pc": 150, "log_Mbh": 7.5, "type": "LRD"},
    {"id": "LRD-2", "z": 6.5, "log_Mstar": 9.8, "Re_pc": 200, "log_Mbh": 7.8, "type": "LRD"},
    {"id": "LRD-3", "z": 8.5, "log_Mstar": 9.2, "Re_pc": 100, "log_Mbh": 7.2, "type": "LRD"},
]

# Normal High-z Galaxies (Comparison)
normal_data = [
    {"id": "Norm-1", "z": 7.0, "log_Mstar": 9.5, "Re_pc": 1500, "log_Mbh": 6.0, "type": "Normal"}, # 10x larger radius
    {"id": "Norm-2", "z": 6.5, "log_Mstar": 9.8, "Re_pc": 2000, "log_Mbh": 6.3, "type": "Normal"},
]

df = pd.DataFrame(lrd_data + normal_data)

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_potentials(df):
    """Calculate potential depth proxies."""
    G = 4.301e-6 # kpc (km/s)^2 / M_sun
    
    results = []
    for _, row in df.iterrows():
        # Estimate Sigma from virial theorem: sigma^2 ~ G M / (5*R_e) (Wolf+10 estimator approx)
        M_star = 10**row['log_Mstar']
        Re_kpc = row['Re_pc'] / 1000.0
        
        # Assume dynamical mass follows stellar mass within Re (baryon dominated?)
        # For LRDs, they are very dense, likely baryon dominated.
        sigma_sq = G * M_star / (3 * Re_kpc) # Approximate virial factor
        sigma = np.sqrt(sigma_sq)
        
        # Calculate Halo Mass equivalent?
        # If TEP depends on POTENTIAL, we should use Sigma directly.
        # Calibrate: What Sigma corresponds to the Reference Halo (Mh=10^12)?
        # At z=5.5, Mh=10^12 -> V_vir ~ 360 km/s.
        # Let's use V_vir as the proxy for potential depth.
        
        # Calculate Gamma_t using a Sigma-based scaling derived from the Halo-based one
        # Gamma_t = exp( alpha(z) * (2/3) * log(M_h/M_ref) * z_fac )
        # Since M_h ~ V^3 (virial), log(M_h) ~ 3 log(V).
        # So log(M_h/M_ref) ~ 3 log(V/V_ref).
        # The equation becomes:
        # Gamma_t = exp( alpha(z) * 2 * log(sigma/sigma_ref) * z_fac )
        
        # Reference Halo properties at z=5.5
        Mh_ref = 10**12
        # V_vir = sqrt(G M / R_vir)
        # R_vir approx 40 kpc physical at z=5.5?
        # Let's compute V_ref precisely
        H_z = cosmo.H(5.5).value
        # rho_c = 3 H^2 / 8 pi G
        # Delta_vir ~ 200
        # M = 4/3 pi R^3 * 200 * rho_c
        # R^3 = M / (800/3 pi rho_c)
        # V^2 = G M / R
        
        # Quick approx: V_c = 200 km/s * (M/1e12)^(1/3) * E(z)^(1/6)
        # E(z=5.5) ~ sqrt(OmegaM * 6.5^3) ~ sqrt(0.3 * 274) ~ sqrt(82) ~ 9
        # V_ref ~ 200 * 1 * 9^(1/3) ~ 200 * 2.08 ~ 416 km/s
        
        V_ref = 400.0 # km/s approx
        
        # Calculate Gamma_t
        ratio_v = np.maximum(sigma / V_ref, 1e-12)
        log_mh_eff = float(LOG_MH_REF) + 3.0 * np.log10(ratio_v)
        gamma_ref = tep_gamma(float(LOG_MH_REF), float(row['z']))
        gamma_t = tep_gamma(log_mh_eff, float(row['z'])) / gamma_ref
        
        # BH Growth Enhancement
        # Standard: M = M0 exp(t_age / t_salpeter)
        # TEP: M = M0 exp(t_eff / t_salpeter)
        # Enhancement factor = exp((t_eff - t_age) / t_salpeter)
        # t_eff = gamma_t * t_age
        
        t_age = cosmo.age(row['z']).value * 1000 # Myr
        t_salpeter = 45.0 # Myr (Standard Eddington-limited e-folding time)
        
        growth_std = np.exp(t_age / t_salpeter)
        growth_tep = np.exp((gamma_t * t_age) / t_salpeter)
        
        # How much larger is the BH mass compared to standard expectation?
        # Mass_ratio = Growth_TEP / Growth_STD = exp( (gamma_t - 1) * t_age / t_salpeter )
        mass_excess_factor = np.exp( (gamma_t - 1) * t_age / t_salpeter )
        
        # Cap for physical realism (can't grow from < 1 M_sun)
        # But this factor explains the *relative* excess
        
        results.append({
            "id": row['id'],
            "type": row['type'],
            "z": row['z'],
            "sigma_kms": sigma,
            "gamma_t": gamma_t,
            "t_age_Myr": t_age,
            "t_eff_Myr": gamma_t * t_age,
            "excess_factor": mass_excess_factor
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = analyze_potentials(df)
    
    print("TEP Analysis of Little Red Dots (LRDs)")
    print("======================================")
    print(results_df.to_string(formatters={
        'sigma_kms': '{:.1f}'.format,
        'gamma_t': '{:.2f}'.format,
        't_age_Myr': '{:.0f}'.format,
        't_eff_Myr': '{:.0f}'.format,
        'excess_factor': '{:.1e}'.format
    }))
    
    # Save for user
    results_df.to_csv("results/outputs/lrd_tep_analysis.csv", index=False)
