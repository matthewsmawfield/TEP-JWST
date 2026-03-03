
"""
TEP-JWST Discovery: The Origin of Overmassive Black Holes
Tests the hypothesis that TEP-driven differential time enhancement (Center vs Disk)
explains the high M_BH / M_star ratios observed at z > 4.

Hypothesis:
    - TEP is a local effect dependent on potential depth Phi(r).
    - The galactic center (BH location) has deeper potential than the bulk stellar disk.
    - Therefore, Gamma_t(Center) > Gamma_t(Disk).
    - Black holes experience more effective time than the host galaxy's stars.
    - BHs grow "faster" (relative to cosmic time) than the stellar population.
    - This leads to M_BH / M_star > Local Value in the early universe.

Methodology:
    1. Define a potential profile Phi(r) for a typical high-z galaxy (Bulge + Disk + Halo).
    2. Calibrate TEP scaling to match the global Halo Mass relation.
    3. Calculate Gamma_t(r=0) for the BH and Gamma_t(r=Re) for the Stars.
    4. Evolve BH and Stellar Mass over cosmic time.
    5. Compare predicted M_BH/M_* ratio to JWST observations (Pacucci+24).

Parameters:
    - z_range: 4 to 10
    - M_halo: 10^11 to 10^12 M_sun
    - alpha_0: 0.58 (Fixed from Cepheids)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.constants as c
import astropy.units as u
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_model import compute_gamma_t as tep_gamma, ALPHA_0

# Constants
G = 4.301e-6  # kpc (km/s)^2 / M_sun
C = 3e5       # km/s

def get_potential_depth(M_h, R_vir, c_conc=5.0):
    """
    Estimate central potential depth for an NFW halo + Baryonic concentration.
    Phi_cen ~ V_circ^2 * f_c
    For NFW, Phi(0) = - V_vir^2 * c / (ln(1+c) - c/(1+c)) ... roughly.
    Actually Phi(0) = - 4 pi G rho_s Rs^2 * 1 -> - V_vir^2 * c / f(c).
    
    Let's use a simpler proxy calibrated to the TEP reference.
    Ref: M_h = 10^12 at z=5.5 -> Gamma_t = 1.0.
    """
    V_vir = np.sqrt(G * M_h / R_vir)
    # Potential depth is proportional to V_vir^2
    return V_vir**2

def solve_bh_growth():
    results = []
    
    # Redshifts to probe
    zs = [4, 5, 6, 7, 8, 9, 10]
    
    print(f"{'z':^4} | {'M_halo':^8} | {'Phi_cen/Phi_vir':^15} | {'Gamma_Halo':^10} | {'Gamma_Cen':^10} | {'Ratio':^8}")
    print("-" * 80)

    for z in zs:
        # Typical LRD/High-z Galaxy Halo
        log_Mh = 11.0 # 10^11 M_sun (Host of 10^9 M_star)
        M_h = 10**log_Mh
        
        # Virial Radius (approx)
        H_z = cosmo.H(z).value
        rho_c = 3 * H_z**2 / (8 * np.pi * 4.3e-6 * 1e-6) # M_sun/kpc^3 ?? No, simplistic scaling
        # R_vir ~ 30 kpc at z=6 for 10^11?
        R_vir = 30.0 * (10/(1+z)) # rough scaling
        
        Phi_vir = G * M_h / R_vir # Proportional to V_vir^2
        
        # Central Potential Enhancement
        # In a baryon-dominated core (Bulge/Disk), potential is deeper than virial.
        # Factor f = Phi_cen / Phi_vir.
        # For NFW c=10, Phi(0)/Phi(Rvir) ~ c / (ln(1+c)-c/(1+c)) ~ 10 / 1.5 ~ 6.
        # Including baryons (dense core), factor could be 10-20.
        # LRDs are COMPACT (Re ~ 100pc). R_vir ~ 30kpc. Ratio 300.
        # Potential scales as 1/r.
        # Phi_cen ~ Phi_vir * (R_vir / R_e).
        # But mass enclosed is small.
        # Wolf Isothermal sphere: sigma ~ const. Phi ~ log(r).
        # Let's assume a moderate enhancement factor for the BH environment.
        # Conservatively: Factor 10 (Deep Core).
        
        concentration_factor = 10.0 
        Phi_cen = Phi_vir * concentration_factor
        
        # TEP Factors
        # 1. Global Halo Gamma (What we use for Stars usually)
        gamma_halo = float(tep_gamma(log_Mh, z))
        
        # 2. Central Gamma (What the BH experiences)
        delta_log_mh_cen = 1.5 * np.log10(concentration_factor)
        gamma_cen = float(tep_gamma(log_Mh + delta_log_mh_cen, z))
        
        # Differential
        ratio = gamma_cen / gamma_halo
        
        # Store
        results.append({
            "z": z,
            "gamma_halo": gamma_halo,
            "gamma_cen": gamma_cen,
            "growth_boost": ratio
        })
        
        print(f"{z:4d} | {log_Mh:8.1f} | {concentration_factor:15.1f} | {gamma_halo:10.2f} | {gamma_cen:10.2f} | {ratio:8.2f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = solve_bh_growth()
    
    # Calculate implied Mass Ratio Evolution
    # If M_bh / M_star ~ exp( (Gamma_cen - Gamma_halo) * t_age / t_salpeter )
    # This assumes both start small and grow exponentially.
    
    t_salpeter = 0.045 # Gyr
    
    print("\nImplied Overmassive BH Factors (relative to local relation):")
    print(f"{'z':^4} | {'t_cosmic (Gyr)':^15} | {'Exp Factor':^12}")
    print("-" * 40)
    
    for _, row in df.iterrows():
        z = row['z']
        t_cosmic = cosmo.age(z).value
        
        # Differential growth rate
        # Delta_Gamma = Gamma_cen - Gamma_halo
        # If stars grow at Gamma_halo rate (approx) and BH at Gamma_cen
        # The ratio evolves as exp( Delta_Gamma * t_cosmic / t_scale )
        # But this assumes growth is time-limited.
        # If t_eff > t_cosmic, BH fills its mass budget?
        
        # Simple boost factor:
        # Extra e-foldings = (Gamma_cen - Gamma_halo) * t_cosmic / t_salpeter
        extra_efolds = (row['gamma_cen'] - row['gamma_halo']) * t_cosmic / t_salpeter
        factor = np.exp(extra_efolds)
        
        print(f"{z:4.1f} | {t_cosmic:15.3f} | {factor:12.1e}")

    # Save
    df.to_csv("results/outputs/bh_tep_solution.csv", index=False)
