
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_model import compute_gamma_t as tep_gamma

# Constants
G = 4.301e-6  # kpc (km/s)^2 / M_sun

def generate_sensitivity():
    # Parameters for a typical LRD at z=6
    z = 6.0
    log_Mh = 11.0 # 10^11 M_sun halo
    M_h = 10**log_Mh
    
    # Virial quantities
    R_vir = 30.0 * (10/(1+z)) # ~43 kpc at z=6
    Phi_vir = G * M_h / R_vir
    
    # Reference (z=5.5 10^12 halo)
    Phi_ref_vir = 360.0**2 
    
    # Radii to probe: 10 pc to 2000 pc (0.01 to 2.0 kpc)
    radii_pc = np.logspace(1, 3.3, 50) # 10 to ~2000
    radii_kpc = radii_pc / 1000.0
    
    results = []
    
    # Assume Phi_cen ~ Phi_vir * (R_vir / r_eff) for an isothermal-like profile 
    # (simplistic scaling for sensitivity map)
    # Actually, for NFW+Baryon, Phi(r) increases as r decreases.
    # Let's use the concentration scaling: Phi_cen/Phi_vir ~ R_vir / r_eff is too steep (1/r).
    # Isothermal: Phi ~ log(r). 
    # Hernquist: Phi = -GM / (r+a). At r->0, Phi -> -GM/a. a ~ r_eff/1.81.
    # So Phi_cen ~ 1/r_eff.
    # Let's use Phi_cen = Phi_vir * (R_vir / (R_eff * 5)) + Phi_vir 
    # (Softened to avoid infinity, calibrated so R_eff=R_vir gives Phi_vir)
    
    for r_eff in radii_kpc:
        # Effective radius in kpc
        
        # Simple potential model for the core
        # Phi_core ~ G * M_baryon / r_eff
        # M_baryon ~ f_b * M_h ~ 0.1 * 10^11 = 10^10
        M_baryon = 10**10
        Phi_core = G * M_baryon / r_eff
        
        # Total central potential (dominates over halo potential at small r)
        Phi_total = Phi_core + Phi_vir
        
        # Halo Gamma (at R_vir or R_eff of disk?)
        # Let's say Halo Gamma is the "Disk" Gamma (r ~ 2 kpc)
        r_disk = 2.0
        Phi_disk = G * M_baryon / r_disk + Phi_vir

        # Map potential ratio to mass-equivalent offset using virial scaling Phi ∝ M^(2/3).
        phi_ratio_cen = np.maximum(Phi_total / Phi_vir, 1e-12)
        phi_ratio_disk = np.maximum(Phi_disk / Phi_vir, 1e-12)
        delta_log_mh_cen = 1.5 * np.log10(phi_ratio_cen)
        delta_log_mh_disk = 1.5 * np.log10(phi_ratio_disk)

        gamma_cen = float(tep_gamma(log_Mh + delta_log_mh_cen, z))
        gamma_disk = float(tep_gamma(log_Mh + delta_log_mh_disk, z))
        
        # Differential boost factor over cosmic time
        # Boost = exp( (Gamma_cen - Gamma_disk) * t_cosmic / t_salpeter )
        t_cosmic = cosmo.age(z).value
        t_salpeter = 0.045
        
        extra_efolds = (gamma_cen - gamma_disk) * t_cosmic / t_salpeter
        boost = np.exp(extra_efolds)
        
        # Cap boost for visualization if needed, but let's keep raw
        
        results.append({
            'r_eff_pc': r_eff * 1000,
            'Phi_cen': Phi_total,
            'gamma_cen': gamma_cen,
            'boost_factor': boost
        })
        
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv('/Users/matthewsmawfield/www/TEP-JWST/results/outputs/lrd_radius_sensitivity.csv', index=False)
    
    # Print critical radius where boost > 100
    crit = df[df['boost_factor'] < 100].iloc[0] if any(df['boost_factor'] < 100) else None
    if crit is not None:
        print(f"Critical Radius for Boost > 100: < {crit['r_eff_pc']:.1f} pc")
    else:
        print("All radii probed give Boost > 100")

if __name__ == "__main__":
    generate_sensitivity()
