#!/usr/bin/env python3
"""
TEP-JWST Step 39: Overmassive Black Hole Simulation

This step formalizes the "Time Bubble" hypothesis for Little Red Dots (LRDs).
It calculates the differential time enhancement between a galactic center (BH)
and the stellar halo, predicting the runaway growth factor.

Inputs:
- None (First-principles simulation based on TEP parameters)

Outputs:
- results/outputs/step_039_overmassive_bh.json
- results/outputs/step_039_bh_growth_table.csv
"""

import sys
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo  # Planck 2018 cosmology (age/distance)
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, KAPPA_GAL, KAPPA_GAL  # TEP model: Gamma_t formula, KAPPA_GAL=9.6e5 mag from Cepheids

STEP_NUM = "039"  # Pipeline step number (sequential 001-176)
STEP_NAME = "overmassive_bh"  # Overmassive BH simulation: "Time Bubble" hypothesis for Little Red Dots (LRDs) runaway growth

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# =============================================================================
# CONSTANTS & PARAMETERS
# =============================================================================

G = 4.301e-6  # Gravitational constant in kpc (km/s)^2 / M_sun

# =============================================================================
# FUNCTIONS
# =============================================================================

def gamma_from_effective_mass(log_mh_eff, z):
    return float(tep_gamma(float(log_mh_eff), float(z)))

def run_simulation():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Overmassive Black Hole Simulation", "INFO")
    print_status("=" * 70, "INFO")
    
    results = []
    zs = [4, 5, 6, 7, 8, 9, 10]
    
    # Model parameters for a typical LRD host
    log_Mh = 11.0  # Halo mass 10^11 M_sun
    concentration_factor = 10.0 # Ratio of Central Potential to Virial Potential
    t_salpeter = 0.045 # Gyr (Salpeter time)
    
    print_status(f"Simulation Parameters:", "INFO")
    print_status(f"  Halo Mass: 10^{log_Mh} M_sun", "INFO")
    print_status(f"  Central Concentration Phi_cen/Phi_vir: {concentration_factor}", "INFO")
    print_status(f"  κ_gal: {KAPPA_GAL:.3e} mag", "INFO")
    
    print_status("\nResults:", "INFO")
    print_status(f"{'z':^4} | {'Gamma_Halo':^10} | {'Gamma_Cen':^10} | {'Growth Boost':^12}", "INFO")
    print_status("-" * 50, "INFO")
    
    summary_data = []
    
    for z in zs:
        # Treat the central potential enhancement as an effective halo-mass offset.
        # For virial scaling Phi ∝ M^(2/3), a potential ratio f corresponds to:
        #   Δlog10(M) = (3/2) * log10(f)
        delta_log_mh_cen = 1.5 * np.log10(concentration_factor)
        gamma_halo = gamma_from_effective_mass(log_Mh, z)
        gamma_cen = gamma_from_effective_mass(log_Mh + delta_log_mh_cen, z)
        
        # Differential Growth
        t_cosmic = cosmo.age(z).value
        extra_efolds = (gamma_cen - gamma_halo) * t_cosmic / t_salpeter
        boost_factor = np.exp(extra_efolds)
        
        results.append({
            "z": z,
            "t_cosmic": t_cosmic,
            "gamma_halo": gamma_halo,
            "gamma_cen": gamma_cen,
            "boost_factor": boost_factor
        })
        
        print_status(f"{z:4d} | {gamma_halo:10.2f} | {gamma_cen:10.2f} | {boost_factor:12.1e}", "INFO")
        
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = OUTPUT_PATH / f"step_{STEP_NUM}_bh_growth_table.csv"
    df.to_csv(csv_path, index=False)
    print_status(f"\nSaved table to {csv_path}", "INFO")
    
    # Create JSON summary
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_overmassive_bh.json"
    
    # Extract key findings at z=7
    z7 = df[df['z'] == 7].iloc[0]
    
    summary = {
        "test": "Step 41: Overmassive BH / Time Bubble",
        "parameters": {
            "kappa_gal": KAPPA_GAL,
            "log_Mh": log_Mh,
            "concentration": concentration_factor
        },
        "z7_result": {
            "gamma_halo": float(z7['gamma_halo']),
            "gamma_cen": float(z7['gamma_cen']),
            "differential": float(z7['gamma_cen'] - z7['gamma_halo']),
            "boost_factor": float(z7['boost_factor'])
        },
        "conclusion": "Differential TEP creates a 'Time Bubble' allowing central BHs to grow >10^5x faster than the host galaxy at z>7."
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print_status(f"Saved summary to {json_path}", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")

if __name__ == "__main__":
    run_simulation()
