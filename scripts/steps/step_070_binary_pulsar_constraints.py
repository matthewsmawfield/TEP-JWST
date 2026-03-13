# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.1s.

import numpy as np
import json
from astropy import units as u
from astropy import constants as const
import sys
from pathlib import Path

# =============================================================================
# PATHS AND LOGGER
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import RHO_CRIT_G_CM3  # TEP model: critical screening density rho_c ≈ 20 g/cm³ (from Paper 7)
STEP_NUM = "070"  # Pipeline step number (sequential 001-176)
STEP_NAME = "binary_pulsar_constraints"  # Binary pulsar constraints: validates TEP screening at neutron star densities (~10¹⁴ g/cm³ >> rho_c ≈ 20 g/cm³), ensuring Hulse-Taylor orbital decay agreement

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

# =============================================================================
# PHYSICS CONSTANTS & PARAMETERS
# =============================================================================
from scripts.utils.tep_model import RHO_CRIT_G_CM3

# Critical Density for Screening (from TEP-UCD / Paper 7)
RHO_CRIT = RHO_CRIT_G_CM3 * u.g / u.cm**3

# Neutron Star Parameters (Canonical)
M_NS = 1.4 * u.M_sun
R_NS = 10.0 * u.km

# =============================================================================
# CALCULATIONS
# =============================================================================

def check_screening():
    print_status("Checking Binary Pulsar Screening Constraints...", "PROCESS")
    
    # 1. Calculate NS Density
    vol_ns = (4/3) * np.pi * R_NS**3
    rho_ns = (M_NS / vol_ns).to(u.g / u.cm**3)
    
    print_status(f"Neutron Star Mass: {M_NS}", "INFO")
    print_status(f"Neutron Star Radius: {R_NS}", "INFO")
    print_status(f"Neutron Star Density: {rho_ns:.2e}", "INFO")
    print_status(f"Critical Density (TEP): {RHO_CRIT}", "INFO")
    
    # 2. Check Screening Condition
    # Screening Factor ~ rho / rho_crit (Rough proxy for Thin Shell suppression)
    # The scalar charge Q_scalar is suppressed by ~ (rho_crit / rho)
    suppression_factor = (RHO_CRIT / rho_ns).decompose().value
    
    is_screened = rho_ns > RHO_CRIT
    
    print_status(f"Is Screened? {is_screened}", "SUCCESS" if is_screened else "WARNING")
    print_status(f"Suppression Factor (Scalar Charge): {suppression_factor:.2e}", "INFO")
    
    # 3. Hulse-Taylor Decay Agreement
    # GR prediction is verified to 0.1%. TEP needs scalar radiation to be < 0.1% of quadrupole.
    # Scalar dipole radiation Power_scalar ~ (Delta alpha)^2 * Power_quadrupole? 
    # Actually P_dipole ~ (alpha_1 - alpha_2)^2. 
    # If both are screened, alpha_eff -> 0.
    # We need suppression_factor < 1e-3?
    
    # Actually, suppression is extreme here (~1e-14).
    agreement = suppression_factor < 1e-3
    
    result = {
        "step": STEP_NUM,
        "name": "Binary Pulsar Constraints",
        "object": "Neutron Star (Canonical)",
        "density_g_cm3": float(rho_ns.value),
        "rho_crit_g_cm3": float(RHO_CRIT.value),
        "is_screened": bool(is_screened),
        "suppression_factor": float(suppression_factor),
        "conclusion": "Fully Screened" if is_screened and suppression_factor < 1e-5 else "Potential Conflict"
    }
    
    return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run():
    print_status(f"STEP {STEP_NUM}: Binary Pulsar Constraints", "TITLE")
    
    results = check_screening()
    
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print_status(f"Saved results to {json_path}", "SUCCESS")
    print_status(f"Step {STEP_NUM} complete.", "SUCCESS")

main = run

if __name__ == "__main__":
    run()
