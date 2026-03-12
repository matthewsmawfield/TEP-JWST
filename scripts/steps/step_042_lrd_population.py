#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.9s.
"""
TEP-JWST Step 42: LRD Population Differential Temporal Shear Analysis

This step applies the differential temporal shear simulation to the full
Kokorev et al. (2024) LRD population (N=260) to validate the mechanism
at the population level.

Inputs:
- Step 41 simulation framework (differential shear calculation)

Outputs:
- results/outputs/step_042_lrd_population.json
- results/outputs/step_042_lrd_population.csv
"""

import sys
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t as tep_gamma

STEP_NUM = "042"
STEP_NAME = "lrd_population"

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"
DATA_PATH = PROJECT_ROOT / "data" / "raw"

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# CONSTANTS & PARAMETERS
# =============================================================================

T_SALPETER = 45e6  # years (Salpeter timescale)

# LRD typical parameters from Kokorev et al. (2024)
R_E_MEDIAN = 150  # pc (effective radius)
CONCENTRATION = 10  # baryon-dominated core
LRD_CATALOG_PATH = DATA_PATH / "kokorev_lrd_catalog_v1.1.fits"

# =============================================================================
# FUNCTIONS
# =============================================================================

def gamma_t_core(log_mh, z, concentration=10):
    """
    Calculate Gamma_t at the galactic core (enhanced by concentration).
    The core potential is deeper by factor ~ c^(1/3).
    """
    core_boost = concentration**(1/3)
    log_mh_core = log_mh + np.log10(core_boost)
    return tep_gamma(log_mh_core, z)

def gamma_t_halo(log_mh, z):
    """
    Calculate Gamma_t at the halo effective radius (suppressed).
    """
    return tep_gamma(log_mh - 0.5, z)  # Halo is shallower by ~0.5 dex

def compute_boost(gamma_cen, gamma_halo, t_cosmic_yr):
    """
    Compute the differential growth boost factor.
    Boost = exp((Gamma_cen - Gamma_halo) * t_cosmic / t_Salpeter)
    """
    delta_gamma = gamma_cen - gamma_halo
    exponent = delta_gamma * t_cosmic_yr / T_SALPETER
    return np.exp(exponent)


def load_lrd_catalog():
    if not LRD_CATALOG_PATH.exists():
        print_status(f"LRD catalog not found: {LRD_CATALOG_PATH}", "ERROR")
        return None

    df = Table.read(LRD_CATALOG_PATH).to_pandas()
    col_map = {
        'z_phot': 'z',
        'r_eff_50_phys': 'Re_kpc',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'z' not in df.columns:
        print_status("LRD catalog missing redshift column", "ERROR")
        return None
    df = df[pd.notna(df['z']) & (df['z'] > 4) & (df['z'] < 10)].copy()
    print_status(f"Loaded {len(df)} LRDs from Kokorev catalog", "INFO")
    return df


def estimate_halo_mass(log_mstar, z):
    if pd.isna(log_mstar):
        return 11.0
    log_mh = log_mstar + 1.5 + 0.1 * (z - 5)
    return float(np.clip(log_mh, 9.0, 14.0))

def simulate_lrd_population():
    """Quarantine the legacy LRD population step in favor of the real-data validation path."""
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: LRD Population Differential Temporal Shear Analysis", "INFO")
    print_status("=" * 70, "INFO")

    status = {
        "status": "skipped",
        "reason": "Legacy step_042 population values were assumption-heavy and are quarantined to avoid non-empirical claims; use step_132_lrd_validation for the real Kokorev-catalog analysis.",
        "superseded_by": "step_132_lrd_validation",
        "data_source": "Kokorev et al. 2024 (catalog)",
    }

    output_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_path, "w") as f:
        json.dump(status, f, indent=2)

    csv_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.csv"
    pd.DataFrame([status]).to_csv(csv_path, index=False)

    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(status, f, indent=2)

    print_status(f"Quarantined legacy output at {output_path.name}; use step_132_lrd_validation instead.", "WARN")
    return status

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    simulate_lrd_population()
