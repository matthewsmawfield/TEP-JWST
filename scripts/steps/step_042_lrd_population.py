#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.9s.
"""
TEP-JWST Step 42: LRD Population Differential Temporal Topology Analysis

This step applies the differential temporal topology simulation to the full
Kokorev et al. (2024) LRD population (N=260) to validate the mechanism
at the population level.

Inputs:
- Step 41 simulation framework (differential topology calculation)

Outputs:
- results/outputs/step_042_lrd_population.json
- results/outputs/step_042_lrd_population.csv
"""

import sys
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo  # Planck 2018 cosmology (age/distance)
from astropy.table import Table  # Astropy FITS table reader
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import compute_gamma_t as tep_gamma  # TEP model: Gamma_t = exp[K_gal · (Phi − Phi_ref) · sqrt(1+z)], K_gal = kappa · ln10 / (2.5n)

STEP_NUM = "042"  # Pipeline step number (sequential 001-176)
STEP_NAME = "lrd_population"  # LRD population differential temporal topology: applies step_41 simulation to Kokorev+24 sample (N=260)

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
DATA_PATH = PROJECT_ROOT / "data" / "raw"  # Raw catalogue directory (FITS files from literature/Kokorev+24)

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# =============================================================================
# CONSTANTS & PARAMETERS
# =============================================================================

T_SALPETER = 45e6  # Salpeter e-folding time for Eddington-limited BH accretion (years)

# LRD typical parameters from Kokorev et al. (2024)
R_E_MEDIAN = 150  # Median effective radius (pc)
CONCENTRATION = 10  # Baryon-dominated core concentration parameter
LRD_CATALOG_PATH = DATA_PATH / "kokorev_lrd_catalog_v1.1.fits"  # Published LRD sample

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
    # LRD-specific Behroozi+19-like relation with z-evolution
    # NOTE: differs from shared stellar_to_halo_mass (+2.0 fixed offset)
    # by using a z-dependent offset (+1.5 at z=5, +1.7 at z=7)
    if pd.isna(log_mstar):
        return 11.0
    log_mh = log_mstar + 1.5 + 0.1 * (z - 5)
    return float(np.clip(log_mh, 9.0, 14.0))

def simulate_lrd_population():
    """Quarantine the legacy LRD population step in favor of the real-data validation path."""
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: LRD Population Differential Temporal Topology Analysis", "INFO")
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
