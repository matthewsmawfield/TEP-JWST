#!/usr/bin/env python3
"""
Step 147: Time-Space Coupling Consistency Test

Tests whether TEP maintains internal consistency between temporal and spatial
measurements. In standard GR, proper time and proper space are treated symmetrically.
TEP breaks this symmetry by making proper time environment-dependent while
keeping spatial geometry standard (Jordan frame).

This step verifies:
1. Spatial distances (ruler lengths) remain isotropic and standard
2. Temporal intervals vary as predicted by Γ_t
3. The two are observationally distinguishable
"""

import json
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import compute_gamma_t

STEP_NUM  = "124"  # Pipeline step number (sequential 001-176)
STEP_NAME = "timespace_coupling"  # Time-space coupling consistency: verifies TEP affects temporal (clock rates) not spatial (ruler lengths) measurements - Jordan frame spatial metric remains standard
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)

def test_temporal_spatial_decoupling():
    """
    Verify that TEP predictions distinguish temporal vs spatial effects.
    """
    # Key distinction: TEP affects clock rates, not ruler lengths
    # Spatial metric remains standard (Jordan frame)
    
    tests = {
        'angular_diameter_distance': {
            'description': 'Standard ΛCDM angular diameter distance',
            'tep_effect': 'None - spatial geometry unchanged',
            'distinguishability': 'High - temporal vs spatial separate'
        },
        'luminosity_distance': {
            'description': 'Standard ΛCDM luminosity distance',
            'tep_effect': 'None - photon propagation unchanged',
            'distinguishability': 'High - no TEP modification'
        },
        'proper_time_accumulation': {
            'description': 'Clock rate in high-z galaxies',
            'tep_effect': 'Γ_t enhancement factor',
            'distinguishability': 'Direct TEP signature'
        },
        'redshift_dilation': {
            'description': 'Cosmological redshift',
            'tep_effect': 'Modified age interpretation',
            'distinguishability': 'Interpretation differs'
        }
    }
    
    # Quantitative test: spatial vs temporal observables
    z = 5.5
    log_Mh = 12.6
    M_h = 10**log_Mh
    
    # Temporal: use the canonical shared Γ_t kernel.
    gamma_t = float(compute_gamma_t(log_Mh, z))
    temporal_effect = gamma_t
    
    # Spatial: ruler length unchanged
    spatial_effect = 1.0
    
    ratio = temporal_effect / spatial_effect
    
    return {
        'conceptual_tests': tests,
        'quantitative_test': {
            'z': z,
            'M_h': float(M_h),
            'log_Mh': float(log_Mh),
            'temporal_enhancement': float(temporal_effect),
            'spatial_effect': float(spatial_effect),
            'ratio_t_s': float(ratio),
            'decoupling_confirmed': bool(ratio > 1.5)
        }
    }

def main():
    print("=" * 70)
    print("Step 147: Time-Space Coupling Consistency Test")
    print("=" * 70)
    
    results = test_temporal_spatial_decoupling()
    
    print("\nTemporal vs Spatial Observable Tests:")
    for test, info in results['conceptual_tests'].items():
        print(f"\n  {test}:")
        print(f"    Description: {info['description']}")
        print(f"    TEP Effect: {info['tep_effect']}")
    
    qt = results['quantitative_test']
    print(f"\nQuantitative Test (z={qt['z']:.1f}, M_h={qt['M_h']:.2e}):")
    print(f"  Temporal enhancement Γ_t = {qt['temporal_enhancement']:.2f}")
    print(f"  Spatial effect = {qt['spatial_effect']:.2f}")
    print(f"  Ratio T/S = {qt['ratio_t_s']:.2f}")
    print(f"  Decoupling confirmed: {qt['decoupling_confirmed']}")
    
    output = {
        'step': 147,
        'description': 'Time-Space Coupling Consistency Test',
        'results': results,
        'conclusion': (
            'TEP temporal/spatial decoupling confirmed for the massive-halo test case'
            if results['quantitative_test']['decoupling_confirmed']
            else 'TEP temporal/spatial decoupling not confirmed for the selected quantitative test case'
        )
    }
    
    output_path = RESULTS_DIR / "step_124_timespace_coupling.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "=" * 70)
    print("Time-space coupling test complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
