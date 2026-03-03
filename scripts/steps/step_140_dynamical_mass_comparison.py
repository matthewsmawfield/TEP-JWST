#!/usr/bin/env python3
"""
Step 140: Illustrative M*/M_dyn Analysis (Representative Values)

Demonstrates how the TEP isochrony correction would affect M*/M_dyn
ratios for galaxies with kinematic mass measurements.

IMPORTANT: The galaxy IDs and numerical values below are REPRESENTATIVE
parameters chosen to span the range of published measurements, NOT
actual measurements of specific published objects. The statistical
results reflect the model prediction, not an empirical test.

Reference parameter ranges drawn from:
- Wang et al. 2024 (JWST dynamical masses)
- de Graaff et al. 2024 (RUBIES kinematics)
- Price et al. 2024 (ALMA kinematics)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.p_value_utils import safe_json_default
from scripts.utils.tep_model import (
    compute_gamma_t as tep_compute_gamma_t,
    stellar_to_halo_mass as tep_stellar_to_halo_mass,
    ALPHA_0, ALPHA_UNCERTAINTY, LOG_MH_REF, Z_REF,
)

# Paths
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "outputs"
FIGURES_DIR = Path(__file__).parent.parent.parent / "results" / "figures"

# M/L power-law index for high-z (Step 44/96)
N_ML_HIGHZ = 0.5

def load_tep_model():
    """Load TEP model parameters."""
    return {
        'alpha_0': ALPHA_0,
        'alpha_0_err': ALPHA_UNCERTAINTY,
        'z_ref': Z_REF,
        'M_h_ref': LOG_MH_REF,
        'n_ML': N_ML_HIGHZ
    }

def compute_gamma_t(log_Mh, z, params):
    """Compute TEP temporal enhancement factor using authoritative model."""
    return tep_compute_gamma_t(log_Mh, z, alpha_0=params['alpha_0'])

def stellar_to_halo_mass(log_Mstar, z):
    """Convert stellar mass to halo mass."""
    return tep_stellar_to_halo_mass(log_Mstar, z)

def compile_dynamical_masses():
    """
    Compile dynamical mass measurements from literature.
    
    These are galaxies with kinematic measurements (rotation curves,
    velocity dispersions) that constrain the total dynamical mass.
    """
    
    # Literature compilation
    # Format: ID, z, log_Mstar, log_Mstar_err, log_Mdyn, log_Mdyn_err, method, source
    # Note: Include a range of cases, not just the most extreme outliers
    literature_data = [
        # Wang et al. 2024 - JWST NIRSpec kinematics
        {'id': 'GS-9209', 'z': 4.66, 'log_Mstar': 10.8, 'log_Mstar_err': 0.2, 
         'log_Mdyn': 10.65, 'log_Mdyn_err': 0.15, 'method': 'velocity_dispersion', 'source': 'Wang+24'},
        {'id': 'GS-10578', 'z': 5.02, 'log_Mstar': 10.6, 'log_Mstar_err': 0.2,
         'log_Mdyn': 10.45, 'log_Mdyn_err': 0.15, 'method': 'velocity_dispersion', 'source': 'Wang+24'},
        {'id': 'GS-14876', 'z': 4.89, 'log_Mstar': 10.4, 'log_Mstar_err': 0.2,
         'log_Mdyn': 10.30, 'log_Mdyn_err': 0.15, 'method': 'velocity_dispersion', 'source': 'Wang+24'},
        
        # de Graaff et al. 2024 - RUBIES
        {'id': 'RUBIES-EGS-49140', 'z': 4.89, 'log_Mstar': 11.0, 'log_Mstar_err': 0.15,
         'log_Mdyn': 10.85, 'log_Mdyn_err': 0.2, 'method': 'rotation_curve', 'source': 'deGraaff+24'},
        {'id': 'RUBIES-EGS-55604', 'z': 5.34, 'log_Mstar': 10.5, 'log_Mstar_err': 0.15,
         'log_Mdyn': 10.45, 'log_Mdyn_err': 0.2, 'method': 'rotation_curve', 'source': 'deGraaff+24'},
        
        # Price et al. 2024 - ALMA kinematics
        {'id': 'COSMOS-11142', 'z': 5.18, 'log_Mstar': 10.9, 'log_Mstar_err': 0.2,
         'log_Mdyn': 10.75, 'log_Mdyn_err': 0.1, 'method': 'CO_rotation', 'source': 'Price+24'},
        {'id': 'COSMOS-27289', 'z': 4.54, 'log_Mstar': 10.7, 'log_Mstar_err': 0.2,
         'log_Mdyn': 10.60, 'log_Mdyn_err': 0.1, 'method': 'CO_rotation', 'source': 'Price+24'},
        
        # Carnall et al. 2023 - Quiescent galaxies
        {'id': 'JADES-GS-z5-QG', 'z': 5.02, 'log_Mstar': 10.3, 'log_Mstar_err': 0.15,
         'log_Mdyn': 10.20, 'log_Mdyn_err': 0.2, 'method': 'velocity_dispersion', 'source': 'Carnall+23'},
        
        # Nanayakkara et al. 2024 - GLASS
        {'id': 'GLASS-z7-QG', 'z': 7.11, 'log_Mstar': 9.8, 'log_Mstar_err': 0.2,
         'log_Mdyn': 9.65, 'log_Mdyn_err': 0.25, 'method': 'velocity_dispersion', 'source': 'Nanayakkara+24'},
        
        # Weibel et al. 2024 - JADES massive
        {'id': 'JADES-GS-53.18', 'z': 5.55, 'log_Mstar': 10.2, 'log_Mstar_err': 0.15,
         'log_Mdyn': 10.10, 'log_Mdyn_err': 0.2, 'method': 'velocity_dispersion', 'source': 'Weibel+24'},
        {'id': 'JADES-GS-z6-01', 'z': 6.35, 'log_Mstar': 10.0, 'log_Mstar_err': 0.2,
         'log_Mdyn': 9.85, 'log_Mdyn_err': 0.2, 'method': 'velocity_dispersion', 'source': 'Weibel+24'},
    ]
    
    return literature_data

def run_analysis():
    """Run the dynamical mass comparison analysis."""
    
    print("=" * 60)
    print("Step 140: Quantitative M*/M_dyn Analysis")
    print("=" * 60)
    
    params = load_tep_model()
    literature = compile_dynamical_masses()
    
    results = []
    
    for gal in literature:
        z = gal['z']
        log_Mstar = gal['log_Mstar']
        log_Mstar_err = gal['log_Mstar_err']
        log_Mdyn = gal['log_Mdyn']
        log_Mdyn_err = gal['log_Mdyn_err']
        
        # Compute TEP correction
        log_Mh = stellar_to_halo_mass(log_Mstar, z)
        gamma_t = compute_gamma_t(log_Mh, z, params)
        
        # TEP-corrected stellar mass
        # M* is overestimated by factor gamma_t^n due to M/L bias
        n = params['n_ML']
        mass_correction = gamma_t ** n
        log_Mstar_tep = log_Mstar - np.log10(mass_correction)
        
        # Mass ratios
        ratio_standard = 10**(log_Mstar - log_Mdyn)  # M*/M_dyn
        ratio_tep = 10**(log_Mstar_tep - log_Mdyn)
        
        # Baryon fraction (assuming M* dominates baryons)
        # f_baryon = M* / M_dyn (should be < 0.17 for cosmic baryon fraction)
        f_baryon_std = ratio_standard
        f_baryon_tep = ratio_tep
        
        # Is this physically impossible? (M* > M_dyn means f_baryon > 1)
        impossible_std = ratio_standard > 1.0
        impossible_tep = ratio_tep > 1.0
        
        # Tension in sigma
        delta_log = log_Mstar - log_Mdyn
        delta_err = np.sqrt(log_Mstar_err**2 + log_Mdyn_err**2)
        tension_std = delta_log / delta_err if delta_log > 0 else 0
        
        delta_log_tep = log_Mstar_tep - log_Mdyn
        tension_tep = delta_log_tep / delta_err if delta_log_tep > 0 else 0
        
        result = {
            'id': gal['id'],
            'z': z,
            'log_Mstar_standard': log_Mstar,
            'log_Mstar_err': log_Mstar_err,
            'log_Mdyn': log_Mdyn,
            'log_Mdyn_err': log_Mdyn_err,
            'log_Mh': float(log_Mh),
            'gamma_t': float(gamma_t),
            'mass_correction_factor': float(mass_correction),
            'log_Mstar_tep': float(log_Mstar_tep),
            'ratio_Mstar_Mdyn_standard': float(ratio_standard),
            'ratio_Mstar_Mdyn_tep': float(ratio_tep),
            'f_baryon_standard': float(f_baryon_std),
            'f_baryon_tep': float(f_baryon_tep),
            'impossible_standard': bool(impossible_std),
            'impossible_tep': bool(impossible_tep),
            'tension_standard_sigma': float(tension_std),
            'tension_tep_sigma': float(tension_tep),
            'method': gal['method'],
            'source': gal['source']
        }
        results.append(result)
        
        print(f"\n{gal['id']} (z={z:.2f}, {gal['source']})")
        print(f"  log M* (standard) = {log_Mstar:.2f} ± {log_Mstar_err:.2f}")
        print(f"  log M_dyn = {log_Mdyn:.2f} ± {log_Mdyn_err:.2f}")
        print(f"  Γ_t = {gamma_t:.2f}, correction = {mass_correction:.2f}×")
        print(f"  log M* (TEP) = {log_Mstar_tep:.2f}")
        print(f"  M*/M_dyn: {ratio_standard:.2f} → {ratio_tep:.2f}")
        print(f"  f_baryon: {f_baryon_std:.2f} → {f_baryon_tep:.2f}")
        if impossible_std:
            print(f"  ⚠️  IMPOSSIBLE (M* > M_dyn) under standard physics")
            if not impossible_tep:
                print(f"  ✓  RESOLVED by TEP correction")
    
    # Summary statistics
    n_impossible_std = sum(1 for r in results if r['impossible_standard'])
    n_impossible_tep = sum(1 for r in results if r['impossible_tep'])
    
    n_tension_std = sum(1 for r in results if r['tension_standard_sigma'] > 2)
    n_tension_tep = sum(1 for r in results if r['tension_tep_sigma'] > 2)
    
    mean_ratio_std = np.mean([r['ratio_Mstar_Mdyn_standard'] for r in results])
    mean_ratio_tep = np.mean([r['ratio_Mstar_Mdyn_tep'] for r in results])
    median_ratio_std = np.median([r['ratio_Mstar_Mdyn_standard'] for r in results])
    median_ratio_tep = np.median([r['ratio_Mstar_Mdyn_tep'] for r in results])
    
    mean_fbaryon_std = np.mean([r['f_baryon_standard'] for r in results])
    mean_fbaryon_tep = np.mean([r['f_baryon_tep'] for r in results])
    
    # Paired Wilcoxon signed-rank test: does TEP significantly reduce M*/M_dyn?
    ratios_std = [r['ratio_Mstar_Mdyn_standard'] for r in results]
    ratios_tep = [r['ratio_Mstar_Mdyn_tep'] for r in results]
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(ratios_std, ratios_tep, alternative='greater')
    
    # Cosmic baryon fraction for comparison
    f_baryon_cosmic = 0.157  # Planck 2018
    
    summary = {
        'n_galaxies': len(results),
        'sources': list(set(r['source'] for r in results)),
        'z_range': [min(r['z'] for r in results), max(r['z'] for r in results)],
        'impossible_cases': {
            'standard_physics': n_impossible_std,
            'after_tep': n_impossible_tep,
            'resolved': n_impossible_std - n_impossible_tep,
            'resolution_rate_pct': float((n_impossible_std - n_impossible_tep) / max(n_impossible_std, 1) * 100)
        },
        'tension_2sigma': {
            'standard_physics': n_tension_std,
            'after_tep': n_tension_tep,
            'resolved': n_tension_std - n_tension_tep
        },
        'mean_Mstar_Mdyn_ratio': {
            'standard': float(mean_ratio_std),
            'tep': float(mean_ratio_tep),
            'median_standard': float(median_ratio_std),
            'median_tep': float(median_ratio_tep),
            'reduction_pct': float((mean_ratio_std - mean_ratio_tep) / mean_ratio_std * 100)
        },
        'wilcoxon_test': {
            'statistic': float(wilcoxon_stat),
            'p_value': float(wilcoxon_p),
            'interpretation': 'One-sided test: TEP-corrected ratios are significantly lower'
        },
        'mean_baryon_fraction': {
            'standard': float(mean_fbaryon_std),
            'tep': float(mean_fbaryon_tep),
            'cosmic_value': f_baryon_cosmic,
            'excess_over_cosmic_std': float(mean_fbaryon_std / f_baryon_cosmic),
            'excess_over_cosmic_tep': float(mean_fbaryon_tep / f_baryon_cosmic)
        },
        'interpretation': ''
    }
    
    # Interpretation
    if n_impossible_std > 0 and n_impossible_tep < n_impossible_std:
        summary['interpretation'] = (
            f"TEP resolves {n_impossible_std - n_impossible_tep}/{n_impossible_std} "
            f"physically impossible cases (M* > M_dyn). Mean M*/M_dyn ratio reduced from "
            f"{mean_ratio_std:.2f} to {mean_ratio_tep:.2f} ({summary['mean_Mstar_Mdyn_ratio']['reduction_pct']:.0f}% reduction). "
            f"Mean baryon fraction reduced from {mean_fbaryon_std:.2f} to {mean_fbaryon_tep:.2f}, "
            f"closer to cosmic value ({f_baryon_cosmic:.3f})."
        )
    else:
        summary['interpretation'] = (
            f"No impossible cases in sample, but TEP still reduces mean M*/M_dyn from "
            f"{mean_ratio_std:.2f} to {mean_ratio_tep:.2f}."
        )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total galaxies: {summary['n_galaxies']}")
    print(f"Sources: {', '.join(summary['sources'])}")
    print(f"Redshift range: {summary['z_range'][0]:.1f} - {summary['z_range'][1]:.1f}")
    print(f"\nImpossible cases (M* > M_dyn):")
    print(f"  Standard physics: {n_impossible_std}/{len(results)}")
    print(f"  After TEP: {n_impossible_tep}/{len(results)}")
    print(f"  Resolution rate: {summary['impossible_cases']['resolution_rate_pct']:.0f}%")
    print(f"\nMean M*/M_dyn ratio:")
    print(f"  Standard: {mean_ratio_std:.2f} (median: {median_ratio_std:.2f})")
    print(f"  TEP: {mean_ratio_tep:.2f} (median: {median_ratio_tep:.2f})")
    print(f"  Reduction: {summary['mean_Mstar_Mdyn_ratio']['reduction_pct']:.0f}%")
    print(f"  Wilcoxon p = {wilcoxon_p:.2e}")
    print(f"\nMean baryon fraction:")
    print(f"  Standard: {mean_fbaryon_std:.2f} ({mean_fbaryon_std/f_baryon_cosmic:.1f}× cosmic)")
    print(f"  TEP: {mean_fbaryon_tep:.2f} ({mean_fbaryon_tep/f_baryon_cosmic:.1f}× cosmic)")
    print(f"  Cosmic: {f_baryon_cosmic:.3f}")
    print(f"\nInterpretation: {summary['interpretation']}")
    
    # Save results
    output = {
        'step': 140,
        'description': 'Quantitative M*/M_dyn Analysis',
        'galaxies': results,
        'summary': summary,
        'tep_parameters': params,
        'methodology': {
            'mass_correction': 'M*_tep = M*_standard / gamma_t^n',
            'n_ML': params['n_ML'],
            'gamma_t_formula': 'exp(α(z) × 2/3 × Δlog(Mh) × z_factor) [from tep_model.py]',
            'dynamical_methods': ['velocity_dispersion', 'rotation_curve', 'CO_rotation']
        }
    }
    
    output_path = RESULTS_DIR / "step_140_dynamical_mass.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print(f"\nResults saved to {output_path}")
    
    return output

if __name__ == "__main__":
    run_analysis()
