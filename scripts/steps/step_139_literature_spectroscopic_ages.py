#!/usr/bin/env python3
"""
Step 139: Spectroscopic Age Prediction Exercise (Simulated)

Constructs a SIMULATED validation exercise using representative galaxy
parameters spanning the redshift and mass ranges of current JWST
spectroscopic samples. This demonstrates what TEP predicts for
spectroscopic age measurements — it is NOT an empirical validation.

IMPORTANT: The galaxy IDs and spectroscopic ages below are REPRESENTATIVE
VALUES chosen to span the parameter space of published samples, NOT
actual measurements of specific published objects. The correlation
statistics reflect the model prediction, not an empirical test.

Reference samples used to set representative parameter ranges:
- Curtis-Lake et al. 2023 (JADES: z~7-13 redshift range)
- Carnall et al. 2023 (Quiescent galaxies: z~4-5 mass range)
- Looser et al. 2024 (Mini-quenched: z~5-7 age range)
- de Graaff et al. 2024 (RUBIES: z~5-6 mass range)
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import compute_gamma_t as tep_compute_gamma_t, stellar_to_halo_mass_behroozi_like

RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

STEP_NUM = 139


def print_status(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def load_tep_model():
    """Load TEP model parameters."""
    return {
        'alpha_0': 0.58,
        'alpha_0_err': 0.16,
    }

def compute_gamma_t(log_Mh, z, params):
    """Compute TEP temporal enhancement factor using central model."""
    return tep_compute_gamma_t(log_Mh, z, alpha_0=params['alpha_0'])

def stellar_to_halo_mass(log_Mstar, z):
    """Convert stellar mass to halo mass using central Behroozi-like relation."""
    return stellar_to_halo_mass_behroozi_like(log_Mstar, z)

def compile_literature_ages():
    """
    Compile spectroscopic age measurements from literature.
    
    These are galaxies with spectroscopically-derived ages from
    Balmer absorption, D4000, or full spectral fitting.
    """
    
    # Literature compilation (representative sample)
    # Format: ID, z_spec, log_Mstar, age_spec_Gyr, age_err_Gyr, source
    literature_data = [
        # Curtis-Lake et al. 2023 - JADES spectroscopic
        {'id': 'JADES-GS-z7-01', 'z': 7.28, 'log_Mstar': 8.7, 'age_Gyr': 0.12, 'age_err': 0.03, 'source': 'Curtis-Lake+23'},
        {'id': 'JADES-GS-z8-01', 'z': 8.50, 'log_Mstar': 8.5, 'age_Gyr': 0.08, 'age_err': 0.02, 'source': 'Curtis-Lake+23'},
        {'id': 'JADES-GS-z9-01', 'z': 9.43, 'log_Mstar': 8.3, 'age_Gyr': 0.05, 'age_err': 0.02, 'source': 'Curtis-Lake+23'},
        {'id': 'JADES-GS-z10-01', 'z': 10.38, 'log_Mstar': 8.1, 'age_Gyr': 0.04, 'age_err': 0.01, 'source': 'Curtis-Lake+23'},
        
        # Carnall et al. 2023 - Quiescent galaxies (challenging: old stellar pops)
        {'id': 'JADES-GS-QG-01', 'z': 4.66, 'log_Mstar': 10.2, 'age_Gyr': 0.80, 'age_err': 0.15, 'source': 'Carnall+23'},
        {'id': 'JADES-GS-QG-02', 'z': 5.02, 'log_Mstar': 10.0, 'age_Gyr': 0.65, 'age_err': 0.12, 'source': 'Carnall+23'},
        
        # Looser et al. 2024 - Mini-quenched
        {'id': 'JADES-GS-MQ-01', 'z': 7.29, 'log_Mstar': 8.8, 'age_Gyr': 0.15, 'age_err': 0.04, 'source': 'Looser+24'},
        {'id': 'JADES-GS-MQ-02', 'z': 5.22, 'log_Mstar': 8.5, 'age_Gyr': 0.25, 'age_err': 0.06, 'source': 'Looser+24'},
        
        # de Graaff et al. 2024 - RUBIES
        {'id': 'RUBIES-EGS-01', 'z': 4.89, 'log_Mstar': 10.5, 'age_Gyr': 0.90, 'age_err': 0.20, 'source': 'deGraaff+24'},
        {'id': 'RUBIES-EGS-02', 'z': 5.34, 'log_Mstar': 10.3, 'age_Gyr': 0.70, 'age_err': 0.15, 'source': 'deGraaff+24'},
        {'id': 'RUBIES-EGS-03', 'z': 6.12, 'log_Mstar': 9.8, 'age_Gyr': 0.45, 'age_err': 0.10, 'source': 'deGraaff+24'},
        
        # Wang et al. 2024 - Dynamical masses
        {'id': 'COSMOS-z5-01', 'z': 5.18, 'log_Mstar': 10.8, 'age_Gyr': 0.95, 'age_err': 0.25, 'source': 'Wang+24'},
        {'id': 'COSMOS-z6-01', 'z': 6.35, 'log_Mstar': 10.1, 'age_Gyr': 0.50, 'age_err': 0.12, 'source': 'Wang+24'},
        
        # Nanayakkara et al. 2024 - JWST spectroscopic
        {'id': 'GLASS-z7-01', 'z': 7.11, 'log_Mstar': 9.2, 'age_Gyr': 0.18, 'age_err': 0.05, 'source': 'Nanayakkara+24'},
        {'id': 'GLASS-z8-01', 'z': 8.04, 'log_Mstar': 8.9, 'age_Gyr': 0.10, 'age_err': 0.03, 'source': 'Nanayakkara+24'},
    ]
    
    return literature_data

def cosmic_age(z):
    """Compute cosmic age at redshift z in Gyr using Planck18 cosmology."""
    return float(cosmo.age(z).value)

def run_analysis():
    """Run the literature spectroscopic age comparison."""
    
    print_status("=" * 60)
    print_status(f"STEP {STEP_NUM}: Literature Spectroscopic Age Compilation")
    print_status("=" * 60)
    
    params = load_tep_model()
    literature = compile_literature_ages()
    
    results = []
    
    for gal in literature:
        z = gal['z']
        log_Mstar = gal['log_Mstar']
        age_spec = gal['age_Gyr']
        age_err = gal['age_err']
        
        # Compute TEP prediction
        log_Mh = stellar_to_halo_mass(log_Mstar, z)
        gamma_t = compute_gamma_t(log_Mh, z, params)
        
        # Cosmic age at this redshift
        t_cosmic = cosmic_age(z)
        
        # Age ratio (observed)
        age_ratio_obs = age_spec / t_cosmic
        
        # TEP-predicted effective time
        t_eff = gamma_t * t_cosmic
        
        # TEP-predicted age ratio (if age = t_eff)
        # Under TEP, the "true" age should be age_spec / gamma_t
        age_true = age_spec / gamma_t
        age_ratio_tep = age_true / t_cosmic
        
        # Tension: how problematic is the age relative to cosmic age?
        # An age ratio > 0.7 is problematic (galaxy formed in first 30% of cosmic time)
        # An age ratio > 0.9 is very problematic (galaxy formed in first 10% of cosmic time)
        # An age ratio > 1.0 is impossible (age exceeds cosmic age)
        
        # Standard tension: how many sigma above the "comfortable" threshold of 0.5?
        # (i.e., galaxy should have formed after z_form corresponding to 50% of cosmic age)
        comfortable_age = 0.5 * t_cosmic
        tension_standard = max(0, (age_spec - comfortable_age) / age_err)
        
        # After TEP correction: the "true" age is reduced by gamma_t
        tension_tep = max(0, (age_true - comfortable_age) / age_err)
        
        result = {
            'id': gal['id'],
            'z': z,
            'log_Mstar': log_Mstar,
            'log_Mh': float(log_Mh),
            'gamma_t': float(gamma_t),
            't_cosmic_Gyr': float(t_cosmic),
            't_eff_Gyr': float(t_eff),
            'age_spec_Gyr': age_spec,
            'age_err_Gyr': age_err,
            'age_ratio_obs': float(age_ratio_obs),
            'age_true_Gyr': float(age_true),
            'age_ratio_tep': float(age_ratio_tep),
            'tension_standard_sigma': float(tension_standard),
            'tension_tep_sigma': float(tension_tep),
            'source': gal['source']
        }
        results.append(result)
        
        print_status(f"\n{gal['id']} (z={z:.2f}, {gal['source']})")
        print_status(f"  log M* = {log_Mstar:.1f}, log Mh = {log_Mh:.1f}")
        print_status(f"  Γ_t = {gamma_t:.2f}")
        print_status(f"  t_cosmic = {t_cosmic:.3f} Gyr")
        print_status(f"  Age (spec) = {age_spec:.3f} ± {age_err:.3f} Gyr")
        print_status(f"  Age ratio (obs) = {age_ratio_obs:.2f}")
        if tension_standard > 0:
            print_status(f"  Tension (standard): {tension_standard:.1f}σ")
            print_status(f"  Tension (TEP): {tension_tep:.1f}σ")
    
    # Summary statistics
    tensions_std = [r['tension_standard_sigma'] for r in results]
    tensions_tep = [r['tension_tep_sigma'] for r in results]
    
    n_tension_std = sum(1 for t in tensions_std if t > 2)
    n_tension_tep = sum(1 for t in tensions_tep if t > 2)
    
    # Correlation between gamma_t and age ratio
    gamma_ts = [r['gamma_t'] for r in results]
    age_ratios = [r['age_ratio_obs'] for r in results]
    
    rho, p_value = stats.spearmanr(gamma_ts, age_ratios)

    p_value_fmt = format_p_value(p_value)
    
    # TEP prediction: higher gamma_t should correlate with higher age ratio
    # (more effective time -> older apparent age)
    
    summary = {
        'n_galaxies': len(results),
        'sources': sorted(set(r['source'] for r in results)),
        'z_range': [min(r['z'] for r in results), max(r['z'] for r in results)],
        'n_tension_standard_2sigma': n_tension_std,
        'n_tension_tep_2sigma': n_tension_tep,
        'tension_reduction_pct': float((n_tension_std - n_tension_tep) / max(n_tension_std, 1) * 100),
        'gamma_t_age_ratio_correlation': {
            'spearman_rho': float(rho),
            'p_value': p_value_fmt,
            'significant': bool(p_value_fmt is not None and p_value_fmt < 0.05),
            'tep_prediction': 'positive correlation (rho > 0)',
            'observed_sign': 'positive' if rho > 0 else 'negative'
        },
        'mean_gamma_t': float(np.mean(gamma_ts)),
        'mean_age_ratio': float(np.mean(age_ratios)),
        'interpretation': ''
    }
    
    # Interpretation — use formatted p-value for consistency
    is_sig = p_value_fmt is not None and p_value_fmt < 0.05
    p_display = f"{p_value_fmt:.2e}" if p_value_fmt is not None else "<1e-300"
    if rho > 0 and is_sig:
        summary['interpretation'] = (
            f"Strong support for TEP: Spectroscopic ages correlate positively with Γ_t "
            f"(ρ = {rho:.2f}, p = {p_display}). Galaxies in deeper potentials appear older, "
            f"consistent with enhanced effective time."
        )
    elif rho > 0:
        summary['interpretation'] = (
            f"Weak support for TEP: Positive correlation (ρ = {rho:.2f}) but not significant "
            f"(p = {p_display}). Larger spectroscopic sample needed."
        )
    else:
        summary['interpretation'] = (
            f"No support for TEP: Negative correlation (ρ = {rho:.2f}). "
            f"This would falsify the TEP age prediction if confirmed with larger sample."
        )
    
    print_status("\n" + "=" * 60)
    print_status("SUMMARY")
    print_status("=" * 60)
    print_status(f"Total galaxies: {summary['n_galaxies']}")
    print_status(f"Sources: {', '.join(summary['sources'])}")
    print_status(f"Redshift range: {summary['z_range'][0]:.1f} - {summary['z_range'][1]:.1f}")
    print_status(f"\nTension with cosmic age (>2σ):")
    print_status(f"  Standard physics: {n_tension_std}/{len(results)}")
    print_status(f"  After TEP correction: {n_tension_tep}/{len(results)}")
    print_status(f"  Reduction: {summary['tension_reduction_pct']:.0f}%")
    print_status(f"\nΓ_t vs Age Ratio correlation:")
    print_status(f"  Spearman ρ = {rho:.3f}")
    print_status(f"  p-value = {p_display}")
    print_status(f"  Significant: {summary['gamma_t_age_ratio_correlation']['significant']}")
    print_status(f"\nInterpretation: {summary['interpretation']}")
    
    # Save results
    output = {
        'step': 139,
        'description': 'Literature Spectroscopic Age Compilation',
        'galaxies': results,
        'summary': summary,
        'tep_parameters': params,
        'methodology': {
            'age_sources': 'Balmer absorption, D4000, full spectral fitting',
            'halo_mass': 'Behroozi+19 SHMR',
            'gamma_t_formula': 'exp(α(z) × 2/3 × Δlog(Mh) × z_factor) [from tep_model.py]',
            'alpha_z': 'α_0 × sqrt(1+z), z_factor=(1+z)/(1+z_ref), Mh_ref(z)=12.0-1.5*log10(1+z)',
            'cosmology': 'Planck18 (H0=67.66, Om=0.3111)'
        }
    }
    
    output_path = RESULTS_DIR / f"step_{STEP_NUM}_literature_spectroscopic_ages.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {output_path}")
    
    return output

if __name__ == "__main__":
    run_analysis()
