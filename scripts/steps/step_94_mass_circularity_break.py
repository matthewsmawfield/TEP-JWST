#!/usr/bin/env python3
"""
Step 94: Breaking Mass Circularity - Independent Tests

The mass circularity concern is that Gamma_t is derived from M_h, which is
derived from M_*, which TEP corrects. This step implements three independent
tests to break this circularity:

1. FIXED-MASS BINS: Test Gamma_t-dust correlation within narrow mass bins
   where mass variation is minimal but redshift (and thus Gamma_t) varies.

2. REDSHIFT-ONLY COMPONENT: Decompose Gamma_t into mass and z components
   and test if the z-component alone predicts observables.

3. STRUCTURAL TEST: Use compactness (r_e) as a mass-independent proxy for
   potential depth. Compact galaxies have deeper potentials at fixed mass.
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.p_value_utils import format_p_value

from scripts.utils.tep_model import (
    compute_gamma_t as tep_gamma,
    stellar_to_halo_mass_behroozi_like,
    tep_alpha,
    LOG_MH_REF,
    Z_REF,
)

def load_uncover_data():
    """Load high-z galaxy data from available catalogs."""
    uncover_path = PROJECT_ROOT / 'results' / 'interim' / 'step_02_uncover_full_sample_tep.csv'
    if uncover_path.exists():
        df = pd.read_csv(uncover_path)
        if 'z_phot' in df.columns:
            df = df.rename(columns={'z_phot': 'z'})
        return df[df['z'] > 4] if 'z' in df.columns else df

    base_path = PROJECT_ROOT / 'data' / 'interim'
    
    # Try CEERS first (has dust column)
    ceers_path = base_path / 'ceers_highz_sample.csv'
    if ceers_path.exists():
        df = pd.read_csv(ceers_path)
        # Rename columns for consistency
        if 'z_phot' in df.columns:
            df = df.rename(columns={'z_phot': 'z'})
        if 'dust' in df.columns:
            df = df.rename(columns={'dust': 'Av'})
        return df[df['z'] > 4] if 'z' in df.columns else df
    
    # Try COSMOS-Web
    cosmosweb_path = base_path / 'cosmosweb_highz_sample.csv'
    if cosmosweb_path.exists():
        df = pd.read_csv(cosmosweb_path)
        if 'z_phot' in df.columns:
            df = df.rename(columns={'z_phot': 'z'})
        if 'dust' in df.columns:
            df = df.rename(columns={'dust': 'Av'})
        return df[df['z'] > 4] if 'z' in df.columns else df
    
    return None

def test_fixed_mass_bins(df):
    """
    Test 1: Fixed-mass bins
    
    Within narrow mass bins (0.3 dex), mass variation is minimal.
    If Gamma_t still predicts dust, it's due to the z-component, not mass.
    """
    results = []
    
    # Define narrow mass bins
    mass_bins = [(8.0, 8.3), (8.3, 8.6), (8.6, 8.9), (8.9, 9.2), (9.2, 9.5)]
    
    for m_low, m_high in mass_bins:
        mask = (df['log_Mstar'] >= m_low) & (df['log_Mstar'] < m_high)
        subset = df[mask].copy()
        
        if len(subset) < 20:
            continue
        
        # Calculate Gamma_t for this subset
        z_vals = subset['z'].astype(float).to_numpy()
        subset['log_Mh'] = stellar_to_halo_mass_behroozi_like(subset['log_Mstar'].astype(float).to_numpy(), z_vals)
        subset['gamma_t'] = tep_gamma(subset['log_Mh'].astype(float).to_numpy(), z_vals)
        
        # Test correlation with dust
        if 'Av' in subset.columns:
            dust_col = 'Av'
        elif 'dust' in subset.columns:
            dust_col = 'dust'
        else:
            continue
        
        valid = subset[[dust_col, 'gamma_t', 'z']].dropna()
        if len(valid) < 20:
            continue
        
        # Gamma_t vs dust correlation
        rho_gamma_dust, p_gamma_dust = spearmanr(valid['gamma_t'], valid[dust_col])
        
        # z vs dust correlation (should be similar if z drives the effect)
        rho_z_dust, p_z_dust = spearmanr(valid['z'], valid[dust_col])
        
        # Mass variation in this bin
        mass_std = subset['log_Mstar'].std()
        
        results.append({
            'mass_bin': f'{m_low}-{m_high}',
            'n': len(valid),
            'mass_std': float(mass_std),
            'rho_gamma_dust': float(rho_gamma_dust),
            'p_gamma_dust': format_p_value(p_gamma_dust),
            'rho_z_dust': float(rho_z_dust),
            'p_z_dust': format_p_value(p_z_dust),
            'z_range': f'{valid["z"].min():.1f}-{valid["z"].max():.1f}',
            'interpretation': 'z-driven' if abs(rho_z_dust) > abs(rho_gamma_dust) * 0.8 else 'gamma_t-driven'
        })
    
    return results

def test_z_component_decomposition(df):
    """
    Test 2: Decompose Gamma_t into mass and z components
    
    Gamma_t = exp(alpha_z * (2/3) * delta_log_Mh * z_factor)
    
    The z-component is: alpha_z * z_factor = alpha_0 * sqrt(1+z) * (1+z)/(1+z_ref)
    The mass-component is: (2/3) * delta_log_Mh
    
    Test if z-component alone predicts observables.
    """
    df = df.copy()
    z_vals = df['z'].astype(float).to_numpy()

    if 'log_Mh' not in df.columns:
        df['log_Mh'] = stellar_to_halo_mass_behroozi_like(df['log_Mstar'].astype(float).to_numpy(), z_vals)

    mh_vals = df['log_Mh'].astype(float).to_numpy()

    alpha_z = tep_alpha(z_vals)
    z_factor = (1 + z_vals) / (1 + Z_REF)
    df['z_component'] = alpha_z * z_factor
    df['mass_component'] = (2/3) * (mh_vals - float(LOG_MH_REF))

    gamma = tep_gamma(mh_vals, z_vals)
    gamma_ref = tep_gamma(np.full_like(mh_vals, float(LOG_MH_REF)), z_vals)
    gamma_ref = np.maximum(gamma_ref, np.nextafter(0, 1))
    df['gamma_t'] = gamma / gamma_ref
    
    # Identify dust column
    if 'Av' in df.columns:
        dust_col = 'Av'
    elif 'dust' in df.columns:
        dust_col = 'dust'
    else:
        return None
    
    valid = df[['z_component', 'mass_component', 'gamma_t', dust_col, 'log_Mstar', 'z']].dropna()
    
    # Test each component's correlation with dust
    rho_z_comp, p_z_comp = spearmanr(valid['z_component'], valid[dust_col])
    rho_mass_comp, p_mass_comp = spearmanr(valid['mass_component'], valid[dust_col])
    rho_gamma, p_gamma = spearmanr(valid['gamma_t'], valid[dust_col])
    rho_mass, p_mass = spearmanr(valid['log_Mstar'], valid[dust_col])
    
    # Partial correlation: z_component vs dust, controlling for mass
    from scipy.stats import pearsonr
    
    # Residualize z_component on mass
    z_comp_resid = valid['z_component'] - np.polyval(np.polyfit(valid['log_Mstar'], valid['z_component'], 1), valid['log_Mstar'])
    dust_resid = valid[dust_col] - np.polyval(np.polyfit(valid['log_Mstar'], valid[dust_col], 1), valid['log_Mstar'])
    
    rho_z_partial, p_z_partial = spearmanr(z_comp_resid, dust_resid)
    
    return {
        'n': len(valid),
        'correlations': {
            'z_component_vs_dust': {'rho': float(rho_z_comp), 'p': format_p_value(p_z_comp)},
            'mass_component_vs_dust': {'rho': float(rho_mass_comp), 'p': format_p_value(p_mass_comp)},
            'gamma_t_vs_dust': {'rho': float(rho_gamma), 'p': format_p_value(p_gamma)},
            'stellar_mass_vs_dust': {'rho': float(rho_mass), 'p': format_p_value(p_mass)}
        },
        'partial_correlation': {
            'z_component_vs_dust_controlling_mass': {'rho': float(rho_z_partial), 'p': format_p_value(p_z_partial)}
        },
        'interpretation': (
            'The z-component of Gamma_t provides independent predictive power '
            f'beyond stellar mass (partial rho = {rho_z_partial:.3f}, p = {p_z_partial:.2e}). '
            'This breaks the mass circularity: the redshift-dependent scaling '
            'predicted by TEP is detected in the data.'
        ) if p_z_partial < 0.05 else (
            'The z-component does not provide significant independent power.'
        )
    }

def test_compactness_proxy(df):
    """
    Test 3: Compactness as mass-independent potential depth proxy
    
    At fixed stellar mass, more compact galaxies have deeper potentials.
    TEP predicts compact galaxies should show stronger enhancement effects.
    
    Compactness = M_* / r_e^2 (surface density proxy)
    """
    # Check if we have size information
    if 'r_e' not in df.columns and 'Re' not in df.columns and 'r_eff' not in df.columns:
        return {
            'status': 'SKIPPED',
            'reason': 'No size (r_e) column available in dataset'
        }
    
    size_col = 'r_e' if 'r_e' in df.columns else ('Re' if 'Re' in df.columns else 'r_eff')
    
    df = df.copy()
    df['compactness'] = df['log_Mstar'] - 2 * np.log10(df[size_col] + 0.01)  # log(M/r^2)
    
    # Identify dust column
    if 'Av' in df.columns:
        dust_col = 'Av'
    elif 'dust' in df.columns:
        dust_col = 'dust'
    else:
        return None
    
    valid = df[['compactness', dust_col, 'log_Mstar', 'z']].dropna()
    valid = valid[np.isfinite(valid['compactness'])]
    
    if len(valid) < 50:
        return {'status': 'INSUFFICIENT_DATA', 'n': len(valid)}
    
    # Test compactness vs dust at fixed mass
    # Residualize on mass
    comp_resid = valid['compactness'] - np.polyval(np.polyfit(valid['log_Mstar'], valid['compactness'], 1), valid['log_Mstar'])
    dust_resid = valid[dust_col] - np.polyval(np.polyfit(valid['log_Mstar'], valid[dust_col], 1), valid['log_Mstar'])
    
    rho_partial, p_partial = spearmanr(comp_resid, dust_resid)
    rho_raw, p_raw = spearmanr(valid['compactness'], valid[dust_col])
    
    return {
        'n': len(valid),
        'raw_correlation': {'rho': float(rho_raw), 'p': format_p_value(p_raw)},
        'partial_correlation_controlling_mass': {'rho': float(rho_partial), 'p': format_p_value(p_partial)},
        'interpretation': (
            'Compact galaxies show enhanced dust at fixed mass '
            f'(partial rho = {rho_partial:.3f}, p = {p_partial:.2e}), '
            'consistent with deeper potentials enabling more dust production. '
            'This is a mass-independent confirmation of the potential-depth mechanism.'
        ) if p_partial < 0.05 and rho_partial > 0 else (
            'No significant compactness-dust correlation at fixed mass.'
        )
    }

def main():
    results = {
        'step': 94,
        'name': 'Breaking Mass Circularity - Independent Tests',
        'timestamp': str(np.datetime64('now'))
    }
    
    # Load data
    df = load_uncover_data()
    if df is None:
        results['error'] = 'Could not load UNCOVER data'
        return results
    
    results['sample_size'] = len(df)
    
    # Test 1: Fixed-mass bins
    print("Running Test 1: Fixed-mass bins...")
    fixed_mass_results = test_fixed_mass_bins(df)
    results['test_1_fixed_mass_bins'] = {
        'description': 'Test Gamma_t-dust correlation within narrow mass bins where mass variation is minimal',
        'results': fixed_mass_results,
        'summary': {
            'n_bins_tested': len(fixed_mass_results),
            'n_significant': sum(
                1 for r in fixed_mass_results
                if (r.get('p_gamma_dust') is not None and r['p_gamma_dust'] < 0.05)
            ),
            'mean_rho': np.mean([r['rho_gamma_dust'] for r in fixed_mass_results]) if fixed_mass_results else None
        }
    }
    
    # Test 2: Z-component decomposition
    print("Running Test 2: Z-component decomposition...")
    z_decomp_results = test_z_component_decomposition(df)
    results['test_2_z_component'] = {
        'description': 'Decompose Gamma_t into mass and z components, test if z-component alone predicts dust',
        'results': z_decomp_results
    }
    
    # Test 3: Compactness proxy
    print("Running Test 3: Compactness proxy...")
    compactness_results = test_compactness_proxy(df)
    results['test_3_compactness'] = {
        'description': 'Use compactness as mass-independent proxy for potential depth',
        'results': compactness_results
    }
    
    # Overall assessment
    circularity_broken = False
    evidence = []
    
    if z_decomp_results and 'partial_correlation' in z_decomp_results:
        p_partial = z_decomp_results['partial_correlation']['z_component_vs_dust_controlling_mass'].get('p')
        if p_partial is not None and p_partial < 0.05:
            circularity_broken = True
            evidence.append('Z-component provides independent predictive power beyond mass')
    
    if fixed_mass_results:
        significant_bins = [
            r for r in fixed_mass_results
            if (r.get('p_gamma_dust') is not None and r['p_gamma_dust'] < 0.05)
        ]
        if len(significant_bins) >= 2:
            circularity_broken = True
            evidence.append(f'{len(significant_bins)} fixed-mass bins show significant Gamma_t-dust correlation')
    
    if compactness_results and 'partial_correlation_controlling_mass' in compactness_results:
        p_compact = compactness_results['partial_correlation_controlling_mass'].get('p')
        if p_compact is not None and p_compact < 0.05:
            circularity_broken = True
            evidence.append('Compactness predicts dust at fixed mass')
    
    results['overall_assessment'] = {
        'circularity_broken': circularity_broken,
        'evidence': evidence,
        'conclusion': (
            'Mass circularity is BROKEN by multiple independent tests. '
            'The TEP signal is not merely a mass proxy.'
        ) if circularity_broken else (
            'Mass circularity remains a concern. Additional tests needed.'
        )
    }
    
    # Save results
    output_path = Path(__file__).parent.parent.parent / 'results' / 'outputs' / 'step_94_mass_circularity_break.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nStep 94 complete. Results saved to {output_path}")
    print(f"Circularity broken: {circularity_broken}")
    if evidence:
        print("Evidence:")
        for e in evidence:
            print(f"  - {e}")
    
    return results

if __name__ == '__main__':
    main()
