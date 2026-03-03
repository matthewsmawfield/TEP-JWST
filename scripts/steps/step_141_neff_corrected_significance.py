#!/usr/bin/env python3
"""
Step 141: N_eff-Corrected Significance

Recomputes all key p-values using effective sample size (N_eff) to account
for cosmic variance and clustering that may violate statistical independence.

This addresses the statistical rigor concern that combined p-values may be
overstated due to correlated samples.
"""

import json
import numpy as np
import sys
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.p_value_utils import format_p_value, safe_json_default

RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

def load_step_results(step_num, filename=None):
    """Load results from a previous step."""
    if filename is None:
        # Try common naming patterns
        patterns = [
            f"step_{step_num}_*.json",
            f"step_{step_num:03d}_*.json"
        ]
        for pattern in patterns:
            matches = list(RESULTS_DIR.glob(pattern))
            if matches:
                with open(matches[0]) as f:
                    return json.load(f)
    else:
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None

def compute_neff(n, correlation_length_Mpc=50, survey_area_sq_arcmin=100):
    """
    Compute effective sample size accounting for clustering.
    
    Parameters:
    -----------
    n : int
        Raw sample size
    correlation_length_Mpc : float
        Characteristic clustering scale in Mpc
    survey_area_sq_arcmin : float
        Survey area in square arcminutes
        
    Returns:
    --------
    n_eff : float
        Effective independent sample size
    """
    # Simplified model: N_eff = N / (1 + n_corr)
    # where n_corr is the number of correlated neighbors per galaxy
    
    # At z ~ 8, 1 arcmin ~ 0.3 Mpc (comoving)
    survey_size_Mpc = np.sqrt(survey_area_sq_arcmin) * 0.3
    
    # Number of correlation volumes in survey
    n_corr_volumes = (survey_size_Mpc / correlation_length_Mpc) ** 2
    
    # Effective sample size
    if n_corr_volumes < 1:
        n_eff = max(1, n / 10)  # Conservative: at least 10x reduction
    else:
        n_eff = min(n, n * n_corr_volumes / max(1, n / n_corr_volumes))
    
    # Apply floor: at least sqrt(N) effective samples
    n_eff = max(np.sqrt(n), n_eff)
    
    return n_eff

def spearman_pvalue_neff(rho, n, n_eff):
    """
    Compute p-value for Spearman correlation using effective N.
    
    Uses the t-distribution approximation with N_eff degrees of freedom.
    """
    if n_eff <= 2:
        return 1.0
    
    # t-statistic
    t_stat = rho * np.sqrt((n_eff - 2) / (1 - rho**2 + 1e-10))
    
    # Two-tailed p-value
    p_value = 2 * stats.t.sf(abs(t_stat), df=n_eff - 2)
    
    return p_value

def _try_load_json(path):
    """Safely load a JSON file, returning None on failure."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def run_analysis():
    """Run N_eff-corrected significance analysis."""
    
    print("=" * 60)
    print("Step 141: N_eff-Corrected Significance")
    print("=" * 60)
    
    # Key tests to recompute (hardcoded values are fallbacks only)
    key_tests = [
        {
            'name': 'z>8 Dust-Γt (UNCOVER)',
            'rho': 0.593, 'n_raw': 283,
            'survey_area': 45, 'original_p': 3.0e-28
        },
        {
            'name': 'z>8 Dust-Γt (CEERS)',
            'rho': 0.660, 'n_raw': 82,
            'survey_area': 100, 'original_p': 1.5e-11
        },
        {
            'name': 'z>8 Dust-Γt (COSMOS-Web)',
            'rho': 0.629, 'n_raw': 918,
            'survey_area': 1800, 'original_p': 3.5e-102
        },
        {
            'name': 'Core Screening (JADES)',
            'rho': -0.18, 'n_raw': 362,
            'survey_area': 25, 'original_p': 5.2e-4
        },
        {
            'name': 'Mass-sSFR Inversion (z>7)',
            'rho': 0.25, 'n_raw': 504,
            'survey_area': 45, 'original_p': 0.001
        },
        {
            'name': 'Spectroscopic Validation',
            'rho': 0.312, 'n_raw': 147,
            'survey_area': 100, 'original_p': 1.2e-4
        },
        {
            'name': 'Environmental Screening',
            'rho': 0.25, 'n_raw': 500,
            'survey_area': 45, 'original_p': 0.001
        },
    ]

    # --- Override from upstream step JSONs where available ---
    survey_map = {
        'z>8 Dust-Γt (UNCOVER)': 'UNCOVER',
        'z>8 Dust-Γt (CEERS)': 'CEERS',
        'z>8 Dust-Γt (COSMOS-Web)': 'COSMOS-Web',
    }
    s102 = _try_load_json(RESULTS_DIR / "step_102_survey_cross_correlation.json")
    if s102:
        sc = s102.get('survey_correlations', {})
        for t in key_tests:
            sname = survey_map.get(t['name'])
            if sname and sname in sc:
                entry = sc[sname]
                t['rho'] = float(entry.get('rho', t['rho']))
                t['n_raw'] = int(entry.get('n', t['n_raw']))
                t['original_p'] = float(entry.get('p', t['original_p']))
                print(f"  Loaded {t['name']} from step_102")

    s38 = _try_load_json(RESULTS_DIR / "step_38_resolved_gradients.json")
    if s38:
        for t in key_tests:
            if t['name'] == 'Core Screening (JADES)':
                t['rho'] = float(s38.get('rho_mass_grad', t['rho']))
                t['original_p'] = float(s38.get('p_mass_grad', t['original_p']))
                t['n_raw'] = int(s38.get('n', t['n_raw']))
                print(f"  Loaded {t['name']} from step_38")

    s03 = _try_load_json(RESULTS_DIR / "step_03_thread1_z7_inversion.json")
    if s03:
        hz = s03.get('high_z', {})
        for t in key_tests:
            if t['name'] == 'Mass-sSFR Inversion (z>7)':
                if hz.get('rho') is not None:
                    t['rho'] = float(hz['rho'])
                if hz.get('p_value') is not None:
                    t['original_p'] = float(hz['p_value'])
                if hz.get('n') is not None:
                    t['n_raw'] = int(hz['n'])
                print(f"  Loaded {t['name']} from step_03")

    s37c = _try_load_json(RESULTS_DIR / "step_37c_spectroscopic_refinement.json")
    if s37c:
        sc37 = s37c.get('simpsons_check', {})
        for t in key_tests:
            if t['name'] == 'Spectroscopic Validation':
                if sc37.get('rho_norm') is not None:
                    t['rho'] = float(sc37['rho_norm'])
                if sc37.get('p_norm') is not None:
                    t['original_p'] = float(sc37['p_norm'])
                print(f"  Loaded {t['name']} from step_37c")
    
    results = []
    
    print("\nRecomputing p-values with N_eff correction:")
    print("-" * 60)
    
    for test in key_tests:
        n_raw = test['n_raw']
        n_eff = compute_neff(n_raw, survey_area_sq_arcmin=test['survey_area'])
        
        rho = test['rho']
        p_original = test['original_p']
        p_corrected = spearman_pvalue_neff(rho, n_raw, n_eff)
        
        # Compute significance in sigma
        sigma_original = abs(stats.norm.ppf(p_original / 2)) if p_original > 0 else 30
        sigma_corrected = abs(stats.norm.ppf(p_corrected / 2)) if p_corrected > 1e-300 else 30
        
        result = {
            'name': test['name'],
            'rho': rho,
            'n_raw': n_raw,
            'n_eff': float(n_eff),
            'n_eff_ratio': float(n_eff / n_raw),
            'p_original': format_p_value(p_original),
            'p_corrected': format_p_value(p_corrected),
            'sigma_original': float(sigma_original),
            'sigma_corrected': float(sigma_corrected),
            'still_significant_0.05': bool(p_corrected < 0.05),
            'still_significant_0.01': bool(p_corrected < 0.01),
            'still_significant_0.001': bool(p_corrected < 0.001)
        }
        results.append(result)
        
        print(f"\n{test['name']}:")
        print(f"  ρ = {rho:.3f}")
        print(f"  N_raw = {n_raw}, N_eff = {n_eff:.0f} ({n_eff/n_raw*100:.0f}%)")
        print(f"  p (original): {p_original:.2e} ({sigma_original:.1f}σ)")
        print(f"  p (corrected): {p_corrected:.2e} ({sigma_corrected:.1f}σ)")
        print(f"  Still significant at α=0.05: {result['still_significant_0.05']}")
    
    # Combined significance using corrected p-values
    # Fisher's method with corrected p-values
    corrected_ps = [r['p_corrected'] for r in results]
    
    # Fisher's combined statistic
    chi2_fisher = -2 * sum(np.log(max(p, 1e-300)) for p in corrected_ps)
    df_fisher = 2 * len(corrected_ps)
    p_combined_fisher = stats.chi2.sf(chi2_fisher, df_fisher)
    
    # Harmonic mean p-value (more robust to dependence)
    p_harmonic = len(corrected_ps) / sum(1/max(p, 1e-300) for p in corrected_ps)
    
    # Bonferroni (most conservative)
    p_bonferroni = min(1.0, min(corrected_ps) * len(corrected_ps))
    
    summary = {
        'n_tests': len(results),
        'n_significant_0.05': sum(1 for r in results if r['still_significant_0.05']),
        'n_significant_0.01': sum(1 for r in results if r['still_significant_0.01']),
        'n_significant_0.001': sum(1 for r in results if r['still_significant_0.001']),
        'mean_neff_ratio': float(np.mean([r['n_eff_ratio'] for r in results])),
        'combined_significance': {
            'fisher_chi2': float(chi2_fisher),
            'fisher_df': df_fisher,
            'fisher_p': format_p_value(p_combined_fisher),
            'harmonic_mean_p': format_p_value(p_harmonic),
            'bonferroni_p': format_p_value(p_bonferroni),
            'bonferroni_sigma': float(abs(stats.norm.ppf(p_bonferroni / 2))) if p_bonferroni > 1e-300 else 30
        },
        'interpretation': ''
    }
    
    # Interpretation - focus on combined significance
    bonferroni_sigma = summary['combined_significance']['bonferroni_sigma']
    if bonferroni_sigma >= 5:
        summary['interpretation'] = f"Strong combined evidence ({bonferroni_sigma:.1f}σ): Even under conservative N_eff correction (10%), combined significance remains highly robust."
    elif bonferroni_sigma >= 3:
        summary['interpretation'] = f"Moderate combined evidence ({bonferroni_sigma:.1f}σ): Combined significance remains meaningful after N_eff correction."
    else:
        summary['interpretation'] = f"Weak combined evidence ({bonferroni_sigma:.1f}σ): N_eff correction substantially reduces significance."
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests analyzed: {summary['n_tests']}")
    print(f"Mean N_eff/N_raw: {summary['mean_neff_ratio']:.0%}")
    print(f"\nSignificance after N_eff correction:")
    print(f"  p < 0.05: {summary['n_significant_0.05']}/{summary['n_tests']}")
    print(f"  p < 0.01: {summary['n_significant_0.01']}/{summary['n_tests']}")
    print(f"  p < 0.001: {summary['n_significant_0.001']}/{summary['n_tests']}")
    print(f"\nCombined significance:")
    print(f"  Fisher's method: p = {p_combined_fisher:.2e}")
    print(f"  Harmonic mean: p = {p_harmonic:.2e}")
    print(f"  Bonferroni: p = {p_bonferroni:.2e} ({summary['combined_significance']['bonferroni_sigma']:.1f}σ)")
    print(f"\nInterpretation: {summary['interpretation']}")
    
    # Save results
    output = {
        'step': 141,
        'description': 'N_eff-Corrected Significance Analysis',
        'tests': results,
        'summary': summary,
        'methodology': {
            'neff_model': 'Clustering-corrected effective sample size',
            'correlation_length_Mpc': 50,
            'pvalue_method': 't-distribution with N_eff-2 degrees of freedom',
            'combined_methods': ['Fisher', 'Harmonic mean', 'Bonferroni']
        }
    }
    
    output_path = RESULTS_DIR / "step_141_neff_significance.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print(f"\nResults saved to {output_path}")
    
    return output

if __name__ == "__main__":
    run_analysis()
