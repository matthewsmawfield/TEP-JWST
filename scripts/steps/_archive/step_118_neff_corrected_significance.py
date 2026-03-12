#!/usr/bin/env python3
"""
Step 118: N_eff-Corrected Significance

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

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM  = "118"
STEP_NAME = "neff_corrected_significance"
LOGS_PATH = PROJECT_ROOT / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

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
        Characteristic clustering scale in comoving Mpc
    survey_area_sq_arcmin : float
        Survey area in square arcminutes
        
    Returns:
    --------
    n_eff : float
        Effective independent sample size
    """
    from astropy.cosmology import Planck18 as cosmo
    
    # Simplified model: N_eff = N / (1 + n_corr)
    # where n_corr is the number of correlated neighbors per galaxy
    
    # 1 arcmin in comoving Mpc at z ~ 8
    arcmin_to_cmpc = cosmo.comoving_distance(8.0).value * (np.pi / (180 * 60))
    survey_size_cMpc = np.sqrt(survey_area_sq_arcmin) * arcmin_to_cmpc
    
    # Number of correlation volumes in survey
    n_corr_volumes = (survey_size_cMpc / correlation_length_Mpc) ** 2
    
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

def _append_test(tests, name, rho, n_raw, original_p, survey_area, source_step, source_metric):
    if rho is None or n_raw is None or original_p is None:
        return
    tests.append({
        'name': name,
        'rho': float(rho),
        'n_raw': int(n_raw),
        'survey_area': float(survey_area),
        'original_p': float(original_p),
        'source_step': source_step,
        'source_metric': source_metric,
        'source_status': 'live_reproducible',
    })

def run_analysis():
    """Run N_eff-corrected significance analysis."""
    
    print("=" * 60)
    print("Step 118: N_eff-Corrected Significance")
    print("=" * 60)
    
    key_tests = []
    excluded_tests = []

    s081 = _try_load_json(RESULTS_DIR / "step_081_survey_cross_correlation.json")
    if s081:
        survey_correlations = s081.get('survey_correlations', {})
        for label, survey, area in [
            ('z>8 Dust-Γt (UNCOVER)', 'UNCOVER', 45),
            ('z>8 Dust-Γt (CEERS)', 'CEERS', 100),
            ('z>8 Dust-Γt (COSMOS-Web)', 'COSMOS-Web', 1800),
        ]:
            entry = survey_correlations.get(survey, {})
            _append_test(
                key_tests,
                label,
                entry.get('rho'),
                entry.get('n'),
                entry.get('p'),
                area,
                'step_081_survey_cross_correlation',
                'survey_correlations',
            )
            if entry:
                print(f"  Loaded {label} from step_081")
    else:
        excluded_tests.append({
            'name': 'L1 survey replication basket',
            'source_step': 'step_081_survey_cross_correlation',
            'reason': 'upstream live survey-correlation output unavailable',
        })

    s037 = _try_load_json(RESULTS_DIR / "step_037_resolved_gradients.json")
    if s037:
        _append_test(
            key_tests,
            'Core Screening (JADES)',
            s037.get('rho_mass_grad'),
            s037.get('n'),
            s037.get('p_mass_grad'),
            25,
            'step_037_resolved_gradients',
            'rho_mass_grad',
        )
        print("  Loaded Core Screening (JADES) from step_037")
    else:
        excluded_tests.append({
            'name': 'Core Screening (JADES)',
            'source_step': 'step_037_resolved_gradients',
            'reason': 'upstream live output unavailable',
        })

    s004 = _try_load_json(RESULTS_DIR / "step_004_thread1_z7_inversion.json")
    if s004:
        high_z = s004.get('high_z', {})
        _append_test(
            key_tests,
            'Mass-sSFR Inversion (z>7)',
            high_z.get('rho'),
            high_z.get('n'),
            high_z.get('p_value'),
            45,
            'step_004_thread1_z7_inversion',
            'high_z_rho',
        )
        print("  Loaded Mass-sSFR Inversion (z>7) from step_004")
    else:
        excluded_tests.append({
            'name': 'Mass-sSFR Inversion (z>7)',
            'source_step': 'step_004_thread1_z7_inversion',
            'reason': 'upstream live output unavailable',
        })

    s035 = _try_load_json(RESULTS_DIR / "step_035_spectroscopic_validation.json")
    if s035:
        z8_sample = s035.get('z8_sample', {})
        _append_test(
            key_tests,
            'Spectroscopic Validation (z>8)',
            z8_sample.get('rho_gamma_age'),
            z8_sample.get('n'),
            z8_sample.get('p_gamma_age'),
            25,
            'step_035_spectroscopic_validation',
            'z8_sample.rho_gamma_age',
        )
        print("  Loaded Spectroscopic Validation (z>8) from step_035")
    else:
        excluded_tests.append({
            'name': 'Spectroscopic Validation (z>8)',
            'source_step': 'step_035_spectroscopic_validation',
            'reason': 'upstream live output unavailable',
        })

    s138 = _try_load_json(RESULTS_DIR / "step_138_environmental_screening_steiger.json")
    if s138:
        excluded_tests.append({
            'name': 'Environmental Screening (z>8)',
            'source_step': 'step_138_environmental_screening_steiger',
            'reason': 'primary statistic is a field-vs-dense delta_rho / Fisher-Z contrast, not a single-sample Spearman rho suitable for the N_eff basket',
        })

    s158 = _try_load_json(RESULTS_DIR / "step_158_dja_balmer_decrement.json")
    if s158 and s158.get('status') == 'SUCCESS_REFERENCE_ONLY':
        excluded_tests.append({
            'name': 'Balmer decrement reference branch',
            'source_step': 'step_158_dja_balmer_decrement',
            'reason': 'reference-only fallback in this workspace; excluded from reproducible combined significance',
        })
    
    results = []
    corrected_ps = []
    
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
            'p_original': float(p_original),
            'p_corrected': float(p_corrected),
            'sigma_original': float(sigma_original),
            'sigma_corrected': float(sigma_corrected),
            'still_significant_0.05': bool(p_corrected < 0.05),
            'still_significant_0.01': bool(p_corrected < 0.01),
            'still_significant_0.001': bool(p_corrected < 0.001),
            'survey_area_sq_arcmin': float(test['survey_area']),
            'source_step': test['source_step'],
            'source_metric': test['source_metric'],
            'source_status': test['source_status'],
        }
        results.append(result)
        corrected_ps.append(float(p_corrected))
        
        print(f"\n{test['name']}:")
        print(f"  ρ = {rho:.3f}")
        print(f"  N_raw = {n_raw}, N_eff = {n_eff:.0f} ({n_eff/n_raw*100:.0f}%)")
        print(f"  p (original): {p_original:.2e} ({sigma_original:.1f}σ)")
        print(f"  p (corrected): {p_corrected:.2e} ({sigma_corrected:.1f}σ)")
        print(f"  Still significant at α=0.05: {result['still_significant_0.05']}")
    
    if corrected_ps:
        chi2_fisher = -2 * sum(np.log(max(p, 1e-300)) for p in corrected_ps)
        df_fisher = 2 * len(corrected_ps)
        p_combined_fisher = stats.chi2.sf(chi2_fisher, df_fisher)
        p_harmonic = len(corrected_ps) / sum(1 / max(p, 1e-300) for p in corrected_ps)
        p_bonferroni = min(1.0, min(corrected_ps) * len(corrected_ps))
        bonferroni_sigma = float(abs(stats.norm.ppf(p_bonferroni / 2))) if p_bonferroni > 1e-300 else 30.0
    else:
        chi2_fisher = 0.0
        df_fisher = 0
        p_combined_fisher = 1.0
        p_harmonic = 1.0
        p_bonferroni = 1.0
        bonferroni_sigma = 0.0
    
    summary = {
        'n_tests': len(results),
        'n_excluded_live_or_reference_tests': len(excluded_tests),
        'n_significant_0.05': sum(1 for r in results if r['still_significant_0.05']),
        'n_significant_0.01': sum(1 for r in results if r['still_significant_0.01']),
        'n_significant_0.001': sum(1 for r in results if r['still_significant_0.001']),
        'mean_neff_ratio': float(np.mean([r['n_eff_ratio'] for r in results])) if results else 0.0,
        'combined_significance': {
            'fisher_chi2': float(chi2_fisher),
            'fisher_df': df_fisher,
            'fisher_p': float(p_combined_fisher),
            'harmonic_mean_p': float(p_harmonic),
            'bonferroni_p': float(p_bonferroni),
            'bonferroni_sigma': bonferroni_sigma
        },
        'interpretation': ''
    }
    
    if not results:
        summary['interpretation'] = "No live rho-based reproducible tests were available for N_eff recomputation."
    elif bonferroni_sigma >= 5:
        summary['interpretation'] = f"Strong combined evidence ({bonferroni_sigma:.1f}σ): the live-source-only basket remains highly significant after conservative N_eff correction."
    elif bonferroni_sigma >= 3:
        summary['interpretation'] = f"Moderate combined evidence ({bonferroni_sigma:.1f}σ): the live-source-only basket remains meaningful after conservative N_eff correction."
    else:
        summary['interpretation'] = f"Weak combined evidence ({bonferroni_sigma:.1f}σ): conservative N_eff correction substantially reduces the live-source-only basket."
    
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
    
    output = {
        'step': '118',
        'description': 'N_eff-Corrected Significance Analysis (live-source-only reproducible basket)',
        'selection_policy': {
            'include_only_live_reproducible_rho_tests': True,
            'exclude_reference_only_or_metric_incompatible_branches': True,
            'note': 'This step no longer falls back to hardcoded placeholder statistics when a live upstream output is absent or not directly compatible with the Spearman N_eff basket.'
        },
        'tests': results,
        'excluded_tests': excluded_tests,
        'summary': summary,
        'methodology': {
            'neff_model': 'Clustering-corrected effective sample size',
            'correlation_length_Mpc': 50,
            'pvalue_method': 't-distribution with N_eff-2 degrees of freedom',
            'combined_methods': ['Fisher', 'Harmonic mean', 'Bonferroni']
        }
    }
    
    output_path = RESULTS_DIR / "step_118_neff_corrected_significance.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print(f"\nResults saved to {output_path}")
    
    return output

if __name__ == "__main__":
    run_analysis()
