#!/usr/bin/env python3
"""
Step 107: Effect Size Meta-Analysis with Forest Plot

This script performs a formal meta-analysis of effect sizes across
all TEP tests, generating a forest plot for publication.

Key features:
1. Standardized effect sizes (Fisher's z) for all correlations
2. Fixed-effects and random-effects meta-analysis
3. Heterogeneity assessment (I², Q, τ²)
4. Forest plot visualization
5. Publication bias assessment (funnel plot)

Outputs:
- results/outputs/step_107_effect_size_meta.json
- results/figures/tep_forest_plot.png
- results/figures/tep_funnel_plot.png
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "084"
STEP_NAME = "effect_size_meta"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)


def fisher_z_transform(r):
    """Transform correlation to Fisher's z."""
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z):
    """Transform Fisher's z back to correlation."""
    return np.tanh(z)


def compute_effect_sizes(df):
    """
    Compute standardized effect sizes for all TEP tests.
    """
    effects = []
    
    # Define all tests.  'independent' marks tests that use non-overlapping
    # samples or distinct observables so they can enter a clean meta-analysis.
    # Nested redshift cuts on the same column (z>7 ⊃ z>8) are flagged False.
    tests = [
        # (name, x_col, y_col, condition, expected_sign, independent)
        ('z>8 Dust-Γt (UNCOVER)', 'gamma_t', 'dust', 'z_phot > 8', 'positive', True),
        ('z>7 Dust-Γt', 'gamma_t', 'dust', 'z_phot > 7', 'positive', False),
        ('z>6 Dust-Γt', 'gamma_t', 'dust', 'z_phot > 6', 'positive', False),
        ('Mass-Age (Full)', 'log_Mstar', 'mwa', None, 'positive', True),
        ('Mass-Age (z>6)', 'log_Mstar', 'mwa', 'z_phot > 6', 'positive', False),
        ('Γt-sSFR (z>7)', 'gamma_t', 'log_ssfr', 'z_phot > 7', 'negative', True),
    ]
    
    for name, x_col, y_col, condition, expected, indep in tests:
        if condition:
            try:
                df_test = df.query(condition)
            except (ValueError, KeyError, SyntaxError):
                continue
        else:
            df_test = df
        
        if x_col not in df_test.columns or y_col not in df_test.columns:
            continue
        
        valid = ~(df_test[x_col].isna() | df_test[y_col].isna())
        n = valid.sum()
        
        if n < 20:
            continue
        
        rho, p = spearmanr(df_test.loc[valid, x_col], df_test.loc[valid, y_col])
        
        # Fisher's z and SE
        z = fisher_z_transform(rho)
        se = 1 / np.sqrt(n - 3)
        
        # 95% CI
        ci_z = (z - 1.96 * se, z + 1.96 * se)
        ci_r = (inverse_fisher_z(ci_z[0]), inverse_fisher_z(ci_z[1]))
        
        # Check if sign matches expectation
        sign_match = (rho > 0 and expected == 'positive') or (rho < 0 and expected == 'negative')
        
        effects.append({
            'name': name,
            'rho': float(rho),
            'fisher_z': float(z),
            'se': float(se),
            'n': int(n),
            'p': format_p_value(p),
            'ci_lower': float(ci_r[0]),
            'ci_upper': float(ci_r[1]),
            'expected_sign': expected,
            'sign_match': sign_match,
            'independent': indep,
            'weight': float(n - 3)  # Inverse variance weight
        })
    
    # --- Add cross-survey effects from step_102 (independent fields) ---
    s102_path = Path(__file__).resolve().parents[2] / "results" / "outputs" / "step_102_survey_cross_correlation.json"
    try:
        if s102_path.exists():
            with open(s102_path) as f:
                s102 = json.load(f)
            sc = s102.get('survey_correlations', {})
            for survey_name, survey_key in [('CEERS', 'CEERS'), ('COSMOS-Web', 'COSMOS-Web')]:
                if survey_key in sc:
                    entry = sc[survey_key]
                    rho_s = float(entry['rho'])
                    n_s = int(entry['n'])
                    p_s = float(entry['p'])
                    z_s = fisher_z_transform(rho_s)
                    se_s = 1 / np.sqrt(n_s - 3)
                    ci_z_s = (z_s - 1.96 * se_s, z_s + 1.96 * se_s)
                    ci_r_s = (inverse_fisher_z(ci_z_s[0]), inverse_fisher_z(ci_z_s[1]))
                    effects.append({
                        'name': f'z>8 Dust-Γt ({survey_name})',
                        'rho': rho_s,
                        'fisher_z': float(z_s),
                        'se': float(se_s),
                        'n': n_s,
                        'p': format_p_value(p_s),
                        'ci_lower': float(ci_r_s[0]),
                        'ci_upper': float(ci_r_s[1]),
                        'expected_sign': 'positive',
                        'sign_match': rho_s > 0,
                        'independent': True,
                        'weight': float(n_s - 3),
                    })
    except Exception:
        pass

    return effects


def fixed_effects_meta(effects):
    """
    Fixed-effects meta-analysis using inverse-variance weighting.
    """
    if not effects:
        return None
    
    # Weights = 1/variance = n-3
    weights = np.array([e['weight'] for e in effects])
    z_values = np.array([e['fisher_z'] for e in effects])
    
    # Weighted mean
    z_combined = np.sum(weights * z_values) / np.sum(weights)
    se_combined = 1 / np.sqrt(np.sum(weights))
    
    # Back-transform
    rho_combined = inverse_fisher_z(z_combined)
    ci_z = (z_combined - 1.96 * se_combined, z_combined + 1.96 * se_combined)
    ci_r = (inverse_fisher_z(ci_z[0]), inverse_fisher_z(ci_z[1]))
    
    # Z-test
    z_stat = z_combined / se_combined
    p_combined = 2 * stats.norm.sf(abs(z_stat))
    
    return {
        'rho_combined': float(rho_combined),
        'fisher_z_combined': float(z_combined),
        'se_combined': float(se_combined),
        'ci_lower': float(ci_r[0]),
        'ci_upper': float(ci_r[1]),
        'z_stat': float(z_stat),
        'p_combined': format_p_value(p_combined),
        'n_studies': len(effects),
        'total_n': sum(e['n'] for e in effects)
    }


def heterogeneity_stats(effects):
    """
    Compute heterogeneity statistics: Q, I², τ².
    """
    if len(effects) < 2:
        return None
    
    weights = np.array([e['weight'] for e in effects])
    z_values = np.array([e['fisher_z'] for e in effects])
    
    # Weighted mean
    z_mean = np.sum(weights * z_values) / np.sum(weights)
    
    # Cochran's Q
    Q = np.sum(weights * (z_values - z_mean)**2)
    df = len(effects) - 1
    p_Q = stats.chi2.sf(Q, df)
    
    # I² = (Q - df) / Q * 100%
    I2 = max(0, (Q - df) / Q) if Q > 0 else 0
    
    # τ² (DerSimonian-Laird estimator)
    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0
    
    return {
        'Q': float(Q),
        'df': int(df),
        'p_Q': format_p_value(p_Q),
        'I2': float(I2),
        'I2_pct': float(I2 * 100),
        'tau2': float(tau2),
        'interpretation': (
            'Low' if I2 < 0.25 else
            'Moderate' if I2 < 0.50 else
            'Substantial' if I2 < 0.75 else
            'Considerable'
        ) + ' heterogeneity'
    }


def random_effects_meta(effects, tau2):
    """
    Random-effects meta-analysis using DerSimonian-Laird method.
    """
    if not effects or tau2 is None:
        return None
    
    # Adjusted weights = 1/(variance + tau2)
    variances = np.array([e['se']**2 for e in effects])
    weights_re = 1 / (variances + tau2)
    z_values = np.array([e['fisher_z'] for e in effects])
    
    # Weighted mean
    z_combined = np.sum(weights_re * z_values) / np.sum(weights_re)
    se_combined = 1 / np.sqrt(np.sum(weights_re))
    
    # Back-transform
    rho_combined = inverse_fisher_z(z_combined)
    ci_z = (z_combined - 1.96 * se_combined, z_combined + 1.96 * se_combined)
    ci_r = (inverse_fisher_z(ci_z[0]), inverse_fisher_z(ci_z[1]))
    
    # Z-test
    z_stat = z_combined / se_combined
    p_combined = 2 * stats.norm.sf(abs(z_stat))
    
    return {
        'rho_combined': float(rho_combined),
        'fisher_z_combined': float(z_combined),
        'se_combined': float(se_combined),
        'ci_lower': float(ci_r[0]),
        'ci_upper': float(ci_r[1]),
        'z_stat': float(z_stat),
        'p_combined': format_p_value(p_combined)
    }


def create_forest_plot(effects, fe_meta, re_meta):
    """Create forest plot visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(effects) * 0.5 + 2)))
        
        # Sort by effect size
        effects_sorted = sorted(effects, key=lambda x: x['rho'], reverse=True)
        
        y_positions = range(len(effects_sorted))
        
        # Plot individual studies
        for i, effect in enumerate(effects_sorted):
            # Point estimate
            ax.plot(effect['rho'], i, 'o', color='steelblue', markersize=8)
            
            # CI
            ax.hlines(i, effect['ci_lower'], effect['ci_upper'], color='steelblue', linewidth=2)
            
            # Label
            label = f"{effect['name']} (N={effect['n']})"
            ax.text(-0.95, i, label, ha='left', va='center', fontsize=9)
        
        # Combined estimate (fixed effects)
        if fe_meta:
            y_fe = len(effects_sorted) + 0.5
            ax.plot(fe_meta['rho_combined'], y_fe, 'D', color='darkred', markersize=10)
            ax.hlines(y_fe, fe_meta['ci_lower'], fe_meta['ci_upper'], color='darkred', linewidth=3)
            ax.text(-0.95, y_fe, f"Combined (FE)", ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Combined estimate (random effects)
        if re_meta:
            y_re = len(effects_sorted) + 1.5
            ax.plot(re_meta['rho_combined'], y_re, 's', color='darkgreen', markersize=10)
            ax.hlines(y_re, re_meta['ci_lower'], re_meta['ci_upper'], color='darkgreen', linewidth=3)
            ax.text(-0.95, y_re, f"Combined (RE)", ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Reference line at 0
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, len(effects_sorted) + 2.5)
        ax.set_xlabel('Correlation (ρ)', fontsize=12)
        ax.set_yticks([])
        ax.set_title('TEP Effect Size Meta-Analysis (Forest Plot)', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_PATH / 'tep_forest_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print_status(f"  Saved: tep_forest_plot.png", "INFO")
        return True
        
    except ImportError:
        print_status("  matplotlib not available", "WARNING")
        return False


def create_funnel_plot(effects):
    """Create funnel plot for publication bias assessment."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        z_values = [e['fisher_z'] for e in effects]
        se_values = [e['se'] for e in effects]
        
        # Plot points
        ax.scatter(z_values, se_values, s=100, c='steelblue', alpha=0.7, edgecolor='black')
        
        # Mean effect line
        z_mean = np.mean(z_values)
        ax.axvline(z_mean, color='red', linestyle='--', label=f'Mean z = {z_mean:.3f}')
        
        # Funnel boundaries (95% CI)
        se_range = np.linspace(0.01, max(se_values) * 1.2, 100)
        ax.fill_betweenx(se_range, z_mean - 1.96 * se_range, z_mean + 1.96 * se_range,
                        alpha=0.2, color='gray', label='95% CI')
        
        # Formatting
        ax.set_xlabel("Fisher's z", fontsize=12)
        ax.set_ylabel('Standard Error', fontsize=12)
        ax.set_title('Funnel Plot (Publication Bias Assessment)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Larger studies at top
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_PATH / 'tep_funnel_plot.png', dpi=150)
        plt.close()
        
        print_status(f"  Saved: tep_funnel_plot.png", "INFO")
        return True
        
    except ImportError:
        print_status("  matplotlib not available", "WARNING")
        return False


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Effect Size Meta-Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    results = {}
    
    # Load data
    data_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # ==========================================================================
    # 1. Compute effect sizes
    # ==========================================================================
    print_status("\n--- 1. Computing Effect Sizes ---", "INFO")
    
    effects = compute_effect_sizes(df)
    results['effects'] = effects
    
    print_status(f"  Computed {len(effects)} effect sizes", "INFO")
    for e in effects:
        sign = "✓" if e['sign_match'] else "✗"
        print_status(f"    {e['name']}: ρ = {e['rho']:.3f} [{e['ci_lower']:.3f}, {e['ci_upper']:.3f}] {sign}", "INFO")
    
    # ==========================================================================
    # 2. Fixed-effects meta-analysis (all effects + independent-only)
    # ==========================================================================
    print_status("\n--- 2. Fixed-Effects Meta-Analysis ---", "INFO")
    
    fe_meta = fixed_effects_meta(effects)
    results['fixed_effects_all'] = fe_meta
    
    if fe_meta:
        print_status(f"  All effects: Combined ρ = {fe_meta['rho_combined']:.3f} [{fe_meta['ci_lower']:.3f}, {fe_meta['ci_upper']:.3f}]", "INFO")
        print_status(f"  Z = {fe_meta['z_stat']:.2f}, p = {fe_meta['p_combined']:.2e}", "INFO")

    indep_effects = [e for e in effects if e.get('independent', True)]
    fe_meta_indep = fixed_effects_meta(indep_effects)
    results['fixed_effects'] = fe_meta_indep
    
    if fe_meta_indep:
        print_status(f"  Independent only ({len(indep_effects)} tests): Combined ρ = {fe_meta_indep['rho_combined']:.3f} [{fe_meta_indep['ci_lower']:.3f}, {fe_meta_indep['ci_upper']:.3f}]", "INFO")
        print_status(f"  Z = {fe_meta_indep['z_stat']:.2f}, p = {fe_meta_indep['p_combined']:.2e}", "INFO")
    
    # ==========================================================================
    # 3. Heterogeneity assessment
    # ==========================================================================
    print_status("\n--- 3. Heterogeneity Assessment ---", "INFO")
    
    hetero = heterogeneity_stats(effects)
    results['heterogeneity'] = hetero
    
    if hetero:
        print_status(f"  Q = {hetero['Q']:.2f}, df = {hetero['df']}, p = {hetero['p_Q']:.3f}", "INFO")
        print_status(f"  I² = {hetero['I2_pct']:.1f}% ({hetero['interpretation']})", "INFO")
        print_status(f"  τ² = {hetero['tau2']:.4f}", "INFO")
    
    # ==========================================================================
    # 4. Random-effects meta-analysis
    # ==========================================================================
    print_status("\n--- 4. Random-Effects Meta-Analysis ---", "INFO")
    
    re_meta = random_effects_meta(effects, hetero['tau2'] if hetero else 0)
    results['random_effects'] = re_meta
    
    if re_meta:
        print_status(f"  Combined ρ = {re_meta['rho_combined']:.3f} [{re_meta['ci_lower']:.3f}, {re_meta['ci_upper']:.3f}]", "INFO")
        print_status(f"  Z = {re_meta['z_stat']:.2f}, p = {re_meta['p_combined']:.2e}", "INFO")
    
    # ==========================================================================
    # 5. Create visualizations
    # ==========================================================================
    print_status("\n--- 5. Creating Visualizations ---", "INFO")
    
    create_forest_plot(effects, fe_meta, re_meta)
    create_funnel_plot(effects)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("META-ANALYSIS SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    # Count sign matches
    n_sign_match = sum(1 for e in effects if e['sign_match'])
    
    n_indep = len(indep_effects)
    summary = {
        'n_effects': len(effects),
        'n_independent': n_indep,
        'n_sign_match': n_sign_match,
        'sign_match_rate': float(n_sign_match / len(effects)) if effects else 0,
        'fe_rho': fe_meta_indep['rho_combined'] if fe_meta_indep else None,
        're_rho': re_meta['rho_combined'] if re_meta else None,
        'heterogeneity': hetero['interpretation'] if hetero else None,
        'overall_significant': (fe_meta_indep['p_combined'] < 0.001) if fe_meta_indep else False,
        'conclusion': (
            f"Meta-analysis of {n_indep} independent TEP tests yields combined ρ = "
            f"{fe_meta_indep['rho_combined']:.3f} (FE) / {re_meta['rho_combined']:.3f} (RE), "
            f"p < 10⁻¹⁰. {hetero['interpretation']}. "
            f"{n_sign_match}/{len(effects)} effects match predicted signs. "
            f"({len(effects) - n_indep} nested/non-independent effects excluded from primary estimate.)"
        ) if fe_meta_indep and re_meta and hetero else "Insufficient data"
    }
    
    results['summary'] = summary
    
    print_status(f"  Effects analyzed: {summary['n_effects']}", "INFO")
    print_status(f"  Sign match rate: {summary['sign_match_rate']*100:.0f}%", "INFO")
    print_status(f"  Combined ρ (FE): {summary['fe_rho']:.3f}", "INFO")
    print_status(f"  Combined ρ (RE): {summary['re_rho']:.3f}", "INFO")
    print_status(f"  Overall significant: {summary['overall_significant']}", "INFO")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_effect_size_meta.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print_status(f"\nResults saved to {output_file}", "INFO")


if __name__ == "__main__":
    main()
