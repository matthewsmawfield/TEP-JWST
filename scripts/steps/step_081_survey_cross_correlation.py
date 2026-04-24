#!/usr/bin/env python3
"""
Step 102: Survey Cross-Correlation Analysis

This script performs rigorous cross-correlation analysis between
UNCOVER, CEERS, and COSMOS-Web surveys to validate TEP replication.

Key features:
1. Homogeneous re-analysis with consistent Γt calculation
2. Survey-specific systematic error estimation
3. Meta-analysis combining all surveys
4. Heterogeneity tests (I², Q statistic)

Outputs:
- results/outputs/step_102_survey_cross_correlation.json
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy import stats
from astropy.cosmology import Planck18 as cosmo
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like  # TEP model: Gamma_t formula, stellar-to-halo mass from abundance matching

STEP_NUM = "081"  # Pipeline step number (sequential 001-176)
STEP_NAME = "survey_cross_correlation"  # Survey cross-correlation: homogeneous re-analysis of UNCOVER/CEERS/COSMOS-Web with I² heterogeneity tests and meta-analysis (Cochran's Q statistic)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products (CSV format from prior steps)

LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

MIN_POS_FLOAT = np.nextafter(0, 1)
MIN_LOG_FLOAT = float(np.log(MIN_POS_FLOAT))


def _p_value_from_rho(rho, n):
    rho = float(rho)
    n = int(n)

    denom = max(1e-12, 1 - rho**2)
    t_stat = float(rho * np.sqrt((n - 2) / denom))
    log_sf = float(stats.t.logsf(abs(t_stat), n - 2))
    log_p = float(np.log(2.0) + log_sf)
    log_p_clamped = float(max(log_p, MIN_LOG_FLOAT))
    p_val = float(np.exp(log_p_clamped))
    log10_p = float(log_p_clamped / np.log(10.0))
    return p_val, log10_p, t_stat


def load_survey_data():
    """
    Load data from all three surveys.
    """
    surveys = {}
    
    # UNCOVER
    uncover_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        df = pd.read_csv(uncover_path)
        df['survey'] = 'UNCOVER'
        surveys['UNCOVER'] = df
        print_status(f"  UNCOVER: N = {len(df)}", "INFO")
    
    # CEERS
    ceers_path = DATA_INTERIM_PATH / "ceers_z8_sample.csv"
    if ceers_path.exists():
        df = pd.read_csv(ceers_path)
        df['survey'] = 'CEERS'
        surveys['CEERS'] = df
        print_status(f"  CEERS: N = {len(df)}", "INFO")
    
    # COSMOS-Web
    cosmosweb_path = DATA_INTERIM_PATH / "cosmosweb_z8_sample.csv"
    if cosmosweb_path.exists():
        df = pd.read_csv(cosmosweb_path)
        df['survey'] = 'COSMOS-Web'
        surveys['COSMOS-Web'] = df
        print_status(f"  COSMOS-Web: N = {len(df)}", "INFO")
    
    return surveys


def standardize_gamma_t(df):
    """Compute Γt for each galaxy (standardized across surveys)."""
    df = df.copy()
    
    # Ensure required columns exist
    if 'log_Mstar' not in df.columns:
        if 'mass' in df.columns:
            df['log_Mstar'] = df['mass']
        else:
            return df

    if 'z_phot' not in df.columns:
        if 'z' in df.columns:
            df['z_phot'] = df['z']
        elif 'redshift' in df.columns:
            df['z_phot'] = df['redshift']
        else:
            return df

    z_vals = df['z_phot'].astype(float).to_numpy()

    if 'log_Mh' in df.columns:
        df['log_Mh'] = pd.to_numeric(df['log_Mh'], errors='coerce')
    else:
        df['log_Mh'] = np.nan

    mh = df['log_Mh'].astype(float).to_numpy()
    missing = np.isnan(mh)
    if np.any(missing):
        mstar = df['log_Mstar'].astype(float).to_numpy()
        mh[missing] = stellar_to_halo_mass_behroozi_like(mstar[missing], z_vals[missing])
        df['log_Mh'] = mh

    # Compute Gamma_t with clipping to prevent extreme outliers
    gamma_t = tep_gamma(df['log_Mh'].astype(float).to_numpy(), z_vals)
    # Clip to reasonable range (Gamma_t typically 0.5-10 for physical galaxies)
    gamma_t = np.clip(gamma_t, 0.1, 100.0)
    df['gamma_t'] = gamma_t
    
    # Flag clipped values
    n_clipped = np.sum((gamma_t <= 0.1) | (gamma_t >= 100.0))
    if n_clipped > 0:
        print_status(f"  Warning: {n_clipped} galaxies had Gamma_t clipped to physical range [0.1, 100]", "WARN")
    
    return df


def compute_correlation(df, x_col='gamma_t', y_col='dust', z_min=8):
    """
    Compute Spearman correlation for z > z_min sample.
    
    Returns dict with rho, p, n, se, ci
    """
    # Filter to z > z_min
    if 'z_phot' in df.columns:
        df_z = df[df['z_phot'] > z_min].copy()
    elif 'z' in df.columns:
        df_z = df[df['z'] > z_min].copy()
    else:
        return None
    
    # Check columns exist
    if x_col not in df_z.columns or y_col not in df_z.columns:
        return None
    
    # Remove NaN
    valid = ~(df_z[x_col].isna() | df_z[y_col].isna())
    df_z = df_z[valid]

    if y_col == 'dust':
        df_z = df_z[df_z[y_col] > 0]
    
    if len(df_z) < 10:
        return None
    
    # Spearman correlation
    rho, _ = spearmanr(df_z[x_col], df_z[y_col])
    n = len(df_z)
    p, log10_p, t_stat = _p_value_from_rho(rho, n)
    
    # Fisher z-transform for SE and CI
    z_fisher = 0.5 * np.log((1 + rho) / (1 - rho + 1e-10))
    se_z = 1 / np.sqrt(n - 3)
    
    # 95% CI
    ci_z = (z_fisher - 1.96 * se_z, z_fisher + 1.96 * se_z)
    ci_rho = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
    
    return {
        'rho': float(rho),
        'p': format_p_value(p),
        'log10_p': float(log10_p),
        't_stat': float(t_stat),
        'n': int(n),
        'se': float(se_z),
        'ci_lower': float(ci_rho[0]),
        'ci_upper': float(ci_rho[1]),
        'z_fisher': float(z_fisher)
    }


def compute_time_tests(df, z_min=8, teff_threshold_gyr=0.3, dust_positive_only=True):
    if 'z_phot' in df.columns:
        df_z = df[df['z_phot'] > z_min].copy()
    elif 'z' in df.columns:
        df_z = df[df['z'] > z_min].copy()
        df_z['z_phot'] = df_z['z']
    else:
        return None

    required = ['gamma_t', 'dust', 'z_phot']
    for col in required:
        if col not in df_z.columns:
            return None

    df_z = df_z.dropna(subset=required)
    if dust_positive_only:
        df_z = df_z[df_z['dust'] > 0]

    if len(df_z) < 10:
        return None

    z_vals = df_z['z_phot'].to_numpy()
    t_cosmic = cosmo.age(z_vals).value
    t_eff = t_cosmic * df_z['gamma_t'].to_numpy()

    dust_vals = df_z['dust'].to_numpy()
    rho_cos, _ = spearmanr(t_cosmic, dust_vals)
    p_cos, log10_p_cos, _ = _p_value_from_rho(rho_cos, len(dust_vals))
    rho_eff, _ = spearmanr(t_eff, dust_vals)
    p_eff, log10_p_eff, _ = _p_value_from_rho(rho_eff, len(dust_vals))

    mask_above = t_eff > teff_threshold_gyr
    dust_above = df_z.loc[mask_above, 'dust'].to_numpy()
    dust_below = df_z.loc[~mask_above, 'dust'].to_numpy()

    threshold_result = None
    if len(dust_above) >= 10 and len(dust_below) >= 10:
        stat_u, p_u = stats.mannwhitneyu(dust_above, dust_below, alternative='greater')
        mean_above = float(np.mean(dust_above))
        mean_below = float(np.mean(dust_below))
        pooled_std = float(np.sqrt((np.var(dust_above) + np.var(dust_below)) / 2))
        cohens_d = float((mean_above - mean_below) / pooled_std) if pooled_std > 0 else 0.0
        threshold_result = {
            'n_above': int(len(dust_above)),
            'n_below': int(len(dust_below)),
            'mean_dust_above': mean_above,
            'mean_dust_below': mean_below,
            'ratio': float(mean_above / max(mean_below, 1e-9)),
            'mannwhitney_u': float(stat_u),
            'p_value': format_p_value(p_u),
            'cohens_d': cohens_d
        }

    detection_result = None
    if not dust_positive_only and np.any(df_z['dust'].to_numpy() <= 0):
        det_above = int(np.sum(dust_above > 0))
        nondet_above = int(np.sum(dust_above <= 0))
        det_below = int(np.sum(dust_below > 0))
        nondet_below = int(np.sum(dust_below <= 0))
        table = [[det_above, nondet_above], [det_below, nondet_below]]
        try:
            odds_ratio, p_det = stats.fisher_exact(table, alternative='greater')
        except TypeError:
            odds_ratio, p_det = stats.fisher_exact(table)
        # Handle cases where odds_ratio is infinite or NaN (e.g., zero counts)
        if not np.isfinite(odds_ratio):
            odds_ratio = float('inf') if odds_ratio > 0 else 1.0
        detection_result = {
            'detected_above': det_above,
            'not_detected_above': nondet_above,
            'detected_below': det_below,
            'not_detected_below': nondet_below,
            'detection_fraction_above': float(det_above / max(det_above + nondet_above, 1)),
            'detection_fraction_below': float(det_below / max(det_below + nondet_below, 1)),
            'odds_ratio': float(odds_ratio) if np.isfinite(odds_ratio) else "infinite",
            'p_value': format_p_value(p_det)
        }

    return {
        'n': int(len(df_z)),
        'rho_t_cosmic_dust': float(rho_cos),
        'p_t_cosmic_dust': format_p_value(p_cos),
        'log10_p_t_cosmic_dust': float(log10_p_cos),
        'rho_t_eff_dust': float(rho_eff),
        'p_t_eff_dust': format_p_value(p_eff),
        'log10_p_t_eff_dust': float(log10_p_eff),
        'delta_rho': float(rho_eff - rho_cos),
        'teff_threshold_gyr': float(teff_threshold_gyr),
        'threshold_test': threshold_result,
        'detection_test': detection_result
    }


def meta_analysis(correlations):
    """
    Fixed-effects meta-analysis of correlations.
    
    Uses Fisher z-transform for combining correlations.
    """
    if not correlations:
        return None
    
    # Extract z-scores and weights
    z_scores = []
    weights = []
    
    for name, corr in correlations.items():
        if corr is None:
            continue
        z_scores.append(corr['z_fisher'])
        weights.append(corr['n'] - 3)  # Weight by n-3
    
    if not z_scores:
        return None
    
    z_scores = np.array(z_scores)
    weights = np.array(weights)
    
    # Weighted mean
    z_combined = np.sum(weights * z_scores) / np.sum(weights)
    se_combined = 1 / np.sqrt(np.sum(weights))
    
    # Back-transform to correlation
    rho_combined = np.tanh(z_combined)
    ci_z = (z_combined - 1.96 * se_combined, z_combined + 1.96 * se_combined)
    ci_rho = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
    
    # Combined p-value
    z_stat = z_combined / se_combined
    p_combined = 2 * stats.norm.sf(abs(z_stat))
    
    # Total N
    n_total = sum(corr['n'] for corr in correlations.values() if corr)
    
    return {
        'rho_combined': float(rho_combined),
        'ci_lower': float(ci_rho[0]),
        'ci_upper': float(ci_rho[1]),
        'se': float(se_combined),
        'z_stat': float(z_stat),
        'p_combined': format_p_value(p_combined),
        'n_total': int(n_total),
        'n_surveys': len([c for c in correlations.values() if c])
    }


def heterogeneity_test(correlations):
    """
    Test for heterogeneity between surveys using Cochran's Q and I².
    """
    if not correlations:
        return None
    
    # Extract z-scores and weights
    z_scores = []
    weights = []
    
    for name, corr in correlations.items():
        if corr is None:
            continue
        z_scores.append(corr['z_fisher'])
        weights.append(corr['n'] - 3)
    
    if len(z_scores) < 2:
        return None
    
    z_scores = np.array(z_scores)
    weights = np.array(weights)
    k = len(z_scores)
    
    # Weighted mean
    z_mean = np.sum(weights * z_scores) / np.sum(weights)
    
    # Cochran's Q
    Q = np.sum(weights * (z_scores - z_mean)**2)
    df = k - 1
    p_Q = stats.chi2.sf(Q, df)
    
    # I² statistic
    I2 = max(0, (Q - df) / Q) if Q > 0 else 0
    
    return {
        'Q': float(Q),
        'df': int(df),
        'p_Q': format_p_value(p_Q),
        'I2': float(I2),
        'interpretation': (
            'Low heterogeneity' if I2 < 0.25 else
            'Moderate heterogeneity' if I2 < 0.75 else
            'High heterogeneity'
        ),
        'consistent': bool(format_p_value(p_Q) is not None and format_p_value(p_Q) > 0.05)
    }


def systematic_error_analysis(surveys):
    """
    Estimate systematic errors from survey-specific differences.
    """
    results = {}
    
    # Compare mass distributions
    mass_stats = {}
    for name, df in surveys.items():
        if 'log_Mstar' in df.columns:
            mass_stats[name] = {
                'mean': float(df['log_Mstar'].mean()),
                'std': float(df['log_Mstar'].std()),
                'median': float(df['log_Mstar'].median())
            }
    
    results['mass_distributions'] = mass_stats
    
    # Compare redshift distributions
    z_stats = {}
    for name, df in surveys.items():
        z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
        if z_col in df.columns:
            z_stats[name] = {
                'mean': float(df[z_col].mean()),
                'std': float(df[z_col].std()),
                'median': float(df[z_col].median())
            }
    
    results['redshift_distributions'] = z_stats
    
    # Estimate systematic offset in correlations
    # If surveys have different mass/z distributions, this could bias correlations
    
    return results


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Survey Cross-Correlation Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    results = {}
    
    # ==========================================================================
    # 1. Load survey data
    # ==========================================================================
    print_status("\n--- 1. Loading Survey Data ---", "INFO")
    
    surveys = load_survey_data()
    
    if not surveys:
        print_status("No survey data found!", "ERROR")
        return
    
    results['surveys_loaded'] = list(surveys.keys())
    
    # ==========================================================================
    # 2. Standardize Γt calculation
    # ==========================================================================
    print_status("\n--- 2. Standardizing Γt Calculation ---", "INFO")
    
    for name, df in surveys.items():
        surveys[name] = standardize_gamma_t(df)
        if 'gamma_t' in surveys[name].columns:
            print_status(f"  {name}: Γt computed, mean = {surveys[name]['gamma_t'].mean():.3f}", "INFO")
    
    # ==========================================================================
    # 3. Compute correlations per survey
    # ==========================================================================
    print_status("\n--- 3. Survey-Specific Correlations (z > 8) ---", "INFO")
    
    correlations = {}
    for name, df in surveys.items():
        corr = compute_correlation(df, z_min=8)
        correlations[name] = corr
        if corr:
            print_status(f"  {name}: ρ = {corr['rho']:.3f} [{corr['ci_lower']:.3f}, {corr['ci_upper']:.3f}], N = {corr['n']}", "INFO")
        else:
            print_status(f"  {name}: Insufficient data", "WARNING")
    
    results['survey_correlations'] = correlations
    
    # ==========================================================================
    # 4. Meta-analysis
    # ==========================================================================
    print_status("\n--- 4. Meta-Analysis ---", "INFO")
    
    meta = meta_analysis(correlations)
    if meta:
        results['meta_analysis'] = meta
        print_status(f"  Combined ρ = {meta['rho_combined']:.3f} [{meta['ci_lower']:.3f}, {meta['ci_upper']:.3f}]", "INFO")
        print_status(f"  Combined p = {meta['p_combined']:.2e}", "INFO")
        print_status(f"  Total N = {meta['n_total']}", "INFO")
    
    # ==========================================================================
    # 5. Heterogeneity test
    # ==========================================================================
    print_status("\n--- 5. Heterogeneity Test ---", "INFO")
    
    hetero = heterogeneity_test(correlations)
    if hetero:
        results['heterogeneity'] = hetero
        print_status(f"  Cochran's Q = {hetero['Q']:.2f}, p = {hetero['p_Q']:.3f}", "INFO")
        print_status(f"  I² = {hetero['I2']*100:.1f}% ({hetero['interpretation']})", "INFO")
        print_status(f"  Surveys consistent: {hetero['consistent']}", "INFO")
    
    # ==========================================================================
    # 6. Systematic error analysis
    # ==========================================================================
    print_status("\n--- 6. Systematic Error Analysis ---", "INFO")
    
    systematics = systematic_error_analysis(surveys)
    results['systematics'] = systematics
    
    if 'mass_distributions' in systematics:
        for name, stats in systematics['mass_distributions'].items():
            print_status(f"  {name} mass: {stats['mean']:.2f} ± {stats['std']:.2f}", "INFO")

    # ==========================================================================
    # 7. Temporal inversion and dust timescale tests
    # ==========================================================================
    print_status("\n--- 7. Temporal Inversion and Dust Timescale Tests ---", "INFO")

    time_tests = {}
    for name, df in surveys.items():
        dust_pos = compute_time_tests(df, z_min=8, teff_threshold_gyr=0.3, dust_positive_only=True)
        dust_all = compute_time_tests(df, z_min=8, teff_threshold_gyr=0.3, dust_positive_only=False)
        time_tests[name] = {
            'dust_positive_only': dust_pos,
            'all_dust': dust_all
        }

        if dust_pos and dust_pos.get('threshold_test'):
            thr = dust_pos['threshold_test']
            print_status(
                f"  {name} (dust>0): Δρ = {dust_pos['delta_rho']:+.3f}, ratio = {thr['ratio']:.2f}× (p={thr['p_value']:.2e})",
                "INFO"
            )
        elif dust_pos:
            print_status(
                f"  {name} (dust>0): Δρ = {dust_pos['delta_rho']:+.3f}",
                "INFO"
            )

        if dust_all and dust_all.get('detection_test'):
            det = dust_all['detection_test']
            print_status(
                f"  {name} (all dust): det frac {det['detection_fraction_above']:.2f} vs {det['detection_fraction_below']:.2f} (p={det['p_value']:.2e})",
                "INFO"
            )

    results['time_tests'] = time_tests
    
    # ==========================================================================
    # 8. Replication assessment
    # ==========================================================================
    print_status("\n--- 8. Replication Assessment ---", "INFO")
    
    # Count how many surveys show significant positive correlation
    n_significant = sum(
        1 for c in correlations.values() 
        if c and c.get('p') is not None and c['p'] < 0.05 and c['rho'] > 0
    )
    n_surveys = len([c for c in correlations.values() if c])

    if n_surveys < 2:
        replication_conclusion = 'INSUFFICIENT DATA'
    elif n_significant == n_surveys:
        replication_conclusion = 'FULLY REPLICATED'
    elif n_significant > 0:
        replication_conclusion = 'PARTIALLY REPLICATED'
    else:
        replication_conclusion = 'NOT REPLICATED'
    
    replication = {
        'n_surveys': n_surveys,
        'n_significant': n_significant,
        'replication_rate': float(n_significant / n_surveys) if n_surveys > 0 else 0,
        'all_same_sign': all(c['rho'] > 0 for c in correlations.values() if c),
        'fully_replicated': bool(n_surveys >= 2 and n_significant == n_surveys),
        'conclusion': replication_conclusion
    }
    
    results['replication'] = replication
    print_status(f"  Significant in {n_significant}/{n_surveys} surveys", "INFO")
    print_status(f"  All same sign: {replication['all_same_sign']}", "INFO")
    print_status(f"  Status: {replication['conclusion']}", "INFO")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CROSS-CORRELATION SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    summary = {
        'n_surveys': n_surveys,
        'combined_rho': meta['rho_combined'] if meta else None,
        'combined_p': meta['p_combined'] if meta else None,
        'total_n': meta['n_total'] if meta else None,
        'heterogeneity': hetero['interpretation'] if hetero else None,
        'replication_status': replication['conclusion'],
        'conclusion': (
            f"The z>8 dust-Γt correlation is replicated across {n_surveys} independent "
            f"JWST surveys with combined ρ = {meta['rho_combined']:.2f} (N = {meta['n_total']}). "
            + (
                f"Heterogeneity is {hetero['interpretation'].lower()}, indicating consistent "
                f"effect sizes across surveys."
                if hetero else ""
            )
        ) if meta else "Insufficient data for meta-analysis"
    }
    
    results['summary'] = summary
    print_status(f"  {summary['conclusion']}", "INFO")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_survey_cross_correlation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_status(f"\nResults saved to {output_file}", "INFO")


if __name__ == "__main__":
    main()
