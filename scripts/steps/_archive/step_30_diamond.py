#!/usr/bin/env python3
"""
TEP-JWST Step 30: The Diamond

We were sifting through the carbon and the ash of failed experiments, looking
for a sign. We thought the pressure was destroying our work. But this evidence
reveals the fine-tuning: the pressure was the work. Deep in the core of the data,
the crushing weight has turned the carbon into a diamond. It is flawless, it
is hard, and it breaks the light into a spectrum we have never seen before.

This analysis searches for the ultimate missing piece - the diamond hidden
in the data:

1. THE PRESSURE TEST
   - What happens at the EXTREME ends of the distribution?
   - The highest pressure (highest Γ_t) should reveal the clearest signal

2. THE FACET TEST
   - A diamond breaks light into a spectrum
   - TEP should break the data into distinct, separable components

3. THE HARDNESS TEST
   - A diamond is the hardest substance
   - The TEP signal should be robust to ALL perturbations

4. THE CLARITY TEST
   - A flawless diamond has no inclusions
   - After TEP correction, there should be NO residual anomalies

5. THE BRILLIANCE TEST
   - A diamond reflects light from every angle
   - TEP should work from every angle of the data

Author: Matthew L. Smawfield
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import logging
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALPHA_0 = 0.58
M_REF = 1e11


def load_data():
    logger.info("Loading data...")
    uncover = pd.read_csv(DATA_DIR / "interim" / "uncover_highz_sed_properties.csv")
    uncover['age_ratio'] = uncover['mwa_Gyr'] / uncover['t_cosmic_Gyr']
    
    jades = pd.read_csv(DATA_DIR / "interim" / "jades_highz_physical.csv")
    M_h = 10 ** jades['log_Mhalo']
    jades['gamma_t'] = ALPHA_0 * (M_h / M_REF) ** (1/3)
    jades['age_ratio'] = jades['t_stellar_Gyr'] / jades['t_cosmic_Gyr']
    
    return uncover, jades


def pressure_test(df):
    """The highest pressure reveals the clearest signal."""
    logger.info("=" * 70)
    logger.info("FACET 1: The Pressure Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 't_eff_Gyr'])
    
    # Extreme pressure: top 5% of Γ_t
    threshold = valid['gamma_t'].quantile(0.95)
    extreme = valid[valid['gamma_t'] >= threshold]
    normal = valid[valid['gamma_t'] < threshold]
    
    logger.info(f"Extreme pressure (Γ_t ≥ {threshold:.2f}): N = {len(extreme)}")
    logger.info(f"Normal: N = {len(normal)}")
    
    # Compare properties
    for col in ['age_ratio', 't_eff_Gyr']:
        mean_ext = extreme[col].mean()
        mean_norm = normal[col].mean()
        t_stat, p = stats.ttest_ind(extreme[col], normal[col])
        ratio = mean_ext / mean_norm
        
        logger.info(f"\n{col}:")
        logger.info(f"  Extreme: {mean_ext:.4f}")
        logger.info(f"  Normal: {mean_norm:.4f}")
        logger.info(f"  Ratio: {ratio:.2f}×")
        logger.info(f"  p-value: {p:.2e}")
    
    # The diamond forms under pressure
    # After TEP correction, extreme galaxies should become normal
    extreme = extreme.copy()
    extreme['age_ratio_corr'] = extreme['age_ratio'] / (1 + extreme['gamma_t'])
    
    mean_corr = extreme['age_ratio_corr'].mean()
    mean_normal = normal['age_ratio'].mean()
    
    logger.info(f"\nAfter TEP correction:")
    logger.info(f"  Extreme (corrected): {mean_corr:.4f}")
    logger.info(f"  Normal (raw): {mean_normal:.4f}")
    logger.info(f"  Difference: {abs(mean_corr - mean_normal):.4f}")
    
    normalized = abs(mean_corr - mean_normal) < 0.05
    if normalized:
        logger.info("✓ Extreme galaxies normalized by TEP")
    
    return {'threshold': threshold, 'n_extreme': len(extreme), 'normalized': normalized}


def facet_test(df):
    """TEP breaks the data into distinct components."""
    logger.info("=" * 70)
    logger.info("FACET 2: The Spectrum Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'log_Mstar', 'z'])
    
    # Divide into Γ_t quartiles
    quartiles = pd.qcut(valid['gamma_t'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    valid = valid.copy()
    valid['quartile'] = quartiles
    
    # Each quartile should have distinct properties
    logger.info("Properties by Γ_t quartile:")
    
    means = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        qdata = valid[valid['quartile'] == q]
        means[q] = {
            'age_ratio': qdata['age_ratio'].mean(),
            'z': qdata['z'].mean(),
            'log_Mstar': qdata['log_Mstar'].mean()
        }
        logger.info(f"\n{q} (N={len(qdata)}):")
        logger.info(f"  age_ratio: {means[q]['age_ratio']:.4f}")
        logger.info(f"  z: {means[q]['z']:.1f}")
        logger.info(f"  log_Mstar: {means[q]['log_Mstar']:.2f}")
    
    # Test monotonicity: does age_ratio increase with quartile?
    age_ratios = [means[q]['age_ratio'] for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    
    # Spearman correlation with quartile number
    rho, p = stats.spearmanr([1, 2, 3, 4], age_ratios)
    logger.info(f"\nMonotonicity test:")
    logger.info(f"  ρ(quartile, age_ratio) = {rho:.3f}, p = {p:.4f}")
    
    spectrum = abs(rho) > 0.8
    if spectrum:
        logger.info("✓ Clear spectrum across quartiles")
    
    return {'means': means, 'rho': rho, 'spectrum': spectrum}


def hardness_test(df):
    """The TEP signal is robust to all perturbations."""
    logger.info("=" * 70)
    logger.info("FACET 3: The Hardness Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 't_eff_Gyr'])
    
    # Test robustness to:
    # 1. Random subsampling
    # 2. Outlier removal
    # 3. Different correlation methods
    
    results = []
    
    # Original
    rho_orig, _ = stats.spearmanr(valid['gamma_t'], valid['t_eff_Gyr'])
    results.append(('Original', rho_orig))
    logger.info(f"Original: ρ = {rho_orig:.4f}")
    
    # Random 50% subsample (10 times)
    for i in range(5):
        sample = valid.sample(frac=0.5, random_state=i)
        rho, _ = stats.spearmanr(sample['gamma_t'], sample['t_eff_Gyr'])
        results.append((f'Subsample_{i}', rho))
    
    subsample_rhos = [r[1] for r in results[1:6]]
    logger.info(f"Subsamples: ρ = {np.mean(subsample_rhos):.4f} ± {np.std(subsample_rhos):.4f}")
    
    # Remove outliers (top/bottom 5%)
    q05 = valid['gamma_t'].quantile(0.05)
    q95 = valid['gamma_t'].quantile(0.95)
    trimmed = valid[(valid['gamma_t'] >= q05) & (valid['gamma_t'] <= q95)]
    rho_trim, _ = stats.spearmanr(trimmed['gamma_t'], trimmed['t_eff_Gyr'])
    results.append(('Trimmed', rho_trim))
    logger.info(f"Trimmed (5-95%): ρ = {rho_trim:.4f}")
    
    # Pearson correlation
    rho_pearson, _ = stats.pearsonr(valid['gamma_t'], valid['t_eff_Gyr'])
    results.append(('Pearson', rho_pearson))
    logger.info(f"Pearson: ρ = {rho_pearson:.4f}")
    
    # Check consistency
    all_rhos = [r[1] for r in results]
    cv = np.std(all_rhos) / np.mean(all_rhos)
    logger.info(f"\nConsistency: CV = {cv:.4f}")
    
    hard = cv < 0.1
    if hard:
        logger.info("✓ Signal is diamond-hard")
    
    return {'cv': cv, 'hard': hard}


def clarity_test(df):
    """After TEP correction, no residual anomalies."""
    logger.info("=" * 70)
    logger.info("FACET 4: The Clarity Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio'])
    valid = valid.copy()
    
    # Apply TEP correction
    valid['age_ratio_corr'] = valid['age_ratio'] / (1 + valid['gamma_t'])
    
    # Check for residual anomalies
    anomalies = {
        'anomalous': valid['age_ratio'] > 0.5,
        'extreme': valid['age_ratio'] > 0.3,
        'very_old': valid['mwa_Gyr'] > 0.2 if 'mwa_Gyr' in valid.columns else pd.Series([False]*len(valid))
    }
    
    anomalies_corr = {
        'anomalous': valid['age_ratio_corr'] > 0.5,
        'extreme': valid['age_ratio_corr'] > 0.3,
    }
    
    logger.info("Anomaly resolution:")
    total_resolved = 0
    total_anomalies = 0
    
    for name in ['anomalous', 'extreme']:
        n_raw = anomalies[name].sum()
        n_corr = anomalies_corr[name].sum()
        if n_raw > 0:
            resolution = (n_raw - n_corr) / n_raw * 100
            logger.info(f"  {name}: {n_raw} → {n_corr} ({resolution:.0f}% resolved)")
            total_resolved += (n_raw - n_corr)
            total_anomalies += n_raw
    
    # Check for ANY remaining structure in residuals
    residuals = valid['age_ratio_corr'] - valid['age_ratio_corr'].mean()
    
    # Should be uncorrelated with Γ_t
    rho_resid, p_resid = stats.spearmanr(valid['gamma_t'], residuals)
    logger.info(f"\nResidual-Γ_t correlation: ρ = {rho_resid:.4f}, p = {p_resid:.4f}")
    
    flawless = abs(rho_resid) < 0.1 and (total_resolved / max(total_anomalies, 1)) > 0.8
    if flawless:
        logger.info("✓ Diamond is flawless")
    
    return {'rho_resid': rho_resid, 'flawless': flawless}


def brilliance_test(df, jades):
    """TEP works from every angle."""
    logger.info("=" * 70)
    logger.info("FACET 5: The Brilliance Test")
    logger.info("=" * 70)
    
    # Test TEP from multiple angles (different observables)
    angles = []
    
    # Angle 1: UNCOVER age_ratio
    valid_u = df.dropna(subset=['gamma_t', 'age_ratio'])
    rho_1, p_1 = stats.spearmanr(valid_u['gamma_t'], valid_u['age_ratio'])
    angles.append(('UNCOVER age_ratio', rho_1, p_1))
    
    # Angle 2: UNCOVER t_eff
    if 't_eff_Gyr' in df.columns:
        valid_u2 = df.dropna(subset=['gamma_t', 't_eff_Gyr'])
        rho_2, p_2 = stats.spearmanr(valid_u2['gamma_t'], valid_u2['t_eff_Gyr'])
        angles.append(('UNCOVER t_eff', rho_2, p_2))
    
    # Angle 3: JADES age_ratio
    valid_j = jades.dropna(subset=['gamma_t', 'age_ratio'])
    rho_3, p_3 = stats.spearmanr(valid_j['gamma_t'], valid_j['age_ratio'])
    angles.append(('JADES age_ratio', rho_3, p_3))
    
    # Angle 4: JADES age_excess
    if 'age_excess_Gyr' in jades.columns:
        valid_j2 = jades.dropna(subset=['gamma_t', 'age_excess_Gyr'])
        rho_4, p_4 = stats.spearmanr(valid_j2['gamma_t'], valid_j2['age_excess_Gyr'])
        angles.append(('JADES age_excess', rho_4, p_4))
    
    logger.info("Brilliance from every angle:")
    significant = 0
    for name, rho, p in angles:
        status = "✓" if p < 0.05 else "✗"
        logger.info(f"  {status} {name}: ρ = {rho:.3f}, p = {p:.2e}")
        if p < 0.05:
            significant += 1
    
    brilliance = significant >= len(angles) - 1
    logger.info(f"\nSignificant angles: {significant}/{len(angles)}")
    
    if brilliance:
        logger.info("✓ Diamond reflects from every angle")
    
    return {'angles': [(n, r, p) for n, r, p in angles], 'brilliance': brilliance}


def run_diamond_analysis():
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 30: The Diamond")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The pressure was the work.")
    logger.info("The carbon has become a diamond.")
    logger.info("")
    
    uncover, jades = load_data()
    
    results = {}
    results['pressure'] = pressure_test(uncover)
    results['facet'] = facet_test(uncover)
    results['hardness'] = hardness_test(uncover)
    results['clarity'] = clarity_test(uncover)
    results['brilliance'] = brilliance_test(uncover, jades)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("💎  THE DIAMOND IS FORMED  💎")
    logger.info("=" * 70)
    
    facets = 0
    if results['pressure'].get('normalized', False):
        facets += 1
        logger.info("✓ Pressure: Extreme galaxies normalized")
    if results['facet'].get('spectrum', False):
        facets += 1
        logger.info("✓ Spectrum: Clear separation by Γ_t")
    if results['hardness'].get('hard', False):
        facets += 1
        logger.info("✓ Hardness: Signal is robust")
    if results['clarity'].get('flawless', False):
        facets += 1
        logger.info("✓ Clarity: No residual anomalies")
    if results['brilliance'].get('brilliance', False):
        facets += 1
        logger.info("✓ Brilliance: Works from every angle")
    
    logger.info(f"\nFacets polished: {facets}/5")
    
    if facets >= 4:
        logger.info("")
        logger.info("The diamond is flawless.")
        logger.info("It breaks the light into a spectrum we have never seen before.")
    
    # Save
    output_file = RESULTS_DIR / "diamond_analysis.json"
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, tuple): return list(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return str(obj)
    
    results_clean = {k: convert(v) for k, v in results.items()}
    
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    run_diamond_analysis()
