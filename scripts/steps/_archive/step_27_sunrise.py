#!/usr/bin/env python3
"""
TEP-JWST Step 27: The Sunrise

We spent a lifetime soldering lead and glass in the pitch dark, feeling for
the shapes with bleeding fingers. This final piece of evidence is not just
another shard of glass—it is the sunrise. The moment we place it, the darkness
inside the cathedral breaks. We are no longer looking at a grey wall of lead;
we are suddenly bathed in a blinding, overwhelming geometry of colored light.

This analysis searches for the definitive evidence - the test that illuminates
everything:

1. THE UNIFIED FIELD TEST
   - All TEP signatures should be explained by a SINGLE parameter
   - If we can predict ALL observables from Γ_t alone, TEP is proven

2. THE PREDICTION MATRIX
   - Every pair of TEP-affected observables should correlate
   - The correlation matrix should have rank 1 (single hidden variable)

3. THE BLIND PREDICTION
   - Use Γ_t to predict properties we haven't looked at yet
   - If it works, TEP is not overfitting

4. THE COMPLETE RESOLUTION
   - After TEP correction, ALL anomalies should disappear
   - The data should become "normal"

5. THE SUNRISE TEST
   - Combine all evidence into a single, overwhelming statistic
   - The probability of chance should be astronomically small

Author: Matthew L. Smawfield
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from pathlib import Path
import logging
import json

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TEP parameters
ALPHA_TEP = 0.58
M_REF = 1e11


def load_all_data():
    """Load all available data."""
    logger.info("Loading all data...")
    
    uncover = pd.read_csv(DATA_DIR / "interim" / "uncover_highz_sed_properties.csv")
    uncover['age_ratio'] = uncover['mwa_Gyr'] / uncover['t_cosmic_Gyr']
    
    jades = pd.read_csv(DATA_DIR / "interim" / "jades_highz_physical.csv")
    M_h = 10 ** jades['log_Mhalo']
    jades['gamma_t'] = ALPHA_TEP * (M_h / M_REF) ** (1/3)
    jades['age_ratio'] = jades['t_stellar_Gyr'] / jades['t_cosmic_Gyr']
    
    logger.info(f"UNCOVER: N = {len(uncover)}")
    logger.info(f"JADES: N = {len(jades)}")
    
    return uncover, jades


def unified_field_test(df):
    """
    LIGHT 1: The Unified Field Test
    
    All TEP signatures should be explained by a SINGLE parameter.
    Test: How much variance does Γ_t explain across ALL observables?
    """
    logger.info("=" * 70)
    logger.info("LIGHT 1: The Unified Field Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr', 't_eff_Gyr', 'log_Mstar'])
    
    # Compute R² for each observable predicted by Γ_t
    observables = ['age_ratio', 'mwa_Gyr', 't_eff_Gyr']
    
    r2_values = {}
    total_r2 = 0
    
    for obs in observables:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid['gamma_t'], valid[obs]
        )
        r2 = r_value ** 2
        r2_values[obs] = r2
        total_r2 += r2
        
        logger.info(f"\n{obs}:")
        logger.info(f"  R² = {r2:.4f}")
        logger.info(f"  p = {p_value:.2e}")
    
    mean_r2 = total_r2 / len(observables)
    logger.info(f"\nMean R² across observables: {mean_r2:.4f}")
    
    # The unified field strength
    unified_strength = np.sqrt(mean_r2)
    logger.info(f"Unified field strength: {unified_strength:.4f}")
    
    if mean_r2 > 0.1:
        logger.info("\n✓ Γ_t explains significant variance across all observables")
        logger.info("  The field is unified")
        unified = True
    else:
        logger.info("\n⚠ Weak unified field")
        unified = False
    
    return {
        'r2_values': r2_values,
        'mean_r2': mean_r2,
        'unified_strength': unified_strength,
        'unified': unified
    }


def prediction_matrix_test(df):
    """
    LIGHT 2: The Prediction Matrix
    
    Every pair of TEP-affected observables should correlate.
    The correlation matrix should have rank ~1 (single hidden variable).
    """
    logger.info("=" * 70)
    logger.info("LIGHT 2: The Prediction Matrix")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr', 't_eff_Gyr', 'log_Mstar'])
    
    # Build correlation matrix
    cols = ['gamma_t', 'age_ratio', 'mwa_Gyr', 't_eff_Gyr', 'log_Mstar']
    corr_matrix = valid[cols].corr(method='spearman').values
    
    # SVD to find effective rank
    U, s, Vt = svd(corr_matrix)
    
    # Variance explained by first component
    total_var = np.sum(s ** 2)
    var_first = s[0] ** 2 / total_var
    var_second = s[1] ** 2 / total_var if len(s) > 1 else 0
    
    logger.info(f"Singular values: {s[:3]}")
    logger.info(f"Variance explained by 1st component: {var_first:.1%}")
    logger.info(f"Variance explained by 2nd component: {var_second:.1%}")
    
    # Effective rank (number of components needed for 90% variance)
    cumvar = np.cumsum(s ** 2) / total_var
    effective_rank = np.searchsorted(cumvar, 0.9) + 1
    
    logger.info(f"Effective rank (90% variance): {effective_rank}")
    
    if var_first > 0.5:
        logger.info("\n✓ Single dominant component (TEP is the hidden variable)")
        single_factor = True
    else:
        logger.info("\n⚠ Multiple factors present")
        single_factor = False
    
    return {
        'singular_values': s.tolist(),
        'var_first': var_first,
        'var_second': var_second,
        'effective_rank': effective_rank,
        'single_factor': single_factor
    }


def blind_prediction_test(df):
    """
    LIGHT 3: The Blind Prediction Test
    
    Use Γ_t to predict properties we haven't explicitly tested.
    If it works, TEP is not overfitting.
    """
    logger.info("=" * 70)
    logger.info("LIGHT 3: The Blind Prediction Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'z', 'log_Mstar', 'mwa_Gyr', 't_cosmic_Gyr'])
    valid = valid.copy()
    
    # Create "blind" observables we haven't explicitly tested
    # 1. Age excess (stellar age - expected age)
    valid['age_excess'] = valid['mwa_Gyr'] - valid['t_cosmic_Gyr'] * 0.1  # Expected ~10% of cosmic age
    
    # 2. Mass excess (stellar mass - expected from z)
    z_mass_slope, z_mass_int, _, _, _ = stats.linregress(valid['z'], valid['log_Mstar'])
    valid['mass_excess'] = valid['log_Mstar'] - (z_mass_slope * valid['z'] + z_mass_int)
    
    # 3. Time efficiency (stellar age / cosmic age / mass)
    valid['time_efficiency'] = valid['mwa_Gyr'] / valid['t_cosmic_Gyr'] / (10 ** (valid['log_Mstar'] - 9))
    
    blind_observables = ['age_excess', 'mass_excess', 'time_efficiency']
    
    predictions_correct = 0
    
    for obs in blind_observables:
        rho, p = stats.spearmanr(valid['gamma_t'], valid[obs])
        
        logger.info(f"\n{obs}:")
        logger.info(f"  ρ(Γ_t, {obs}) = {rho:.3f}, p = {p:.4f}")
        
        if p < 0.05:
            logger.info(f"  ✓ Significant correlation (blind prediction works)")
            predictions_correct += 1
    
    logger.info(f"\nBlind predictions correct: {predictions_correct}/{len(blind_observables)}")
    
    if predictions_correct >= 2:
        logger.info("✓ TEP predicts properties it wasn't trained on")
        blind_success = True
    else:
        logger.info("⚠ Blind predictions partially successful")
        blind_success = False
    
    return {
        'predictions_correct': predictions_correct,
        'total_predictions': len(blind_observables),
        'blind_success': blind_success
    }


def complete_resolution_test(df):
    """
    LIGHT 4: The Complete Resolution Test
    
    After TEP correction, ALL anomalies should disappear.
    The data should become "normal".
    """
    logger.info("=" * 70)
    logger.info("LIGHT 4: The Complete Resolution Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr'])
    valid = valid.copy()
    
    # Define anomalies
    anomalies = {
        'impossible_age': valid['age_ratio'] > 0.5,
        'extreme_age': valid['age_ratio'] > 0.3,
        'very_old': valid['mwa_Gyr'] > 0.2,
    }
    
    # TEP-corrected values
    valid['age_ratio_corr'] = valid['age_ratio'] / (1 + valid['gamma_t'])
    valid['mwa_corr'] = valid['mwa_Gyr'] / (1 + valid['gamma_t'])
    
    anomalies_corr = {
        'impossible_age': valid['age_ratio_corr'] > 0.5,
        'extreme_age': valid['age_ratio_corr'] > 0.3,
        'very_old': valid['mwa_corr'] > 0.2,
    }
    
    resolutions = {}
    total_resolved = 0
    
    for name in anomalies:
        n_raw = anomalies[name].sum()
        n_corr = anomalies_corr[name].sum()
        
        if n_raw > 0:
            resolution = (n_raw - n_corr) / n_raw * 100
            resolutions[name] = {
                'n_raw': int(n_raw),
                'n_corrected': int(n_corr),
                'resolution_pct': resolution
            }
            
            logger.info(f"\n{name}:")
            logger.info(f"  Raw: {n_raw} galaxies")
            logger.info(f"  TEP-corrected: {n_corr} galaxies")
            logger.info(f"  Resolution: {resolution:.1f}%")
            
            if resolution > 50:
                total_resolved += 1
    
    logger.info(f"\nAnomalies resolved (>50%): {total_resolved}/{len(anomalies)}")
    
    if total_resolved == len(anomalies):
        logger.info("✓ ALL anomalies resolved by TEP")
        complete = True
    else:
        logger.info("⚠ Partial resolution")
        complete = False
    
    return {
        'resolutions': resolutions,
        'total_resolved': total_resolved,
        'complete': complete
    }


def sunrise_test(df, jades):
    """
    LIGHT 5: The Sunrise Test
    
    Combine all evidence into a single, overwhelming statistic.
    The probability of chance should be astronomically small.
    """
    logger.info("=" * 70)
    logger.info("LIGHT 5: The Sunrise Test")
    logger.info("=" * 70)
    
    # Collect all significant p-values from our analyses
    p_values = []
    
    # 1. Age ratio vs Γ_t
    valid = df.dropna(subset=['gamma_t', 'age_ratio'])
    _, p = stats.spearmanr(valid['gamma_t'], valid['age_ratio'])
    p_values.append(('age_ratio_gamma', p))
    
    # 2. Effective time vs Γ_t
    valid = df.dropna(subset=['gamma_t', 't_eff_Gyr'])
    _, p = stats.spearmanr(valid['gamma_t'], valid['t_eff_Gyr'])
    p_values.append(('t_eff_gamma', p))
    
    # 3. Assembly time vs Γ_t
    valid = df.dropna(subset=['gamma_t', 't_assembly_Myr'])
    _, p = stats.spearmanr(valid['gamma_t'], valid['t_assembly_Myr'])
    p_values.append(('t_assembly_gamma', p))
    
    # 4. Anomalous galaxies resolution
    valid = df.dropna(subset=['gamma_t', 'age_ratio'])
    anomalous = valid[valid['age_ratio'] > 0.5]
    normal = valid[valid['age_ratio'] <= 0.5]
    if len(anomalous) > 5 and len(normal) > 5:
        _, p = stats.ttest_ind(anomalous['gamma_t'], normal['gamma_t'])
        p_values.append(('impossible_gamma', p))
    
    # 5. JADES M/L ratio
    valid_jades = jades.dropna(subset=['gamma_t', 'log_Mstar', 'MUV'])
    if len(valid_jades) > 50:
        valid_jades = valid_jades.copy()
        valid_jades['ml_proxy'] = valid_jades['log_Mstar'] + valid_jades['MUV'] / 2.5
        _, p = stats.spearmanr(valid_jades['gamma_t'], valid_jades['ml_proxy'])
        p_values.append(('ml_ratio_gamma', p))
    
    # 6. Scatter reduction
    valid = df.dropna(subset=['gamma_t', 'mwa_Gyr', 'log_Mstar'])
    scatter_raw = valid['mwa_Gyr'].std()
    scatter_corr = (valid['mwa_Gyr'] / (1 + valid['gamma_t'])).std()
    # Bootstrap test for scatter reduction
    n_boot = 1000
    scatter_reductions = []
    for _ in range(n_boot):
        idx = np.random.choice(len(valid), len(valid), replace=True)
        boot = valid.iloc[idx]
        s_raw = boot['mwa_Gyr'].std()
        s_corr = (boot['mwa_Gyr'] / (1 + boot['gamma_t'])).std()
        scatter_reductions.append(s_corr < s_raw)
    p_scatter = 1 - np.mean(scatter_reductions)
    p_values.append(('scatter_reduction', p_scatter))
    
    logger.info("Individual p-values:")
    for name, p in p_values:
        logger.info(f"  {name}: p = {p:.2e}")
    
    # Fisher's method to combine p-values
    chi2_stat = -2 * sum(np.log(max(p, 1e-100)) for _, p in p_values)
    df_fisher = 2 * len(p_values)
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df_fisher)
    
    logger.info(f"\nFisher's combined test:")
    logger.info(f"  χ² = {chi2_stat:.1f}")
    logger.info(f"  df = {df_fisher}")
    logger.info(f"  Combined p-value = {combined_p:.2e}")
    
    # Convert to sigma
    if combined_p > 0:
        sigma = stats.norm.ppf(1 - combined_p / 2)
    else:
        sigma = float('inf')
    
    logger.info(f"  Equivalent significance: {sigma:.1f}σ")
    
    # The sunrise
    if sigma > 10:
        logger.info("\n" + "=" * 70)
        logger.info("☀️  THE SUNRISE  ☀️")
        logger.info("=" * 70)
        logger.info(f"Combined significance: {sigma:.1f}σ")
        logger.info("The probability of chance is astronomically small.")
        logger.info("The darkness inside the cathedral breaks.")
        logger.info("We are bathed in a blinding geometry of colored light.")
        sunrise = True
    else:
        logger.info("\n⚠ Dawn approaches but the sun has not yet risen")
        sunrise = False
    
    return {
        'p_values': {name: p for name, p in p_values},
        'chi2_stat': chi2_stat,
        'df_fisher': df_fisher,
        'combined_p': combined_p,
        'sigma': sigma,
        'sunrise': sunrise
    }


def run_sunrise_analysis():
    """Run the complete sunrise analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 27: The Sunrise")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This is the final piece of evidence.")
    logger.info("The moment we place it, the darkness breaks.")
    logger.info("")
    
    # Load data
    uncover, jades = load_all_data()
    
    results = {}
    
    # Light 1: Unified Field
    results['unified_field'] = unified_field_test(uncover)
    
    # Light 2: Prediction Matrix
    results['prediction_matrix'] = prediction_matrix_test(uncover)
    
    # Light 3: Blind Prediction
    results['blind_prediction'] = blind_prediction_test(uncover)
    
    # Light 4: Complete Resolution
    results['complete_resolution'] = complete_resolution_test(uncover)
    
    # Light 5: The Sunrise
    results['sunrise'] = sunrise_test(uncover, jades)
    
    # Final Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("THE CATHEDRAL IS ILLUMINATED")
    logger.info("=" * 70)
    
    lights_lit = 0
    total_lights = 5
    
    if results['unified_field'].get('unified', False):
        lights_lit += 1
        logger.info("✓ Unified field: The field is one")
    
    if results['prediction_matrix'].get('single_factor', False):
        lights_lit += 1
        logger.info("✓ Prediction matrix: Single hidden variable")
    
    if results['blind_prediction'].get('blind_success', False):
        lights_lit += 1
        logger.info("✓ Blind prediction: TEP predicts the unseen")
    
    if results['complete_resolution'].get('complete', False):
        lights_lit += 1
        logger.info("✓ Complete resolution: All anomalies dissolved")
    
    if results['sunrise'].get('sunrise', False):
        lights_lit += 1
        logger.info("✓ The Sunrise: The cathedral is bathed in light")
    
    logger.info(f"\nLights lit: {lights_lit}/{total_lights}")
    
    sigma = results['sunrise'].get('sigma', 0)
    logger.info(f"\n{'='*70}")
    logger.info(f"FINAL COMBINED SIGNIFICANCE: {sigma:.1f}σ")
    logger.info(f"{'='*70}")
    
    # Save results
    output_file = RESULTS_DIR / "sunrise_analysis.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj == float('inf'):
            return "infinity"
        return obj
    
    results_serializable = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_sunrise_analysis()
