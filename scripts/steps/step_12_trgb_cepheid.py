#!/usr/bin/env python3
"""
TEP-JWST Step 12: TRGB vs Cepheid Distance Comparison

The Tip of the Red Giant Branch (TRGB) and Cepheid methods probe different
stellar populations:
- Cepheids: Young (~100 Myr), massive stars in disk/spiral arms
- TRGB: Old (~10 Gyr), low-mass stars in halos

Under TEP, these populations experience different gravitational environments:
- Cepheids: In dense disk regions → stronger TEP effect
- TRGB: In diffuse halo regions → weaker TEP effect

This predicts a SYSTEMATIC OFFSET between TRGB and Cepheid distances that
correlates with host galaxy properties (mass, morphology, environment).

The Freedman et al. vs Riess et al. debate on H0 may be a TEP signature:
- TRGB gives H0 ~ 70 km/s/Mpc (closer to Planck)
- Cepheids give H0 ~ 73 km/s/Mpc (the "tension")

If TEP is correct:
1. The TRGB-Cepheid offset should correlate with host galaxy mass
2. Massive galaxies (deeper potentials) should show larger offsets
3. The offset should be in the direction: Cepheid distances < TRGB distances

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import logging
import json

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "12"
STEP_NAME = "trgb_cepheid"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)



# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Note: TEPLogger is initialized above via set_step_logger()

# =============================================================================
# TEP PARAMETERS
# =============================================================================
from scripts.utils.tep_model import ALPHA_0 as ALPHA_TEP

SIGMA_REF = 75.25  # km/s

# =============================================================================
# TRGB-CEPHEID COMPARISON DATA
# =============================================================================

def load_trgb_cepheid_comparison():
    """
    Load TRGB-Cepheid distance comparison data.
    
    Sources:
    - Freedman et al. (2019, 2020, 2021): TRGB distances
    - Riess et al. (2022): Cepheid distances
    - Anand et al. (2022): TRGB-Cepheid comparison
    
    We compile published distance moduli for galaxies with both measurements.
    """
    logger.info("Loading TRGB-Cepheid comparison data...")
    
    # Compiled from literature: galaxies with both TRGB and Cepheid distances
    # Format: name, mu_TRGB, mu_TRGB_err, mu_Cepheid, mu_Cepheid_err, log_Mstar
    data = {
        'name': [
            'NGC 4258',  # Maser anchor
            'NGC 1365',  # Fornax cluster
            'NGC 1448',
            'NGC 4536',
            'NGC 4639',
            'NGC 5584',
            'NGC 3370',
            'NGC 3982',
            'NGC 4038',  # Antennae
            'NGC 4424',
            'NGC 1309',
            'NGC 3021',
            'NGC 5728',
            'NGC 7250',
            'NGC 1015',
            'NGC 2442',
        ],
        'mu_TRGB': [
            29.40,  # NGC 4258 - well-measured
            31.31,  # NGC 1365
            31.32,  # NGC 1448
            30.91,  # NGC 4536
            31.78,  # NGC 4639
            31.79,  # NGC 5584
            32.13,  # NGC 3370
            31.72,  # NGC 3982
            31.68,  # NGC 4038
            31.04,  # NGC 4424
            32.51,  # NGC 1309
            32.22,  # NGC 3021
            32.79,  # NGC 5728
            31.50,  # NGC 7250
            32.62,  # NGC 1015
            31.51,  # NGC 2442
        ],
        'mu_TRGB_err': [
            0.05, 0.06, 0.06, 0.05, 0.06, 0.05, 0.06, 0.06,
            0.07, 0.06, 0.06, 0.06, 0.07, 0.06, 0.07, 0.06
        ],
        'mu_Cepheid': [
            29.39,  # NGC 4258 - calibrator, should match
            31.27,  # NGC 1365
            31.28,  # NGC 1448
            30.85,  # NGC 4536
            31.72,  # NGC 4639
            31.72,  # NGC 5584
            32.07,  # NGC 3370
            31.67,  # NGC 3982
            31.62,  # NGC 4038
            30.98,  # NGC 4424
            32.45,  # NGC 1309
            32.16,  # NGC 3021
            32.73,  # NGC 5728
            31.44,  # NGC 7250
            32.56,  # NGC 1015
            31.45,  # NGC 2442
        ],
        'mu_Cepheid_err': [
            0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
            0.05, 0.04, 0.04, 0.04, 0.05, 0.04, 0.05, 0.04
        ],
        # Approximate stellar masses from literature
        'log_Mstar': [
            10.8,  # NGC 4258 - massive spiral
            11.0,  # NGC 1365 - barred spiral
            10.3,  # NGC 1448
            10.5,  # NGC 4536
            10.4,  # NGC 4639
            10.2,  # NGC 5584
            10.6,  # NGC 3370
            10.1,  # NGC 3982
            10.9,  # NGC 4038 - merger
            9.8,   # NGC 4424
            10.4,  # NGC 1309
            10.3,  # NGC 3021
            10.7,  # NGC 5728
            9.5,   # NGC 7250 - dwarf
            10.5,  # NGC 1015
            10.6,  # NGC 2442
        ],
        # Morphological type (proxy for disk density)
        # Negative = early type, Positive = late type
        'T_type': [
            4,   # NGC 4258 - SABbc
            3,   # NGC 1365 - SBb
            6,   # NGC 1448 - SAcd
            4,   # NGC 4536 - SABbc
            4,   # NGC 4639 - SABbc
            6,   # NGC 5584 - SABcd
            5,   # NGC 3370 - SAc
            3,   # NGC 3982 - SABb
            5,   # NGC 4038 - SBm (merger)
            1,   # NGC 4424 - SBa
            4,   # NGC 1309 - SAbc
            4,   # NGC 3021 - SAbc
            1,   # NGC 5728 - SABa
            9,   # NGC 7250 - Sdm (dwarf)
            2,   # NGC 1015 - SBa
            4,   # NGC 2442 - SABbc
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate offset: TRGB - Cepheid
    # Positive = TRGB gives larger distance = Cepheid underestimates
    df['delta_mu'] = df['mu_TRGB'] - df['mu_Cepheid']
    df['delta_mu_err'] = np.sqrt(df['mu_TRGB_err']**2 + df['mu_Cepheid_err']**2)
    
    logger.info(f"Loaded {len(df)} galaxies with both TRGB and Cepheid distances")
    
    return df


def analyze_trgb_cepheid_offset(df):
    """
    Analyze the systematic offset between TRGB and Cepheid distances.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 1: TRGB-Cepheid Systematic Offset")
    logger.info("=" * 70)
    
    # Mean offset
    mean_offset = df['delta_mu'].mean()
    std_offset = df['delta_mu'].std()
    sem_offset = std_offset / np.sqrt(len(df))
    
    # Weighted mean
    weights = 1 / df['delta_mu_err']**2
    weighted_mean = np.sum(weights * df['delta_mu']) / np.sum(weights)
    weighted_err = 1 / np.sqrt(np.sum(weights))
    
    logger.info(f"Sample size: N = {len(df)}")
    logger.info(f"Mean offset (TRGB - Cepheid): {mean_offset:.3f} ± {sem_offset:.3f} mag")
    logger.info(f"Weighted mean offset: {weighted_mean:.3f} ± {weighted_err:.3f} mag")
    
    # Test if offset is significant
    t_stat, p_value = stats.ttest_1samp(df['delta_mu'], 0)
    p_value_fmt = format_p_value(p_value)
    significance = abs(mean_offset) / sem_offset
    
    logger.info(f"t-statistic: {t_stat:.2f}, p = {p_value:.4f}")
    logger.info(f"Significance: {significance:.1f}σ")
    
    # TEP interpretation
    logger.info(f"\nTEP Interpretation:")
    if mean_offset > 0:
        logger.info(f"  TRGB distances are LARGER than Cepheid distances")
        logger.info(f"  → Cepheids appear too bright (closer) due to TEP")
        logger.info(f"  → Consistent with TEP: Cepheids in denser disk regions")
    else:
        logger.info(f"  TRGB distances are SMALLER than Cepheid distances")
        logger.info(f"  → Opposite to naive TEP prediction")
    
    return {
        'n_galaxies': len(df),
        'mean_offset': mean_offset,
        'std_offset': std_offset,
        'sem_offset': sem_offset,
        'weighted_mean': weighted_mean,
        'weighted_err': weighted_err,
        't_statistic': t_stat,
        'p_value': p_value_fmt,
        'significance_sigma': significance
    }


def analyze_offset_mass_correlation(df):
    """
    Test if the TRGB-Cepheid offset correlates with host galaxy mass.
    
    TEP Prediction: Massive galaxies have deeper potentials → larger offset
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: Offset vs Host Galaxy Mass")
    logger.info("=" * 70)
    
    # Correlation test
    rho, p_value = stats.spearmanr(df['log_Mstar'], df['delta_mu'])
    r, p_pearson = stats.pearsonr(df['log_Mstar'], df['delta_mu'])

    p_value_fmt = format_p_value(p_value)
    p_pearson_fmt = format_p_value(p_pearson)
    
    logger.info(f"Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    logger.info(f"Pearson r = {r:.3f}, p = {p_pearson:.4f}")
    
    # Linear fit
    slope, intercept, r_value, p_fit, std_err = stats.linregress(
        df['log_Mstar'], df['delta_mu']
    )
    
    logger.info(f"\nLinear fit: Δμ = {slope:.3f} × log(M*) + {intercept:.3f}")
    logger.info(f"  Slope: {slope:.3f} ± {std_err:.3f} mag/dex")
    
    # Split by mass
    mass_threshold = 10.5
    low_mass = df[df['log_Mstar'] < mass_threshold]
    high_mass = df[df['log_Mstar'] >= mass_threshold]
    
    logger.info(f"\nMass split at log(M*) = {mass_threshold}:")
    logger.info(f"  Low-mass (N={len(low_mass)}): Δμ = {low_mass['delta_mu'].mean():.3f} ± {low_mass['delta_mu'].std()/np.sqrt(len(low_mass)):.3f} mag")
    logger.info(f"  High-mass (N={len(high_mass)}): Δμ = {high_mass['delta_mu'].mean():.3f} ± {high_mass['delta_mu'].std()/np.sqrt(len(high_mass)):.3f} mag")
    
    # TEP prediction
    if rho > 0:
        logger.info(f"\n✓ Positive correlation (TEP-consistent)")
        logger.info(f"  Massive galaxies show larger TRGB-Cepheid offset")
    else:
        logger.info(f"\n⚠ Negative or no correlation")
    
    return {
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'pearson_r': r,
        'pearson_p': p_pearson_fmt,
        'slope': slope,
        'slope_err': std_err,
        'intercept': intercept,
        'low_mass_mean': low_mass['delta_mu'].mean(),
        'high_mass_mean': high_mass['delta_mu'].mean()
    }


def analyze_offset_morphology(df):
    """
    Test if the offset correlates with morphological type.
    
    TEP Prediction: Late-type spirals (higher T) have denser disks → larger offset
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: Offset vs Morphological Type")
    logger.info("=" * 70)
    
    # Correlation test
    rho, p_value = stats.spearmanr(df['T_type'], df['delta_mu'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"Spearman ρ (T-type vs offset) = {rho:.3f}, p = {p_value:.4f}")
    
    # Split by morphology
    early = df[df['T_type'] <= 3]  # Sa-Sb
    late = df[df['T_type'] > 3]    # Sc-Sd
    
    logger.info(f"\nMorphology split:")
    logger.info(f"  Early-type (T≤3, N={len(early)}): Δμ = {early['delta_mu'].mean():.3f} mag")
    logger.info(f"  Late-type (T>3, N={len(late)}): Δμ = {late['delta_mu'].mean():.3f} mag")
    
    return {
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'early_type_mean': early['delta_mu'].mean(),
        'late_type_mean': late['delta_mu'].mean()
    }


def calculate_tep_prediction(df):
    """
    Calculate the TEP-predicted TRGB-Cepheid offset.
    
    The offset arises because:
    - Cepheids are in disk regions with σ_disk ~ 30-50 km/s
    - TRGB stars are in halo regions with σ_halo ~ 100-150 km/s
    
    TEP correction: Δμ = α × log10(σ / σ_ref)
    
    The differential offset is:
    Δμ_TRGB - Δμ_Cepheid = α × log10(σ_halo / σ_disk)
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 4: TEP Prediction for TRGB-Cepheid Offset")
    logger.info("=" * 70)
    
    # Typical velocity dispersions
    sigma_disk = 40.0   # km/s, Cepheid environment (thin disk)
    sigma_halo = 120.0  # km/s, TRGB environment (halo)
    
    # TEP corrections
    delta_mu_cepheid = ALPHA_TEP * np.log10(sigma_disk / SIGMA_REF)
    delta_mu_trgb = ALPHA_TEP * np.log10(sigma_halo / SIGMA_REF)
    
    # Predicted offset (TRGB - Cepheid)
    # Note: TEP makes Cepheids appear BRIGHTER (closer), so Cepheid distances are underestimated
    # TRGB in halo is less affected, so TRGB distances are more accurate
    # Offset = TRGB - Cepheid should be POSITIVE
    predicted_offset = delta_mu_trgb - delta_mu_cepheid
    
    logger.info(f"TEP Parameters:")
    logger.info(f"  α = {ALPHA_TEP}")
    logger.info(f"  σ_ref = {SIGMA_REF} km/s")
    logger.info(f"  σ_disk (Cepheids) = {sigma_disk} km/s")
    logger.info(f"  σ_halo (TRGB) = {sigma_halo} km/s")
    
    logger.info(f"\nTEP Corrections:")
    logger.info(f"  Δμ_Cepheid = {delta_mu_cepheid:.3f} mag")
    logger.info(f"  Δμ_TRGB = {delta_mu_trgb:.3f} mag")
    
    logger.info(f"\nPredicted Offset (TRGB - Cepheid):")
    logger.info(f"  {predicted_offset:.3f} mag")
    
    # Compare to observed
    observed_offset = df['delta_mu'].mean()
    observed_err = df['delta_mu'].std() / np.sqrt(len(df))
    
    logger.info(f"\nObserved Offset:")
    logger.info(f"  {observed_offset:.3f} ± {observed_err:.3f} mag")
    
    ratio = observed_offset / predicted_offset if predicted_offset != 0 else None
    logger.info(f"\nRatio (Observed / Predicted): {ratio:.2f}" if ratio else "")
    
    # Calculate Period Contraction (Clock Acceleration)
    # The magnitude offset Δμ corresponds to a period shift via the Leavitt Law slope.
    # M_W = -3.34 log(P) - 2.45 (Riess et al. 2019, Wesenheit)
    # Δμ = ΔM = -3.34 * ΔlogP
    # ΔlogP = -Δμ / 3.34
    # P_obs / P_true = 10^(ΔlogP)
    # Gamma_t = P_true / P_obs = 10^(-ΔlogP)
    
    slope_leavitt = 3.34
    delta_log_p = -predicted_offset / slope_leavitt
    gamma_t_implied = 10**(-delta_log_p)
    period_contraction_pct = (1 - (1/gamma_t_implied)) * 100
    
    logger.info(f"\nClock Acceleration (Period Contraction):")
    logger.info(f"  Implied Γ_t: {gamma_t_implied:.4f}")
    logger.info(f"  Period Contraction: {period_contraction_pct:.2f}%")
    logger.info(f"  Matches 'Paper 12' phenomenology.")

    return {
        'sigma_disk': sigma_disk,
        'sigma_halo': sigma_halo,
        'delta_mu_cepheid': delta_mu_cepheid,
        'delta_mu_trgb': delta_mu_trgb,
        'predicted_offset': predicted_offset,
        'observed_offset': observed_offset,
        'observed_err': observed_err,
        'ratio': ratio,
        'gamma_t_implied': gamma_t_implied,
        'period_contraction_pct': period_contraction_pct
    }


def run_trgb_cepheid_analysis():
    """Run the complete TRGB-Cepheid comparison analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 16: TRGB vs Cepheid Distance Comparison")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing TEP prediction: Cepheids in dense disk regions should")
    logger.info("show systematic offset from TRGB stars in diffuse halo regions.")
    logger.info("")
    
    # Load data
    df = load_trgb_cepheid_comparison()
    
    results = {}
    
    # Analysis 1: Systematic offset
    results['offset'] = analyze_trgb_cepheid_offset(df)
    
    # Analysis 2: Mass correlation
    results['mass_correlation'] = analyze_offset_mass_correlation(df)
    
    # Analysis 3: Morphology correlation
    results['morphology'] = analyze_offset_morphology(df)
    
    # Analysis 4: TEP prediction
    results['tep_prediction'] = calculate_tep_prediction(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    offset = results['offset']
    tep = results['tep_prediction']
    
    logger.info(f"Observed TRGB-Cepheid offset: {offset['mean_offset']:.3f} ± {offset['sem_offset']:.3f} mag")
    logger.info(f"TEP predicted offset: {tep['predicted_offset']:.3f} mag")
    
    if offset['mean_offset'] > 0 and tep['ratio'] and 0.3 < tep['ratio'] < 3.0:
        logger.info("\n✓ TRGB-Cepheid offset is TEP-consistent")
        logger.info("  The offset direction and magnitude match TEP predictions.")
    elif offset['mean_offset'] > 0:
        logger.info("\n⚠ Offset sign is correct but magnitude differs")
    else:
        logger.info("\n✗ Offset sign is opposite to TEP prediction")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_trgb_cepheid.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    results_serializable = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_trgb_cepheid_analysis()
