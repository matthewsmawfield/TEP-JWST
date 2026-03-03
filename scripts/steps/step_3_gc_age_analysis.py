#!/usr/bin/env python3
"""
Step 3: Globular Cluster Age-Galactocentric Distance Analysis

Tests the TEP prediction that GCs in deeper galactic potentials
should show enhanced proper time accumulation (appearing older).

Data Source: VandenBerg et al. 2013, ApJ 775, 134
             Harris 1996 (2010 edition) catalog

TEP Prediction:
- At fixed metallicity, GCs at smaller R_gc (deeper potential) should
  appear older than GCs at larger R_gc
- The age excess should scale with potential depth: Δt ∝ Φ ∝ 1/R_gc

Author: TEP Research Program
Date: 2026-01-15
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# VandenBerg+2013 Table 2 data (55 GCs with ages and galactocentric distances)
# Columns: Name, [Fe/H], Age (Gyr), R_gc (kpc)
# Transcribed from ApJ 775, 134
VANDENBERG_2013_DATA = """
NGC104,-0.72,12.75,7.4
NGC288,-1.32,12.50,12.0
NGC362,-1.26,11.50,9.4
NGC1261,-1.27,11.50,18.1
NGC1851,-1.18,11.00,16.6
NGC1904,-1.60,12.50,18.8
NGC2298,-1.92,13.00,15.8
NGC2808,-1.14,11.00,11.1
NGC3201,-1.59,12.00,8.8
NGC4147,-1.80,12.50,21.4
NGC4590,-2.23,12.50,33.8
NGC4833,-1.85,13.00,6.6
NGC5024,-2.10,13.00,18.3
NGC5053,-2.27,12.50,17.4
NGC5272,-1.50,12.50,12.0
NGC5286,-1.69,13.00,8.9
NGC5466,-1.98,13.00,16.3
NGC5897,-1.90,12.50,7.3
NGC5904,-1.29,12.25,6.2
NGC5927,-0.49,11.00,4.6
NGC5986,-1.59,12.50,4.8
NGC6093,-1.75,13.00,3.8
NGC6101,-1.98,13.00,11.1
NGC6121,-1.16,12.50,5.9
NGC6144,-1.76,13.00,2.6
NGC6171,-1.02,12.00,3.3
NGC6205,-1.53,12.00,8.4
NGC6218,-1.37,13.00,4.5
NGC6254,-1.56,12.50,4.6
NGC6304,-0.45,11.50,2.3
NGC6341,-2.31,13.00,9.6
NGC6352,-0.64,11.50,3.3
NGC6362,-1.07,12.50,5.1
NGC6388,-0.55,11.50,3.1
NGC6397,-2.02,13.00,6.0
NGC6441,-0.46,11.50,3.9
NGC6496,-0.46,11.50,4.0
NGC6535,-1.79,12.00,3.9
NGC6541,-1.81,13.00,2.2
NGC6584,-1.50,12.00,7.0
NGC6624,-0.44,11.50,1.2
NGC6637,-0.59,11.50,1.6
NGC6652,-0.81,11.50,2.4
NGC6656,-1.70,13.00,4.9
NGC6681,-1.62,12.50,2.2
NGC6717,-1.26,12.50,2.4
NGC6723,-1.10,12.50,2.6
NGC6752,-1.54,12.50,5.2
NGC6779,-1.98,13.00,9.4
NGC6809,-1.94,13.00,3.9
NGC6838,-0.78,11.50,6.7
NGC6934,-1.47,12.00,12.8
NGC7006,-1.52,12.50,38.5
NGC7078,-2.37,13.00,10.4
NGC7099,-2.27,13.00,7.1
"""

def load_gc_data():
    """Load VandenBerg+2013 globular cluster data."""
    from io import StringIO
    
    df = pd.read_csv(StringIO(VANDENBERG_2013_DATA.strip()), 
                     names=['Name', 'FeH', 'Age_Gyr', 'R_gc_kpc'])
    
    logger.info(f"Loaded {len(df)} globular clusters from VandenBerg+2013")
    logger.info(f"  [Fe/H] range: {df['FeH'].min():.2f} to {df['FeH'].max():.2f}")
    logger.info(f"  Age range: {df['Age_Gyr'].min():.1f} to {df['Age_Gyr'].max():.1f} Gyr")
    logger.info(f"  R_gc range: {df['R_gc_kpc'].min():.1f} to {df['R_gc_kpc'].max():.1f} kpc")
    
    return df

def estimate_galactic_potential(R_gc_kpc):
    """
    Estimate the Milky Way gravitational potential at galactocentric distance R_gc.
    
    Uses a simple NFW-like profile:
    Φ(r) ∝ -ln(1 + r/r_s) / r
    
    For simplicity, we use log(R_gc) as a proxy for potential depth.
    Deeper potential = smaller R_gc = more negative Φ.
    """
    # Potential depth proxy: -log(R_gc) 
    # More negative = deeper potential = smaller R_gc
    return -np.log10(R_gc_kpc)

def analyze_age_rgc_correlation(df, metallicity_cut=None, label="Full Sample"):
    """
    Analyze correlation between GC age and galactocentric distance.
    
    TEP Prediction: Negative correlation (smaller R_gc → older apparent age)
    Standard Expectation: No correlation at fixed metallicity
    """
    if metallicity_cut is not None:
        mask = df['FeH'] < metallicity_cut
        sample = df[mask].copy()
        label = f"Metal-poor ([Fe/H] < {metallicity_cut})"
    else:
        sample = df.copy()
    
    n = len(sample)
    
    # Calculate potential proxy
    sample['Phi_proxy'] = estimate_galactic_potential(sample['R_gc_kpc'])
    
    # Correlation: Age vs R_gc
    r_age_rgc, p_age_rgc = stats.spearmanr(sample['Age_Gyr'], sample['R_gc_kpc'])
    
    # Correlation: Age vs Potential (should be positive under TEP: deeper potential → older)
    r_age_phi, p_age_phi = stats.spearmanr(sample['Age_Gyr'], sample['Phi_proxy'])
    
    # Linear regression: Age = a + b * log(R_gc)
    log_rgc = np.log10(sample['R_gc_kpc'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_rgc, sample['Age_Gyr'])
    
    # Partial correlation controlling for metallicity
    # Residualize both Age and R_gc against [Fe/H]
    slope_age_feh, intercept_age_feh, _, _, _ = stats.linregress(sample['FeH'], sample['Age_Gyr'])
    slope_rgc_feh, intercept_rgc_feh, _, _, _ = stats.linregress(sample['FeH'], sample['R_gc_kpc'])
    
    age_resid = sample['Age_Gyr'] - (slope_age_feh * sample['FeH'] + intercept_age_feh)
    rgc_resid = sample['R_gc_kpc'] - (slope_rgc_feh * sample['FeH'] + intercept_rgc_feh)
    
    r_partial, p_partial = stats.spearmanr(age_resid, rgc_resid)
    
    # Bootstrap confidence interval
    n_boot = 1000
    boot_slopes = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_sample = sample.iloc[idx]
        boot_log_rgc = np.log10(boot_sample['R_gc_kpc'])
        s, _, _, _, _ = stats.linregress(boot_log_rgc, boot_sample['Age_Gyr'])
        boot_slopes.append(s)
    
    slope_ci_low = np.percentile(boot_slopes, 2.5)
    slope_ci_high = np.percentile(boot_slopes, 97.5)
    
    results = {
        'label': label,
        'n_clusters': n,
        'age_mean': float(sample['Age_Gyr'].mean()),
        'age_std': float(sample['Age_Gyr'].std()),
        'rgc_mean': float(sample['R_gc_kpc'].mean()),
        'rgc_std': float(sample['R_gc_kpc'].std()),
        'spearman_age_rgc': float(r_age_rgc),
        'spearman_age_rgc_p': float(p_age_rgc),
        'spearman_age_phi': float(r_age_phi),
        'spearman_age_phi_p': float(p_age_phi),
        'slope_age_logRgc': float(slope),
        'slope_err': float(std_err),
        'slope_ci_low': float(slope_ci_low),
        'slope_ci_high': float(slope_ci_high),
        'intercept': float(intercept),
        'r_squared': float(r_value**2),
        'partial_r_age_rgc_feh': float(r_partial),
        'partial_p': float(p_partial),
    }
    
    return results, sample

def tep_prediction(alpha=0.58, R_gc_inner=2.0, R_gc_outer=20.0):
    """
    Calculate TEP-predicted age enhancement for inner vs outer GCs.
    
    Under TEP, proper time accumulation scales with potential depth:
    Γ_t ∝ α * |Φ|^(1/3)
    
    For a simple 1/r potential: Φ ∝ -1/r
    So Γ_t ∝ α * (1/r)^(1/3) = α * r^(-1/3)
    
    Age enhancement: t_apparent / t_true = 1 + Γ_t
    """
    # Potential depth ratio
    phi_ratio = R_gc_outer / R_gc_inner  # Outer is shallower
    
    # TEP enhancement ratio (inner has more enhancement)
    gamma_inner = alpha * (1.0 / R_gc_inner)**(1/3)
    gamma_outer = alpha * (1.0 / R_gc_outer)**(1/3)
    
    # Relative age enhancement
    delta_gamma = gamma_inner - gamma_outer
    
    # For a 12.5 Gyr true age, what's the apparent age difference?
    true_age = 12.5  # Gyr
    age_inner = true_age * (1 + gamma_inner)
    age_outer = true_age * (1 + gamma_outer)
    delta_age = age_inner - age_outer
    
    return {
        'alpha': alpha,
        'R_gc_inner_kpc': R_gc_inner,
        'R_gc_outer_kpc': R_gc_outer,
        'gamma_inner': float(gamma_inner),
        'gamma_outer': float(gamma_outer),
        'delta_gamma': float(delta_gamma),
        'predicted_delta_age_Gyr': float(delta_age),
        'true_age_assumed_Gyr': true_age,
    }

def main():
    """Run the globular cluster age analysis."""
    logger.info("="*60)
    logger.info("TEP-JWST: Globular Cluster Age-Distance Analysis")
    logger.info("="*60)
    
    # Load data
    df = load_gc_data()
    
    # Save processed data
    df.to_csv(DATA_DIR / "interim" / "vandenberg_2013_gc_ages.csv", index=False)
    
    results = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'data_source': 'VandenBerg et al. 2013, ApJ 775, 134',
        'n_total': len(df),
    }
    
    # Full sample analysis
    logger.info("\n--- Full Sample Analysis ---")
    full_results, _ = analyze_age_rgc_correlation(df, label="Full Sample")
    results['full_sample'] = full_results
    
    logger.info(f"  N = {full_results['n_clusters']}")
    logger.info(f"  Age-R_gc Spearman ρ = {full_results['spearman_age_rgc']:.3f} (p = {full_results['spearman_age_rgc_p']:.4f})")
    logger.info(f"  Slope (Age vs log R_gc) = {full_results['slope_age_logRgc']:.3f} ± {full_results['slope_err']:.3f} Gyr/dex")
    logger.info(f"  Partial ρ (controlling [Fe/H]) = {full_results['partial_r_age_rgc_feh']:.3f} (p = {full_results['partial_p']:.4f})")
    
    # Metal-poor subsample (to control for age-metallicity relation)
    logger.info("\n--- Metal-Poor Subsample ([Fe/H] < -1.5) ---")
    mp_results, mp_sample = analyze_age_rgc_correlation(df, metallicity_cut=-1.5)
    results['metal_poor'] = mp_results
    
    logger.info(f"  N = {mp_results['n_clusters']}")
    logger.info(f"  Age-R_gc Spearman ρ = {mp_results['spearman_age_rgc']:.3f} (p = {mp_results['spearman_age_rgc_p']:.4f})")
    logger.info(f"  Slope = {mp_results['slope_age_logRgc']:.3f} ± {mp_results['slope_err']:.3f} Gyr/dex")
    
    # Very metal-poor subsample
    logger.info("\n--- Very Metal-Poor Subsample ([Fe/H] < -1.7) ---")
    vmp_results, vmp_sample = analyze_age_rgc_correlation(df, metallicity_cut=-1.7)
    results['very_metal_poor'] = vmp_results
    
    logger.info(f"  N = {vmp_results['n_clusters']}")
    logger.info(f"  Age-R_gc Spearman ρ = {vmp_results['spearman_age_rgc']:.3f} (p = {vmp_results['spearman_age_rgc_p']:.4f})")
    logger.info(f"  Slope = {vmp_results['slope_age_logRgc']:.3f} ± {vmp_results['slope_err']:.3f} Gyr/dex")
    
    # Inner vs Outer comparison
    logger.info("\n--- Inner vs Outer Halo Comparison ---")
    inner_mask = df['R_gc_kpc'] < 8.0
    outer_mask = df['R_gc_kpc'] >= 8.0
    
    inner_age = df[inner_mask]['Age_Gyr'].mean()
    outer_age = df[outer_mask]['Age_Gyr'].mean()
    inner_std = df[inner_mask]['Age_Gyr'].std()
    outer_std = df[outer_mask]['Age_Gyr'].std()
    n_inner = inner_mask.sum()
    n_outer = outer_mask.sum()
    
    # t-test
    t_stat, t_p = stats.ttest_ind(df[inner_mask]['Age_Gyr'], df[outer_mask]['Age_Gyr'])
    
    results['inner_outer'] = {
        'R_gc_cut_kpc': 8.0,
        'n_inner': int(n_inner),
        'n_outer': int(n_outer),
        'age_inner_mean': float(inner_age),
        'age_inner_std': float(inner_std),
        'age_outer_mean': float(outer_age),
        'age_outer_std': float(outer_std),
        'delta_age': float(inner_age - outer_age),
        't_statistic': float(t_stat),
        't_p_value': float(t_p),
    }
    
    logger.info(f"  Inner (R_gc < 8 kpc): N = {n_inner}, Age = {inner_age:.2f} ± {inner_std:.2f} Gyr")
    logger.info(f"  Outer (R_gc ≥ 8 kpc): N = {n_outer}, Age = {outer_age:.2f} ± {outer_std:.2f} Gyr")
    logger.info(f"  ΔAge (Inner - Outer) = {inner_age - outer_age:.2f} Gyr")
    logger.info(f"  t-test p-value = {t_p:.4f}")
    
    # TEP prediction
    logger.info("\n--- TEP Prediction ---")
    tep_pred = tep_prediction(alpha=0.58)
    results['tep_prediction'] = tep_pred
    
    logger.info(f"  Using α = {tep_pred['alpha']} from TEP-H0")
    logger.info(f"  Predicted Δt (R=2 vs R=20 kpc) = {tep_pred['predicted_delta_age_Gyr']:.2f} Gyr")
    
    # Interpretation
    logger.info("\n" + "="*60)
    logger.info("INTERPRETATION")
    logger.info("="*60)
    
    if full_results['spearman_age_rgc'] < 0:
        logger.info("✓ NEGATIVE correlation detected: Inner GCs appear OLDER")
        logger.info("  This is CONSISTENT with TEP prediction")
    else:
        logger.info("✗ Positive/null correlation: Inner GCs NOT systematically older")
        logger.info("  This does NOT support TEP prediction")
    
    if full_results['spearman_age_rgc_p'] < 0.05:
        logger.info(f"  Correlation is SIGNIFICANT (p = {full_results['spearman_age_rgc_p']:.4f})")
    else:
        logger.info(f"  Correlation is NOT significant (p = {full_results['spearman_age_rgc_p']:.4f})")
    
    # Save results
    output_file = RESULTS_DIR / "gc_age_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    main()
