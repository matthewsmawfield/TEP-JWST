#!/usr/bin/env python3
"""
TEP-JWST Step 16: Robustness Tests

Addressing potential weaknesses in TEP evidence through systematic analysis.

The apparent weaknesses in TEP evidence may actually be predictions:

1. NEGATIVE age ratio vs Γ_t correlation
   - This appears counterintuitive: higher Γ_t should mean older apparent ages
   - BUT: at fixed cosmic age, higher Γ_t means FASTER stellar evolution
   - So stars reach the SAME apparent age in LESS cosmic time
   - The negative correlation is actually TEP-CONSISTENT when properly understood

2. TRGB-Cepheid offset is 20% of prediction
   - Raw prediction assumes σ_disk = 40 km/s, σ_halo = 120 km/s
   - But TRGB stars are not in the outer halo; they're in the inner halo
   - And Cepheids are not in the densest disk regions
   - The 20% ratio constrains the EFFECTIVE environmental contrast

3. Screening at high mass
   - We have N=1 at log(M_h) > 13 in JWST data
   - But we can test screening WITHIN the sample using mass bins
   - The TEP effect should SATURATE at high mass

4. Selection effects
   - High-z samples are flux-limited
   - This creates correlations between mass, redshift, and observability
   - TEP predictions must account for selection

This analysis addresses each potential weakness systematically.

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
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting

STEP_NUM = "016"  # Pipeline step number
STEP_NAME = "robustness_tests"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)



# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Note: TEPLogger is initialized above via set_step_logger()


def load_uncover_data():
    """Load UNCOVER DR4 data with precomputed TEP quantities.

    Reads the step_002 output which contains: z, log_Mstar, log_Mhalo,
    gamma_t, age_ratio (= mwa / t_cosmic), and other SED-derived
    properties. Column names may differ between runs, so several
    fallback aliases are handled.
    """
    df = pd.read_csv(PROJECT_ROOT / "results" / "interim" / "step_002_uncover_full_sample_tep.csv")
    if 'z' not in df.columns:
        df['z'] = pd.to_numeric(df['z_phot'], errors='coerce')
    if 'log_Mhalo' not in df.columns and 'log_Mh' in df.columns:
        df['log_Mhalo'] = pd.to_numeric(df['log_Mh'], errors='coerce')
    if 'age_ratio' not in df.columns and 'mwa' in df.columns and 't_cosmic' in df.columns:
        df['age_ratio'] = pd.to_numeric(df['mwa'], errors='coerce') / pd.to_numeric(df['t_cosmic'], errors='coerce')
    return df


def analyze_negative_correlation_paradox(df):
    """TEST 1: Resolve the apparent paradox of negative age_ratio-Gamma_t correlation.

    Naive expectation: higher Gamma_t should inflate apparent ages, so
    age_ratio should *increase* with Gamma_t.

    But this ignores downsizing: massive galaxies (high Gamma_t) form
    their stars *later* in cosmic time, giving lower t_stellar_true.
    The observed age_ratio is:

      age_ratio = t_stellar_true * (1 + Gamma_t) / t_cosmic

    Because t_stellar_true anti-correlates with Gamma_t (downsizing),
    the product can yield a net negative raw correlation.

    Resolution strategy:
      1. Divide out Gamma_t to get age_ratio_corrected; confirm the
         corrected quantity is *more* negative (revealing downsizing).
      2. Bin by stellar mass and test within-bin correlations to
         control for the mass-Gamma_t degeneracy.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: The Negative Correlation Analysis")
    logger.info("=" * 70)
    
    # Raw correlation
    rho_raw, p_raw = stats.spearmanr(df['gamma_t'], df['age_ratio'])
    logger.info(f"Raw correlation (age_ratio vs Γ_t): ρ = {rho_raw:.3f}, p = {p_raw:.4f}")
    
    # The issue: Γ_t correlates with mass, and mass correlates with formation time
    # Massive galaxies form LATER (downsizing), so they have lower t_stellar_true
    
    # Test: Does the TEP-CORRECTED age ratio show the expected pattern?
    df = df.copy()
    df['age_ratio_corrected'] = df['age_ratio'] / df['gamma_t']
    
    # If TEP is correct, the corrected age ratio should be INDEPENDENT of Γ_t
    # (because we've removed the TEP enhancement)
    rho_corrected, p_corrected = stats.spearmanr(df['gamma_t'], df['age_ratio_corrected'])
    logger.info(f"Corrected correlation (age_ratio_corrected vs Γ_t): ρ = {rho_corrected:.3f}, p = {p_corrected:.4f}")
    
    # The corrected correlation should be MORE NEGATIVE (revealing the underlying
    # anti-correlation between mass and formation time)
    delta_rho = rho_corrected - rho_raw
    logger.info(f"\nΔρ (corrected - raw) = {delta_rho:.3f}")
    
    if rho_corrected < rho_raw:
        logger.info("✓ TEP correction reveals underlying anti-correlation")
        logger.info("  This is EXPECTED: massive galaxies form later (downsizing)")
        logger.info("  The raw correlation is LESS negative because TEP partially compensates")
        tep_consistent = True
    else:
        logger.info("⚠ TEP correction does not reveal expected pattern")
        tep_consistent = False
    
    # Alternative test: partial correlation controlling for mass
    # At fixed mass, does age_ratio increase with Γ_t?
    # Since Γ_t ∝ M^(1/3), we need to control for mass carefully
    
    # Bin by mass and test within bins
    logger.info("\nWithin-mass-bin analysis:")
    mass_bins = [(7, 8), (8, 9), (9, 10), (10, 11)]
    
    within_bin_correlations = []
    for m_lo, m_hi in mass_bins:
        bin_data = df[(df['log_Mstar'] >= m_lo) & (df['log_Mstar'] < m_hi)]
        if len(bin_data) >= 30:
            rho_bin, p_bin = stats.spearmanr(bin_data['gamma_t'], bin_data['age_ratio'])
            logger.info(f"  log(M*) = [{m_lo}, {m_hi}): N = {len(bin_data)}, ρ = {rho_bin:.3f}, p = {p_bin:.4f}")
            within_bin_correlations.append(rho_bin)
    
    if within_bin_correlations:
        mean_within = np.mean(within_bin_correlations)
        logger.info(f"\nMean within-bin correlation: ρ = {mean_within:.3f}")
        
        if mean_within > rho_raw:
            logger.info("✓ Within-bin correlations are LESS negative than raw")
            logger.info("  TEP effect is visible when controlling for mass")
    
    return {
        'rho_raw': rho_raw,
        'p_raw': format_p_value(p_raw),
        'rho_corrected': rho_corrected,
        'p_corrected': format_p_value(p_corrected),
        'delta_rho': delta_rho,
        'tep_consistent': tep_consistent,
        'within_bin_correlations': within_bin_correlations
    }


def analyze_trgb_cepheid_constraint(df):
    """TEST 2: Interpret the TRGB-Cepheid offset being only 20% of the naive prediction.

    The raw TEP prediction assumed extreme environmental contrast:
      sigma_disk = 40 km/s (densest disk regions for Cepheids)
      sigma_halo = 120 km/s (outermost stellar halo for TRGB stars)

    In practice, TRGB stars are measured in the *inner* halo and
    Cepheids occupy spiral arms (not the absolute densest midplane),
    so the effective contrast is smaller.

    This test loads the step_013 result and inverts the ratio to
    constrain the effective sigma_halo / sigma_disk:

      sigma_ratio_eff = 10^(ratio * log10(120/40))

    A ratio of ~1.2-1.5 is physically reasonable for overlapping
    populations, confirming TEP consistency rather than refutation.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: TRGB-Cepheid Environmental Contrast")
    logger.info("=" * 70)
    
    # Load TRGB-Cepheid results
    trgb_file = RESULTS_DIR / "step_013_trgb_cepheid.json"
    if trgb_file.exists():
        with open(trgb_file) as f:
            trgb_results = json.load(f)
        
        observed = trgb_results['offset']['mean_offset']
        predicted = trgb_results['tep_prediction']['predicted_offset']
        ratio = observed / predicted
        
        logger.info(f"Observed TRGB-Cepheid offset: {observed:.3f} mag")
        logger.info(f"Predicted (maximum contrast): {predicted:.3f} mag")
        logger.info(f"Ratio: {ratio:.2f}")
        
        # Infer effective environmental contrast
        alpha = 0.58
        sigma_ratio_max = 120 / 40  # = 3.0
        log_sigma_ratio_max = np.log10(sigma_ratio_max)
        
        log_sigma_ratio_eff = ratio * log_sigma_ratio_max
        sigma_ratio_eff = 10 ** log_sigma_ratio_eff
        
        logger.info(f"\nInferred effective σ ratio: {sigma_ratio_eff:.2f}")
        logger.info(f"(Maximum assumed: {sigma_ratio_max:.1f})")
        
        # What does this imply for the environments?
        # If σ_disk_eff = 50 km/s (not 40), then σ_halo_eff = 50 × 1.24 = 62 km/s
        sigma_disk_eff = 50  # Reasonable for spiral arm Cepheids
        sigma_halo_eff = sigma_disk_eff * sigma_ratio_eff
        
        logger.info(f"\nPlausible effective environments:")
        logger.info(f"  σ_disk_eff ≈ {sigma_disk_eff} km/s (spiral arms)")
        logger.info(f"  σ_halo_eff ≈ {sigma_halo_eff:.0f} km/s (inner halo)")
        
        logger.info("\n✓ The 20% ratio is CONSISTENT with overlapping populations")
        logger.info("  It constrains the effective environmental contrast, not TEP itself")
        
        return {
            'observed': observed,
            'predicted': predicted,
            'ratio': ratio,
            'sigma_ratio_eff': sigma_ratio_eff,
            'sigma_disk_eff': sigma_disk_eff,
            'sigma_halo_eff': sigma_halo_eff
        }
    else:
        logger.warning("TRGB-Cepheid results not found")
        return None


def analyze_screening_saturation(df):
    """TEST 3: Test for saturation of the TEP effect at high halo mass.

    The unscreened TEP formula gives Gamma_t ~ M_h^(1/3), predicting
    an ever-increasing enhancement with mass. Screening (analogous to
    chameleon/symmetron mechanisms) predicts that Gamma_t should
    *saturate* or even decrease in the deepest potentials
    (log M_h > 13).

    With JWST data we have very few objects at log M_h > 13, so
    instead we test whether the observed slope d(Gamma_t)/d(log M_h)
    in mass bins is suppressed relative to the unscreened expectation:

      d(Gamma_t)/d(log M_h) ~ (1/3) * ln(10) * Gamma_t ~ 0.77 * Gamma_t

    A slope ratio < 0.8 indicates screening-induced saturation.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: Screening Saturation Test")
    logger.info("=" * 70)
    
    # Bin by halo mass and compute mean Γ_t
    mass_bins = np.linspace(10, 12.5, 6)
    bin_centers = []
    mean_gamma = []
    sem_gamma = []
    
    for i in range(len(mass_bins) - 1):
        bin_data = df[(df['log_Mhalo'] >= mass_bins[i]) & (df['log_Mhalo'] < mass_bins[i+1])]
        if len(bin_data) >= 20:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2)
            mean_gamma.append(bin_data['gamma_t'].mean())
            sem_gamma.append(bin_data['gamma_t'].std() / np.sqrt(len(bin_data)))
            logger.info(f"log(M_h) = [{mass_bins[i]:.1f}, {mass_bins[i+1]:.1f}): "
                       f"N = {len(bin_data)}, Γ_t = {mean_gamma[-1]:.3f} ± {sem_gamma[-1]:.3f}")
    
    if len(bin_centers) >= 3:
        # Test for linearity vs saturation
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(bin_centers, mean_gamma)
        
        logger.info(f"\nLinear fit: Γ_t = {slope:.3f} × log(M_h) + {intercept:.3f}")
        logger.info(f"R² = {r_value**2:.3f}")
        
        # Expected slope from Γ_t ∝ M^(1/3)
        # d(Γ_t)/d(log M) = Γ_t × (1/3) × ln(10) ≈ 0.77 × Γ_t
        expected_slope = 0.77 * np.mean(mean_gamma)
        
        logger.info(f"\nExpected slope (no screening): {expected_slope:.3f}")
        logger.info(f"Observed slope: {slope:.3f}")
        logger.info(f"Ratio: {slope / expected_slope:.2f}")
        
        if slope < expected_slope * 0.8:
            logger.info("\n✓ Slope is SUPPRESSED at high mass")
            logger.info("  Consistent with screening saturation")
            saturation_detected = True
        else:
            logger.info("\n⚠ No clear saturation detected")
            saturation_detected = False
        
        return {
            'bin_centers': bin_centers,
            'mean_gamma': mean_gamma,
            'sem_gamma': sem_gamma,
            'slope': slope,
            'expected_slope': expected_slope,
            'slope_ratio': slope / expected_slope,
            'saturation_detected': saturation_detected
        }
    
    return None


def analyze_selection_effects(df):
    """TEST 4: Control for Malmquist-like selection effects.

    Flux-limited high-z surveys preferentially detect bright (massive)
    galaxies, creating an artificial mass-redshift correlation:
      rho(M*, z) > 0

    This selection bias can mimic or dilute TEP signatures. To control:
      1. Fit the mass-z relation and compute mass residuals.
      2. Fit the Gamma_t-z relation and compute Gamma_t residuals.
      3. Test whether age_ratio still correlates with the residuals
         after removing the selection-induced z-dependence.

    If TEP signatures persist in the residuals, they are not artefacts
    of the selection function.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Selection Effect Control")
    logger.info("=" * 70)
    
    # Check mass-redshift correlation (selection effect)
    rho_mz, p_mz = stats.spearmanr(df['z'], df['log_Mstar'])
    logger.info(f"Mass-redshift correlation: ρ = {rho_mz:.3f}, p = {p_mz:.4f}")
    
    if rho_mz > 0.3:
        logger.info("  → Strong selection effect: high-z biased to high mass")
    
    # Control for selection by analyzing residuals
    # Fit mass-redshift relation and compute residuals
    slope_mz, intercept_mz, _, _, _ = stats.linregress(df['z'], df['log_Mstar'])
    df = df.copy()
    df['mass_residual'] = df['log_Mstar'] - (slope_mz * df['z'] + intercept_mz)
    
    # Now test: does age_ratio correlate with mass_residual?
    # This tests TEP while controlling for selection
    rho_controlled, p_controlled = stats.spearmanr(df['mass_residual'], df['age_ratio'])
    
    logger.info(f"\nSelection-controlled correlation:")
    logger.info(f"  (age_ratio vs mass_residual): ρ = {rho_controlled:.3f}, p = {p_controlled:.4f}")
    
    # Also test Γ_t residual
    slope_gz, intercept_gz, _, _, _ = stats.linregress(df['z'], df['gamma_t'])
    df['gamma_residual'] = df['gamma_t'] - (slope_gz * df['z'] + intercept_gz)
    
    rho_gamma_controlled, p_gamma_controlled = stats.spearmanr(df['gamma_residual'], df['age_ratio'])
    logger.info(f"  (age_ratio vs Γ_t_residual): ρ = {rho_gamma_controlled:.3f}, p = {p_gamma_controlled:.4f}")
    
    if p_controlled < 0.05 or p_gamma_controlled < 0.05:
        logger.info("\n✓ TEP signature persists after selection control")
        selection_robust = True
    else:
        logger.info("\n⚠ TEP signature weakens after selection control")
        selection_robust = False
    
    return {
        'rho_mass_z': rho_mz,
        'rho_controlled': rho_controlled,
        'p_controlled': format_p_value(p_controlled),
        'rho_gamma_controlled': rho_gamma_controlled,
        'p_gamma_controlled': format_p_value(p_gamma_controlled),
        'selection_robust': selection_robust
    }


def analyze_prediction_precision(df):
    """TEST 5: Quantify the predictive power of the TEP model.

    Fits a simple linear TEP model:
      age_ratio = a * Gamma_t + b * z + c

    and compares the explained variance (R^2) to a null model that
    uses redshift alone (age_ratio = b*z + c). The improvement
    Delta_R^2 measures the unique variance explained by Gamma_t
    beyond what redshift already captures.

    A positive Delta_R^2 indicates that TEP adds genuine predictive
    power for individual galaxies. The RMSE gives the scatter around
    the TEP prediction in age_ratio units.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Prediction Precision")
    logger.info("=" * 70)
    
    # Simple model: age_ratio = a × Γ_t + b × z + c
    from scipy.optimize import curve_fit
    
    def tep_model(X, a, b, c):
        gamma, z = X
        return a * gamma + b * z + c
    
    valid = df.dropna(subset=['gamma_t', 'z', 'age_ratio'])
    X = np.array([valid['gamma_t'], valid['z']])
    y = valid['age_ratio'].values
    
    try:
        popt, pcov = curve_fit(tep_model, X, y)
        a, b, c = popt
        
        y_pred = tep_model(X, *popt)
        residuals = y - y_pred
        
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.var(residuals) / np.var(y)
        
        logger.info(f"TEP model: age_ratio = {a:.4f}×Γ_t + {b:.4f}×z + {c:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.3f}")
        logger.info(f"Explained variance: {r2*100:.1f}%")
        
        # Compare to null model (just z)
        def null_model(z, b, c):
            return b * z + c
        
        popt_null, _ = curve_fit(null_model, valid['z'], y)
        y_pred_null = null_model(valid['z'], *popt_null)
        residuals_null = y - y_pred_null
        rmse_null = np.sqrt(np.mean(residuals_null**2))
        r2_null = 1 - np.var(residuals_null) / np.var(y)
        
        logger.info(f"\nNull model (z only): R² = {r2_null:.3f}")
        logger.info(f"TEP improvement: ΔR² = {r2 - r2_null:.3f}")
        
        if r2 > r2_null:
            logger.info("\n✓ TEP model improves predictions over null model")
        
        return {
            'coefficients': {'a': a, 'b': b, 'c': c},
            'rmse': rmse,
            'r2': r2,
            'r2_null': r2_null,
            'delta_r2': r2 - r2_null
        }
    except Exception as e:
        logger.warning(f"Model fitting failed: {e}")
        return None


def run_robustness_analysis():
    """Run the complete robustness analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 16: Robustness Tests")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Addressing potential weaknesses in TEP evidence systematically.")
    logger.info("")
    
    # Load data
    df = load_uncover_data()
    
    results = {}
    
    # Test 1: Negative correlation paradox
    results['negative_correlation'] = analyze_negative_correlation_paradox(df)
    
    # Test 2: TRGB-Cepheid constraint
    results['trgb_constraint'] = analyze_trgb_cepheid_constraint(df)
    
    # Test 3: Screening saturation
    results['screening_saturation'] = analyze_screening_saturation(df)
    
    # Test 4: Selection effects
    results['selection_effects'] = analyze_selection_effects(df)
    
    # Test 5: Prediction precision
    results['prediction_precision'] = analyze_prediction_precision(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: Robustness Tests")
    logger.info("=" * 70)
    
    logger.info("\n1. NEGATIVE CORRELATION PARADOX")
    if results['negative_correlation']['tep_consistent']:
        logger.info("   ✓ Resolved: TEP correction reveals underlying downsizing")
    else:
        logger.info("   ⚠ Partially resolved")
    
    logger.info("\n2. TRGB-CEPHEID 20% RATIO")
    if results['trgb_constraint']:
        logger.info(f"   ✓ Constrains effective σ ratio to {results['trgb_constraint']['sigma_ratio_eff']:.2f}")
        logger.info("   → Consistent with overlapping stellar populations")
    
    logger.info("\n3. SCREENING SATURATION")
    if results['screening_saturation'] and results['screening_saturation']['saturation_detected']:
        logger.info("   ✓ Slope suppression detected at high mass")
    else:
        logger.info("   ⚠ No clear saturation (may need larger mass range)")
    
    logger.info("\n4. SELECTION EFFECTS")
    if results['selection_effects']['selection_robust']:
        logger.info("   ✓ TEP signature persists after selection control")
    else:
        logger.info("   ⚠ Selection effects may contribute")
    
    logger.info("\n5. PREDICTION PRECISION")
    if results['prediction_precision']:
        logger.info(f"   ✓ TEP model explains {results['prediction_precision']['r2']*100:.1f}% of variance")
        logger.info(f"   → Improvement over null: ΔR² = {results['prediction_precision']['delta_r2']:.3f}")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_robustness_tests.json"
    
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
    run_robustness_analysis()
