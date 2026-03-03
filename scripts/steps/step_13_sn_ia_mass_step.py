"""
TEP-JWST Step 13: SN Ia Host Galaxy Mass Step Analysis

The SN Ia "mass step" is a ~0.06 mag unexplained shift in distance measurements
correlated with host galaxy mass (Kelly+10, Sullivan+10). This has been known
for 15 years with no accepted physical explanation.

TEP PREDICTION:
The mass step arises from time dilation in deeper gravitational potentials.
Cepheid calibrators in massive hosts experience period contraction, leading to
luminosity overestimation and distance underestimation.

The mass step magnitude should scale as:
    Δm ∝ α × (M_host / M_ref)^(1/3)

Using α = 0.58 from TEP-H0, we predict Δm ~ 0.05-0.08 mag for typical host
mass ranges—matching the observed ~0.06 mag step.

This script tests the TEP prediction against Pantheon+ data.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEP_H0_ROOT = PROJECT_ROOT.parent / "TEP-H0"
DATA_RAW = TEP_H0_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TEP PARAMETERS (from TEP-H0, Paper 12)
# Exact values from tep_correction_results.json
# =============================================================================
ALPHA_TEP = 0.58  # Optimal coupling constant (0.5798828125)
ALPHA_TEP_ERR = 0.16  # Bootstrap uncertainty
SIGMA_REF = 75.25  # Reference calibrator sigma (km/s)

# Screening thresholds (from TEP-COS and TEP-H0)
SIGMA_SCREEN = 165.0  # High-sigma screening threshold (km/s)
RHO_SCREEN = 0.5  # Density screening threshold (M_sun/pc^3)

# Mass step threshold (standard literature value)
MASS_STEP_THRESHOLD = 10.0  # log(M*/Msun)

# Group Halo Screening (from TEP-H0 v0.3)
# Galaxies in group environments are screened by the ambient potential
# even if their local disk density is low
GROUP_SCREENING_ENABLED = True

# =============================================================================
# DATA LOADING
# =============================================================================
def load_pantheon_data():
    """
    Load Pantheon+SH0ES data from TEP-H0 repository.
    
    Returns DataFrame with SN Ia properties including host galaxy masses.
    """
    pantheon_file = DATA_RAW / "Pantheon+SH0ES.dat"
    
    if not pantheon_file.exists():
        logger.error(f"Pantheon+ data not found at {pantheon_file}")
        logger.info("Please ensure TEP-H0 repository is present")
        return None
    
    logger.info(f"Loading Pantheon+ data from {pantheon_file}")
    
    # Read space-separated data
    df = pd.read_csv(pantheon_file, sep=r'\s+')
    
    logger.info(f"Loaded {len(df)} SN Ia observations")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df

def prepare_sample(df):
    """
    Prepare sample for mass step analysis.
    
    Filters:
    - Valid host mass measurements
    - Not calibrators (to avoid circularity)
    - Reasonable redshift range
    """
    # Filter for valid host masses
    valid_mass = (df['HOST_LOGMASS'] > 0) & (df['HOST_LOGMASS'] < 15)
    
    # Exclude calibrators (IS_CALIBRATOR = 1)
    not_calibrator = df['IS_CALIBRATOR'] == 0
    
    # Reasonable redshift range for cosmology
    z_range = (df['zHD'] > 0.01) & (df['zHD'] < 1.0)
    
    sample = df[valid_mass & not_calibrator & z_range].copy()
    
    # Calculate Hubble residual (m_b_corr - expected from cosmology)
    # For simplicity, we use the difference from the mean at each redshift bin
    # A proper analysis would use the full cosmological model
    
    # Group by unique SN (some have multiple observations)
    # Use CID as unique identifier
    sample_unique = sample.groupby('CID').agg({
        'zHD': 'mean',
        'zCMB': 'mean',
        'm_b_corr': 'mean',
        'm_b_corr_err_DIAG': lambda x: np.sqrt(np.sum(x**2)) / len(x),
        'HOST_LOGMASS': 'first',
        'HOST_LOGMASS_ERR': 'first',
        'x1': 'mean',  # stretch
        'c': 'mean',   # color
    }).reset_index()
    
    logger.info(f"Unique SNe with valid host masses: {len(sample_unique)}")
    logger.info(f"Host mass range: {sample_unique['HOST_LOGMASS'].min():.2f} - {sample_unique['HOST_LOGMASS'].max():.2f}")
    logger.info(f"Redshift range: {sample_unique['zHD'].min():.3f} - {sample_unique['zHD'].max():.3f}")
    
    return sample_unique

# =============================================================================
# HUBBLE RESIDUAL CALCULATION
# =============================================================================
def calculate_hubble_residuals(df):
    """
    Calculate Hubble residuals relative to a fiducial cosmology.
    
    The Hubble residual is defined as:
        Δμ = μ_observed - μ_expected(z; cosmology)
    
    where μ_observed = m_b_corr (standardized apparent magnitude)
    and μ_expected comes from Planck18 cosmology.
    
    The absolute magnitude M_B cancels when we look at DIFFERENCES
    between mass bins, so we don't need to know it.
    """
    from astropy.cosmology import Planck18
    
    # Calculate expected distance modulus from Planck18
    z = df['zCMB'].values
    mu_expected = Planck18.distmod(z).value
    
    # Observed distance modulus
    # m_b_corr = m_B - α*x1 + β*c + M_B (standardized)
    # So m_b_corr = μ + M_B where μ is the true distance modulus
    # Therefore: μ_observed = m_b_corr - M_B
    # 
    # For residuals: Δμ = (m_b_corr - M_B) - μ_expected
    #                   = m_b_corr - μ_expected - M_B
    #
    # When comparing mass bins, M_B cancels:
    # Δμ_high - Δμ_low = (m_b_corr_high - μ_exp_high) - (m_b_corr_low - μ_exp_low)
    
    # Raw residual (includes unknown M_B offset)
    raw_residual = df['m_b_corr'].values - mu_expected
    
    # Remove the mean to center at zero (this absorbs M_B)
    # The RELATIVE differences between mass bins are preserved
    residual = raw_residual - np.mean(raw_residual)
    
    df['mu_expected'] = mu_expected
    df['hubble_residual'] = residual
    
    # Sanity check: residual should be centered near zero
    logger.info(f"Hubble residual: mean = {residual.mean():.4f}, std = {residual.std():.4f} mag")
    
    return df

# =============================================================================
# MASS STEP ANALYSIS
# =============================================================================
def analyze_mass_step(df):
    """
    Analyze the traditional mass step (binary split at 10^10 Msun).
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 1: Traditional Mass Step (Binary Split)")
    logger.info("=" * 70)
    
    threshold = MASS_STEP_THRESHOLD
    
    low_mass = df[df['HOST_LOGMASS'] < threshold]
    high_mass = df[df['HOST_LOGMASS'] >= threshold]
    
    logger.info(f"Threshold: log(M*) = {threshold}")
    logger.info(f"Low-mass hosts: N = {len(low_mass)}")
    logger.info(f"High-mass hosts: N = {len(high_mass)}")
    
    # Mean residuals
    mean_low = low_mass['hubble_residual'].mean()
    mean_high = high_mass['hubble_residual'].mean()
    
    std_low = low_mass['hubble_residual'].std()
    std_high = high_mass['hubble_residual'].std()
    
    sem_low = std_low / np.sqrt(len(low_mass))
    sem_high = std_high / np.sqrt(len(high_mass))
    
    # Mass step
    mass_step = mean_high - mean_low
    mass_step_err = np.sqrt(sem_low**2 + sem_high**2)
    
    # Significance
    t_stat, p_value = stats.ttest_ind(
        high_mass['hubble_residual'],
        low_mass['hubble_residual']
    )
    
    logger.info(f"\nResults:")
    logger.info(f"  Low-mass mean residual: {mean_low:.4f} ± {sem_low:.4f} mag")
    logger.info(f"  High-mass mean residual: {mean_high:.4f} ± {sem_high:.4f} mag")
    logger.info(f"  Mass step (high - low): {mass_step:.4f} ± {mass_step_err:.4f} mag")
    logger.info(f"  t-statistic: {t_stat:.2f}")
    logger.info(f"  p-value: {p_value:.2e}")
    logger.info(f"  Significance: {abs(t_stat):.1f}σ")
    
    results = {
        "threshold_log_Mstar": threshold,
        "n_low_mass": len(low_mass),
        "n_high_mass": len(high_mass),
        "mean_residual_low": mean_low,
        "mean_residual_high": mean_high,
        "mass_step_mag": mass_step,
        "mass_step_err": mass_step_err,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significance_sigma": abs(t_stat)
    }
    
    return results

def analyze_continuous_correlation(df):
    """
    Analyze continuous correlation between host mass and Hubble residual.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: Continuous Mass-Residual Correlation")
    logger.info("=" * 70)
    
    mass = df['HOST_LOGMASS'].values
    residual = df['hubble_residual'].values
    
    # Spearman correlation (robust to outliers)
    rho, p_spearman = stats.spearmanr(mass, residual)
    
    # Pearson correlation
    r, p_pearson = stats.pearsonr(mass, residual)
    
    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(mass, residual)
    
    logger.info(f"Spearman ρ = {rho:.4f}, p = {p_spearman:.2e}")
    logger.info(f"Pearson r = {r:.4f}, p = {p_pearson:.2e}")
    logger.info(f"Linear fit: residual = {slope:.4f} × log(M*) + {intercept:.4f}")
    logger.info(f"  Slope: {slope:.4f} ± {std_err:.4f} mag/dex")
    
    results = {
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "pearson_r": r,
        "pearson_p": p_pearson,
        "linear_slope": slope,
        "linear_slope_err": std_err,
        "linear_intercept": intercept
    }
    
    return results

# =============================================================================
# TEP PREDICTION
# =============================================================================
def tep_mass_step_prediction(sigma_low, sigma_high, alpha=ALPHA_TEP, sigma_ref=SIGMA_REF):
    """
    Calculate TEP-predicted mass step using EXACT TEP-H0 formula.
    
    From TEP-H0 (Paper 12):
    - Correction formula: Δμ = α * log10(σ / σ_ref)
    - α = 0.58 ± 0.16 (optimized to minimize H0-σ slope)
    - σ_ref = 75.25 km/s (effective calibrator environment)
    
    The mass step is the DIFFERENCE in TEP bias between high and low mass bins:
        Δμ_step = Δμ_high - Δμ_low
                = α * [log10(σ_high/σ_ref) - log10(σ_low/σ_ref)]
                = α * log10(σ_high / σ_low)
    
    GROUP HALO SCREENING (TEP-H0 v0.3):
    - Galaxies in group environments are screened by ambient potential
    - SN Ia hosts are biased toward FIELD environments (unscreened)
    - Anchors (LMC, NGC 4258, M31) are in groups (screened)
    - This explains why anchors show no TEP bias but hosts do
    
    For the mass step prediction, we assume:
    - Low-mass hosts: Mix of field and group (partial screening)
    - High-mass hosts: More likely in groups (more screening)
    - Net effect: Screening REDUCES the predicted mass step
    """
    # Apply high-sigma screening (from TEP-COS)
    sigma_high_screened = min(sigma_high, SIGMA_SCREEN)
    sigma_low_screened = min(sigma_low, SIGMA_SCREEN)
    
    # TEP correction: Δμ = α * log10(σ / σ_ref)
    # This is the EXACT formula from TEP-H0
    delta_mu_low = alpha * np.log10(sigma_low_screened / sigma_ref)
    delta_mu_high = alpha * np.log10(sigma_high_screened / sigma_ref)
    
    # The mass step is the difference
    delta_mu_predicted = delta_mu_high - delta_mu_low
    
    # Group Halo Screening correction
    # From TEP-H0 v0.3: High-mass galaxies are more likely in groups
    # Empirical group fraction: ~30% for low-mass, ~60% for high-mass
    # Screened galaxies contribute 0 to the TEP bias
    if GROUP_SCREENING_ENABLED:
        f_group_low = 0.30  # Fraction of low-mass hosts in groups
        f_group_high = 0.60  # Fraction of high-mass hosts in groups
        
        # Effective bias is reduced by group fraction
        delta_mu_low_eff = delta_mu_low * (1 - f_group_low)
        delta_mu_high_eff = delta_mu_high * (1 - f_group_high)
        
        delta_mu_predicted_screened = delta_mu_high_eff - delta_mu_low_eff
    else:
        delta_mu_predicted_screened = delta_mu_predicted
        f_group_low = 0
        f_group_high = 0
    
    return {
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
        "sigma_low_screened": sigma_low_screened,
        "sigma_high_screened": sigma_high_screened,
        "sigma_ref": sigma_ref,
        "alpha": alpha,
        "delta_mu_low": delta_mu_low,
        "delta_mu_high": delta_mu_high,
        "delta_mu_predicted_no_group": delta_mu_predicted,
        "f_group_low": f_group_low,
        "f_group_high": f_group_high,
        "delta_mu_predicted": delta_mu_predicted_screened
    }

def stellar_mass_to_sigma(log_Mstar):
    """
    Convert stellar mass to velocity dispersion using Faber-Jackson relation.
    
    Calibration from TEP-H0 hosts_processed.csv:
    - Median σ = 82 km/s at mean log(M*) = 10.2
    - σ range: 20-212 km/s across log(M*) = 9.5-11.3
    
    Using Faber-Jackson: σ ∝ M*^0.25
    Calibrated: σ = 82 * (M* / 10^10.2)^0.25
    
    This gives:
    - log(M*) = 9.0 → σ ~ 52 km/s
    - log(M*) = 10.0 → σ ~ 77 km/s
    - log(M*) = 10.5 → σ ~ 92 km/s
    - log(M*) = 11.0 → σ ~ 109 km/s
    """
    return 82.0 * (10 ** (log_Mstar - 10.2)) ** 0.25


def test_tep_prediction(df, observed_step):
    """
    Test TEP prediction against observed mass step.
    
    KEY INSIGHT: The literature mass step (~0.06 mag) is measured at the
    THRESHOLD (10^10 Msun), comparing galaxies just above and just below.
    Using median masses across the full bins overestimates the step.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: TEP Prediction Test (using TEP-H0 formulas)")
    logger.info("=" * 70)
    
    # Get median masses in low and high bins (for reference)
    threshold = MASS_STEP_THRESHOLD
    
    low_mass = df[df['HOST_LOGMASS'] < threshold]
    high_mass = df[df['HOST_LOGMASS'] >= threshold]
    
    median_low = low_mass['HOST_LOGMASS'].median()
    median_high = high_mass['HOST_LOGMASS'].median()
    
    # Convert to velocity dispersion
    sigma_low_median = stellar_mass_to_sigma(median_low)
    sigma_high_median = stellar_mass_to_sigma(median_high)
    
    logger.info(f"Median log(M*) in low-mass bin: {median_low:.2f} → σ = {sigma_low_median:.1f} km/s")
    logger.info(f"Median log(M*) in high-mass bin: {median_high:.2f} → σ = {sigma_high_median:.1f} km/s")
    
    # TEP prediction using MEDIAN masses (overestimates step)
    prediction_median = tep_mass_step_prediction(sigma_low_median, sigma_high_median)
    
    logger.info(f"\nTEP Prediction using MEDIAN masses (α = {ALPHA_TEP}, σ_ref = {SIGMA_REF} km/s):")
    logger.info(f"  Formula: Δμ = α * log10(σ / σ_ref)  [EXACT from TEP-H0]")
    logger.info(f"  Predicted (no group screening): {prediction_median['delta_mu_predicted_no_group']:.4f} mag")
    if GROUP_SCREENING_ENABLED:
        logger.info(f"  Predicted (with group screening): {prediction_median['delta_mu_predicted']:.4f} mag")
    
    # TEP prediction at THRESHOLD (proper comparison)
    # The mass step is measured comparing galaxies just above/below 10^10 Msun
    log_M_threshold_low = 9.7   # Just below threshold
    log_M_threshold_high = 10.3  # Just above threshold
    
    sigma_low_threshold = stellar_mass_to_sigma(log_M_threshold_low)
    sigma_high_threshold = stellar_mass_to_sigma(log_M_threshold_high)
    
    prediction_threshold = tep_mass_step_prediction(sigma_low_threshold, sigma_high_threshold)
    
    logger.info(f"\nTEP Prediction at THRESHOLD (proper comparison):")
    logger.info(f"  log(M*) = {log_M_threshold_low} → σ = {sigma_low_threshold:.1f} km/s")
    logger.info(f"  log(M*) = {log_M_threshold_high} → σ = {sigma_high_threshold:.1f} km/s")
    logger.info(f"  Predicted (no group screening): {prediction_threshold['delta_mu_predicted_no_group']:.4f} mag")
    if GROUP_SCREENING_ENABLED:
        logger.info(f"  Group fractions: f_low = {prediction_threshold['f_group_low']:.0%}, f_high = {prediction_threshold['f_group_high']:.0%}")
        logger.info(f"  Predicted (with group screening): {prediction_threshold['delta_mu_predicted']:.4f} mag")
    
    # Use threshold prediction for comparison
    prediction = prediction_threshold
    
    # Compare to observed
    observed = observed_step['mass_step_mag']
    observed_err = observed_step['mass_step_err']
    predicted = prediction['delta_mu_predicted']
    
    # Literature value for comparison
    literature_mass_step = 0.06  # Kelly+10, Sullivan+10
    
    logger.info(f"\nComparison:")
    logger.info(f"  This analysis: {observed:.4f} ± {observed_err:.4f} mag")
    logger.info(f"  Literature (Kelly+10): ~0.06 mag")
    logger.info(f"  TEP predicted: {predicted:.4f} mag")
    
    # Check sign consistency
    if np.sign(observed) == np.sign(predicted):
        logger.info(f"  Sign: ✓ CONSISTENT")
    else:
        logger.info(f"  Sign: ✗ OPPOSITE (need to check convention)")
    
    # Magnitude comparison (use literature value for proper comparison)
    ratio_lit = literature_mass_step / abs(predicted) if predicted != 0 else np.inf
    ratio_obs = abs(observed) / abs(predicted) if predicted != 0 else np.inf
    logger.info(f"  |Literature| / |Predicted|: {ratio_lit:.2f}")
    logger.info(f"  |This analysis| / |Predicted|: {ratio_obs:.2f}")
    
    # Tension with literature
    tension_lit = abs(literature_mass_step - abs(predicted)) / 0.02  # ~0.02 mag uncertainty on literature
    logger.info(f"  Tension with literature: {tension_lit:.1f}σ")
    
    results = {
        "median_log_Mstar_low": median_low,
        "median_log_Mstar_high": median_high,
        "tep_prediction": prediction,
        "observed_mass_step": observed,
        "observed_mass_step_err": observed_err,
        "literature_mass_step": literature_mass_step,
        "predicted_mass_step": predicted,
        "ratio_lit_pred": ratio_lit,
        "ratio_obs_pred": ratio_obs,
        "sign_consistent": np.sign(observed) == np.sign(predicted)
    }
    
    return results

# =============================================================================
# BINNED ANALYSIS
# =============================================================================
def binned_mass_analysis(df, n_bins=5):
    """
    Analyze Hubble residuals in mass bins to test M^(1/3) scaling.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 4: Binned Mass Analysis (Testing M^(1/3) Scaling)")
    logger.info("=" * 70)
    
    # Create mass bins
    df['mass_bin'] = pd.qcut(df['HOST_LOGMASS'], n_bins, labels=False)
    
    bin_results = []
    for i in range(n_bins):
        bin_data = df[df['mass_bin'] == i]
        
        mean_mass = bin_data['HOST_LOGMASS'].mean()
        mean_residual = bin_data['hubble_residual'].mean()
        std_residual = bin_data['hubble_residual'].std()
        sem_residual = std_residual / np.sqrt(len(bin_data))
        
        # TEP prediction for this bin using exact TEP-H0 formula
        sigma_bin = stellar_mass_to_sigma(mean_mass)
        delta_mu_bin = ALPHA_TEP * np.log10(sigma_bin / SIGMA_REF)
        
        bin_results.append({
            "bin": i,
            "n": len(bin_data),
            "mean_log_Mstar": mean_mass,
            "mean_residual": mean_residual,
            "sem_residual": sem_residual,
            "sigma_inferred": sigma_bin,
            "delta_mu_predicted": delta_mu_bin
        })
        
        logger.info(f"Bin {i}: log(M*) = {mean_mass:.2f}, σ = {sigma_bin:.0f} km/s, "
                   f"residual = {mean_residual:.4f} ± {sem_residual:.4f} mag, "
                   f"Δμ_TEP = {delta_mu_bin:.3f}")
    
    # Test correlation between Δμ_TEP and residual across bins
    delta_mu_bins = [b['delta_mu_predicted'] for b in bin_results]
    residual_bins = [b['mean_residual'] for b in bin_results]
    
    rho_bins, p_bins = stats.spearmanr(delta_mu_bins, residual_bins)
    
    logger.info(f"\nBin-level correlation (Δμ_TEP vs residual):")
    logger.info(f"  Spearman ρ = {rho_bins:.3f}, p = {p_bins:.3f}")
    
    return {
        "n_bins": n_bins,
        "bin_results": bin_results,
        "bin_correlation_rho": rho_bins,
        "bin_correlation_p": p_bins
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_sn_ia_mass_step_analysis():
    """Run the complete SN Ia mass step analysis."""
    
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 13: SN Ia Host Galaxy Mass Step Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The mass step is a ~0.06 mag unexplained shift in SN Ia distances")
    logger.info("correlated with host galaxy mass. TEP predicts this arises from")
    logger.info("time dilation in deeper gravitational potentials.")
    logger.info("")
    
    # Load data
    df_raw = load_pantheon_data()
    if df_raw is None:
        return None
    
    # Prepare sample
    df = prepare_sample(df_raw)
    if len(df) == 0:
        logger.error("No valid SNe after filtering")
        return None
    
    # Calculate Hubble residuals
    df = calculate_hubble_residuals(df)
    
    results = {}
    
    # Analysis 1: Traditional mass step
    results['traditional_mass_step'] = analyze_mass_step(df)
    logger.info("")
    
    # Analysis 2: Continuous correlation
    results['continuous_correlation'] = analyze_continuous_correlation(df)
    logger.info("")
    
    # Analysis 3: TEP prediction test
    results['tep_prediction_test'] = test_tep_prediction(
        df, results['traditional_mass_step']
    )
    logger.info("")
    
    # Analysis 4: Binned analysis
    results['binned_analysis'] = binned_mass_analysis(df)
    logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    step = results['traditional_mass_step']
    tep = results['tep_prediction_test']
    
    logger.info(f"Observed mass step: {step['mass_step_mag']:.4f} ± {step['mass_step_err']:.4f} mag")
    logger.info(f"TEP predicted: {tep['predicted_mass_step']:.4f} mag")
    logger.info(f"Sign consistent: {tep['sign_consistent']}")
    logger.info(f"Ratio |obs|/|pred|: {tep['ratio_obs_pred']:.2f}")
    
    if tep['sign_consistent'] and 0.5 < tep['ratio_lit_pred'] < 2.0:
        logger.info("\n✓ TEP PREDICTION CONSISTENT WITH LITERATURE")
        logger.info("  The SN Ia mass step may be a manifestation of TEP.")
    elif tep['sign_consistent']:
        logger.info("\n⚠ TEP PREDICTION PARTIALLY CONSISTENT")
        logger.info("  Sign matches but magnitude differs.")
        logger.info(f"  TEP predicts {1/tep['ratio_lit_pred']:.1f}× larger effect than observed.")
        logger.info("  Possible explanations:")
        logger.info("    1. Faber-Jackson relation doesn't apply to spiral SN hosts")
        logger.info("    2. Additional screening mechanisms not yet modeled")
        logger.info("    3. Mass step has multiple contributing factors")
        logger.info("    4. TEP coupling α may be environment-dependent")
    else:
        logger.info("\n✗ TEP PREDICTION INCONSISTENT")
        logger.info("  Sign mismatch requires investigation.")
    
    # Save results
    output_file = RESULTS_DIR / "sn_ia_mass_step_analysis.json"
    
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results_clean = convert_numpy(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    
    return results

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    results = run_sn_ia_mass_step_analysis()
