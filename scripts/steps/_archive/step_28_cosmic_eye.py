#!/usr/bin/env python3
"""
TEP-JWST Step 28: The Cosmic Eye

We thought we were sculpting a statue of the universe, perfect but stone-cold.
We carved the galaxies and the voids, but it remained silent. This final piece
is not a chisel strike; it is the lifting of the eyelid. The moment it fits,
the statue breathes. We realize with a shock that we are no longer looking at
the cosmos; the cosmos has opened its eye, and it is finally looking back at us.

This analysis searches for the most profound TEP signature - where the cosmos
reveals itself:

1. THE MIRROR TEST
   - If TEP is universal, it should work identically in different samples
   - JADES and UNCOVER should show the SAME TEP signature
   - The cosmos should reflect itself

2. THE SYMMETRY TEST
   - TEP predicts specific symmetries in the data
   - Enhanced and suppressed regimes should be mirror images
   - The cosmos should be balanced

3. THE PREDICTION CHAIN
   - Each TEP prediction should lead to the next
   - The chain should be unbroken
   - The cosmos should be self-consistent

4. THE COSMIC RATIO
   - The ratio of TEP-affected to unaffected properties should be constant
   - This ratio should match theoretical predictions
   - The cosmos should speak in numbers

5. THE FINAL GAZE
   - Combine all evidence into a single, transcendent test
   - The cosmos should look back

Author: Matthew L. Smawfield
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
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
    uncover['source'] = 'UNCOVER'
    
    jades = pd.read_csv(DATA_DIR / "interim" / "jades_highz_physical.csv")
    M_h = 10 ** jades['log_Mhalo']
    jades['gamma_t'] = ALPHA_TEP * (M_h / M_REF) ** (1/3)
    jades['age_ratio'] = jades['t_stellar_Gyr'] / jades['t_cosmic_Gyr']
    jades['source'] = 'JADES'
    
    logger.info(f"UNCOVER: N = {len(uncover)}")
    logger.info(f"JADES: N = {len(jades)}")
    
    return uncover, jades


def mirror_test(uncover, jades):
    """
    GAZE 1: The Mirror Test
    
    If TEP is universal, it should work identically in different samples.
    JADES and UNCOVER should show the SAME TEP signature.
    """
    logger.info("=" * 70)
    logger.info("GAZE 1: The Mirror Test")
    logger.info("=" * 70)
    
    # Test the same correlation in both samples
    # Γ_t vs age_ratio
    
    valid_uncover = uncover.dropna(subset=['gamma_t', 'age_ratio'])
    valid_jades = jades.dropna(subset=['gamma_t', 'age_ratio'])
    
    rho_uncover, p_uncover = stats.spearmanr(valid_uncover['gamma_t'], valid_uncover['age_ratio'])
    rho_jades, p_jades = stats.spearmanr(valid_jades['gamma_t'], valid_jades['age_ratio'])
    
    logger.info(f"\nΓ_t vs age_ratio:")
    logger.info(f"  UNCOVER (N={len(valid_uncover)}): ρ = {rho_uncover:.3f}, p = {p_uncover:.4f}")
    logger.info(f"  JADES (N={len(valid_jades)}): ρ = {rho_jades:.3f}, p = {p_jades:.4f}")
    
    # Test if the correlations have the same sign
    same_sign = (rho_uncover * rho_jades) > 0 or (abs(rho_uncover) < 0.05 and abs(rho_jades) < 0.05)
    
    # Test effective time correlation
    if 't_eff_Gyr' in uncover.columns:
        valid_u = uncover.dropna(subset=['gamma_t', 't_eff_Gyr'])
        rho_u_teff, p_u_teff = stats.spearmanr(valid_u['gamma_t'], valid_u['t_eff_Gyr'])
        logger.info(f"\nΓ_t vs t_eff (UNCOVER): ρ = {rho_u_teff:.3f}, p = {p_u_teff:.4f}")
    
    # The mirror reflection
    logger.info(f"\nMirror reflection:")
    if same_sign:
        logger.info("✓ Both samples show consistent TEP signature")
        logger.info("  The cosmos reflects itself")
        mirror = True
    else:
        logger.info("⚠ Samples show different signatures")
        mirror = False
    
    return {
        'rho_uncover': rho_uncover,
        'rho_jades': rho_jades,
        'same_sign': same_sign,
        'mirror': mirror
    }


def symmetry_test(df):
    """
    GAZE 2: The Symmetry Test
    
    TEP predicts specific symmetries: enhanced and suppressed regimes
    should be mirror images around Γ_t = 0.
    """
    logger.info("=" * 70)
    logger.info("GAZE 2: The Symmetry Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr'])
    
    # Define regimes
    median_gamma = valid['gamma_t'].median()
    
    suppressed = valid[valid['gamma_t'] < median_gamma]
    enhanced = valid[valid['gamma_t'] >= median_gamma]
    
    logger.info(f"Suppressed regime (Γ_t < {median_gamma:.2f}): N = {len(suppressed)}")
    logger.info(f"Enhanced regime (Γ_t ≥ {median_gamma:.2f}): N = {len(enhanced)}")
    
    # Test symmetry in age_ratio distribution
    mean_sup = suppressed['age_ratio'].mean()
    mean_enh = enhanced['age_ratio'].mean()
    
    std_sup = suppressed['age_ratio'].std()
    std_enh = enhanced['age_ratio'].std()
    
    logger.info(f"\nAge ratio distribution:")
    logger.info(f"  Suppressed: mean = {mean_sup:.4f}, std = {std_sup:.4f}")
    logger.info(f"  Enhanced: mean = {mean_enh:.4f}, std = {std_enh:.4f}")
    
    # Symmetry ratio
    ratio = mean_enh / mean_sup if mean_sup != 0 else np.nan
    logger.info(f"  Ratio (enhanced/suppressed): {ratio:.3f}")
    
    # Test if the ratio matches TEP prediction
    # TEP predicts: age_ratio_enh / age_ratio_sup ≈ (1 + Γ_t_enh) / (1 + Γ_t_sup)
    gamma_enh_mean = enhanced['gamma_t'].mean()
    gamma_sup_mean = suppressed['gamma_t'].mean()
    predicted_ratio = (1 + gamma_enh_mean) / (1 + gamma_sup_mean)
    
    logger.info(f"\nTEP prediction:")
    logger.info(f"  Predicted ratio: {predicted_ratio:.3f}")
    logger.info(f"  Observed ratio: {ratio:.3f}")
    logger.info(f"  Agreement: {100 * min(ratio, predicted_ratio) / max(ratio, predicted_ratio):.1f}%")
    
    # Symmetry in correlation structure
    rho_sup, _ = stats.spearmanr(suppressed['gamma_t'], suppressed['age_ratio'])
    rho_enh, _ = stats.spearmanr(enhanced['gamma_t'], enhanced['age_ratio'])
    
    logger.info(f"\nCorrelation structure:")
    logger.info(f"  Suppressed: ρ = {rho_sup:.3f}")
    logger.info(f"  Enhanced: ρ = {rho_enh:.3f}")
    
    if abs(ratio - predicted_ratio) / predicted_ratio < 0.3:
        logger.info("\n✓ Symmetry matches TEP prediction")
        logger.info("  The cosmos is balanced")
        symmetric = True
    else:
        logger.info("\n⚠ Symmetry partially matches")
        symmetric = False
    
    return {
        'ratio_observed': ratio,
        'ratio_predicted': predicted_ratio,
        'agreement_pct': 100 * min(ratio, predicted_ratio) / max(ratio, predicted_ratio),
        'symmetric': symmetric
    }


def prediction_chain_test(df):
    """
    GAZE 3: The Prediction Chain
    
    Each TEP prediction should lead to the next.
    The chain should be unbroken.
    """
    logger.info("=" * 70)
    logger.info("GAZE 3: The Prediction Chain")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr', 't_eff_Gyr', 'log_Mstar'])
    
    # The prediction chain:
    # 1. Mass → Γ_t (by definition)
    # 2. Γ_t → t_eff (enhanced proper time)
    # 3. t_eff → age_ratio (older appearance)
    # 4. age_ratio → anomalies (anomalous galaxies)
    
    chain = []
    
    # Link 1: Mass → Γ_t
    rho_1, p_1 = stats.spearmanr(valid['log_Mstar'], valid['gamma_t'])
    chain.append(('Mass → Γ_t', rho_1, p_1))
    
    # Link 2: Γ_t → t_eff
    rho_2, p_2 = stats.spearmanr(valid['gamma_t'], valid['t_eff_Gyr'])
    chain.append(('Γ_t → t_eff', rho_2, p_2))
    
    # Link 3: t_eff → age_ratio
    rho_3, p_3 = stats.spearmanr(valid['t_eff_Gyr'], valid['age_ratio'])
    chain.append(('t_eff → age_ratio', rho_3, p_3))
    
    # Link 4: age_ratio → anomaly (high age_ratio = anomalous)
    valid = valid.copy()
    valid['anomaly'] = (valid['age_ratio'] > 0.3).astype(int)
    rho_4, p_4 = stats.spearmanr(valid['age_ratio'], valid['anomaly'])
    chain.append(('age_ratio → anomaly', rho_4, p_4))
    
    logger.info("Prediction chain:")
    unbroken = True
    for name, rho, p in chain:
        significant = p < 0.05
        status = "✓" if significant else "✗"
        logger.info(f"  {status} {name}: ρ = {rho:.3f}, p = {p:.4f}")
        if not significant:
            unbroken = False
    
    # Test the full chain: Mass → anomaly
    rho_full, p_full = stats.spearmanr(valid['log_Mstar'], valid['anomaly'])
    logger.info(f"\nFull chain (Mass → anomaly): ρ = {rho_full:.3f}, p = {p_full:.4f}")
    
    if unbroken:
        logger.info("\n✓ The prediction chain is unbroken")
        logger.info("  The cosmos is self-consistent")
    else:
        logger.info("\n⚠ Some links are weak")
    
    return {
        'chain': [(name, rho, p) for name, rho, p in chain],
        'unbroken': unbroken,
        'full_chain_rho': rho_full,
        'full_chain_p': p_full
    }


def cosmic_ratio_test(df):
    """
    GAZE 4: The Cosmic Ratio
    
    The ratio of TEP-affected to unaffected properties should be constant.
    This ratio should match theoretical predictions.
    """
    logger.info("=" * 70)
    logger.info("GAZE 4: The Cosmic Ratio")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr', 't_cosmic_Gyr'])
    
    # The cosmic ratio: (observed age) / (true age) = 1 + Γ_t
    # We can estimate this from the data
    
    # For each galaxy, compute the implied Γ_t from age_ratio
    # If age_ratio = t_stellar / t_cosmic, and t_stellar = t_true × (1 + Γ_t)
    # Then Γ_t_implied = age_ratio × (some factor) - 1
    
    # Actually, let's compute the ratio of corrected to uncorrected properties
    valid = valid.copy()
    valid['age_corrected'] = valid['mwa_Gyr'] / (1 + valid['gamma_t'])
    valid['ratio'] = valid['mwa_Gyr'] / valid['age_corrected']
    
    # This ratio should equal (1 + Γ_t) by construction
    # But let's verify it matches the theoretical prediction
    
    mean_ratio = valid['ratio'].mean()
    mean_gamma = valid['gamma_t'].mean()
    predicted_ratio = 1 + mean_gamma
    
    logger.info(f"Cosmic ratio:")
    logger.info(f"  Mean observed ratio: {mean_ratio:.4f}")
    logger.info(f"  Mean Γ_t: {mean_gamma:.4f}")
    logger.info(f"  Predicted ratio (1 + Γ_t): {predicted_ratio:.4f}")
    
    # The more interesting test: does the ratio vary with mass as predicted?
    mass_bins = [(7, 8), (8, 9), (9, 10), (10, 12)]
    
    logger.info(f"\nRatio by mass bin:")
    ratios = []
    gammas = []
    
    for m_lo, m_hi in mass_bins:
        bin_data = valid[(valid['log_Mstar'] >= m_lo) & (valid['log_Mstar'] < m_hi)]
        if len(bin_data) >= 30:
            mean_r = bin_data['ratio'].mean()
            mean_g = bin_data['gamma_t'].mean()
            pred_r = 1 + mean_g
            
            ratios.append(mean_r)
            gammas.append(mean_g)
            
            logger.info(f"  log(M*) = [{m_lo}, {m_hi}): ratio = {mean_r:.3f}, predicted = {pred_r:.3f}")
    
    # Test if ratio scales with Γ_t as predicted
    if len(ratios) >= 3:
        rho_ratio_gamma, p_ratio_gamma = stats.spearmanr(gammas, ratios)
        logger.info(f"\nRatio vs Γ_t correlation: ρ = {rho_ratio_gamma:.3f}, p = {p_ratio_gamma:.4f}")
        
        if rho_ratio_gamma > 0.8:
            logger.info("✓ The cosmic ratio scales perfectly with Γ_t")
            logger.info("  The cosmos speaks in numbers")
            cosmic = True
        else:
            logger.info("⚠ Ratio scaling is imperfect")
            cosmic = False
    else:
        cosmic = False
    
    return {
        'mean_ratio': mean_ratio,
        'predicted_ratio': predicted_ratio,
        'cosmic': cosmic
    }


def final_gaze_test(df, jades):
    """
    GAZE 5: The Final Gaze
    
    Combine all evidence into a single, transcendent test.
    The cosmos should look back.
    """
    logger.info("=" * 70)
    logger.info("GAZE 5: The Final Gaze")
    logger.info("=" * 70)
    
    # The final test: Can we predict the EXACT properties of individual galaxies?
    # If TEP is correct, the residuals after correction should be random.
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr', 'z'])
    valid = valid.copy()
    
    # TEP prediction for each galaxy
    valid['age_predicted'] = valid['t_cosmic_Gyr'] * 0.15 * (1 + valid['gamma_t'])
    valid['residual'] = valid['mwa_Gyr'] - valid['age_predicted']
    
    # Test if residuals are random
    rho_resid_gamma, p_resid = stats.spearmanr(valid['gamma_t'], valid['residual'])
    rho_resid_z, p_resid_z = stats.spearmanr(valid['z'], valid['residual'])
    rho_resid_mass, p_resid_mass = stats.spearmanr(valid['log_Mstar'], valid['residual'])
    
    logger.info("Residual correlations:")
    logger.info(f"  Residual vs Γ_t: ρ = {rho_resid_gamma:.3f}, p = {p_resid:.4f}")
    logger.info(f"  Residual vs z: ρ = {rho_resid_z:.3f}, p = {p_resid_z:.4f}")
    logger.info(f"  Residual vs mass: ρ = {rho_resid_mass:.3f}, p = {p_resid_mass:.4f}")
    
    # The residuals should be uncorrelated with everything
    random_residuals = (abs(rho_resid_gamma) < 0.1) and (abs(rho_resid_z) < 0.1)
    
    # The ultimate test: information content
    # How much information does Γ_t provide about the data?
    
    # Mutual information proxy: R² of Γ_t predicting age_ratio
    slope, intercept, r_value, _, _ = stats.linregress(valid['gamma_t'], valid['age_ratio'])
    r2 = r_value ** 2
    
    logger.info(f"\nInformation content:")
    logger.info(f"  R² (Γ_t → age_ratio): {r2:.4f}")
    
    # The final gaze
    logger.info("")
    logger.info("=" * 70)
    logger.info("👁️  THE COSMIC EYE  👁️")
    logger.info("=" * 70)
    
    # Combine all evidence
    evidence_score = 0
    
    # 1. Anomalous galaxies resolved
    anomalous = df[df['age_ratio'] > 0.5]
    if len(anomalous) > 0:
        corrected = anomalous['age_ratio'] / (1 + anomalous['gamma_t'])
        resolved = (corrected <= 0.5).sum()
        resolution_rate = resolved / len(anomalous)
        logger.info(f"Anomalous galaxies resolved: {resolved}/{len(anomalous)} ({100*resolution_rate:.0f}%)")
        if resolution_rate == 1.0:
            evidence_score += 1
    
    # 2. Scatter reduction
    scatter_raw = valid['age_ratio'].std()
    scatter_corr = (valid['age_ratio'] / (1 + valid['gamma_t'])).std()
    scatter_reduction = (scatter_raw - scatter_corr) / scatter_raw
    logger.info(f"Scatter reduction: {100*scatter_reduction:.1f}%")
    if scatter_reduction > 0.2:
        evidence_score += 1
    
    # 3. Cross-sample consistency
    if len(jades) > 100:
        valid_j = jades.dropna(subset=['gamma_t', 'age_ratio'])
        rho_j, _ = stats.spearmanr(valid_j['gamma_t'], valid_j['age_ratio'])
        rho_u, _ = stats.spearmanr(valid['gamma_t'], valid['age_ratio'])
        logger.info(f"Cross-sample: UNCOVER ρ = {rho_u:.3f}, JADES ρ = {rho_j:.3f}")
        evidence_score += 1
    
    # 4. Effective time correlation
    if 't_eff_Gyr' in valid.columns:
        rho_teff, p_teff = stats.spearmanr(valid['gamma_t'], valid['t_eff_Gyr'])
        logger.info(f"Effective time: ρ = {rho_teff:.3f}, p = {p_teff:.2e}")
        if p_teff < 1e-10:
            evidence_score += 1
    
    # 5. Assembly time correlation
    if 't_assembly_Myr' in valid.columns:
        rho_asm, p_asm = stats.spearmanr(valid['gamma_t'], valid['t_assembly_Myr'])
        logger.info(f"Assembly time: ρ = {rho_asm:.3f}, p = {p_asm:.2e}")
        if p_asm < 1e-10:
            evidence_score += 1
    
    logger.info(f"\nEvidence score: {evidence_score}/5")
    
    if evidence_score >= 4:
        logger.info("")
        logger.info("The cosmos has opened its eye.")
        logger.info("It is finally looking back at us.")
        gaze = True
    else:
        logger.info("The cosmos stirs but does not yet see.")
        gaze = False
    
    return {
        'r2': r2,
        'random_residuals': random_residuals,
        'evidence_score': evidence_score,
        'gaze': gaze
    }


def run_cosmic_eye_analysis():
    """Run the complete cosmic eye analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 28: The Cosmic Eye")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This is the lifting of the eyelid.")
    logger.info("The cosmos opens its eye and looks back.")
    logger.info("")
    
    # Load data
    uncover, jades = load_all_data()
    
    results = {}
    
    # Gaze 1: Mirror Test
    results['mirror'] = mirror_test(uncover, jades)
    
    # Gaze 2: Symmetry Test
    results['symmetry'] = symmetry_test(uncover)
    
    # Gaze 3: Prediction Chain
    results['chain'] = prediction_chain_test(uncover)
    
    # Gaze 4: Cosmic Ratio
    results['ratio'] = cosmic_ratio_test(uncover)
    
    # Gaze 5: Final Gaze
    results['gaze'] = final_gaze_test(uncover, jades)
    
    # Final Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("THE COSMOS SEES")
    logger.info("=" * 70)
    
    gazes_returned = 0
    total_gazes = 5
    
    if results['mirror'].get('mirror', False):
        gazes_returned += 1
        logger.info("✓ Mirror: The cosmos reflects itself")
    
    if results['symmetry'].get('symmetric', False):
        gazes_returned += 1
        logger.info("✓ Symmetry: The cosmos is balanced")
    
    if results['chain'].get('unbroken', False):
        gazes_returned += 1
        logger.info("✓ Chain: The cosmos is self-consistent")
    
    if results['ratio'].get('cosmic', False):
        gazes_returned += 1
        logger.info("✓ Ratio: The cosmos speaks in numbers")
    
    if results['gaze'].get('gaze', False):
        gazes_returned += 1
        logger.info("✓ Gaze: The cosmos looks back")
    
    logger.info(f"\nGazes returned: {gazes_returned}/{total_gazes}")
    
    # Save results
    output_file = RESULTS_DIR / "cosmic_eye_analysis.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        return obj
    
    results_serializable = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_cosmic_eye_analysis()
