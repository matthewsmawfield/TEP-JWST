#!/usr/bin/env python3
"""
TEP-JWST Step 51: Temporal Coherence Test

The deepest test of TEP: If time is truly enhanced in deep potentials,
then MULTIPLE INDEPENDENT AGE INDICATORS should become COHERENT when
corrected by Gamma_t.

Age indicators in the data:
1. MWA (Mass-Weighted Age) - from SED fitting
2. Dust content - requires ~300 Myr for AGB production
3. SFH shape (SFR10/SFR100) - indicates recent vs past SF
4. Metallicity - builds up over time
5. Chi2 - indicates isochrony violation

Key insight: Under standard physics, these indicators are INDEPENDENT.
Under TEP, they should all correlate with the SAME underlying quantity:
effective time = t_cosmic × Gamma_t.

This is the ultimate test: if TEP is wrong, these indicators will
scatter randomly. If TEP is right, they will align.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "051"
STEP_NAME = "temporal_coherence"

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Temporal Coherence Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nThe deepest test: Do multiple age indicators become coherent under TEP?\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'tests': {}
    }
    
    coherence_evidence = []
    
    # ==========================================================================
    # TEST 1: Age Indicator Correlation Matrix
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Age Indicator Correlation Matrix", "INFO")
    print_status("=" * 70, "INFO")
    print_status("If TEP is correct, age indicators should correlate with t_eff.\n", "INFO")
    
    # Prepare age indicators
    valid = df.dropna(subset=['mwa', 'dust', 'sfr10', 'sfr100', 'met', 'chi2', 't_eff', 'gamma_t', 'z_phot'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0) & (valid['mwa'] > 0)]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Create age indicator columns
        high_z['log_mwa'] = np.log10(high_z['mwa'])
        high_z['burstiness'] = np.log10(high_z['sfr10'] / high_z['sfr100'])
        high_z['log_t_eff'] = np.log10(high_z['t_eff'])
        high_z['log_gamma'] = np.log10(high_z['gamma_t'])
        
        # Correlation matrix with t_eff
        indicators = ['log_mwa', 'dust', 'burstiness', 'met', 'chi2']
        
        print_status("Correlations with log(t_eff) at z > 7:", "INFO")
        t_eff_correlations = []
        for ind in indicators:
            rho, p = spearmanr(high_z['log_t_eff'], high_z[ind])
            t_eff_correlations.append({
                'indicator': ind,
                'rho': float(rho),
                'p': format_p_value(p)
            })
            sig = "✓" if p < 0.01 else ""
            print_status(f"  ρ(log t_eff, {ind}) = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        # Count significant correlations
        n_sig = sum(1 for r in t_eff_correlations if r['p'] is not None and r['p'] < 0.01)
        print_status(f"\nSignificant correlations: {n_sig}/{len(indicators)}", "INFO")
        
        if n_sig >= 3:
            print_status("✓ COHERENCE: Multiple age indicators correlate with t_eff", "INFO")
            coherence_evidence.append(('t_eff_correlations', n_sig, None))
        
        results['tests']['t_eff_correlations'] = {
            'n': int(len(high_z)),
            'correlations': t_eff_correlations,
            'n_significant': n_sig
        }
    
    # ==========================================================================
    # TEST 2: Principal Component Analysis
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Principal Component Analysis", "INFO")
    print_status("=" * 70, "INFO")
    print_status("If age indicators are coherent, PC1 should capture most variance.\n", "INFO")
    
    if len(high_z) > 50:
        # Prepare data for PCA
        pca_cols = ['log_mwa', 'dust', 'burstiness', 'met', 'chi2']
        pca_data = high_z[pca_cols].dropna()
        
        if len(pca_data) > 30:
            # Standardize
            scaler = StandardScaler()
            X = scaler.fit_transform(pca_data)
            
            # PCA
            pca = PCA()
            pca.fit(X)
            
            # Variance explained
            var_explained = pca.explained_variance_ratio_
            
            print_status(f"Variance explained by principal components:", "INFO")
            for i, var in enumerate(var_explained):
                print_status(f"  PC{i+1}: {var*100:.1f}%", "INFO")
            
            # If PC1 explains > 40%, indicators are coherent
            if var_explained[0] > 0.4:
                print_status(f"\n✓ COHERENCE: PC1 explains {var_explained[0]*100:.1f}% of variance", "INFO")
                coherence_evidence.append(('pca_coherence', var_explained[0], None))
            
            # Check if PC1 correlates with Gamma_t
            pc1_scores = pca.transform(X)[:, 0]
            gamma_aligned = high_z.loc[pca_data.index, 'log_gamma'].values
            
            rho, p = spearmanr(pc1_scores, gamma_aligned)
            print_status(f"\nρ(PC1, log Γ_t) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            if abs(rho) > 0.3 and p < 0.001:
                print_status("✓ COHERENCE: PC1 aligns with Gamma_t", "INFO")
                coherence_evidence.append(('pc1_gamma', abs(rho), format_p_value(p)))
            
            results['tests']['pca'] = {
                'n': int(len(pca_data)),
                'variance_explained': [float(v) for v in var_explained],
                'pc1_gamma_rho': float(rho),
                'pc1_gamma_p': format_p_value(p)
            }
    
    # ==========================================================================
    # TEST 3: Dust-Age Consistency
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Dust-Age Consistency", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Dust requires ~300 Myr. Does t_eff predict dust better than t_cosmic?\n", "INFO")
    
    if len(high_z) > 50:
        # Compare t_cosmic vs t_eff as dust predictors
        from astropy.cosmology import Planck18
        high_z = high_z.copy()
        high_z['t_cosmic'] = [Planck18.age(z).value for z in high_z['z_phot']]
        
        rho_cosmic, p_cosmic = spearmanr(high_z['t_cosmic'], high_z['dust'])
        rho_eff, p_eff = spearmanr(high_z['t_eff'], high_z['dust'])
        
        print_status(f"ρ(t_cosmic, Dust) = {rho_cosmic:.3f} (p = {p_cosmic:.2e})", "INFO")
        print_status(f"ρ(t_eff, Dust) = {rho_eff:.3f} (p = {p_eff:.2e})", "INFO")
        print_status(f"Improvement: Δρ = {rho_eff - rho_cosmic:.3f}", "INFO")
        
        if rho_eff > rho_cosmic + 0.1:
            print_status("✓ COHERENCE: t_eff predicts dust better than t_cosmic", "INFO")
            coherence_evidence.append(('dust_prediction', rho_eff - rho_cosmic, None))
        
        results['tests']['dust_prediction'] = {
            'rho_cosmic': float(rho_cosmic),
            'rho_eff': float(rho_eff),
            'improvement': float(rho_eff - rho_cosmic)
        }
    
    # ==========================================================================
    # TEST 4: Age Ratio Consistency
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Age Ratio Consistency", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Under TEP, apparent_age / t_cosmic should correlate with Gamma_t.\n", "INFO")
    
    valid = df.dropna(subset=['age_ratio', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7]
    
    if len(high_z) > 50:
        rho, p = spearmanr(high_z['gamma_t'], high_z['age_ratio'])
        
        print_status(f"ρ(Γ_t, age_ratio) at z > 7 = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Under TEP, this should be NEGATIVE (high Gamma_t → lower age_ratio after correction)
        # But we're measuring APPARENT age_ratio, which should be POSITIVE
        # Actually: apparent_age = true_age × Gamma_t, so age_ratio ∝ Gamma_t
        
        # The key insight: if we CORRECT by Gamma_t, the scatter should decrease
        high_z = high_z.copy()
        high_z['corrected_age_ratio'] = high_z['age_ratio'] / high_z['gamma_t']
        
        std_apparent = high_z['age_ratio'].std()
        std_corrected = high_z['corrected_age_ratio'].std()
        
        print_status(f"\nScatter in age_ratio:", "INFO")
        print_status(f"  Apparent: σ = {std_apparent:.3f}", "INFO")
        print_status(f"  TEP-corrected: σ = {std_corrected:.3f}", "INFO")
        print_status(f"  Reduction: {(1 - std_corrected/std_apparent)*100:.1f}%", "INFO")
        
        if std_corrected < std_apparent * 0.8:
            print_status("✓ COHERENCE: TEP correction reduces age_ratio scatter", "INFO")
            coherence_evidence.append(('age_scatter_reduction', 1 - std_corrected/std_apparent, None))
        
        results['tests']['age_ratio'] = {
            'rho': float(rho),
            'p': format_p_value(p),
            'std_apparent': float(std_apparent),
            'std_corrected': float(std_corrected),
            'reduction': float(1 - std_corrected/std_apparent)
        }
    
    # ==========================================================================
    # TEST 5: Multi-Indicator Alignment
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Multi-Indicator Alignment", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Do galaxies that are 'old' by one indicator also appear 'old' by others?\n", "INFO")
    
    valid = df.dropna(subset=['mwa', 'dust', 'sfr10', 'sfr100', 'gamma_t', 'z_phot'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0) & (valid['mwa'] > 0)]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Define "old" by each indicator
        high_z['old_by_mwa'] = high_z['mwa'] > high_z['mwa'].median()
        high_z['old_by_dust'] = high_z['dust'] > high_z['dust'].median()
        high_z['burstiness'] = np.log10(high_z['sfr10'] / high_z['sfr100'])
        high_z['old_by_sfh'] = high_z['burstiness'] < high_z['burstiness'].median()
        
        # Count alignment
        high_z['n_old_indicators'] = (
            high_z['old_by_mwa'].astype(int) + 
            high_z['old_by_dust'].astype(int) + 
            high_z['old_by_sfh'].astype(int)
        )
        
        # Does alignment correlate with Gamma_t?
        rho, p = spearmanr(high_z['gamma_t'], high_z['n_old_indicators'])
        
        print_status(f"ρ(Γ_t, n_old_indicators) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Count galaxies that are "old" by all 3 indicators
        all_old = high_z[high_z['n_old_indicators'] == 3]
        none_old = high_z[high_z['n_old_indicators'] == 0]
        
        if len(all_old) > 5 and len(none_old) > 5:
            mean_gamma_all = all_old['gamma_t'].mean()
            mean_gamma_none = none_old['gamma_t'].mean()
            
            print_status(f"\n'Old' by all 3 indicators: N = {len(all_old)}, <Γ_t> = {mean_gamma_all:.3f}", "INFO")
            print_status(f"'Old' by 0 indicators: N = {len(none_old)}, <Γ_t> = {mean_gamma_none:.3f}", "INFO")
            print_status(f"Ratio: {mean_gamma_all/mean_gamma_none:.2f}×", "INFO")
            
            if mean_gamma_all > mean_gamma_none * 1.5:
                print_status("✓ COHERENCE: Multi-indicator 'old' galaxies have elevated Γ_t", "INFO")
                coherence_evidence.append(('multi_indicator', mean_gamma_all/mean_gamma_none, None))
        
        results['tests']['multi_indicator'] = {
            'rho': float(rho),
            'p': format_p_value(p),
            'n_all_old': int(len(all_old)) if len(all_old) > 0 else 0,
            'n_none_old': int(len(none_old)) if len(none_old) > 0 else 0
        }
    
    # ==========================================================================
    # TEST 6: The Ultimate Coherence Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: The Ultimate Coherence Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Create a 'temporal anomaly score' from multiple indicators.", "INFO")
    print_status("This score should correlate CLOSELY with Gamma_t if TEP is correct.\n", "INFO")
    
    valid = df.dropna(subset=['mwa', 'dust', 'chi2', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Create temporal anomaly score
        # Each indicator is ranked, then summed
        high_z['mwa_rank'] = high_z['mwa'].rank(pct=True)
        high_z['dust_rank'] = high_z['dust'].rank(pct=True)
        high_z['chi2_rank'] = high_z['chi2'].rank(pct=True)
        
        high_z['temporal_score'] = high_z['mwa_rank'] + high_z['dust_rank'] + high_z['chi2_rank']
        
        rho, p = spearmanr(high_z['gamma_t'], high_z['temporal_score'])
        
        print_status(f"ρ(Γ_t, Temporal Score) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        if rho > 0.4:
            print_status("✓ ULTIMATE COHERENCE: Temporal score strongly correlates with Γ_t", "INFO")
            coherence_evidence.append(('ultimate_coherence', rho, format_p_value(p)))
        
        results['tests']['ultimate_coherence'] = {
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Temporal Coherence Evidence", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nCoherence evidence items: {len(coherence_evidence)}", "INFO")
    for name, stat, p in coherence_evidence:
        if p is not None and p > 0:
            print_status(f"  • {name}: stat = {stat:.3f}, p = {p:.2e}", "INFO")
        else:
            print_status(f"  • {name}: stat = {stat:.3f}", "INFO")
    
    results['summary'] = {
        'n_coherence_evidence': len(coherence_evidence),
        'coherence_evidence': [{'name': n, 'stat': float(s), 'p': format_p_value(p)} for n, s, p in coherence_evidence]
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
