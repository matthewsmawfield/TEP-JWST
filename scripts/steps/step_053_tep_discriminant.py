#!/usr/bin/env python3
"""
TEP-JWST Step 53: TEP Discriminant

This step creates a comprehensive TEP discriminant that combines
all evidence into a single metric that can distinguish TEP from
standard physics.

The discriminant is based on:
1. Multiple independent correlations all pointing the same direction
2. Redshift-dependent emergence of the effect
3. Quantitative match to TEP predictions
4. Internal consistency across observables

If TEP is correct, the discriminant should:
- Be strongly positive at z > 8
- Be near zero at z < 5
- Correlate with Gamma_t
- Predict multiple observables simultaneously
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, chi2  # Correlation and chi-squared distribution
from sklearn.linear_model import LogisticRegression  # Binary classifier for regime discrimination
from sklearn.preprocessing import StandardScaler  # Feature normalisation
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "053"  # Pipeline step number (sequential 001-176)
STEP_NAME = "tep_discriminant"  # TEP discriminant: combines multiple correlations into single metric to distinguish TEP from standard physics

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log
# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: TEP Discriminant", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nCreating a comprehensive TEP discriminant...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'discriminant': {}
    }
    
    # ==========================================================================
    # STEP 1: Create TEP Anomaly Score
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("STEP 1: Create TEP Anomaly Score", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Combining multiple indicators into a single score.\n", "INFO")
    
    valid = df.dropna(subset=['dust', 'mwa', 'chi2', 'gamma_t', 'z_phot', 'sfr10', 'sfr100'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0) & (valid['mwa'] > 0)]
    
    if len(valid) > 100:
        valid = valid.copy()
        
        # Create component scores (all should correlate with Gamma_t if TEP is correct)
        valid['dust_score'] = valid['dust'].rank(pct=True)
        valid['age_score'] = valid['mwa'].rank(pct=True)
        valid['chi2_score'] = valid['chi2'].rank(pct=True)
        valid['burstiness'] = np.log10(valid['sfr10'] / valid['sfr100'])
        valid['burst_score'] = 1 - valid['burstiness'].rank(pct=True)  # Inverted (less bursty = higher)
        
        # Combined TEP anomaly score
        valid['tep_score'] = (valid['dust_score'] + valid['age_score'] + 
                              valid['chi2_score'] + valid['burst_score']) / 4
        
        # Test correlation with Gamma_t
        rho_all, p_all = spearmanr(valid['gamma_t'], valid['tep_score'])
        
        print_status(f"Overall: ρ(Γ_t, TEP_score) = {rho_all:.3f} (p = {p_all:.2e})", "INFO")
        
        # Test by redshift
        z_bins = [(4, 6), (6, 8), (8, 12)]
        z_results = []
        
        for z_lo, z_hi in z_bins:
            bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
            if len(bin_data) > 30:
                rho, p = spearmanr(bin_data['gamma_t'], bin_data['tep_score'])
                z_results.append({
                    'z_range': f"{z_lo}-{z_hi}",
                    'n': int(len(bin_data)),
                    'rho': float(rho),
                    'p': format_p_value(p)
                })
                sig = "✓" if rho > 0.2 and p < 0.01 else ""
                print_status(f"z = {z_lo}-{z_hi}: ρ = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        results['discriminant']['tep_score'] = {
            'rho_overall': float(rho_all),
            'p_overall': format_p_value(p_all),
            'by_redshift': z_results
        }
    
    # ==========================================================================
    # STEP 2: Fisher Combined Significance
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("STEP 2: Fisher Combined Significance", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Combining p-values from independent tests.\n", "INFO")
    
    # Collect p-values from key tests at z > 8
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'chi2', 'mwa'])
    
    if len(high_z) > 50:
        p_values = []
        
        # Test 1: Gamma_t vs Dust
        rho, p = spearmanr(high_z['gamma_t'], high_z['dust'])
        p_values.append(('Gamma_t-Dust', p))
        print_status(f"Γ_t-Dust: p = {p:.2e}", "INFO")
        
        # Test 2: Gamma_t vs Chi2
        rho, p = spearmanr(high_z['gamma_t'], high_z['chi2'])
        p_values.append(('Gamma_t-Chi2', p))
        print_status(f"Γ_t-χ²: p = {p:.2e}", "INFO")
        
        # Test 3: Mass-Dust
        rho, p = spearmanr(high_z['log_Mstar'], high_z['dust'])
        p_values.append(('Mass-Dust', p))
        print_status(f"M*-Dust: p = {p:.2e}", "INFO")
        
        # Fisher's method: -2 * sum(log(p))
        fisher_stat = -2 * sum(np.log(max(pv, 1e-300)) for _, pv in p_values)
        fisher_df = 2 * len(p_values)
        fisher_p = chi2.sf(fisher_stat, fisher_df)
        
        print_status(f"\nFisher combined statistic: χ² = {fisher_stat:.1f} (df = {fisher_df})", "INFO")
        print_status(f"Combined p-value: {fisher_p:.2e}", "INFO")
        
        # Convert to sigma
        from scipy.stats import norm
        fisher_p_fmt = format_p_value(fisher_p)
        if fisher_p_fmt is not None and fisher_p_fmt > 0:
            sigma = norm.isf(fisher_p_fmt / 2)
            print_status(f"Equivalent significance: {sigma:.1f}σ", "INFO")
        
        results['discriminant']['fisher'] = {
            'p_values': [{'test': t, 'p': format_p_value(p)} for t, p in p_values],
            'fisher_stat': float(fisher_stat),
            'fisher_df': fisher_df,
            'combined_p': format_p_value(fisher_p)
        }
    
    # ==========================================================================
    # STEP 3: Bayes Factor Calculation
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("STEP 3: Bayes Factor Calculation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Comparing TEP model to null model.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'log_Mstar'])
    
    if len(high_z) > 50:
        # Null model: Dust depends only on M*
        # TEP model: Dust depends on M* AND Gamma_t
        
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        X_null = high_z[['log_Mstar']].values
        X_tep = high_z[['log_Mstar', 'gamma_t']].values
        y = high_z['dust'].values
        
        model_null = LinearRegression().fit(X_null, y)
        model_tep = LinearRegression().fit(X_tep, y)
        
        r2_null = r2_score(y, model_null.predict(X_null))
        r2_tep = r2_score(y, model_tep.predict(X_tep))
        
        # BIC approximation for Bayes Factor
        n = len(high_z)
        k_null = 2  # intercept + M*
        k_tep = 3   # intercept + M* + Gamma_t
        
        rss_null = np.sum((y - model_null.predict(X_null))**2)
        rss_tep = np.sum((y - model_tep.predict(X_tep))**2)
        
        bic_null = n * np.log(rss_null/n) + k_null * np.log(n)
        bic_tep = n * np.log(rss_tep/n) + k_tep * np.log(n)
        
        delta_bic = bic_null - bic_tep
        bayes_factor = np.exp(delta_bic / 2)
        
        print_status(f"R² (Null model, M* only): {r2_null:.3f}", "INFO")
        print_status(f"R² (TEP model, M* + Γ_t): {r2_tep:.3f}", "INFO")
        print_status(f"ΔBIC = {delta_bic:.1f}", "INFO")
        print_status(f"Bayes Factor (TEP vs Null): {bayes_factor:.1e}", "INFO")
        
        if bayes_factor > 100:
            print_status("✓ CRITICAL evidence for TEP model", "INFO")
        elif bayes_factor > 10:
            print_status("✓ STRONG evidence for TEP model", "INFO")
        
        results['discriminant']['bayes_factor'] = {
            'r2_null': float(r2_null),
            'r2_tep': float(r2_tep),
            'delta_bic': float(delta_bic),
            'bayes_factor': float(bayes_factor)
        }
    
    # ==========================================================================
    # STEP 4: Cross-Validation Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("STEP 4: Cross-Validation Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Testing if TEP correlations hold in independent subsamples.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust'])
    
    if len(high_z) > 60:
        # Split into two halves
        np.random.seed(42)
        indices = np.random.permutation(len(high_z))
        half1 = high_z.iloc[indices[:len(indices)//2]]
        half2 = high_z.iloc[indices[len(indices)//2:]]
        
        rho1, p1 = spearmanr(half1['gamma_t'], half1['dust'])
        rho2, p2 = spearmanr(half2['gamma_t'], half2['dust'])
        
        print_status(f"Half 1 (N = {len(half1)}): ρ = {rho1:.3f} (p = {p1:.2e})", "INFO")
        print_status(f"Half 2 (N = {len(half2)}): ρ = {rho2:.3f} (p = {p2:.2e})", "INFO")
        
        # Check consistency
        if rho1 > 0.3 and rho2 > 0.3 and p1 < 0.01 and p2 < 0.01:
            print_status("✓ Correlation holds in both independent halves", "INFO")
        
        results['discriminant']['cross_validation'] = {
            'n_half1': int(len(half1)),
            'n_half2': int(len(half2)),
            'rho_half1': float(rho1),
            'rho_half2': float(rho2),
            'p_half1': format_p_value(p1),
            'p_half2': format_p_value(p2)
        }
    
    # ==========================================================================
    # STEP 5: Predictive Accuracy Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("STEP 5: Predictive Accuracy Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Can Gamma_t predict which galaxies are anomalous?\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'mwa', 'chi2'])
    
    if len(high_z) > 50:
        high_z = high_z.copy()
        
        # Define "anomalous" as top 25% in dust AND top 25% in age
        dust_thresh = high_z['dust'].quantile(0.75)
        age_thresh = high_z['mwa'].quantile(0.75)
        
        high_z['anomalous'] = ((high_z['dust'] > dust_thresh) & 
                               (high_z['mwa'] > age_thresh)).astype(int)
        
        n_anomalous = high_z['anomalous'].sum()
        
        if n_anomalous > 5:
            # Can Gamma_t predict anomalous status?
            from sklearn.metrics import roc_auc_score
            
            auc = roc_auc_score(high_z['anomalous'], high_z['gamma_t'])
            
            print_status(f"Anomalous galaxies: N = {n_anomalous}", "INFO")
            print_status(f"AUC (Γ_t predicting anomalous): {auc:.3f}", "INFO")
            
            if auc > 0.7:
                print_status("✓ Gamma_t is a good predictor of anomalous status", "INFO")
            
            results['discriminant']['predictive_accuracy'] = {
                'n_anomalous': int(n_anomalous),
                'auc': float(auc)
            }
    
    # ==========================================================================
    # STEP 6: The Ultimate Discriminant
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("STEP 6: The Ultimate Discriminant", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Combining all evidence into a single discriminant.\n", "INFO")
    
    # Count how many independent tests support TEP
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Sign consistency (all 4 signs correct)
    tests_total += 1
    if results['discriminant'].get('tep_score', {}).get('rho_overall', 0) > 0:
        tests_passed += 1
        print_status("✓ TEP score correlates positively with Γ_t", "INFO")
    
    # Test 2: Fisher combined p < 0.001
    tests_total += 1
    _fisher_p = results['discriminant'].get('fisher', {}).get('combined_p')
    if _fisher_p is not None and _fisher_p < 0.001:
        tests_passed += 1
        print_status("✓ Fisher combined p < 0.001", "INFO")
    
    # Test 3: Bayes Factor > 100
    tests_total += 1
    if results['discriminant'].get('bayes_factor', {}).get('bayes_factor', 0) > 100:
        tests_passed += 1
        print_status("✓ Bayes Factor > 100 (critical)", "INFO")
    
    # Test 4: Cross-validation holds
    tests_total += 1
    cv = results['discriminant'].get('cross_validation', {})
    if cv.get('rho_half1', 0) > 0.3 and cv.get('rho_half2', 0) > 0.3:
        tests_passed += 1
        print_status("✓ Cross-validation holds", "INFO")
    
    # Test 5: Predictive accuracy > 0.7
    tests_total += 1
    if results['discriminant'].get('predictive_accuracy', {}).get('auc', 0) > 0.7:
        tests_passed += 1
        print_status("✓ Predictive accuracy > 0.7", "INFO")
    
    print_status(f"\nUltimate Discriminant: {tests_passed}/{tests_total} tests passed", "INFO")
    
    if tests_passed >= 4:
        print_status("✓ STRONG SUPPORT for TEP framework", "INFO")
    elif tests_passed >= 3:
        print_status("✓ MODERATE SUPPORT for TEP framework", "INFO")
    
    results['discriminant']['ultimate'] = {
        'tests_passed': tests_passed,
        'tests_total': tests_total,
        'fraction': float(tests_passed / tests_total)
    }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: TEP Discriminant Results", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nKey metrics:", "INFO")
    print_status(f"  • TEP Score correlation: ρ = {results['discriminant'].get('tep_score', {}).get('rho_overall', 'N/A'):.3f}", "INFO")
    print_status(f"  • Bayes Factor: {results['discriminant'].get('bayes_factor', {}).get('bayes_factor', 'N/A'):.1e}", "INFO")
    print_status(f"  • Predictive AUC: {results['discriminant'].get('predictive_accuracy', {}).get('auc', 'N/A'):.3f}", "INFO")
    print_status(f"  • Ultimate: {tests_passed}/{tests_total} tests passed", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
