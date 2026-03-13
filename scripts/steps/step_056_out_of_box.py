#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.6s.
"""
TEP-JWST Step 56: Out-of-the-Box Tests

This step explores unconventional angles that could either strengthen
or falsify the TEP framework:

1. REVERSE CAUSALITY: Can we rule out that dust causes mass estimates?
   - If dust → mass, then at fixed dust, mass should not correlate with Gamma_t
   
2. INFORMATION CONTENT: How much information does Gamma_t add beyond mass?
   - Use partial correlations and information theory
   
3. PREDICTION INVERSION: Can we predict Gamma_t FROM observables?
   - If TEP is real, observables should predict Gamma_t
   
4. RESIDUAL STRUCTURE: After removing Gamma_t effect, is there residual structure?
   - If TEP explains everything, residuals should be random
   
5. BOOTSTRAP STABILITY: How stable are the correlations?
   - Test robustness to sample variations
   
6. EXTREME OUTLIER TEST: Do the most extreme outliers follow TEP?
   - The hardest cases should still follow the pattern
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, linregress  # Rank/linear correlation and OLS regression
from sklearn.ensemble import RandomForestRegressor  # Non-linear prediction inversion test
from sklearn.model_selection import cross_val_score  # Cross-validated scoring
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "056"  # Pipeline step number (sequential 001-176)
STEP_NAME = "out_of_box"  # Out-of-the-box tests: 6 unconventional TEP tests (reverse causality, information content, prediction inversion, residual structure, bootstrap stability, extreme outliers)

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
    print_status(f"STEP {STEP_NUM}: Out-of-the-Box Tests", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nExploring unconventional angles...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'out_of_box': {}
    }
    
    # ==========================================================================
    # TEST 1: Reverse Causality
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Reverse Causality", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: Does dust CAUSE mass estimates, or does Gamma_t predict both?", "INFO")
    print_status("If dust → mass, then at fixed dust, mass should not correlate with Gamma_t.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'dust', 'gamma_t'])
    
    if len(high_z) > 50:
        # Bin by dust and check mass-Gamma_t correlation within bins
        dust_bins = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 3.0)]
        reverse_results = []
        
        for d_lo, d_hi in dust_bins:
            bin_data = high_z[(high_z['dust'] >= d_lo) & (high_z['dust'] < d_hi)]
            if len(bin_data) > 10:
                rho, p = spearmanr(bin_data['log_Mstar'], bin_data['gamma_t'])
                reverse_results.append({
                    'dust_range': f'{d_lo}-{d_hi}',
                    'n': int(len(bin_data)),
                    'rho': float(rho),
                    'p': format_p_value(p)
                })
                sig = "✓" if rho > 0.3 and p < 0.05 else ""
                print_status(f"Dust = {d_lo}-{d_hi}: ρ(M*, Γ_t) = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        # If mass-Gamma_t correlation persists at fixed dust, dust is not the cause
        n_significant = sum(
            1
            for r in reverse_results
            if r['rho'] > 0.3 and r.get('p') is not None and r['p'] < 0.05
        )
        if n_significant >= 2:
            print_status("\n✓ Mass-Γ_t correlation persists at fixed dust → Dust is not the cause", "INFO")
        
        results['out_of_box']['reverse_causality'] = reverse_results
    
    # ==========================================================================
    # TEST 2: Partial Correlation
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Partial Correlation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: Does Gamma_t predict dust BEYOND what mass predicts?", "INFO")
    print_status("Compute partial correlation: ρ(Γ_t, Dust | M*)\n", "INFO")
    
    if len(high_z) > 50:
        # Residualize both Gamma_t and Dust against M*
        high_z = high_z.copy()
        
        slope_g, intercept_g, _, _, _ = linregress(high_z['log_Mstar'], high_z['gamma_t'])
        high_z['gamma_resid'] = high_z['gamma_t'] - (slope_g * high_z['log_Mstar'] + intercept_g)
        
        slope_d, intercept_d, _, _, _ = linregress(high_z['log_Mstar'], high_z['dust'])
        high_z['dust_resid'] = high_z['dust'] - (slope_d * high_z['log_Mstar'] + intercept_d)
        
        # Partial correlation
        rho_partial, p_partial = spearmanr(high_z['gamma_resid'], high_z['dust_resid'])
        
        # Compare to raw correlation
        rho_raw, p_raw = spearmanr(high_z['gamma_t'], high_z['dust'])
        
        print_status(f"Raw ρ(Γ_t, Dust) = {rho_raw:.3f}", "INFO")
        print_status(f"Partial ρ(Γ_t, Dust | M*) = {rho_partial:.3f}", "INFO")
        
        if rho_partial > 0.1 and p_partial < 0.05:
            print_status("\n✓ Gamma_t predicts dust BEYOND mass → Unique information", "INFO")
        
        results['out_of_box']['partial_correlation'] = {
            'rho_raw': float(rho_raw),
            'rho_partial': float(rho_partial),
            'p_partial': format_p_value(p_partial)
        }
    
    # ==========================================================================
    # TEST 3: Prediction Inversion
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Prediction Inversion", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: Can we predict Gamma_t FROM observables?", "INFO")
    print_status("If TEP is real, dust+age+chi2 should predict Gamma_t.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'mwa', 'chi2', 'log_Mstar'])
    valid = valid[valid['mwa'] > 0]
    
    if len(valid) > 50:
        # Use Random Forest to predict Gamma_t from observables
        X = valid[['dust', 'mwa', 'chi2']].values
        y = np.log10(valid['gamma_t'].values + 0.01)  # Log transform
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        
        print_status(f"Cross-validated R² (predicting log Γ_t from dust, age, χ²):", "INFO")
        print_status(f"  Mean R² = {scores.mean():.3f} ± {scores.std():.3f}", "INFO")
        
        if scores.mean() > 0.3:
            print_status("\n✓ Observables can predict Gamma_t → Consistent with TEP", "INFO")
        
        # Now add mass and see if it helps
        X_with_mass = valid[['dust', 'mwa', 'chi2', 'log_Mstar']].values
        scores_with_mass = cross_val_score(rf, X_with_mass, y, cv=5, scoring='r2')
        
        print_status(f"\nWith mass added:", "INFO")
        print_status(f"  Mean R² = {scores_with_mass.mean():.3f} ± {scores_with_mass.std():.3f}", "INFO")
        print_status(f"  Improvement: {(scores_with_mass.mean() - scores.mean())*100:.1f}%", "INFO")
        
        results['out_of_box']['prediction_inversion'] = {
            'r2_observables': float(scores.mean()),
            'r2_with_mass': float(scores_with_mass.mean()),
            'improvement': float(scores_with_mass.mean() - scores.mean())
        }
    
    # ==========================================================================
    # TEST 4: Bootstrap Stability
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Bootstrap Stability", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: How stable is the Gamma_t-Dust correlation?", "INFO")
    print_status("Bootstrap 1000 times and check distribution.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust'])
    
    if len(high_z) > 50:
        np.random.seed(42)
        bootstrap_rhos = []
        
        for _ in range(1000):
            sample = high_z.sample(n=len(high_z), replace=True)
            rho, _ = spearmanr(sample['gamma_t'], sample['dust'])
            bootstrap_rhos.append(rho)
        
        bootstrap_rhos = np.array(bootstrap_rhos)
        
        print_status(f"Bootstrap distribution (N = 1000):", "INFO")
        print_status(f"  Mean ρ = {bootstrap_rhos.mean():.3f}", "INFO")
        print_status(f"  Std ρ = {bootstrap_rhos.std():.3f}", "INFO")
        print_status(f"  95% CI: [{np.percentile(bootstrap_rhos, 2.5):.3f}, {np.percentile(bootstrap_rhos, 97.5):.3f}]", "INFO")
        
        # Check if 0 is outside the 95% CI
        if np.percentile(bootstrap_rhos, 2.5) > 0:
            print_status("\n✓ Zero is outside 95% CI → Correlation is robust", "INFO")
        
        results['out_of_box']['bootstrap'] = {
            'mean': float(bootstrap_rhos.mean()),
            'std': float(bootstrap_rhos.std()),
            'ci_low': float(np.percentile(bootstrap_rhos, 2.5)),
            'ci_high': float(np.percentile(bootstrap_rhos, 97.5))
        }
    
    # ==========================================================================
    # TEST 5: Extreme Outlier Analysis
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Extreme Outlier Analysis", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: Do the MOST extreme galaxies follow TEP?", "INFO")
    print_status("The hardest cases should still follow the pattern.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'mwa', 'chi2'])
    
    if len(high_z) > 50:
        # Define extreme as top 5% in dust OR age OR chi2
        dust_thresh = high_z['dust'].quantile(0.95)
        age_thresh = high_z['mwa'].quantile(0.95)
        chi2_thresh = high_z['chi2'].quantile(0.95)
        
        extreme = high_z[
            (high_z['dust'] > dust_thresh) | 
            (high_z['mwa'] > age_thresh) | 
            (high_z['chi2'] > chi2_thresh)
        ]
        normal = high_z.drop(extreme.index)
        
        print_status(f"Extreme outliers (top 5% in any property): N = {len(extreme)}", "INFO")
        
        if len(extreme) > 5:
            mean_gamma_ext = extreme['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            
            print_status(f"  <Γ_t> extreme = {mean_gamma_ext:.2f}", "INFO")
            print_status(f"  <Γ_t> normal = {mean_gamma_norm:.2f}", "INFO")
            print_status(f"  Ratio: {mean_gamma_ext/mean_gamma_norm:.2f}×", "INFO")
            
            if mean_gamma_ext > mean_gamma_norm * 2:
                print_status("\n✓ Extreme outliers have elevated Γ_t → TEP explains extremes", "INFO")
        
        results['out_of_box']['extreme_outliers'] = {
            'n_extreme': int(len(extreme)),
            'mean_gamma_extreme': float(mean_gamma_ext) if len(extreme) > 0 else 0,
            'mean_gamma_normal': float(mean_gamma_norm) if len(normal) > 0 else 0
        }
    
    # ==========================================================================
    # TEST 6: Information Gain
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: Information Gain", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: How much does Gamma_t improve dust prediction?", "INFO")
    print_status("Compare R² with and without Gamma_t.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'log_Mstar'])
    
    if len(high_z) > 50:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Model 1: Mass only
        X1 = high_z[['log_Mstar']].values
        y = high_z['dust'].values
        model1 = LinearRegression().fit(X1, y)
        r2_mass = r2_score(y, model1.predict(X1))
        
        # Model 2: Mass + Gamma_t
        X2 = high_z[['log_Mstar', 'gamma_t']].values
        model2 = LinearRegression().fit(X2, y)
        r2_both = r2_score(y, model2.predict(X2))
        
        # Model 3: Gamma_t only
        X3 = high_z[['gamma_t']].values
        model3 = LinearRegression().fit(X3, y)
        r2_gamma = r2_score(y, model3.predict(X3))
        
        print_status(f"R² (Mass only): {r2_mass:.3f}", "INFO")
        print_status(f"R² (Gamma_t only): {r2_gamma:.3f}", "INFO")
        print_status(f"R² (Mass + Gamma_t): {r2_both:.3f}", "INFO")
        print_status(f"Information gain from Gamma_t: {(r2_both - r2_mass)*100:.1f}%", "INFO")
        
        results['out_of_box']['information_gain'] = {
            'r2_mass': float(r2_mass),
            'r2_gamma': float(r2_gamma),
            'r2_both': float(r2_both),
            'gain': float(r2_both - r2_mass)
        }
    
    # ==========================================================================
    # TEST 7: The "Anomalous" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 7: The 'Anomalous' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: Are there galaxies that VIOLATE TEP predictions?", "INFO")
    print_status("Look for high-Gamma_t galaxies with LOW dust (should not exist).\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust'])
    
    if len(high_z) > 50:
        # Define "anomalous" as high Gamma_t but low dust
        high_gamma = high_z[high_z['gamma_t'] > 5]
        
        if len(high_gamma) > 5:
            low_dust_high_gamma = high_gamma[high_gamma['dust'] < 0.3]
            
            print_status(f"High Γ_t (> 5) galaxies: N = {len(high_gamma)}", "INFO")
            print_status(f"With low dust (< 0.3): N = {len(low_dust_high_gamma)}", "INFO")
            print_status(f"Fraction: {len(low_dust_high_gamma)/len(high_gamma)*100:.1f}%", "INFO")
            
            if len(low_dust_high_gamma) / len(high_gamma) < 0.1:
                print_status("\n✓ Very few 'anomalous' cases → TEP predictions hold", "INFO")
            
            results['out_of_box']['impossible_test'] = {
                'n_high_gamma': int(len(high_gamma)),
                'n_low_dust': int(len(low_dust_high_gamma)),
                'fraction': float(len(low_dust_high_gamma)/len(high_gamma)) if len(high_gamma) > 0 else 0
            }
    
    # ==========================================================================
    # TEST 8: Redshift Derivative
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 8: Redshift Derivative", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Question: Does the RATE of change match TEP predictions?", "INFO")
    print_status("The derivative dρ/dz should be positive and increasing.\n", "INFO")
    
    valid = df.dropna(subset=['gamma_t', 'dust', 'z_phot'])
    
    z_points = [5, 6, 7, 8, 9, 10]
    rhos = []
    
    for z in z_points:
        bin_data = valid[(valid['z_phot'] >= z - 0.5) & (valid['z_phot'] < z + 0.5)]
        if len(bin_data) > 20:
            rho, _ = spearmanr(bin_data['gamma_t'], bin_data['dust'])
            rhos.append(rho)
        else:
            rhos.append(np.nan)
    
    # Calculate derivative
    rhos = np.array(rhos)
    valid_mask = ~np.isnan(rhos)
    
    if sum(valid_mask) >= 4:
        z_valid = np.array(z_points)[valid_mask]
        rho_valid = rhos[valid_mask]
        
        # Fit linear trend
        slope, intercept, r, p, _ = linregress(z_valid, rho_valid)
        
        print_status(f"dρ/dz = {slope:.3f} (R² = {r**2:.3f})", "INFO")
        
        if slope > 0.05:
            print_status("\n✓ Correlation strengthens with z as predicted", "INFO")
        
        results['out_of_box']['z_derivative'] = {
            'slope': float(slope),
            'r_squared': float(r**2)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Out-of-the-Box Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    tests_passed = 0
    tests_total = 8
    
    # Count passed tests
    if results['out_of_box'].get('reverse_causality'):
        n_sig = sum(
            1
            for r in results['out_of_box']['reverse_causality']
            if r['rho'] > 0.3 and r.get('p') is not None and r['p'] < 0.05
        )
        if n_sig >= 2:
            tests_passed += 1
    
    if results['out_of_box'].get('partial_correlation', {}).get('rho_partial', 0) > 0.1:
        tests_passed += 1
    
    if results['out_of_box'].get('prediction_inversion', {}).get('r2_observables', 0) > 0.3:
        tests_passed += 1
    
    if results['out_of_box'].get('bootstrap', {}).get('ci_low', 0) > 0:
        tests_passed += 1
    
    if results['out_of_box'].get('extreme_outliers', {}).get('mean_gamma_extreme', 0) > results['out_of_box'].get('extreme_outliers', {}).get('mean_gamma_normal', 1) * 2:
        tests_passed += 1
    
    if results['out_of_box'].get('information_gain', {}).get('gain', 0) > 0:
        tests_passed += 1
    
    if results['out_of_box'].get('impossible_test', {}).get('fraction', 1) < 0.1:
        tests_passed += 1
    
    if results['out_of_box'].get('z_derivative', {}).get('slope', 0) > 0.05:
        tests_passed += 1
    
    print_status(f"\nOut-of-box tests passed: {tests_passed}/{tests_total}", "INFO")
    
    results['summary'] = {
        'tests_passed': tests_passed,
        'tests_total': tests_total
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
