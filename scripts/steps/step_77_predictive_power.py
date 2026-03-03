#!/usr/bin/env python3
"""
TEP-JWST Step 77: Predictive Power Analysis

The ultimate test of any theory is its PREDICTIVE POWER.
Can TEP predict properties that standard physics cannot?

1. DUST PREDICTION: Can Gamma_t predict dust content?
2. CHI2 PREDICTION: Can Gamma_t predict SED fit quality?
3. AGE PREDICTION: Can Gamma_t predict apparent stellar age?
4. CROSS-PROPERTY PREDICTION: Can one property predict another via Gamma_t?
5. OUT-OF-SAMPLE PREDICTION: Does the model generalize?
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM = "77"
STEP_NAME = "predictive_power"

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
    print_status(f"STEP {STEP_NUM}: Predictive Power Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'predictive_power': {}
    }
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'chi2', 'log_Mstar', 't_eff'])
    
    # ==========================================================================
    # TEST 1: Dust Prediction
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Dust Prediction", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Can we predict dust content from different variables?\n", "INFO")
    
    if len(high_z) > 50:
        y = high_z['dust'].values
        
        # Model 1: Mass only
        X_mass = high_z[['log_Mstar']].values
        model_mass = LinearRegression().fit(X_mass, y)
        r2_mass = r2_score(y, model_mass.predict(X_mass))
        
        # Model 2: Gamma_t only
        X_gamma = high_z[['gamma_t']].values
        model_gamma = LinearRegression().fit(X_gamma, y)
        r2_gamma = r2_score(y, model_gamma.predict(X_gamma))
        
        # Model 3: t_eff only
        X_teff = high_z[['t_eff']].values
        model_teff = LinearRegression().fit(X_teff, y)
        r2_teff = r2_score(y, model_teff.predict(X_teff))
        
        # Model 4: Mass + Gamma_t
        X_both = high_z[['log_Mstar', 'gamma_t']].values
        model_both = LinearRegression().fit(X_both, y)
        r2_both = r2_score(y, model_both.predict(X_both))
        
        print_status(f"R² (Mass only): {r2_mass:.4f}", "INFO")
        print_status(f"R² (Gamma_t only): {r2_gamma:.4f}", "INFO")
        print_status(f"R² (t_eff only): {r2_teff:.4f}", "INFO")
        print_status(f"R² (Mass + Gamma_t): {r2_both:.4f}", "INFO")
        print_status(f"\nGamma_t adds: {(r2_both - r2_mass)*100:.1f}% explained variance", "INFO")
        
        results['predictive_power']['dust_prediction'] = {
            'r2_mass': float(r2_mass),
            'r2_gamma': float(r2_gamma),
            'r2_teff': float(r2_teff),
            'r2_both': float(r2_both),
            'gamma_adds': float(r2_both - r2_mass)
        }
    
    # ==========================================================================
    # TEST 2: Cross-Validated Prediction
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Cross-Validated Prediction", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Does the prediction generalize to unseen data?\n", "INFO")
    
    if len(high_z) > 50:
        X = high_z[['log_Mstar', 'gamma_t']].values
        y = high_z['dust'].values
        
        # Cross-validation
        model = Ridge(alpha=1.0)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        print_status(f"5-fold CV R² scores: {cv_scores}", "INFO")
        print_status(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", "INFO")
        
        results['predictive_power']['cv_prediction'] = {
            'mean_r2': float(cv_scores.mean()),
            'std_r2': float(cv_scores.std())
        }
    
    # ==========================================================================
    # TEST 3: Out-of-Sample Prediction
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Out-of-Sample Prediction", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Train on z = 8-9, predict z = 9-10.\n", "INFO")
    
    train_data = df[(df['z_phot'] >= 8) & (df['z_phot'] < 9)].dropna(subset=['gamma_t', 'dust', 'log_Mstar'])
    test_data = df[(df['z_phot'] >= 9) & (df['z_phot'] < 10)].dropna(subset=['gamma_t', 'dust', 'log_Mstar'])
    
    if len(train_data) > 30 and len(test_data) > 30:
        X_train = train_data[['log_Mstar', 'gamma_t']].values
        y_train = train_data['dust'].values
        X_test = test_data[['log_Mstar', 'gamma_t']].values
        y_test = test_data['dust'].values
        
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2_oos = r2_score(y_test, y_pred)
        rmse_oos = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print_status(f"Training set (z = 8-9): N = {len(train_data)}", "INFO")
        print_status(f"Test set (z = 9-10): N = {len(test_data)}", "INFO")
        print_status(f"Out-of-sample R²: {r2_oos:.4f}", "INFO")
        print_status(f"Out-of-sample RMSE: {rmse_oos:.4f}", "INFO")
        
        if r2_oos > 0.2:
            print_status("\n✓ Model generalizes to higher redshift", "INFO")
        
        results['predictive_power']['oos_prediction'] = {
            'n_train': int(len(train_data)),
            'n_test': int(len(test_data)),
            'r2_oos': float(r2_oos),
            'rmse_oos': float(rmse_oos)
        }
    
    # ==========================================================================
    # TEST 4: Feature Importance
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Feature Importance", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Which features are most important for dust prediction?\n", "INFO")
    
    if len(high_z) > 50:
        X = high_z[['log_Mstar', 'gamma_t', 't_eff']].values
        y = high_z['dust'].values
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        features = ['log_Mstar', 'gamma_t', 't_eff']
        
        print_status("Feature importances (Random Forest):", "INFO")
        for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
            print_status(f"  {feat}: {imp:.3f}", "INFO")
        
        results['predictive_power']['feature_importance'] = {
            feat: float(imp) for feat, imp in zip(features, importances)
        }
    
    # ==========================================================================
    # TEST 5: Prediction Residuals
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Prediction Residuals", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Are prediction residuals random or structured?\n", "INFO")
    
    if len(high_z) > 50:
        X = high_z[['log_Mstar', 'gamma_t']].values
        y = high_z['dust'].values
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Check if residuals correlate with anything
        rho_z, p_z = spearmanr(high_z['z_phot'], residuals)
        rho_chi2, p_chi2 = spearmanr(high_z['chi2'], residuals)
        
        print_status(f"Residual correlation with z: ρ = {rho_z:.3f} (p = {p_z:.2e})", "INFO")
        print_status(f"Residual correlation with χ²: ρ = {rho_chi2:.3f} (p = {p_chi2:.2e})", "INFO")
        
        if abs(rho_z) < 0.1 and abs(rho_chi2) < 0.1:
            print_status("\n✓ Residuals are approximately random", "INFO")
        
        results['predictive_power']['residuals'] = {
            'rho_z': float(rho_z),
            'rho_chi2': float(rho_chi2)
        }
    
    # ==========================================================================
    # TEST 6: Comparison with Null Model
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: Comparison with Null Model", "INFO")
    print_status("=" * 70, "INFO")
    print_status("How much better is TEP than a null model?\n", "INFO")
    
    if len(high_z) > 50:
        y = high_z['dust'].values
        
        # Null model: predict mean
        y_null = np.full_like(y, y.mean())
        r2_null = r2_score(y, y_null)  # Should be 0
        
        # TEP model
        X_tep = high_z[['gamma_t']].values
        model_tep = LinearRegression().fit(X_tep, y)
        r2_tep = r2_score(y, model_tep.predict(X_tep))
        
        print_status(f"R² (Null model): {r2_null:.4f}", "INFO")
        print_status(f"R² (TEP model): {r2_tep:.4f}", "INFO")
        print_status(f"Improvement: {(r2_tep - r2_null)*100:.1f}%", "INFO")
        
        results['predictive_power']['null_comparison'] = {
            'r2_null': float(r2_null),
            'r2_tep': float(r2_tep),
            'improvement': float(r2_tep - r2_null)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Predictive Power Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status("\nKey findings:", "INFO")
    
    dp = results['predictive_power']
    
    if 'dust_prediction' in dp:
        print_status(f"  • Gamma_t adds {dp['dust_prediction']['gamma_adds']*100:.1f}% explained variance", "INFO")
    
    if 'cv_prediction' in dp:
        print_status(f"  • Cross-validated R²: {dp['cv_prediction']['mean_r2']:.3f}", "INFO")
    
    if 'oos_prediction' in dp:
        print_status(f"  • Out-of-sample R²: {dp['oos_prediction']['r2_oos']:.3f}", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
