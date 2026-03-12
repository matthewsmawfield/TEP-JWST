#!/usr/bin/env python3
"""
TEP-JWST Step 062: Predictive Power Analysis

A fundamental test of any theoretical framework is its predictive capability.
This step evaluates whether the Temporal Enhancement of Potentials (TEP) framework,
specifically the effective time t_eff, can predict observed galaxy properties
more accurately than models relying solely on standard cosmic time or physical mass.

Key evaluations:
1. DUST PREDICTION: Can Gamma_t provide additional predictive power for dust content beyond physical mass?
2. CROSS-VALIDATED PREDICTION: Are the predictions robust to data partitioning (K-fold CV)?
3. OUT-OF-SAMPLE PREDICTION: Does the model successfully generalize to unseen redshift bins?
4. FEATURE IMPORTANCE: How does t_eff rank against physical mass in non-linear models (Random Forests)?
5. NULL COMPARISON: Does the framework perform significantly better than a baseline mean-prediction model?

Mathematical logic:
The core analysis relies on ordinary least squares (OLS) regression and non-linear Random Forest regressors.
For linear models: y = beta_0 + beta_1 * X_1 + ... + epsilon
We evaluate the coefficient of determination (R^2), which quantifies the proportion of the variance
in the dependent variable (e.g., dust attenuation A_V) that is predictable from the independent variables.
R^2 = 1 - (SS_res / SS_tot), where SS_res is the sum of squares of residuals and SS_tot is the total sum of squares.
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
from scripts.utils.p_value_utils import safe_json_default

STEP_NUM = "062"
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
    data_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data not found: {data_path}. Run step_002 first.", "ERROR")
        return
        
    df = pd.read_csv(data_path)
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'predictive_power': {}
    }
    
    # Isolate high-redshift sample where TEP effects are most pronounced
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'chi2', 'log_Mstar', 't_eff'])
    
    # ==========================================================================
    # TEST 1: Dust Prediction via Linear Models
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Dust Prediction via Linear Models", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Evaluating the linear correlation between mass, Gamma_t, t_eff and dust content.\n", "INFO")
    
    if len(high_z) > 50:
        y = high_z['dust'].values
        
        # Model 1: Physical stellar mass only
        X_mass = high_z[['log_Mstar']].values
        model_mass = LinearRegression().fit(X_mass, y)
        r2_mass = r2_score(y, model_mass.predict(X_mass))
        
        # Model 2: Temporal Enhancement Factor only
        X_gamma = high_z[['gamma_t']].values
        model_gamma = LinearRegression().fit(X_gamma, y)
        r2_gamma = r2_score(y, model_gamma.predict(X_gamma))
        
        # Model 3: Effective time (t_eff = t_cosmic * Gamma_t) only
        X_teff = high_z[['t_eff']].values
        model_teff = LinearRegression().fit(X_teff, y)
        r2_teff = r2_score(y, model_teff.predict(X_teff))
        
        # Model 4: Bivariate model (Mass + Gamma_t)
        X_both = high_z[['log_Mstar', 'gamma_t']].values
        model_both = LinearRegression().fit(X_both, y)
        r2_both = r2_score(y, model_both.predict(X_both))
        
        print_status(f"Variance explained by Mass alone: R² = {r2_mass:.4f}", "INFO")
        print_status(f"Variance explained by Gamma_t alone: R² = {r2_gamma:.4f}", "INFO")
        print_status(f"Variance explained by t_eff alone: R² = {r2_teff:.4f}", "INFO")
        print_status(f"Variance explained by joint Mass + Gamma_t model: R² = {r2_both:.4f}", "INFO")
        
        added_variance = (r2_both - r2_mass)
        print_status(f"\nMarginal predictive power: Gamma_t adds {added_variance*100:.1f}% explained variance over mass alone.", "INFO")
        
        results['predictive_power']['dust_prediction'] = {
            'r2_mass': float(r2_mass),
            'r2_gamma': float(r2_gamma),
            'r2_teff': float(r2_teff),
            'r2_both': float(r2_both),
            'gamma_adds': float(added_variance)
        }
    
    # ==========================================================================
    # TEST 2: Cross-Validated Prediction (Ridge Regression)
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Cross-Validated Prediction robustness", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Testing if the joint model's predictive power is robust to data partitioning (K-Fold CV).\n", "INFO")
    
    if len(high_z) > 50:
        X = high_z[['log_Mstar', 'gamma_t']].values
        y = high_z['dust'].values
        
        # 5-fold cross-validation with Ridge regression (L2 regularization) to prevent overfitting
        model = Ridge(alpha=1.0)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        print_status(f"5-fold CV R² scores: {np.array2string(cv_scores, precision=4)}", "INFO")
        print_status(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", "INFO")
        
        results['predictive_power']['cv_prediction'] = {
            'mean_r2': float(cv_scores.mean()),
            'std_r2': float(cv_scores.std())
        }
    
    # ==========================================================================
    # TEST 3: Out-of-Sample Prediction (Generalization across redshift)
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Out-of-Sample Generalization", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Training the joint model on z = 8-9, and evaluating its performance on an unseen higher-z (9-10) epoch.\n", "INFO")
    
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
        
        if r2_oos > 0.0:
            print_status("\n-> The model retains positive predictive power on structurally distinct out-of-sample data.", "INFO")
        else:
            print_status("\n-> The linear model fails to generalize to the higher redshift bin.", "INFO")
            
        results['predictive_power']['oos_prediction'] = {
            'n_train': int(len(train_data)),
            'n_test': int(len(test_data)),
            'r2_oos': float(r2_oos),
            'rmse_oos': float(rmse_oos)
        }
    
    # ==========================================================================
    # TEST 4: Feature Importance Analysis (Random Forest)
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Non-linear Feature Importance", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Using a non-linear ensemble tree model to evaluate relative feature information content.\n", "INFO")
    
    if len(high_z) > 50:
        X = high_z[['log_Mstar', 'gamma_t', 't_eff']].values
        y = high_z['dust'].values
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        features = ['log_Mstar', 'gamma_t', 't_eff']
        
        print_status("Feature importances (Random Forest Impurity reduction):", "INFO")
        for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
            print_status(f"  {feat}: {imp:.3f}", "INFO")
        
        results['predictive_power']['feature_importance'] = {
            feat: float(imp) for feat, imp in zip(features, importances)
        }
    
    # ==========================================================================
    # TEST 5: Null Model Baseline Comparison
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Baseline Null Model Comparison", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Comparing the TEP parameterization against a purely random, mean-predictive null distribution.\n", "INFO")
    
    if len(high_z) > 50:
        y = high_z['dust'].values
        
        # Null model: consistently predict the sample mean
        y_null = np.full_like(y, y.mean())
        r2_null = r2_score(y, y_null)  # Precisely 0 by definition
        
        # TEP model
        X_tep = high_z[['gamma_t']].values
        model_tep = LinearRegression().fit(X_tep, y)
        r2_tep = r2_score(y, model_tep.predict(X_tep))
        
        print_status(f"R² (Null model - mean predictor): {r2_null:.4f}", "INFO")
        print_status(f"R² (TEP Gamma_t model): {r2_tep:.4f}", "INFO")
        print_status(f"Calibrated predictive gain over baseline: {(r2_tep - r2_null)*100:.1f}%", "INFO")
        
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
