import pandas as pd
import numpy as np
from scipy import stats

def calculate_aic(n, k, rss):
    if rss <= 0: return np.inf
    aic = n * np.log(rss/n) + 2*k
    return aic

def fit_linear_model(X, y):
    X_with_intercept = np.column_stack([np.ones(len(y)), X])
    try:
        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except:
        return None, np.inf
    y_pred = X_with_intercept @ coeffs
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    return coeffs, rss

try:
    df = pd.read_csv('/Users/matthewsmawfield/www/TEP-JWST/results/interim/step_01_uncover_full_sample.csv')
    df['z'] = df['z_phot']
    df['Av'] = df['dust']
    df['log_Mhalo'] = df['log_Mstar'] + 2.0
    ALPHA_TEP = 0.58
    df['gamma_t'] = ALPHA_TEP * np.sqrt(1 + df['z']) * np.power(10, (df['log_Mhalo'] - 11) / 3)
    
    z8 = df[(df['z'] > 8) & (df['log_Mstar'] > 7)].dropna(subset=['Av', 'log_Mstar', 'gamma_t'])
    
    if len(z8) > 10:
        y = z8['Av']
        n = len(y)
        
        # Mass Model (Mass + Z) -> Standard
        # k = 3 (intercept, M, Z)
        X_mass = np.column_stack([z8['log_Mstar'], z8['z']])
        _, rss_mass = fit_linear_model(X_mass, y)
        aic_mass = calculate_aic(n, 3, rss_mass)
        
        # TEP Model (Gamma only) -> Physical Prediction
        # k = 2 (intercept, Gamma)
        # Gamma already contains Mass and Z info
        X_tep = np.column_stack([z8['gamma_t']])
        _, rss_tep = fit_linear_model(X_tep, y)
        aic_tep = calculate_aic(n, 2, rss_tep)
        
        print(f"z > 8 (N={n}):")
        print(f"  AIC Mass (M, Z): {aic_mass:.2f}")
        print(f"  AIC TEP (Gamma): {aic_tep:.2f}")
        print(f"  Delta AIC (TEP - Mass): {aic_tep - aic_mass:.2f}")
        
except Exception as e:
    print(e)
