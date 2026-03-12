import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def calculate_aic(n, k, rss):
    """Calculate AIC from Residual Sum of Squares."""
    if rss <= 0: return np.inf
    aic = n * np.log(rss/n) + 2*k
    return aic

try:
    # Load data
    df = pd.read_csv('/Users/matthewsmawfield/www/TEP-JWST/results/outputs/step_53_dust_yield_comparison.csv')
    
    # Filter for z > 8 (The manuscript compares models for z > 8, N=283)
    # The file step_53 might not have all 283 objects if it's a yield comparison subset.
    # Let's check the length.
    print(f"Total rows in step_53 csv: {len(df)}")
    
    df_z8 = df[df['z'] > 8]
    print(f"Rows with z > 8: {len(df_z8)}")
    
    # We need the full sample for AIC to match N=283.
    # If this file is small, we can't reproduce the exact AIC but can check the trend.
    
    # Let's try to load the full interim file if possible.
    # But for now, let's look at the "deficit" correlation.
    
    # Calculate simple AIC for Mass vs Dust in this subset
    # Model 1: Dust ~ Mass (Standard)
    # Model 2: Dust ~ Gamma (TEP)
    
    if len(df_z8) > 10:
        y = df_z8['av_obs']
        x_mass = df_z8['m_star'] # Obs mass? Or log mass?
        # Assuming m_star is linear, let's take log
        log_m = np.log10(df_z8['m_star'])
        x_gamma = df_z8['gamma_t']
        
        # Null model (constant)
        rss_null = np.sum((y - np.mean(y))**2)
        aic_null = calculate_aic(len(y), 1, rss_null)
        
        # Mass model
        coeffs_m = np.polyfit(log_m, y, 1)
        pred_m = np.polyval(coeffs_m, log_m)
        rss_m = np.sum((y - pred_m)**2)
        aic_m = calculate_aic(len(y), 2, rss_m)
        
        # TEP model (Gamma)
        coeffs_g = np.polyfit(x_gamma, y, 1)
        pred_g = np.polyval(coeffs_g, x_gamma)
        rss_g = np.sum((y - pred_g)**2)
        aic_g = calculate_aic(len(y), 2, rss_g)
        
        print(f"AIC Null: {aic_null:.2f}")
        print(f"AIC Mass: {aic_m:.2f}")
        print(f"AIC TEP: {aic_g:.2f}")
        print(f"Delta AIC (Mass vs Null): {aic_m - aic_null:.2f}")
        print(f"Delta AIC (TEP vs Null): {aic_g - aic_null:.2f}")
        
except Exception as e:
    print(f"Error: {e}")
