
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def analyze_z67_dip():
    # Load data
    df = pd.read_csv('/Users/matthewsmawfield/www/TEP-JWST/results/interim/step_01_uncover_full_sample.csv')
    
    # Filter for z=6-7
    z_min, z_max = 6.0, 7.0
    mask = (df['z_phot'] >= z_min) & (df['z_phot'] < z_max) & (df['dust'] > -99) & (df['log_Mstar'] > 7)
    subset = df[mask].copy()
    
    print(f"Total N in z={z_min}-{z_max} bin: {len(subset)}")
    
    # Calculate overall correlation
    rho_overall, p_overall = spearmanr(subset['log_Mstar'], subset['dust'])
    print(f"Overall Mass-Dust Correlation: rho={rho_overall:.3f}, p={p_overall:.3e}")
    
    # Split by sSFR (Specific Star Formation Rate)
    # log_ssfr is already in the file
    median_ssfr = subset['log_ssfr'].median()
    print(f"Median log(sSFR): {median_ssfr:.3f} yr^-1")
    
    high_ssfr = subset[subset['log_ssfr'] > median_ssfr]
    low_ssfr = subset[subset['log_ssfr'] <= median_ssfr]
    
    print(f"High sSFR N: {len(high_ssfr)}")
    print(f"Low sSFR N: {len(low_ssfr)}")
    
    # Calculate correlations for subsamples
    rho_high, p_high = spearmanr(high_ssfr['log_Mstar'], high_ssfr['dust'])
    rho_low, p_low = spearmanr(low_ssfr['log_Mstar'], low_ssfr['dust'])
    
    print(f"High sSFR Correlation (Destruction Dominated?): rho={rho_high:.3f}, p={p_high:.3e}")
    print(f"Low sSFR Correlation (Production Dominated?): rho={rho_low:.3f}, p={p_low:.3e}")
    
    # Save results for manuscript verification
    results = {
        'z_range': f"{z_min}-{z_max}",
        'N_total': len(subset),
        'rho_overall': rho_overall,
        'median_ssfr': median_ssfr,
        'high_ssfr': {'N': len(high_ssfr), 'rho': rho_high, 'p': p_high},
        'low_ssfr': {'N': len(low_ssfr), 'rho': rho_low, 'p': p_low}
    }
    
    import json
    with open('/Users/matthewsmawfield/www/TEP-JWST/results/outputs/misc/z67_ssfr_split_test.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    analyze_z67_dip()
