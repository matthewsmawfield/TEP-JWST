
import pandas as pd
from scipy import stats
from pathlib import Path
import numpy as np

# Load data
try:
    df = pd.read_csv("/Users/matthewsmawfield/www/TEP-JWST/data/interim/combined_spectroscopic_catalog.csv")
    # Columns: id,ra,dec,z_spec,log_Mstar,mwa,dust,met,source_catalog,...
    
    # Filter for z > 4
    df = df[df['z_spec'] > 4]
    
    # Calculate Age Ratio
    from astropy.cosmology import Planck18 as cosmo
    df['t_cosmic'] = cosmo.age(df['z_spec']).value
    df['age_ratio'] = df['mwa'] / df['t_cosmic']
    
    # Calculate Gamma_t
    ALPHA_0 = 0.58
    Z_REF = 5.5
    LOG_MH_REF = 12.0
    
    df['log_Mh'] = df['log_Mstar'] + 2.0 # Simple proxy for check
    
    def get_gamma(row):
        alpha = ALPHA_0 * np.sqrt(1 + row['z_spec'])
        delta_mh = row['log_Mh'] - LOG_MH_REF
        z_fac = (1 + row['z_spec']) / (1 + Z_REF)
        return np.exp(alpha * (2/3) * delta_mh * z_fac)
        
    df['gamma_t'] = df.apply(get_gamma, axis=1)
    
    # Define Quiescent: sSFR < 1e-10 (approx). 
    # Catalog might not have sSFR directly. 
    # If not, we can't verify easily without the specific definition used in the paper.
    # Let's check columns.
    print("Columns:", df.columns)
    
    # If sSFR is missing, we might look for 'uvj_class' or similar if it exists.
    # Assuming 'ssfr' column exists or can be derived if 'sfr' exists.
    if 'sfr' in df.columns:
        df['ssfr'] = df['sfr'] / (10**df['log_Mstar'])
        quiescent = df[df['ssfr'] < 1e-10]
        star_forming = df[df['ssfr'] >= 1e-10]
        
        if len(quiescent) > 5:
            rho, p = stats.spearmanr(quiescent['gamma_t'], quiescent['age_ratio'])
            print(f"Quiescent (N={len(quiescent)}): rho={rho:.3f}, p={p:.4f}")
            print(f"Mean Gamma Quiescent: {quiescent['gamma_t'].mean():.2f}")
        else:
            print("Not enough quiescent galaxies found with sSFR < 1e-10")
    else:
        print("SFR column missing, cannot verify quiescent subset directly.")
            
except Exception as e:
    print(f"Error: {e}")
