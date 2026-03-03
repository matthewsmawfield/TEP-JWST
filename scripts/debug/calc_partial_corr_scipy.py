
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def partial_corr(x, y, covar):
    # Convert to ranks
    x_r = stats.rankdata(x)
    y_r = stats.rankdata(y)
    
    if covar.ndim == 1:
        covar = covar.reshape(-1, 1)
        
    covar_r = np.column_stack([stats.rankdata(covar[:, i]) for i in range(covar.shape[1])])
    
    # Linear regression to get residuals
    def get_residuals(data, confounders):
        # Add intercept
        A = np.column_stack([np.ones(len(confounders)), confounders])
        lstsq = np.linalg.lstsq(A, data, rcond=None)
        return data - A @ lstsq[0]

    res_x = get_residuals(x_r, covar_r)
    res_y = get_residuals(y_r, covar_r)
    
    return stats.pearsonr(res_x, res_y)

# Load
try:
    df = pd.read_csv("results/interim/step_02_uncover_full_sample_tep.csv")
    mask = (df['z_phot'] > 8) & df['dust'].notna() & df['gamma_t'].notna() & df['log_Mstar'].notna()
    sample = df[mask]

    print(f"N = {len(sample)}")

    # Variables
    g = sample['gamma_t'].values
    d = sample['dust'].values
    m = sample['log_Mstar'].values
    z = sample['z_phot'].values

    # 1. Gamma vs Dust (Raw)
    r, p = stats.spearmanr(g, d)
    print(f"Raw: rho={r:.3f}, p={p:.2e}")

    # 2. Control Mass
    r_pm, p_pm = partial_corr(g, d, m)
    print(f"Control Mass: rho={r_pm:.3f}, p={p_pm:.2e}")

    # 3. Control Redshift
    r_pz, p_pz = partial_corr(g, d, z)
    print(f"Control Redshift: rho={r_pz:.3f}, p={p_pz:.2e}")

    # 4. Control Both
    r_pd, p_pd = partial_corr(g, d, np.column_stack([m, z]))
    print(f"Control Both: rho={r_pd:.3f}, p={p_pd:.2e}")

except Exception as e:
    print(f"Error: {e}")
