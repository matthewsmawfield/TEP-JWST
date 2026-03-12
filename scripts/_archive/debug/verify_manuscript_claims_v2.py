import pandas as pd
import numpy as np
from scipy import stats
from astropy.cosmology import Planck18

def calculate_aic(n, k, rss):
    """Calculate AIC from Residual Sum of Squares."""
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

# Load Data
print("Loading data...")
try:
    df = pd.read_csv('/Users/matthewsmawfield/www/TEP-JWST/results/interim/step_01_uncover_full_sample.csv')
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Failed to load full sample: {e}")
    exit()

# Map columns
df['z'] = df['z_phot']
df['Av'] = df['dust']
# log_Mstar is already log_Mstar
# mwa is in years (based on previous script dividing by 1e9)
df['mwa_Gyr'] = df['mwa'] / 1e9

# Calculate Gamma_t
# TEP parameters
ALPHA_TEP = 0.58
# log_Mhalo = log_Mstar + 2.0 (standard assumption in this pipeline)
df['log_Mhalo'] = df['log_Mstar'] + 2.0

# Formula from memory/manuscript:
# gamma = alpha * (M_h / M_ref)^(1/3) * sqrt(1+z)
# M_ref = 1e11 (log 11)
# gamma = 0.58 * 10^((log_Mhalo - 11)/3) * sqrt(1+z)
df['gamma_t'] = ALPHA_TEP * np.sqrt(1 + df['z']) * np.power(10, (df['log_Mhalo'] - 11) / 3)

# Filter valid data
df = df.dropna(subset=['z', 'log_Mstar', 'Av', 'mwa_Gyr', 'gamma_t'])
df = df[df['log_Mstar'] > 7] # Clean low mass noise

# TEST 1: SCATTER REDUCTION (SUBSETS)
print("\n--- TEST 1: SCATTER REDUCTION ---")
# Mass-Age relation
# We need age.
if 'age_Gyr' in df.columns:
    df['mwa_Gyr'] = df['age_Gyr']
elif 'mwa' in df.columns:
    df['mwa_Gyr'] = df['mwa'] / 1e9
else:
    # Try to calculate or mock if missing, but better to check columns first.
    pass

if 'mwa_Gyr' in df.columns:
    subsets = {
        'Full Sample': df,
        'Mass > 9': df[df['log_Mstar'] > 9],
        'Mass > 10': df[df['log_Mstar'] > 10],
        'z > 8': df[df['z'] > 8]
    }
    
    for name, sub in subsets.items():
        if len(sub) < 10: continue
        valid = sub.dropna(subset=['log_Mstar', 'mwa_Gyr', 'gamma_t'])
        
        # Raw
        slope, intercept, r, p, err = stats.linregress(valid['log_Mstar'], valid['mwa_Gyr'])
        scatter_raw = (valid['mwa_Gyr'] - (slope*valid['log_Mstar'] + intercept)).std()
        
        # Corrected: t_true = t_obs / gamma
        valid['mwa_corr'] = valid['mwa_Gyr'] / valid['gamma_t']
        slope_c, intercept_c, r_c, p_c, err_c = stats.linregress(valid['log_Mstar'], valid['mwa_corr'])
        scatter_corr = (valid['mwa_corr'] - (slope_c*valid['log_Mstar'] + intercept_c)).std()
        
        pct_change = 100 * (scatter_raw - scatter_corr) / scatter_raw
        print(f"{name}: N={len(valid)}, Raw={scatter_raw:.3f}, Corr={scatter_corr:.3f}, Change={pct_change:.1f}%")

# TEST 2: AIC FOR DUST (Z > 8)
print("\n--- TEST 2: AIC FOR DUST (Z > 8) ---")
if 'Av' in df.columns:
    z8 = df[df['z'] > 8].dropna(subset=['Av', 'log_Mstar', 'gamma_t'])
    if len(z8) > 10:
        y = z8['Av']
        
        # Null (Constant) - effectively z is constant in high-z bin? Or include z?
        # Manuscript usually compares to Mass model.
        
        # Mass Model
        X_mass = np.column_stack([z8['log_Mstar'], z8['z']])
        _, rss_mass = fit_linear_model(X_mass, y)
        aic_mass = calculate_aic(len(y), 3, rss_mass)
        
        # TEP Model
        X_tep = np.column_stack([z8['gamma_t'], z8['z']])
        _, rss_tep = fit_linear_model(X_tep, y)
        aic_tep = calculate_aic(len(y), 3, rss_tep)
        
        print(f"z > 8 (N={len(z8)}):")
        print(f"  AIC Mass: {aic_mass:.2f}")
        print(f"  AIC TEP:  {aic_tep:.2f}")
        print(f"  Delta AIC (TEP - Mass): {aic_tep - aic_mass:.2f}")
        
        # Check pure correlation
        rho_m, _ = stats.spearmanr(z8['log_Mstar'], z8['Av'])
        rho_g, _ = stats.spearmanr(z8['gamma_t'], z8['Av'])
        print(f"  Rho Mass: {rho_m:.3f}")
        print(f"  Rho Gamma: {rho_g:.3f}")

