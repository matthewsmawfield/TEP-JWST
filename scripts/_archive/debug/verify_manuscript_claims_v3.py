import pandas as pd
import numpy as np
from scipy import stats

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
print("Mapping columns...")
if 'z_phot' in df.columns:
    df['z'] = df['z_phot']
else:
    print("Column z_phot not found!")
    print(df.columns.tolist())
    exit()

if 'dust' in df.columns:
    df['Av'] = df['dust']
else:
    print("Column dust not found!")

# mwa is in years, convert to Gyr
df['mwa_Gyr'] = df['mwa'] / 1e9

# Calculate Gamma_t
# TEP parameters
ALPHA_TEP = 0.58
# log_Mhalo = log_Mstar + 2.0 (standard assumption in this pipeline)
df['log_Mhalo'] = df['log_Mstar'] + 2.0

# Formula:
# gamma = alpha * (M_h / M_ref)^(1/3) * sqrt(1+z)
# M_ref = 1e11 (log 11)
df['gamma_t'] = ALPHA_TEP * np.sqrt(1 + df['z']) * np.power(10, (df['log_Mhalo'] - 11) / 3)

# Filter valid data
print("Filtering data...")
initial_len = len(df)
df = df.dropna(subset=['z', 'log_Mstar', 'Av', 'mwa_Gyr', 'gamma_t'])
print(f"Dropped {initial_len - len(df)} rows with NaNs.")

df = df[df['log_Mstar'] > 7] # Clean low mass noise
print(f"Final N={len(df)}")

# TEST 1: SCATTER REDUCTION (SUBSETS)
print("\n--- TEST 1: SCATTER REDUCTION (Mass-Age) ---")

subsets = {
    'Full Sample': df,
    'Mass > 9': df[df['log_Mstar'] > 9],
    'Mass > 10': df[df['log_Mstar'] > 10],
    'z > 8': df[df['z'] > 8]
}

for name, sub in subsets.items():
    if len(sub) < 10: 
        print(f"{name}: N={len(sub)} (Too small)")
        continue
    
    # Raw Scatter
    slope, intercept, r, p, err = stats.linregress(sub['log_Mstar'], sub['mwa_Gyr'])
    residuals_raw = sub['mwa_Gyr'] - (slope*sub['log_Mstar'] + intercept)
    scatter_raw = residuals_raw.std()
    
    # Corrected: t_true = t_obs / gamma
    sub = sub.copy()
    sub['mwa_corr'] = sub['mwa_Gyr'] / sub['gamma_t']
    slope_c, intercept_c, r_c, p_c, err_c = stats.linregress(sub['log_Mstar'], sub['mwa_corr'])
    residuals_corr = sub['mwa_corr'] - (slope_c*sub['log_Mstar'] + intercept_c)
    scatter_corr = residuals_corr.std()
    
    # Check if we should use Coefficient of Variation instead?
    cv_raw = sub['mwa_Gyr'].std() / sub['mwa_Gyr'].mean()
    cv_corr = sub['mwa_corr'].std() / sub['mwa_corr'].mean()
    
    pct_change = 100 * (scatter_raw - scatter_corr) / scatter_raw
    print(f"{name}: N={len(sub)}")
    print(f"  Scatter: Raw={scatter_raw:.6f}, Corr={scatter_corr:.6f}, Change={pct_change:.1f}%")
    print(f"  CV:      Raw={cv_raw:.4f},    Corr={cv_corr:.4f}")
    print(f"  R2:      Raw={r**2:.4f},      Corr={r_c**2:.4f}")

# TEST 2: AIC FOR DUST (Z > 8)
print("\n--- TEST 2: AIC FOR DUST (Z > 8) ---")
z8 = df[df['z'] > 8]
if len(z8) > 10:
    y = z8['Av']
    
    # Null Model (Constant mean)
    # y = c
    rss_null = np.sum((y - np.mean(y))**2)
    aic_null = calculate_aic(len(y), 1, rss_null) # k=1 (intercept)
    
    # Mass Model
    X_mass = np.column_stack([z8['log_Mstar'], z8['z']])
    _, rss_mass = fit_linear_model(X_mass, y)
    aic_mass = calculate_aic(len(y), 3, rss_mass)
    
    # TEP Model
    X_tep = np.column_stack([z8['gamma_t'], z8['z']])
    _, rss_tep = fit_linear_model(X_tep, y)
    aic_tep = calculate_aic(len(y), 3, rss_tep)
    
    print(f"z > 8 (N={len(z8)}):")
    print(f"  AIC Null: {aic_null:.2f}")
    print(f"  AIC Mass: {aic_mass:.2f}")
    print(f"  AIC TEP:  {aic_tep:.2f}")
    print(f"  Delta AIC (TEP vs Null): {aic_tep - aic_null:.2f}")
    print(f"  Delta AIC (Mass vs Null): {aic_mass - aic_null:.2f}")
    print(f"  Delta AIC (TEP vs Mass): {aic_tep - aic_mass:.2f}")
    
    # Correlations
    rho_m, _ = stats.spearmanr(z8['log_Mstar'], z8['Av'])
    rho_g, _ = stats.spearmanr(z8['gamma_t'], z8['Av'])
    print(f"  Rho Mass: {rho_m:.3f}")
    print(f"  Rho Gamma: {rho_g:.3f}")

# TEST 3: SELF-CONSISTENCY (Quick Check)
print("\n--- TEST 3: SELF-CONSISTENCY (Quick Check) ---")
# Optimize alpha for age_ratio flatness
# target: age_ratio / gamma(alpha) should be constant-ish
from scipy.optimize import minimize_scalar

def objective(alpha, data):
    # gamma = alpha * ...
    # We need to recompute gamma for varying alpha
    # gamma = alpha * sqrt(1+z) * 10^((logM - 11)/3)
    gamma_factor = np.sqrt(1 + data['z']) * np.power(10, (data['log_Mhalo'] - 11) / 3)
    gamma = alpha * gamma_factor
    corrected = data['age_ratio'] / gamma
    if corrected.mean() == 0: return np.inf
    return corrected.std() / abs(corrected.mean())

# Use high-z sample where TEP is valid
valid_sample = df[df['z'] > 8]
res_z8 = minimize_scalar(lambda a: objective(a, valid_sample), bounds=(0.1, 2.0), method='bounded')

# Use full sample
res_full = minimize_scalar(lambda a: objective(a, df), bounds=(0.1, 2.0), method='bounded')

print(f"Optimal Alpha (z > 8): {res_z8.x:.3f}")
print(f"Optimal Alpha (Full):  {res_full.x:.3f}")
