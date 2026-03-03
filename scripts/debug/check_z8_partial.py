
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, linregress

def partial_correlation_double(x, y, z_control, mass_control, use_log_x=False):
    if use_log_x:
        x_for_resid = np.log(np.maximum(x, 1e-10))
    else:
        x_for_resid = x
    
    X_controls = np.column_stack([z_control, mass_control, np.ones(len(z_control))])
    coeffs_x = np.linalg.lstsq(X_controls, x_for_resid, rcond=None)[0]
    x_resid = x_for_resid - X_controls @ coeffs_x
    
    coeffs_y = np.linalg.lstsq(X_controls, y, rcond=None)[0]
    y_resid = y - X_controls @ coeffs_y
    
    return spearmanr(x_resid, y_resid)

# Use absolute path
df = pd.read_csv("/Users/matthewsmawfield/www/TEP-JWST/results/interim/step_02_uncover_full_sample_tep.csv")
z8 = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'log_Mstar'])

print(f"N (z>8) = {len(z8)}")
gamma = z8['gamma_t'].values
dust = z8['dust'].values
z = z8['z_phot'].values
mass = z8['log_Mstar'].values

rho, p = spearmanr(gamma, dust)
print(f"Raw rho(Gamma, Dust): {rho:.3f}")

rho_part, p_part = partial_correlation_double(gamma, dust, z, mass, use_log_x=True)
print(f"Partial rho(Gamma, Dust | z, M): {rho_part:.3f}")

# Check Partial controlling ONLY for Mass
def partial_correlation_single(x, y, control, use_log_x=False):
    if use_log_x:
        x_for_resid = np.log(np.maximum(x, 1e-10))
    else:
        x_for_resid = x
        
    slope_x, int_x, _, _, _ = linregress(control, x_for_resid)
    x_resid = x_for_resid - (slope_x * control + int_x)
    
    slope_y, int_y, _, _, _ = linregress(control, y)
    y_resid = y - (slope_y * control + int_y)
    
    return spearmanr(x_resid, y_resid)

rho_mass_control, p_mass = partial_correlation_single(gamma, dust, mass, use_log_x=True)
print(f"Partial rho(Gamma, Dust | M) [Log-Control]: {rho_mass_control:.3f}")

# Test improper linear control (Hypothesis for 0.48)
rho_linear, p_lin = partial_correlation_single(gamma, dust, mass, use_log_x=False)
print(f"Partial rho(Gamma, Dust | M) [Linear-Control]: {rho_linear:.3f}")

# Test t_eff vs Dust controlling for Mass
t_eff = z8['t_eff'].values
rho_teff_mass, p_tm = partial_correlation_single(t_eff, dust, mass, use_log_x=True)
print(f"Partial rho(t_eff, Dust | M): {rho_teff_mass:.3f}")

# Test t_eff vs Dust controlling for Mass AND Redshift (should be similar to Gamma|M,z)
rho_teff_all, p_tall = partial_correlation_double(t_eff, dust, z, mass, use_log_x=True)
print(f"Partial rho(t_eff, Dust | z, M): {rho_teff_all:.3f}")
