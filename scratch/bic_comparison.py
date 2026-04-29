
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-JWST")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_model import compute_gamma_t, compute_gamma_t_logmass, stellar_to_halo_mass_behroozi_like

def calc_bic(log_gamma, dust):
    X = np.column_stack([np.ones_like(log_gamma), log_gamma])
    beta = np.linalg.lstsq(X, dust, rcond=None)[0]
    residuals = dust - X @ beta
    sigma2 = np.mean(residuals**2)
    L = -0.5 * len(dust) * (np.log(2 * np.pi * sigma2) + 1)
    k = 3 # a, b, sigma
    return k * np.log(len(dust)) - 2 * L

df = pd.read_csv(PROJECT_ROOT / "results/interim/step_002_uncover_full_sample_tep.csv")
df = df[df['z_phot'] >= 7].dropna(subset=['dust', 'log_Mstar', 'z_phot'])

mass = df['log_Mstar'].values
z = df['z_phot'].values
dust = df['dust'].values
log_mh = stellar_to_halo_mass_behroozi_like(mass, z)

# Potential-Linear
gamma_pl = compute_gamma_t(log_mh, z)
log_gamma_pl = np.log10(gamma_pl)

# Log-Mass
gamma_lm = compute_gamma_t_logmass(log_mh, z)
log_gamma_lm = np.log10(gamma_lm)

# Standard Physics (M+z)
X_std = np.column_stack([np.ones_like(mass), mass, z])
beta_std = np.linalg.lstsq(X_std, dust, rcond=None)[0]
res_std = dust - X_std @ beta_std
s2_std = np.mean(res_std**2)
L_std = -0.5 * len(dust) * (np.log(2 * np.pi * s2_std) + 1)
bic_std = 4 * np.log(len(dust)) - 2 * L_std

bic_pl = calc_bic(log_gamma_pl, dust)
bic_lm = calc_bic(log_gamma_lm, dust)

print(f"BIC Standard Physics: {bic_std:.2f}")
print(f"BIC Potential-Linear TEP: {bic_pl:.2f}")
print(f"BIC Log-Mass TEP: {bic_lm:.2f}")

print(f"\nDelta BIC (LM - Std): {bic_lm - bic_std:.2f}")
print(f"Delta BIC (PL - Std): {bic_pl - bic_std:.2f}")
