
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-JWST")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like

def log_likelihood(nu, mass, z, dust):
    # Modified gamma_t with free nu
    # Gamma_t = exp[ k * (phi - phi_ref) * (1+z)^nu ]
    # In log space: log_gamma = k' * (mass^2/3) * (1+z)^nu + const
    # We fit a simple linear model: dust ~ a + b * (mass^2/3 * (1+z)^nu)
    
    phi = 10**((2/3) * stellar_to_halo_mass_behroozi_like(mass, z))
    predictor = phi * (1 + z)**nu
    
    # Linear regression to find a, b
    X = np.column_stack([np.ones_like(predictor), predictor])
    beta = np.linalg.lstsq(X, dust, rcond=None)[0]
    residuals = dust - X @ beta
    sigma2 = np.mean(residuals**2)
    
    return 0.5 * len(dust) * np.log(2 * np.pi * sigma2) + 0.5 * len(dust)

# Load data
df = pd.read_csv(PROJECT_ROOT / "results/interim/step_002_uncover_full_sample_tep.csv")
df = df[df['z_phot'] >= 7].dropna(subset=['dust', 'log_Mstar', 'z_phot'])

mass = df['log_Mstar'].values
z = df['z_phot'].values
dust = df['dust'].values

# Scan nu
nu_vals = np.linspace(-1, 2, 31)
ll_vals = [log_likelihood(nu, mass, z, dust) for nu in nu_vals]

print("Nu scan results (Minimize LL):")
for nu, ll in zip(nu_vals, ll_vals):
    print(f"nu={nu:.1f}, LL={ll:.2f}")

best_nu = nu_vals[np.argmin(ll_vals)]
print(f"\nBest Nu: {best_nu}")

# Compare to Nu=0.5 (Standard TEP)
ll_tep = log_likelihood(0.5, mass, z, dust)
ll_best = log_likelihood(best_nu, mass, z, dust)
print(f"Delta LL (Best - TEP): {ll_tep - ll_best:.2f}")
