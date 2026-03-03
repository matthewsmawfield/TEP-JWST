
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "results" / "interim" / "step_02_uncover_full_sample_tep.csv"

# Load data
print(f"Loading {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

impossible = df[df['age_ratio'] > 0.5].copy()
print(f"Found {len(impossible)} impossible galaxies.")

if len(impossible) == 0:
    print("No impossible galaxies found.")
    exit()

# Parameters
alpha_0 = 0.58
z_ref = 5.5
log_Mh_ref_static = 12.0

# 1. Standard Model Calculation
print("\n--- Standard Model ---")
impossible['alpha_z'] = alpha_0 * np.sqrt(1 + impossible['z_phot'])
impossible['z_factor'] = (1 + impossible['z_phot']) / (1 + z_ref)
impossible['gamma_std'] = np.exp(impossible['alpha_z'] * (2/3) * (impossible['log_Mh'] - log_Mh_ref_static) * impossible['z_factor'])
impossible['age_ratio_std'] = impossible['age_ratio'] / impossible['gamma_std']

resolved_std = (impossible['age_ratio_std'] <= 0.5).sum()
print(f"Resolved: {resolved_std}/{len(impossible)}")
print(f"Mean Gamma: {impossible['gamma_std'].mean():.3f}")

# 2. Revised Model (Redshift Dependent Reference)
# Logic: Fixed virial velocity reference sigma_ref ~ 75 km/s
# M_h ~ sigma^3 / (1+z)^1.5 (roughly in matter domination/high-z)
# log M_h_ref(z) = log M_h_ref(0) - 1.5 * log10(1+z)
# But we anchor at z_ref=5.5? 
# Or just use the scaling from z=0?
# Memory said: log M_h_ref(z)
# Let's try the scaling: log M_h_ref(z) = 12.0 - 1.5 * log10((1+z)/(1+0)) ? 
# Actually TEP-H0 reference is at z=0? No, TEP-JWST says z_ref=5.5.
# Let's assume the physical scaling M ~ (1+z)^-1.5 for fixed sigma.

print("\n--- Revised Model (Fixed Sigma Scaling) ---")
# Scaling relative to the static reference which is presumably valid at some z. 
# If 12.0 is valid at z=0, then at z=8 it should be much lower.
# log_Mh_ref(z) = 12.0 - 1.5 * log10(1+z)

impossible['log_Mh_ref_z'] = 12.0 - 1.5 * np.log10(1 + impossible['z_phot'])
impossible['gamma_new'] = np.exp(impossible['alpha_z'] * (2/3) * (impossible['log_Mh'] - impossible['log_Mh_ref_z']) * impossible['z_factor'])
impossible['age_ratio_new'] = impossible['age_ratio'] / impossible['gamma_new']

resolved_new = (impossible['age_ratio_new'] <= 0.5).sum()
print(f"Resolved: {resolved_new}/{len(impossible)}")
print(f"Mean Gamma: {impossible['gamma_new'].mean():.3f}")
print(f"Mean Corrected Age Ratio: {impossible['age_ratio_new'].mean():.3f}")

print("\n--- Individual Results (New Model) ---")
print(impossible[['z_phot', 'log_Mh', 'age_ratio', 'gamma_new', 'age_ratio_new']].sort_values('age_ratio', ascending=False).to_string())
