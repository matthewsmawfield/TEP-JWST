
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger

# Load data
df = pd.read_csv(PROJECT_ROOT / "results" / "interim" / "step_02_uncover_full_sample_tep.csv")
impossible = df[df['age_ratio'] > 0.5].copy()

print(f"Loaded {len(impossible)} impossible galaxies.")
print(f"Mean Age Ratio (Observed): {impossible['age_ratio'].mean():.3f}")

# Standard Model (Current)
# log_Mh_ref = 12.0
# alpha(z) = 0.58 * sqrt(1+z)
# z_factor = (1+z)/(1+5.5)
# gamma = exp( alpha * (2/3) * (logMh - 12.0) * z_factor )

def compute_gamma_standard(row):
    log_Mh = row['log_Mh']
    z = row['z_phot']
    alpha_0 = 0.58
    z_ref = 5.5
    
    alpha_z = alpha_0 * np.sqrt(1 + z)
    z_factor = (1 + z) / (1 + z_ref)
    delta_log_Mh = log_Mh - 12.0
    
    return np.exp(alpha_z * (2/3) * delta_log_Mh * z_factor)

# New Model (Variable Reference)
# log_Mh_ref(z) = 12.0 - 1.5 * log10(1+z)
# Based on constant virial velocity sigma_ref

def compute_gamma_new(row):
    log_Mh = row['log_Mh']
    z = row['z_phot']
    alpha_0 = 0.58
    z_ref = 5.5
    
    # New Reference Mass Scaling
    # sigma ~ M^(1/3) * (1+z)^(1/2)
    # sigma_ref = const => M_ref^(1/3) * (1+z)^(1/2) = const
    # M_ref ~ (1+z)^(-1.5)
    # log M_ref = log M_ref_0 - 1.5 * log10(1+z)
    
    log_mh_ref_z = 12.0 - 1.5 * np.log10(1 + z)
    
    alpha_z = alpha_0 * np.sqrt(1 + z)
    z_factor = (1 + z) / (1 + z_ref)
    delta_log_Mh = log_Mh - log_mh_ref_z
    
    return np.exp(alpha_z * (2/3) * delta_log_Mh * z_factor)

impossible['gamma_std'] = impossible.apply(compute_gamma_standard, axis=1)
impossible['gamma_new'] = impossible.apply(compute_gamma_new, axis=1)

impossible['age_ratio_std'] = impossible['age_ratio'] / impossible['gamma_std']
impossible['age_ratio_new'] = impossible['age_ratio'] / impossible['gamma_new']

print("\nStandard Model Results:")
n_resolved_std = (impossible['age_ratio_std'] <= 0.5).sum()
print(f"Resolved: {n_resolved_std}/{len(impossible)}")
print(f"Mean Corrected Age Ratio: {impossible['age_ratio_std'].mean():.3f}")

print("\nNew Model (Variable Reference) Results:")
n_resolved_new = (impossible['age_ratio_new'] <= 0.5).sum()
print(f"Resolved: {n_resolved_new}/{len(impossible)}")
print(f"Mean Corrected Age Ratio: {impossible['age_ratio_new'].mean():.3f}")

print("\nDetailed Comparison (Top 5):")
print(impossible[['z_phot', 'log_Mh', 'age_ratio', 'gamma_std', 'gamma_new', 'age_ratio_new']].head(5))
