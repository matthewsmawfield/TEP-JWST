
import pandas as pd
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "results" / "interim" / "step_02_uncover_full_sample_tep.csv"

df = pd.read_csv(DATA_PATH)
impossible = df[df['age_ratio'] > 0.5].copy()

# Recalculate new gamma locally to verify
import numpy as np
alpha_0 = 0.58
z_ref = 5.5
impossible['log_Mh_ref_z'] = 12.0 - 1.5 * np.log10(1 + impossible['z_phot'])
impossible['alpha_z'] = alpha_0 * np.sqrt(1 + impossible['z_phot'])
impossible['z_factor'] = (1 + impossible['z_phot']) / (1 + z_ref)
impossible['gamma_new'] = np.exp(impossible['alpha_z'] * (2/3) * (impossible['log_Mh'] - impossible['log_Mh_ref_z']) * impossible['z_factor'])
impossible['age_ratio_new'] = impossible['age_ratio'] / impossible['gamma_new']

print("Impossible Galaxies Properties:")
print(f"{'ID':<6} {'z':<6} {'logM*':<8} {'logMh':<8} {'AgeRat':<8} {'Gamma':<8} {'NewRat':<8} {'Chi2':<8} {'Use':<4}")
print("-" * 70)

for idx, row in impossible.iterrows():
    # Check if it would be excluded by standard quality cuts if we were stricter
    # Standard cuts in step 1 were: mstar > 8, z > 4.
    # Check chi2
    print(f"{idx:<6} {row['z_phot']:<6.2f} {row['log_Mstar']:<8.2f} {row['log_Mh']:<8.2f} {row['age_ratio']:<8.2f} {row['gamma_new']:<8.2f} {row['age_ratio_new']:<8.2f} {row['chi2']:<8.2f} {row['use_phot']}")
