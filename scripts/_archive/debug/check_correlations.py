
import pandas as pd
import numpy as np
import pingouin as pg
from pathlib import Path

# Load data
data_path = Path("results/interim/step_02_uncover_full_sample_tep.csv")
df = pd.read_csv(data_path)

# Filter for z > 8 and valid data
mask = (df['z_phot'] > 8) & df['dust'].notna() & df['gamma_t'].notna() & df['log_Mstar'].notna()
sample = df[mask]

print(f"Sample size (z > 8): {len(sample)}")

# 1. Raw Correlation
corr_raw = pg.corr(sample['gamma_t'], sample['dust'], method='spearman')
print(f"Raw Gamma_t vs Dust: rho = {corr_raw['r'].values[0]:.3f}, p = {corr_raw['p-val'].values[0]:.2e}")

# 2. Partial (Control z)
corr_pz = pg.partial_corr(data=sample, x='gamma_t', y='dust', covar='z_phot', method='spearman')
print(f"Partial (Control z): rho = {corr_pz['r'].values[0]:.3f}, p = {corr_pz['p-val'].values[0]:.2e}")

# 3. Partial (Control M*)
corr_pm = pg.partial_corr(data=sample, x='gamma_t', y='dust', covar='log_Mstar', method='spearman')
print(f"Partial (Control M*): rho = {corr_pm['r'].values[0]:.3f}, p = {corr_pm['p-val'].values[0]:.2e}")

# 4. Double Control (z + M*)
corr_pd = pg.partial_corr(data=sample, x='gamma_t', y='dust', covar=['z_phot', 'log_Mstar'], method='spearman')
print(f"Double Control (z + M*): rho = {corr_pd['r'].values[0]:.3f}, p = {corr_pd['p-val'].values[0]:.2e}")
