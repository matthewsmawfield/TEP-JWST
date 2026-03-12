
import pandas as pd
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "results" / "interim" / "step_01_uncover_full_sample.csv"

df = pd.read_csv(DATA_PATH)
print(f"Current Full Sample: {len(df)}")
print(f"use_phot value counts:")
print(df['use_phot'].value_counts())

impossible = df[df['age_ratio'] > 0.5]
print(f"\nImpossible Galaxies ({len(impossible)}):")
print(impossible['use_phot'].value_counts())
