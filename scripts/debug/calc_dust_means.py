import pandas as pd
import numpy as np

try:
    df = pd.read_csv('/Users/matthewsmawfield/www/TEP-JWST/results/outputs/step_53_dust_yield_comparison.csv')
    mean_deficit_std = df['deficit_std'].mean()
    mean_deficit_tep = df['deficit_tep'].mean()
    print(f"Mean Deficit (Standard): {mean_deficit_std:.4f}")
    print(f"Mean Deficit (TEP): {mean_deficit_tep:.4f}")
except Exception as e:
    print(f"Error: {e}")
