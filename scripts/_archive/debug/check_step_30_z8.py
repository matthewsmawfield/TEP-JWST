
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing functions from step_30
from scripts.steps.step_30_model_comparison import load_data, compare_models_for_property

def check_z8_aic():
    print("Loading data...")
    df = load_data()
    
    # Filter for z > 8
    df_z8 = df[df['z'] > 8].copy()
    print(f"Data loaded. Total N={len(df)}, z>8 N={len(df_z8)}")
    
    print("\n--- Running Model Comparison for Dust (Av) at z > 8 ---")
    results = compare_models_for_property(df_z8, 'Av', 'Dust (A_V)')
    
    if results:
        print("\nResults for z > 8:")
        delta_aic = results['comparison']['delta_aic']
        print(f"Null (z) AIC: {results['models']['null']['aic']:.2f}")
        print(f"Mass (M, z) AIC: {results['models']['mass']['aic']:.2f}")
        print(f"TEP (G, z) AIC: {results['models']['tep']['aic']:.2f}")
        print(f"Full (M, G, z) AIC: {results['models']['full']['aic']:.2f}")
        
        print("\nDelta AIC (relative to best):")
        print(f"Null: {delta_aic['null']:.2f}")
        print(f"Mass: {delta_aic['mass']:.2f}")
        print(f"TEP:  {delta_aic['tep']:.2f}")
        print(f"Full: {delta_aic['full']:.2f}")
        
        full_vs_mass = results['comparison']['full_vs_mass_aic']
        print(f"\nFull vs Mass AIC: {full_vs_mass:.2f}")
        if full_vs_mass > 2:
            print("Full model preferred over Mass (TEP adds value)")
        else:
            print("Full model NOT preferred over Mass")
            
        # Check coefficient
        if results['models']['full']['coefficients']:
            print(f"\nFull Model Gamma Coefficient: {results['models']['full']['coefficients']['gamma_t']:.4f}")

if __name__ == "__main__":
    check_z8_aic()
