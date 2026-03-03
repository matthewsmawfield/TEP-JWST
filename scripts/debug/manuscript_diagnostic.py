
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Paths
base_path = Path("/Users/matthewsmawfield/www/TEP-JWST")
full_sample_path = base_path / "results/interim/step_02_uncover_full_sample_tep.csv"
multi_sample_path = base_path / "results/interim/step_02_uncover_multi_property_sample_tep.csv"

def analyze():
    print("--- DIAGNOSTIC START ---")
    
    # Load data
    try:
        df_full = pd.read_csv(full_sample_path)
        df_multi = pd.read_csv(multi_sample_path)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # Check Gamma_t range
    print(f"Full Sample N={len(df_full)}")
    print(f"Gamma_t range: {df_full['gamma_t'].min()} to {df_full['gamma_t'].max()}")
    print(f"Log Gamma_t range: {np.log10(df_full['gamma_t']).min()} to {np.log10(df_full['gamma_t']).max()}")
    
    # Regime Counts (Full)
    enhanced_full = df_full[df_full['gamma_t'] > 1]
    suppressed_full = df_full[df_full['gamma_t'] <= 1]
    print(f"Full Enhanced (Gt>1): N={len(enhanced_full)}")
    print(f"Full Suppressed (Gt<=1): N={len(suppressed_full)}")
    
    # Regime Counts (Multi)
    enhanced_multi = df_multi[df_multi['gamma_t'] > 1]
    suppressed_multi = df_multi[df_multi['gamma_t'] <= 1]
    print(f"Multi Enhanced (Gt>1): N={len(enhanced_multi)}")
    print(f"Multi Suppressed (Gt<=1): N={len(suppressed_multi)}")
    
    # Table 11 verification (Regime Properties)
    # Regimes: <0.5, 0.5-1, >=1
    r1 = df_multi[df_multi['gamma_t'] < 0.5]
    r2 = df_multi[(df_multi['gamma_t'] >= 0.5) & (df_multi['gamma_t'] < 1)]
    r3 = df_multi[df_multi['gamma_t'] >= 1]
    
    print("\nTable 11 Re-calc (Multi Sample):")
    print(f"Suppressed (<0.5): N={len(r1)}, Mean Gt={r1['gamma_t'].mean():.2f}")
    print(f"Neutral (0.5-1): N={len(r2)}, Mean Gt={r2['gamma_t'].mean():.2f}")
    print(f"Enhanced (>=1): N={len(r3)}, Mean Gt={r3['gamma_t'].mean():.2f}")
    
    # Table 15 verification (Correlation Matrix)
    # Using Full or Multi? Text says N=1952+355=2307 (Close to Full 2315)
    # Let's check Full sample correlations if columns exist
    # Columns needed: age_ratio (or similar), dust, metallicity
    # Multi sample has these. Full might not.
    print(f"\nMulti Columns: {df_multi.columns.tolist()}")
    
    # Table 16 verification (Extreme vs Normal)
    # Top 5% of age_ratio
    top5_age = df_multi[df_multi['age_ratio'] > df_multi['age_ratio'].quantile(0.95)]
    normal_age = df_multi[df_multi['age_ratio'] <= df_multi['age_ratio'].quantile(0.95)]
    print("\nTable 16 Re-calc:")
    print(f"Top 5% Age: Mean Gt={top5_age['gamma_t'].mean():.2f}, Log Gt={np.log10(top5_age['gamma_t']).mean():.2f}")
    print(f"Normal Age: Mean Gt={normal_age['gamma_t'].mean():.2f}, Log Gt={np.log10(normal_age['gamma_t']).mean():.2f}")
    
    print("--- DIAGNOSTIC END ---")

if __name__ == "__main__":
    analyze()
