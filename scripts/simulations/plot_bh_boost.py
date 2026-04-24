#!/usr/bin/env python3
"""
Plot TEP Black Hole Growth Boost
Visualizes the results from Step 41 (Overmassive Black Hole Resolution).
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "results" / "outputs" / "step_41_bh_growth_table.csv"
FIGURE_OUTPUT = PROJECT_ROOT / "site" / "figures" / "bh_growth_boost.png"

def main():
    # Load simulation data
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run step_41_overmassive_bh.py first.")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    
    # Setup Plot
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Boost Factor (Log Scale)
    color = '#ff4d4d' # Red for LRDs
    ax1.set_xlabel('Redshift (z)', fontsize=12)
    ax1.set_ylabel('TEP Growth Boost Factor', color=color, fontsize=12)
    
    # Plot the curve
    line1, = ax1.plot(df['z'], df['boost_factor'], color=color, linewidth=3, marker='o', label='TEP Boost')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add effective time context on secondary axis
    ax2 = ax1.twinx()
    color2 = '#4dff4d' # Green
    
    # Calculate ratio of effective times
    time_ratio = df['gamma_cen'] / df['gamma_halo']
    # Handle division by zero or small values safely
    time_ratio = np.where(df['gamma_halo'] > 0.001, df['gamma_cen'] / df['gamma_halo'], df['gamma_cen']/0.01)
    
    line2, = ax2.plot(df['z'], time_ratio, color=color2, linewidth=2, linestyle='--', marker='s', label='Time Rate Ratio')
    ax2.set_ylabel(r'Time Rate Ratio ($\Gamma_{cen} / \Gamma_{halo}$)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Annotations
    ax1.grid(True, alpha=0.2)
    plt.title('TEP Resolution of Overmassive Black Holes', fontsize=16, pad=20)
    
    # Add text box explaining the mechanism
    textstr = '\n'.join((
        r'$\bf{Differential\ Temporal\ Topology}$',
        r'Deep Core ($\Gamma_t \approx 1.5$)',
        r'vs Diffuse Halo ($\Gamma_t \approx 0.0$)',
        r'Result: Runaway Growth',
        r'at $z > 6$'
    ))
    props = dict(boxstyle='round', facecolor='#222222', alpha=0.8, edgecolor='gray')
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    # Highlight the critical z=8 region
    z_target = 8
    boost_val = df.loc[df['z'] == z_target, 'boost_factor'].values[0]
    ax1.annotate(f'z=8 LRDs\nBoost ~ {boost_val:.0e}x', xy=(z_target, boost_val), xytext=(z_target-1.5, boost_val/100),
                 arrowprops=dict(facecolor='white', shrink=0.05), fontsize=10, color='white')

    # Save
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT, dpi=300)
    print(f"Figure saved to {FIGURE_OUTPUT}")

if __name__ == "__main__":
    main()
