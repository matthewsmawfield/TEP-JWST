
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

def plot_z67_split():
    set_pub_style(scale=1.0)
    # Load data
    df = pd.read_csv(PROJECT_ROOT / 'results/interim/step_001_uncover_full_sample.csv')
    
    # Filter for z=6-7
    z_min, z_max = 6.0, 7.0
    mask = (df['z_phot'] >= z_min) & (df['z_phot'] < z_max) & (df['dust'] > -99) & (df['log_Mstar'] > 7)
    subset = df[mask].copy()
    
    median_ssfr = subset['log_ssfr'].median()
    subset['sSFR_Class'] = np.where(subset['log_ssfr'] > median_ssfr, 'High sSFR', 'Low sSFR')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE.get('web_two_panel', (12, 5)), sharey=True)
    
    # High sSFR
    high = subset[subset['sSFR_Class'] == 'High sSFR']
    # Manual regplot
    axes[0].scatter(high['log_Mstar'], high['dust'], alpha=0.5, color=COLORS['blue'], label='Data')
    if len(high) > 1:
        m, b = np.polyfit(high['log_Mstar'], high['dust'], 1)
        axes[0].plot(high['log_Mstar'], m*high['log_Mstar'] + b, color=COLORS['primary'], label='Fit')
    
    axes[0].set_title(f"High sSFR (Above Median)\nN={len(high)}, rho={high[['log_Mstar','dust']].corr(method='spearman').iloc[0,1]:.2f}")
    axes[0].set_xlabel("log(Stellar Mass)")
    axes[0].set_ylabel("Dust (Av)")
    axes[0].set_ylim(0, 3)
    axes[0].legend()
    
    # Low sSFR
    low = subset[subset['sSFR_Class'] == 'Low sSFR']
    axes[1].scatter(low['log_Mstar'], low['dust'], alpha=0.5, color=COLORS['accent'], label='Data')
    if len(low) > 1:
        m, b = np.polyfit(low['log_Mstar'], low['dust'], 1)
        axes[1].plot(low['log_Mstar'], m*low['log_Mstar'] + b, color=COLORS['highlight'], label='Fit')
        
    axes[1].set_title(f"Low sSFR (Below Median)\nN={len(low)}, rho={low[['log_Mstar','dust']].corr(method='spearman').iloc[0,1]:.2f}")
    axes[1].set_xlabel("log(Stellar Mass)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'site/public/figures/figure_z67_split.png')
    plt.close()
    print("Saved figure_z67_split.png")

def plot_lrd_sensitivity():
    set_pub_style(scale=1.0)
    # Load data
    try:
        df = pd.read_csv(PROJECT_ROOT / 'results/outputs/lrd_radius_sensitivity.csv')
    except FileNotFoundError:
        print("LRD Sensitivity CSV not found, skipping plot.")
        return

    # Handle infinite/overflow values for plotting
    max_plot_val = 1e20
    df['boost_factor_plot'] = df['boost_factor'].replace([np.inf, -np.inf], max_plot_val)
    df['boost_factor_plot'] = df['boost_factor_plot'].clip(upper=max_plot_val)

    fig, ax = plt.subplots(figsize=FIG_SIZE['web_standard'])
    
    # Plot Boost vs Radius
    ax.plot(df['r_eff_pc'], df['boost_factor_plot'], linewidth=3, color=COLORS['highlight'])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('Effective Radius $r_e$ [pc]')
    ax.set_ylabel('Growth Boost Factor (Overmassive BH)')
    ax.set_title('LRD Sensitivity: Boost vs Compactness')
    
    # Add threshold lines
    ax.axhline(100, color=COLORS['gray'], linestyle='--', label='Boost = 100x')
    ax.axhline(10, color=COLORS['gray'], linestyle=':', label='Boost = 10x')
    
    # Find critical radius for 100x
    crit = df[df['boost_factor'] < 100].iloc[0] if any(df['boost_factor'] < 100) else None
    if crit is not None:
        ax.axvline(crit['r_eff_pc'], color=COLORS['text'], linestyle='-.')
        ax.text(crit['r_eff_pc']*1.1, 150, f"Critical Radius\n< {crit['r_eff_pc']:.0f} pc", verticalalignment='bottom')
    
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'site/public/figures/figure_lrd_sensitivity.png')
    plt.close()
    print("Saved figure_lrd_sensitivity.png")

if __name__ == "__main__":
    plot_z67_split()
    plot_lrd_sensitivity()
