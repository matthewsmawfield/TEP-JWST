
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

def plot_sensitivity():
    # Load data
    try:
        df = pd.read_csv(PROJECT_ROOT / 'results/outputs/step_40_sensitivity.csv')
    except FileNotFoundError:
        print("Error: Sensitivity data not found. Run step_40_sensitivity_analysis.py first.")
        return
    
    # Constants (from tep_model.py, Cepheid-calibrated magnitude sector)
    KAPPA_GAL = 9.6e5  # mag
    KAPPA_GAL_UNCERTAINTY = 4.0e5  # mag
    
    # Set style
    set_pub_style(scale=1.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE['web_standard'])
    
    # Plot 1: Dust Correlation vs κ_gal
    # Highlight the nominal region
    ax1.axvspan(KAPPA_GAL - KAPPA_GAL_UNCERTAINTY, KAPPA_GAL + KAPPA_GAL_UNCERTAINTY,
                color=COLORS['gray'], alpha=0.15, label=r'Cepheid Calibration ($1\sigma$)')
    
    ax1.plot(df['kappa_gal'], df['rho_dust_z8'], color=COLORS['primary'], linewidth=1.5, label=r'$\rho(M_*, \mathrm{Dust})$ at $z>8$')
    
    # Mark nominal
    # Find closest to nominal
    idx_nom = (df['kappa_gal'] - KAPPA_GAL).abs().argmin()
    nom_rho = df.iloc[idx_nom]['rho_dust_z8']
    
    ax1.plot(KAPPA_GAL, nom_rho, 'o', color=COLORS['highlight'], markersize=8, label='Nominal Model', zorder=5)
    
    ax1.set_xlabel(r'Coupling Parameter $\kappa_gal$')
    ax1.set_ylabel(r'Correlation Strength ($\rho$)')
    ax1.set_title(r'(a) Robustness of the $z>8$ Dust Anomaly')
    ax1.set_xlim(0, 2e6)
    ax1.set_ylim(0, 0.8) 
    ax1.legend(loc='lower right', frameon=False)
    
    # Plot 2: P-value (log scale)
    ax2.axvspan(KAPPA_GAL - KAPPA_GAL_UNCERTAINTY, KAPPA_GAL + KAPPA_GAL_UNCERTAINTY,
                color=COLORS['gray'], alpha=0.15, label=r'Cepheid Calibration ($1\sigma$)')
    
    # Avoid log(0)
    p_vals = df['p_dust_z8'].replace(0, 1e-50)
    ax2.semilogy(df['kappa_gal'], p_vals, color=COLORS['primary'], linewidth=1.5)
    
    # Significance threshold line
    ax2.axhline(0.05, color=COLORS['highlight'], linestyle='--', linewidth=1.0, label=r'Significance Threshold ($p=0.05$)')
    
    ax2.set_xlabel(r'Coupling Parameter $\kappa_gal$')
    ax2.set_ylabel(r'$p$-value (Spearman)')
    ax2.set_title(r'(b) Statistical Significance')
    ax2.set_xlim(0, 2e6)

    # Add annotation for robust region (centered on nominal kappa_gal)
    ax2.text(KAPPA_GAL, 1e-10, "Highly Significant\nRegion", ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax2.legend(loc='upper right', frameon=False)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / 'site/public/figures/figure_9_sensitivity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_sensitivity()
