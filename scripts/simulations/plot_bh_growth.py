
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

def plot_bh_growth():
    # Load data
    try:
        df = pd.read_csv(PROJECT_ROOT / 'results/outputs/step_41_bh_growth_table.csv')
    except FileNotFoundError:
        print("Error: Results file not found. Run scripts/steps/step_41_overmassive_bh.py first.")
        return

    # Set style
    set_pub_style(scale=1.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE['web_standard'])
    
    # Plot 1: Gamma differential vs Redshift
    ax1.plot(df['z'], df['gamma_cen'], 'o-', color=COLORS['highlight'], label=r'Central BH ($\Gamma_{\rm cen}$)', linewidth=1.5, markersize=6)
    ax1.plot(df['z'], df['gamma_halo'], 's--', color=COLORS['accent'], label=r'Stellar Halo ($\Gamma_{\rm halo}$)', linewidth=1.5, markersize=6)
    
    ax1.fill_between(df['z'], df['gamma_halo'], df['gamma_cen'], color=COLORS['gray'], alpha=0.1)
    
    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel(r'Chronological Enhancement $\Gamma_t$')
    ax1.set_title(r'(a) The "Time Bubble" Effect')
    ax1.legend(loc='upper left', frameon=False)
    ax1.set_ylim(0, 2.5)
    
    # Annotate differential
    ax1.annotate('Differential Time Rate:\nBH evolves ~2x faster', 
                 xy=(9, 1.3), xytext=(6, 1.8),
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], linewidth=0.3, mutation_scale=4),
                 fontsize=10, ha='center')
    
    # Plot 2: Growth Boost Factor (Log Scale)
    ax2.plot(df['z'], df['boost_factor'], 'D-', color=COLORS['primary'], linewidth=1.5, markersize=6)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Redshift $z$')
    ax2.set_ylabel(r'Mass Growth Boost ($M_{\rm TEP} / M_{\rm Std}$)')
    ax2.set_title(r'(b) Runaway Black Hole Growth')
    ax2.set_ylim(1e0, 1e10)
    
    # Annotate significant points
    ax2.annotate(r'$\times 10^6$ Boost', xy=(6, 5.6e5), xytext=(5, 1e2),
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], linewidth=0.3, mutation_scale=4),
                 fontsize=10)
                 
    # Add comparison line for local ratio
    ax2.axhline(100, color=COLORS['gray'], linestyle=':', label=r'Observed LRD Excess ($\sim 100\times$)')
    ax2.legend(loc='upper left', frameon=False)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / 'site/public/figures/figure_11_bh_growth.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_bh_growth()
