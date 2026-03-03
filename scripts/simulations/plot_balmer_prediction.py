
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import sys
from pathlib import Path

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

def plot_balmer_prediction():
    # Load data
    try:
        df = pd.read_csv(PROJECT_ROOT / 'scripts/simulations/balmer_prediction.csv')
    except FileNotFoundError:
        print("Error: Data file not found. Run predict_balmer_lines.py first.")
        return

    # Set style
    set_pub_style(scale=1.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE['web_standard'])
    
    # Plot 1: H_delta Strength vs Stellar Mass
    # Bin the data for clearer trends
    bins = np.linspace(8, 11, 8)
    df['mass_bin'] = pd.cut(df['log_m_star'], bins)
    
    # Calculate means and errors
    grouped = df.groupby('mass_bin', observed=True).agg({
        'log_m_star': 'mean',
        'Hd_standard': ['mean', 'std'],
        'Hd_tep': ['mean', 'std']
    }).reset_index()
    
    # Extract for plotting
    x = grouped['log_m_star']['mean']
    y_std = grouped['Hd_standard']['mean']
    err_std = grouped['Hd_standard']['std']
    y_tep = grouped['Hd_tep']['mean']
    err_tep = grouped['Hd_tep']['std']
    
    # Plot Standard Model
    ax1.errorbar(x, y_std, yerr=err_std, fmt='o-', label='Standard Physics', 
                 capsize=3, color=COLORS['gray'], alpha=0.7, markersize=4)
    
    # Plot TEP Prediction
    ax1.errorbar(x, y_tep, yerr=err_tep, fmt='s-', label='TEP Prediction', 
                 capsize=3, color=COLORS['primary'], linewidth=1.0, markersize=5)
    
    ax1.set_xlabel(r'Stellar Mass ($\log M_*/M_\odot$)')
    ax1.set_ylabel(r'H$\delta$ Absorption Strength ($\AA$)')
    ax1.set_title(r'(a) Predicted Balmer Absorption ($z \sim 7$)')
    ax1.legend(loc='upper right')
    
    # Plot 2: H_delta vs Gamma_t
    # Scatter plot
    ax2.scatter(df['gamma_t'], df['Hd_standard'], alpha=0.15, s=10, label='Standard', color=COLORS['gray'], edgecolors='none')
    ax2.scatter(df['gamma_t'], df['Hd_tep'], alpha=0.15, s=10, label='TEP', color=COLORS['primary'], edgecolors='none')
    
    # Fit trends
    sort_idx = np.argsort(df['gamma_t'])
    sorted_gamma = df['gamma_t'].iloc[sort_idx]
    
    # Simple polynomial fit for trend lines (log-linear)
    z_std = np.polyfit(np.log10(df['gamma_t']), df['Hd_standard'], 1)
    p_std = np.poly1d(z_std)
    ax2.plot(sorted_gamma, p_std(np.log10(sorted_gamma)), color=COLORS['gray'], linestyle='--', linewidth=1.0, label='Standard Trend')
    
    z_tep = np.polyfit(np.log10(df['gamma_t']), df['Hd_tep'], 1)
    p_tep = np.poly1d(z_tep)
    ax2.plot(sorted_gamma, p_tep(np.log10(sorted_gamma)), color=COLORS['primary'], linewidth=1.5, label='TEP Trend')
    
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Enhancement Factor $\Gamma_t$')
    ax2.set_ylabel(r'H$\delta$ Absorption Strength ($\AA$)')
    ax2.set_title(r'(b) Predicted Correlation')
    
    # Create custom legend for readability
    custom_lines = [Line2D([0], [0], color=COLORS['gray'], lw=1.0, linestyle='--'),
                    Line2D([0], [0], color=COLORS['primary'], lw=1.5)]
    ax2.legend(custom_lines, ['Standard Physics', 'TEP Prediction'], loc='upper left')
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / 'site/public/figures/figure_10_balmer.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_balmer_prediction()
