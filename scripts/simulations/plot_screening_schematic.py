
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

def plot_screening_schematic():
    """
    Create a schematic illustrating the two screening mechanisms:
    1. Group Halo Screening (Environmental)
    2. Core Screening (Internal)
    """
    # Set style
    set_pub_style(scale=1.0)
    
    # Schematics look better with minimal grids
    plt.rcParams['axes.grid'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE['web_standard'])
    
    # ---------------------------------------------------------
    # Panel 1: Group Halo Screening (Potential Depth vs Position)
    # ---------------------------------------------------------
    
    x = np.linspace(-10, 10, 300)
    
    # Deep Group Potential
    phi_group = -1.0 * np.exp(-x**2 / 20)  # Broad deep well
    
    # Galaxy potentials
    # Galaxy A: In the group (deep potential)
    phi_gal_A = -0.5 * np.exp(-(x + 2)**2 / 0.5)
    
    # Galaxy B: In the field (shallow background)
    phi_field = -0.1 # Constant background
    phi_gal_B = -0.5 * np.exp(-(x - 5)**2 / 0.5) 
    
    # Plot Group Potential Background
    ax1.plot(x, phi_group, '--', color=COLORS['text'], label='Ambient Potential', alpha=0.4, linewidth=1.0)
    ax1.axhline(-0.1, color=COLORS['gray'], linestyle=':', alpha=0.4) # Field level
    
    # Plot Galaxy A (Screened)
    ax1.plot(x, phi_group + phi_gal_A, color=COLORS['highlight'], linewidth=1.5, label='Galaxy in Group (Screened)')
    
    # Plot Galaxy B (Unscreened)
    ax1.plot(x, phi_field + phi_gal_B, color=COLORS['accent'], linewidth=1.5, label='Galaxy in Field (Unscreened)')
    
    ax1.set_title('(a) Environmental Screening')
    ax1.set_ylabel(r'Gravitational Potential $\Phi$')
    ax1.set_xlabel('Spatial Position')
    ax1.set_yticks([])
    ax1.set_xticks([])
    
    # Annotate
    ax1.annotate('Deep Potential Well\n(TEP Saturation)', 
                 xy=(-2, -1.2), xytext=(-2, -1.6),
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], linewidth=0.3, mutation_scale=4),
                 ha='center', fontsize=10)
    
    ax1.annotate('Shallow Background\n(Strong TEP Effect)', 
                 xy=(5, -0.6), xytext=(5, -0.9),
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], linewidth=0.3, mutation_scale=4),
                 ha='center', fontsize=10)
    
    ax1.legend(loc='upper right', frameon=False)
    
    # ---------------------------------------------------------
    # Panel 2: Core Screening (Gamma_t vs Radius)
    # ---------------------------------------------------------
    
    r = np.linspace(0.1, 5, 200)
    
    # Unscreened Profile (Field) - High enhancement in core
    gamma_unscreened = 1 + 2.5 * np.exp(-r/0.8)
    
    # Comparison line (Cosmic Mean)
    ax2.axhline(1.0, color=COLORS['gray'], linestyle='--', label=r'Cosmic Mean ($\Gamma_t=1$)', linewidth=1.0)
    
    ax2.plot(r, gamma_unscreened, color=COLORS['primary'], linewidth=2.0, label=r'Predicted $\Gamma_t$ Profile')
    
    ax2.set_title('(b) Radial TEP Gradient ("Inside-Out")')
    ax2.set_ylabel(r'Enhancement Factor $\Gamma_t$')
    ax2.set_xlabel('Galactocentric Radius')
    ax2.set_ylim(0.5, 4.0)
    
    # Annotate Core
    ax2.annotate(r'Core Region: High $\Gamma_t$ → Older Apparent Age', 
                 xy=(0.2, 3.2), xytext=(1.8, 3.2),
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], linewidth=0.3, mutation_scale=4),
                 fontsize=10)
                 
    # Annotate Outskirts
    ax2.annotate(r'Outskirts: $\Gamma_t \to 1$ → True Age', 
                 xy=(4.5, 1.05), xytext=(3.5, 1.8),
                 arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], linewidth=0.3, mutation_scale=4),
                 ha='center', fontsize=10)
                 
    ax2.legend(loc='upper right', frameon=False)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / 'site/public/figures/figure_8_screening.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_screening_schematic()
