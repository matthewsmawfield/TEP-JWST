
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE
from scripts.utils.tep_model import KAPPA_GAL, KAPPA_GAL_UNCERTAINTY

def plot_rm_sensitivity():
    # Red Monster Data (from step_90_sign_paradox.json)
    # S1: z=5.3, log_Mh=12.8, Gamma=2.12
    # S2: z=5.5, log_Mh=12.6, Gamma=1.81
    # S3: z=5.9, log_Mh=13.0, Gamma=2.94
    
    # We can infer the exponent K_i for each galaxy where Gamma_i = exp(kappa_gal * K_i)
    # K_i = ln(Gamma_i) / 9.6e5
    
    gammas_nominal = np.array([2.12, 1.81, 2.94])
    alpha_nominal = KAPPA_GAL
    K_factors = np.log(gammas_nominal) / alpha_nominal
    
    sfe_obs = np.array([0.50, 0.48, 0.52])
    sfe_limit = 0.20
    
    # Alpha range
    alphas = np.linspace(0.0, 1.2, 100)
    
    resolved_fractions = []
    
    for alpha in alphas:
        # Calculate Gamma for this alpha
        # Gamma_new = exp(alpha * K_i)
        gammas_new = np.exp(alpha * K_factors)
        
        # Calculate SFE_true
        # SFE_true = SFE_obs / Gamma^0.7
        sfe_true = sfe_obs / (gammas_new**0.7)
        
        # Calculate Resolved Fraction
        # Excess_obs = SFE_obs - 0.20
        # Excess_true = SFE_true - 0.20
        # If Excess_true < 0, fully resolved (clipped at 100% or just handle sign)
        # Resolved = (Excess_obs - Excess_true) / Excess_obs
        
        excess_obs = sfe_obs - sfe_limit
        excess_true = sfe_true - sfe_limit
        
        # Calculate individual resolved fraction
        fracs = (excess_obs - excess_true) / excess_obs
        # Cap at 1.0 (100%) if SFE_true < 0.20
        fracs = np.clip(fracs, None, 1.0)
        
        # Mean resolved fraction
        mean_resolved = np.mean(fracs) * 100 # percentage
        resolved_fractions.append(mean_resolved)
        
    resolved_fractions = np.array(resolved_fractions)
    
    # Plot
    set_pub_style(scale=1.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE['web_standard'])
    
    ax.plot(alphas, resolved_fractions, color=COLORS['primary'], linewidth=2, label='Red Monsters (Mean)')
    
    # Add error band for alpha
    ax.axvspan(KAPPA_GAL - KAPPA_GAL_UNCERTAINTY, KAPPA_GAL + KAPPA_GAL_UNCERTAINTY, color=COLORS['gray'], alpha=0.2, label=r'Cepheid $\kappa_gal$ ($1\sigma$)')
    ax.axvline(KAPPA_GAL, color=COLORS['text'], linestyle='--', label=rf'Nominal $\kappa={KAPPA_GAL:.2f}$')
    
    ax.set_xlabel(r'Coupling Constant $\kappa_gal$')
    ax.set_ylabel('Anomaly Resolved (%)')
    ax.set_title(r'Red Monster SFE Anomaly Resolution vs $\kappa_gal$')
    ax.set_ylim(-20, 120)
    ax.set_xlim(0, 1.2)
    
    # Highlight the value at KAPPA_GAL
    val_nom = np.interp(KAPPA_GAL, alphas, resolved_fractions)
    ax.scatter([KAPPA_GAL], [val_nom], color=COLORS['highlight'], zorder=5)
    ax.text(0.60, val_nom - 10, f"{val_nom:.1f}%", color=COLORS['highlight'], fontweight='bold')
    
    ax.axhline(0, color=COLORS['text'], linewidth=0.5)
    ax.axhline(100, color=COLORS['gray'], linestyle=':')
    
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'site/public/figures/figure_sensitivity_rm.png')
    plt.close()
    print("Saved figure_sensitivity_rm.png")

if __name__ == "__main__":
    plot_rm_sensitivity()
