
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "site" / "public" / "figures"

# Constants
SN_RATE = 0.01 # SN per Msun formed
F_GAS = 1.0 # Standard assumption
AGB_DELAY = 0.500 # Canonical delay (was 0.300)

def get_cosmic_time(z):
    # Approximation for flat LambdaCDM (matter dominated at high z)
    # Pre-factor = 14.5 * (2/3) / sqrt(0.315) = 17.2
    return 17.2 * (1+z)**(-1.5)

def calculate_max_production(m_star, z, yield_sn, gamma_t=1.0):
    """
    Calculate max dust mass for a given effective yield.
    Simplified analytic approximation to the previous numerical model.
    """
    t_cosmic = get_cosmic_time(z)
    t_eff = t_cosmic * gamma_t
    
    # 1. SN Dust
    # Total stars formed = m_star
    # Total SN = m_star * SN_RATE
    # Dust = Total SN * yield_sn
    # Note: This ignores destruction, so it's a true UPPER LIMIT on production capability
    # If even this is insufficient, it's strong evidence.
    m_dust_sn = m_star * SN_RATE * yield_sn
    
    # 2. AGB Dust
    # Only stars older than 300 Myr contribute
    # Fraction of stars > 300 Myr depends on SFH. Constant SFR:
    if t_eff > AGB_DELAY:
        f_agb = (t_eff - AGB_DELAY) / t_eff
        # Yield AGB is roughly 0.01 per star, but only ~10% are AGB progenitors in range
        # Let's parameterize AGB yield relative to SN yield for simplicity, 
        # or stick to fixed AGB yield = 0.01
        m_dust_agb = (m_star * f_agb) * 0.1 * 0.01
    else:
        m_dust_agb = 0
        
    return m_dust_sn + m_dust_agb

def convert_av_to_mdust(av, m_star):
    r_kpc = 0.5 * (m_star/1e9)**0.2
    r_cm = r_kpc * 3.086e21
    area = np.pi * r_cm**2
    kappa = 5e4 
    m_dust_g = (av / 1.086) * area / kappa
    return m_dust_g / 1.989e33

def main():
    print("Running Required Yield Analysis...")
    set_pub_style(scale=1.0)
    try:
        df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    except:
        return

    high_z = df[(df['z_phot'] >= 8) & (df['log_Mstar'] > 9)].copy()
    
    required_yields_std = []
    required_yields_tep = []
    
    for idx, row in high_z.iterrows():
        m_star = 10**row['log_Mstar']
        m_dust_obs = convert_av_to_mdust(row['dust'], m_star)
        
        # We need to solve for yield_sn such that Production >= m_dust_obs
        # M_dust_obs = (M_star * SN_RATE * Y) + M_agb(fixed)
        # Y = (M_dust_obs - M_agb) / (M_star * SN_RATE)
        
        # Standard
        gamma_std = 1.0
        t_eff_std = get_cosmic_time(row['z_phot']) * gamma_std
        if t_eff_std > AGB_DELAY:
            f_agb_std = (t_eff_std - AGB_DELAY) / t_eff_std
            m_agb_std = (m_star * f_agb_std) * 0.1 * 0.01
        else:
            m_agb_std = 0
            
        req_y_std = (m_dust_obs - m_agb_std) / (m_star * SN_RATE)
        required_yields_std.append(req_y_std)
        
        # TEP
        gamma_tep = row['gamma_t']
        t_eff_tep = get_cosmic_time(row['z_phot']) * gamma_tep
        if t_eff_tep > AGB_DELAY:
            f_agb_tep = (t_eff_tep - AGB_DELAY) / t_eff_tep
            m_agb_tep = (m_star * f_agb_tep) * 0.1 * 0.01
        else:
            m_agb_tep = 0
            
        req_y_tep = (m_dust_obs - m_agb_tep) / (m_star * SN_RATE)
        required_yields_tep.append(req_y_tep)

    # Analyze
    req_y_std = np.array(required_yields_std)
    req_y_tep = np.array(required_yields_tep)
    
    print(f"Median Required Yield (Standard): {np.median(req_y_std):.4f} Msun/SN")
    print(f"Median Required Yield (TEP): {np.median(req_y_tep):.4f} Msun/SN")
    
    print(f"Fraction requiring > 0.05 Msun/SN (Standard): {np.sum(req_y_std > 0.05) / len(req_y_std):.2f}")
    print(f"Fraction requiring > 0.05 Msun/SN (TEP): {np.sum(req_y_tep > 0.05) / len(req_y_tep):.2f}")
    
    # Save Plot
    fig, ax = plt.subplots(figsize=FIG_SIZE['web_standard'])
    
    ax.hist(req_y_std, bins=20, alpha=0.5, label='Standard Physics', color=COLORS['secondary'], range=(-0.05, 0.3))
    ax.hist(req_y_tep, bins=20, alpha=0.5, label='TEP', color=COLORS['highlight'], range=(-0.05, 0.3))
    
    ax.axvline(0.02, color=COLORS['text'], linestyle='--', label='Canonical Yield (0.02)')
    ax.axvline(0.10, color=COLORS['accent'], linestyle=':', label='Theoretical Max (0.1)')
    
    ax.set_xlabel(r'Required Effective Dust Yield ($M_\odot$ per SN)')
    ax.set_ylabel('Number of Galaxies')
    ax.set_title('Required SN Dust Yields to Explain z > 8 Observations')
    ax.legend(facecolor=COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "figure_required_yields.png")
    print(f"Saved figure to {FIGURES_PATH / 'figure_required_yields.png'}")

if __name__ == "__main__":
    main()
