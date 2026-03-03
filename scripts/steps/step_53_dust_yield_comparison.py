
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path
import sys
import json

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "site" / "public" / "figures"

# Constants
Z_SOLAR = 0.014
YIELD_SN = 0.02  # Dust yield per SN (Msun) - average
YIELD_AGB = 0.01 # Dust yield per AGB (Msun) - average high
SN_RATE = 0.01   # SN per Msun formed (Salpeter-like)
AGB_DELAY = 0.500 # Gyr (500 Myr standard for significant AGB dust)

class DustEvolutionModel:
    def __init__(self, m_star_final, z_obs, gamma_t=1.0):
        self.m_star_final = m_star_final
        self.z_obs = z_obs
        self.gamma_t = gamma_t
        self.t_cosmic = self.get_cosmic_time(z_obs)
        self.t_eff_total = self.t_cosmic * gamma_t
        
    def get_cosmic_time(self, z):
        # Approximation for flat LambdaCDM (matter dominated at high z)
        # t = 2 / (3 * H0 * sqrt(Omega_m)) * (1+z)^-1.5
        # H0 = 67.4 km/s/Mpc -> 1/H0 = 14.5 Gyr
        # Omega_m = 0.315
        # Pre-factor = 14.5 * (2/3) / sqrt(0.315) = 17.2
        return 17.2 * (1+z)**(-1.5)

    def run_evolution(self):
        # Time array (Gyr)
        t = np.linspace(0, self.t_eff_total, 1000)
        dt = t[1] - t[0]
        
        # Assume Constant SFR for simplicity to reach M_star_final
        sfr = self.m_star_final / self.t_eff_total / 1e9 # Msun/yr
        
        m_dust_sn = np.zeros_like(t)
        m_dust_agb = np.zeros_like(t)
        m_dust_growth = np.zeros_like(t)
        
        # Integration
        cumulative_mass = 0
        for i in range(1, len(t)):
            # Star formation
            dm_star = sfr * dt * 1e9
            cumulative_mass += dm_star
            
            # 1. Supernovae (Instantaneous)
            # Rate ~ SFR * 0.01
            d_dust_sn = dm_star * SN_RATE * YIELD_SN
            
            # 2. AGB Stars (Delayed)
            # Only turn on if t > AGB_DELAY
            if t[i] > AGB_DELAY:
                # Simple step function for onset, could be smoother
                # Assuming mass returning from AGB phase
                d_dust_agb = dm_star * 0.1 * YIELD_AGB # 10% mass evolves via AGB
            else:
                d_dust_agb = 0
                
            # 3. Grain Growth (ISM)
            # Timescale ~ 1 Gyr / Z' (very rough)
            # dM/dt = M_dust / tau_growth
            # Here we assume minimal growth at very low Z/high z, but let's add a term
            # Growth mostly happens when M_dust is already seeded
            
            # Destruction (SN shocks)
            # M_cleared = 1000 Msun per SN (very rough average for ISM sweeping)
            # Fraction destroyed = M_cleared * SN_Rate / M_gas
            # Assume M_gas ~ M_star (f_gas ~ 0.5)
            m_gas = cumulative_mass # Rough approximation
            if m_gas > 0:
                # Destruction rate proportional to current dust mass
                current_dust = m_dust_sn[i-1] + m_dust_agb[i-1]
                # SN_RATE is per Msun formed. dm_star is mass formed this step.
                n_sn = dm_star * SN_RATE
                m_swept = n_sn * 1000 # Msun swept
                f_dest = m_swept / m_gas
                destruction = current_dust * f_dest
            else:
                destruction = 0
            
            # Apply updates
            # Distribute destruction proportionally
            if (m_dust_sn[i-1] + m_dust_agb[i-1]) > 0:
                f_sn = m_dust_sn[i-1] / (m_dust_sn[i-1] + m_dust_agb[i-1])
                dest_sn = destruction * f_sn
                dest_agb = destruction * (1 - f_sn)
            else:
                dest_sn = 0
                dest_agb = 0

            m_dust_sn[i] = m_dust_sn[i-1] + d_dust_sn - dest_sn
            m_dust_agb[i] = m_dust_agb[i-1] + d_dust_agb - dest_agb
            
            # Ensure non-negative
            m_dust_sn[i] = max(0, m_dust_sn[i])
            m_dust_agb[i] = max(0, m_dust_agb[i])
            
        return {
            'time': t,
            'dust_sn': m_dust_sn,
            'dust_agb': m_dust_agb,
            'dust_total': m_dust_sn + m_dust_agb,
            'final_dust': m_dust_sn[-1] + m_dust_agb[-1]
        }

def convert_av_to_mdust(av, m_star, z):
    # Approximation from Li et al. (2019) / standard dust-to-gas
    # M_dust = (A_V * 1.086 * S) / kappa
    # S = pi * R^2
    # R (kpc) = 0.15 * (1+z)^-1 * M^0.3 (Mass-Size relation high-z)
    
    # R in cm
    r_kpc = 0.5 * (m_star/1e9)**0.2 # Typical size ~0.5 kpc for 10^9
    r_cm = r_kpc * 3.086e21
    area = np.pi * r_cm**2
    
    # kappa ~ 1e4 - 1e5 cm2/g
    kappa = 5e4 
    
    m_dust_g = (av / 1.086) * area / kappa
    m_dust_msun = m_dust_g / 1.989e33
    
    return m_dust_msun

def main():
    print("Running Dust Yield Comparison...")
    set_pub_style(scale=1.0)
    
    # 1. Load Data
    try:
        df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    except:
        print("Could not load data file.")
        return

    # Filter z > 8 and massive
    high_z = df[(df['z_phot'] >= 8) & (df['log_Mstar'] > 9)].copy()
    
    print(f"Analyzing {len(high_z)} massive galaxies at z > 8")
    
    results = []
    
    for idx, row in high_z.iterrows():
        m_star = 10**row['log_Mstar']
        z = row['z_phot']
        av = row['dust']
        gamma_t = row['gamma_t']
        
        # Estimate Observed Dust Mass
        m_dust_obs = convert_av_to_mdust(av, m_star, z)
        
        # Run Standard Model (Gamma = 1)
        std_model = DustEvolutionModel(m_star, z, gamma_t=1.0)
        res_std = std_model.run_evolution()
        
        # Run TEP Model (Gamma = Obs)
        tep_model = DustEvolutionModel(m_star, z, gamma_t=gamma_t)
        res_tep = tep_model.run_evolution()
        
        results.append({
            'id': row['id'],
            'z': z,
            'm_star': m_star,
            'av_obs': av,
            'm_dust_obs': m_dust_obs,
            'm_dust_std_max': res_std['final_dust'],
            'm_dust_tep_max': res_tep['final_dust'],
            'gamma_t': gamma_t
        })
        
    res_df = pd.DataFrame(results)
    
    # Calculate Deficits
    res_df['deficit_std'] = res_df['m_dust_obs'] / res_df['m_dust_std_max']
    res_df['deficit_tep'] = res_df['m_dust_obs'] / res_df['m_dust_tep_max']
    
    print("\nResults Summary:")
    print(f"Mean Dust Excess (Standard): {res_df['deficit_std'].median():.2f}x")
    print(f"Mean Dust Excess (TEP): {res_df['deficit_tep'].median():.2f}x")
    
    # Count "Impossible" (Deficit > 2x to account for uncertainties)
    impossible_std = len(res_df[res_df['deficit_std'] > 2])
    impossible_tep = len(res_df[res_df['deficit_tep'] > 2])
    
    print(f"Impossible in Standard: {impossible_std}/{len(res_df)}")
    print(f"Impossible in TEP: {impossible_tep}/{len(res_df)}")
    
    # Save results
    res_df.to_csv(OUTPUT_PATH / "step_53_dust_yield_comparison.csv", index=False)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=FIG_SIZE['web_standard'])
    
    # Shaded Region (Ratio > 1 => Log10 > 0)
    ax.axhspan(0, 3, color='#95a5a6', alpha=0.1, zorder=0)
    ax.text(8.1, 0.5, "Standard Model\nYield Limit Exceeded\n(Yield > 100%)", 
             color='#7f8c8d', fontsize=10, fontweight='normal', va='center')

    ax.scatter(res_df['z'], np.log10(res_df['deficit_std']), label='Standard Physics', 
                color=COLORS['secondary'], alpha=0.7, s=60, edgecolors='w', zorder=2)
    ax.scatter(res_df['z'], np.log10(res_df['deficit_tep']), label='TEP (Time Enhanced)', 
                color=COLORS['highlight'], alpha=0.7, s=60, edgecolors='w', zorder=2)
    
    ax.axhline(0, color=COLORS['text'], linestyle='--', linewidth=1.5, label='Theoretical Limit (100% Yield)')
    
    # Arrows indicating TEP shift
    # Draw a few representative arrows
    for i in range(0, len(res_df), 5): # Every 5th point to avoid clutter
        z = res_df.iloc[i]['z']
        y1 = np.log10(res_df.iloc[i]['deficit_std'])
        y2 = np.log10(res_df.iloc[i]['deficit_tep'])
        ax.annotate("", xy=(z, y2), xytext=(z, y1),
                     arrowprops=dict(arrowstyle="->", color=COLORS['gray'], alpha=0.5))

    ax.set_ylabel(r'Dust Deficit $\log_{10}(M_{dust}^{obs} / M_{dust}^{max})$')
    ax.set_xlabel('Redshift $z$')
    ax.set_title('Dust Yield Comparison: Standard vs TEP')
    ax.legend(loc='lower left', frameon=True, facecolor=COLORS['background'])
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "figure_dust_deficit.png", dpi=300)
    print(f"Figure saved to {FIGURES_PATH / 'figure_dust_deficit.png'}")

if __name__ == "__main__":
    main()
