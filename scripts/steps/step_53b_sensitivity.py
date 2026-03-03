
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"

# Constants
YIELD_SN = 0.1  # Optimistic SN Yield (standard was 0.02)
YIELD_AGB = 0.01
SN_RATE = 0.01
AGB_DELAY = 0.300
F_GAS = 3.0 # High gas fraction (M_gas = 3 * M_star) -> Reduces destruction

class DustEvolutionModel:
    def __init__(self, m_star_final, z_obs, gamma_t=1.0):
        self.m_star_final = m_star_final
        self.z_obs = z_obs
        self.gamma_t = gamma_t
        self.t_cosmic = 14.5 * (2/3) * (1+z_obs)**(-1.5)
        self.t_eff_total = self.t_cosmic * gamma_t
        
    def run_evolution(self):
        t = np.linspace(0, self.t_eff_total, 1000)
        dt = t[1] - t[0]
        sfr = self.m_star_final / self.t_eff_total / 1e9
        
        m_dust_sn = np.zeros_like(t)
        m_dust_agb = np.zeros_like(t)
        
        cumulative_mass = 0
        
        for i in range(1, len(t)):
            dm_star = sfr * dt * 1e9
            cumulative_mass += dm_star
            
            # Optimistic Production
            d_dust_sn = dm_star * SN_RATE * YIELD_SN
            
            if t[i] > AGB_DELAY:
                d_dust_agb = dm_star * 0.1 * YIELD_AGB
            else:
                d_dust_agb = 0
                
            # Reduced Destruction
            # M_gas is larger now
            m_gas = cumulative_mass * F_GAS
            
            if m_gas > 0:
                current_dust = m_dust_sn[i-1] + m_dust_agb[i-1]
                n_sn = dm_star * SN_RATE
                m_swept = n_sn * 1000
                f_dest = m_swept / m_gas
                destruction = current_dust * f_dest
            else:
                destruction = 0
            
            # Apply
            if (m_dust_sn[i-1] + m_dust_agb[i-1]) > 0:
                f_sn = m_dust_sn[i-1] / (m_dust_sn[i-1] + m_dust_agb[i-1])
                dest_sn = destruction * f_sn
                dest_agb = destruction * (1 - f_sn)
            else:
                dest_sn = 0
                dest_agb = 0

            m_dust_sn[i] = m_dust_sn[i-1] + d_dust_sn - dest_sn
            m_dust_agb[i] = m_dust_agb[i-1] + d_dust_agb - dest_agb
            
            m_dust_sn[i] = max(0, m_dust_sn[i])
            m_dust_agb[i] = max(0, m_dust_agb[i])
            
        return m_dust_sn[-1] + m_dust_agb[-1]

def convert_av_to_mdust(av, m_star):
    r_kpc = 0.5 * (m_star/1e9)**0.2
    r_cm = r_kpc * 3.086e21
    area = np.pi * r_cm**2
    kappa = 5e4 
    m_dust_g = (av / 1.086) * area / kappa
    return m_dust_g / 1.989e33

def main():
    print("Running Sensitivity Analysis (Optimistic Physics)...")
    try:
        df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    except:
        return

    high_z = df[(df['z_phot'] >= 8) & (df['log_Mstar'] > 9)].copy()
    
    impossible_std = 0
    impossible_tep = 0
    
    for idx, row in high_z.iterrows():
        m_star = 10**row['log_Mstar']
        m_dust_obs = convert_av_to_mdust(row['dust'], m_star)
        
        # Standard
        model_std = DustEvolutionModel(m_star, row['z_phot'], gamma_t=1.0)
        max_std = model_std.run_evolution()
        
        # TEP
        model_tep = DustEvolutionModel(m_star, row['z_phot'], gamma_t=row['gamma_t'])
        max_tep = model_tep.run_evolution()
        
        if m_dust_obs > 2 * max_std:
            impossible_std += 1
        if m_dust_obs > 2 * max_tep:
            impossible_tep += 1
            
    print(f"Impossible (Optimistic Standard): {impossible_std}/{len(high_z)}")
    print(f"Impossible (Optimistic TEP): {impossible_tep}/{len(high_z)}")

if __name__ == "__main__":
    main()
