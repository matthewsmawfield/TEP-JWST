
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import Planck18
import astropy.units as u

def effective_time_factor(log_mh, z, kappa_gal=9.6e5, z_ref=5.5):
    """
    Compute Gamma_t based on halo mass and redshift.
    """
    # TEP coupling scaling
    alpha = kappa_gal * np.sqrt(1 + z)
    
    # Potential depth proxy relative to reference
    # Reference mass log_mh = 12
    delta_log_mh = log_mh - 12.0
    
    # Redshift factor relative to z_ref
    z_factor = (1 + z) / (1 + z_ref)
    
    # Gamma_t
    gamma_t = np.exp(alpha * (2/3) * delta_log_mh * z_factor)
    return gamma_t

def simple_hdelta_model(age_gyr):
    """
    Toy model for H_delta absorption strength (Angstroms) as function of age.
    Based roughly on spectral synthesis models (e.g., Worthey & Ottaviani 1997).
    Peaks around 0.5-1.0 Gyr (A-type stars).
    """
    # Approximate shape: rises to peak at ~0.8 Gyr, then declines slowly
    # Very young (< 0.01): weak
    # 0.1 - 1.0: strong (A stars)
    # > 1.0: declines (F/G stars)
    
    # Phenomenological function
    # log-normal-ish shape in log-age
    
    log_age = np.log10(age_gyr + 1e-5) # age in Gyr
    
    # Peak at 0.5 Gyr (log_age = -0.3)
    # Width sigma = 0.5 dex
    peak_val = 8.0 # Angstroms
    peak_pos = -0.3
    width = 0.5
    
    strength = peak_val * np.exp(-0.5 * ((log_age - peak_pos)/width)**2)
    
    # Cutoff for very young ages where O/B stars dominate (emission fills in absorption)
    # This is absorption strength, so emission would make it lower/negative.
    # We'll just model the absorption component.
    
    return strength

def simulate_spectroscopic_prediction(n_galaxies=500, z_mean=7.0, z_width=1.0):
    """
    Generate synthetic population and compare Standard vs TEP predictions for Balmer lines.
    """
    print(f"Simulating {n_galaxies} galaxies at z ~ {z_mean}...")
    
    np.random.seed(42)
    
    # 1. Generate Population
    z_obs = np.random.normal(z_mean, z_width, n_galaxies)
    z_obs = z_obs[z_obs > 4] # Cutoff
    n_actual = len(z_obs)
    
    # Cosmic age at z_obs
    t_cosmic = Planck18.age(z_obs).value # Gyr
    
    # Stellar Mass (log solar)
    # Sample from schematic mass function 8 to 11
    log_m_star = np.random.uniform(8.0, 11.0, n_actual)
    
    # Abundance matching proxy for Halo Mass (simplified)
    # M_h ~ M_*^beta ? Let's just use the inverse of the one in the paper or a simple scaling
    # Paper uses pipeline relation. We'll approximate:
    # log M_h approx log M_* + 2.0 (very rough, but captures the scaling)
    # Actually, let's assume a slightly non-linear one standard in high-z
    log_m_halo = 2.5 + 0.8 * log_m_star + 0.5 * (log_m_star - 10)**2
    # Normalize roughly so 10 -> 11.5, 9 -> 10.5
    # Let's just use the linear proxy for simulation visualization purposes as slope matters most
    log_m_halo = log_m_star + 2.0 
    
    # 2. Compute TEP Factors
    gamma_t = effective_time_factor(log_m_halo, z_obs)
    
    # 3. Assign Star Formation Histories (simplified to single-burst age)
    # In reality, galaxies have complex SFH. 
    # Let's assume galaxies formed 'formation_time' ago.
    # formation_z roughly z+2 to z+5
    
    # Fraction of cosmic time spent forming stars
    # Random fraction 0.1 to 0.9 of cosmic age
    age_fraction = np.random.uniform(0.1, 0.9, n_actual)
    
    # Standard Age (Coordinate Time)
    age_standard = t_cosmic * age_fraction
    
    # TEP Age (Effective Time)
    # t_eff_age = age_standard * Gamma_t
    age_tep = age_standard * gamma_t
    
    # 4. Predict Observables (H_delta strength)
    h_delta_standard = simple_hdelta_model(age_standard)
    h_delta_tep = simple_hdelta_model(age_tep)
    
    # Add measurement noise
    noise = np.random.normal(0, 0.5, n_actual)
    h_delta_standard_obs = h_delta_standard + noise
    h_delta_tep_obs = h_delta_tep + noise
    
    # 5. Analyze Results
    df = pd.DataFrame({
        'z': z_obs,
        'log_m_star': log_m_star,
        'log_m_halo': log_m_halo,
        'gamma_t': gamma_t,
        'age_standard': age_standard,
        'age_tep': age_tep,
        'Hd_standard': h_delta_standard_obs,
        'Hd_tep': h_delta_tep_obs
    })
    
    # Bin by mass
    bins = np.linspace(8, 11, 7)
    df['mass_bin'] = pd.cut(df['log_m_star'], bins)
    
    grouped = df.groupby('mass_bin', observed=True).agg({
        'Hd_standard': ['mean', 'std'],
        'Hd_tep': ['mean', 'std'],
        'gamma_t': 'mean'
    })
    
    print("\nPrediction Comparison (H_delta Strength [Angstroms]):")
    print(f"{'Mass Bin':<15} | {'Gamma_t':<8} | {'Standard':<15} | {'TEP':<15} | {'Diff':<10}")
    print("-" * 75)
    
    for i in range(len(grouped)):
        idx = grouped.index[i]
        gt = grouped[('gamma_t', 'mean')].iloc[i]
        std_mean = grouped[('Hd_standard', 'mean')].iloc[i]
        tep_mean = grouped[('Hd_tep', 'mean')].iloc[i]
        diff = tep_mean - std_mean
        
        print(f"{str(idx):<15} | {gt:.2f}     | {std_mean:.2f}            | {tep_mean:.2f}            | {diff:+.2f}")

    # Output CSV
    outfile = '/Users/matthewsmawfield/www/TEP-JWST/scripts/simulations/balmer_prediction.csv'
    df.to_csv(outfile, index=False)
    print(f"\nSimulation data saved to {outfile}")
    
    return df

if __name__ == "__main__":
    simulate_spectroscopic_prediction()
