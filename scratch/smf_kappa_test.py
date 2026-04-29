
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-JWST")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like, correct_stellar_mass

def smf_resolution(kappa, z=9, log_mstar=10.5):
    log_mh = stellar_to_halo_mass_behroozi_like(np.array([log_mstar]), np.array([z]))[0]
    gamma_t = compute_gamma_t(log_mh, z, kappa=kappa)
    bias_dex = 0.5 * np.log10(gamma_t) # n=0.5 at z>6
    # SMF slope is ~-1.5. To resolve excess of 1.1 dex (at z=9), we need correction in logN of 1.1.
    # delta_logN = 1.5 * bias_dex
    return 1.5 * bias_dex

print("SMF Resolution (delta log N) vs Kappa (z=9, logM=10.5):")
for k in [1e5, 5e5, 9.6e5, 1.5e6, 2e6, 3e6, 4e6, 5e6]:
    res = smf_resolution(k)
    print(f"kappa={k:.1e}, delta_logN={res:.3f} (Target=1.1)")

# Find kappa for delta_logN = 1.1
target = 1.1
# 1.1 = 1.5 * 0.5 * log10(gamma_t)
# log10(gamma_t) = 1.1 / 0.75 = 1.466
# gamma_t = 10^1.466 = 29.2
# Gamma_t = exp(argument) => argument = ln(29.2) = 3.37
# argument = K_exp * (phi - phi_ref) * sqrt(1+z)
# K_exp = kappa * ln10 / (2.5 * 0.7)
phi = 1.6e-7 * (10**stellar_to_halo_mass_behroozi_like(10.5, 9) / 1e12)**(2/3)
phi_ref = 1.6e-7
sqrt_z = np.sqrt(10)
k_exp_required = 3.37 / ((phi - phi_ref) * sqrt_z)
kappa_required = k_exp_required * (2.5 * 0.7) / np.log(10)

print(f"\nRequired Kappa for 100% SMF resolution at z=9: {kappa_required:.2e}")
