
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-JWST")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.tep_model import compute_gamma_t_logmass, stellar_to_halo_mass_behroozi_like

def smf_resolution_logmass(alpha_0, z=9, log_mstar=10.5):
    log_mh = stellar_to_halo_mass_behroozi_like(np.array([log_mstar]), np.array([z]))[0]
    gamma_t = compute_gamma_t_logmass(log_mh, z, alpha_0=alpha_0)
    bias_dex = 0.5 * np.log10(gamma_t)
    return 1.5 * bias_dex

print("SMF Resolution (delta log N) vs Alpha_0 (Log-Mass model):")
for a in [0.1, 0.3, 0.5, 0.58, 1.0, 1.5]:
    res = smf_resolution_logmass(a)
    print(f"alpha_0={a:.2f}, delta_logN={res:.3f} (Target=1.1)")

target = 1.1
# 1.1 = 1.5 * 0.5 * log10(gamma_t)
# log10(gamma_t) = 1.466
# Gamma_t = exp(argument) => argument = 3.37
# argument = alpha_z * (2/3) * delta_log_Mh * z_factor
z = 9
log_mh = stellar_to_halo_mass_behroozi_like(10.5, 9)
log_mh_ref_z = 12.0 - 1.5 * np.log10(1 + z)
delta_log_Mh = log_mh - log_mh_ref_z
z_factor = (1 + z) / (1 + 5.5)
alpha_required = 3.37 / ((2/3) * delta_log_Mh * z_factor * np.sqrt(1 + z))

print(f"\nRequired Alpha_0 for 100% SMF resolution: {alpha_required:.2f}")
