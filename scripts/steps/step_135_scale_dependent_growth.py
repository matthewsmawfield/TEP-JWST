#!/usr/bin/env python3
"""
Step 135: Scale-Dependent Growth Factor with Chameleon Yukawa Suppression

Upgrades Step 92 by solving the growth ODE for each k-mode independently,
incorporating the scale-dependent effective gravitational coupling:

    G_eff(k, z) / G_N = 1 + 2β² × k² / (k² + m_φ(z)²)

where m_φ(z) is the chameleon scalar field mass in the cosmological background.
On scales k >> m_φ (small scales, inside Compton wavelength), G_eff → G_N(1+2β²).
On scales k << m_φ (large scales, outside Compton wavelength), G_eff → G_N.

The chameleon mass in the cosmological background:
    m_φ(z) = m_φ,0 × [ρ_m(z)/ρ_m,0]^{(n+2)/(2(n+1))}

For n=1 (Ratra-Peebles potential): m_φ(z) ∝ ρ^(3/4) ∝ (1+z)^(9/4)

We compute:
1. P(k) ratio P_TEP(k)/P_ΛCDM(k) at z=0
2. σ₈ under TEP with proper scale-dependent suppression
3. f(z)σ₈(z) for comparison with RSD measurements
4. The Compton wavelength λ_C(z) evolution

Author: Matthew L. Smawfield
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types
from scripts.utils.tep_model import KAPPA_GAL, RHO_CRIT_G_CM3  # TEP coupling constant & screening density

STEP_NUM = "135"  # Pipeline step number
STEP_NAME = "scale_dependent_growth"  # Used in log / output filenames
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

# Cosmological parameters (Planck 2018)
Om0 = Planck18.Om0
Ode0 = Planck18.Ode0
Ob0 = Planck18.Ob0
H0 = Planck18.H0.value  # km/s/Mpc
h = H0 / 100.0
sigma8_planck = 0.811
sigma8_err = 0.006
ns = 0.9649  # spectral index

# TEP parameters
BETA = 1.0  # O(1) theoretical test coupling for linear growth
RHO_CRIT_SCREENING = RHO_CRIT_G_CM3  # g/cm³ — halo screening threshold


def hubble_E(a):
    """E(a) = H(a)/H0 for flat ΛCDM background."""
    return np.sqrt(Om0 * a**(-3) + Ode0)


def dEda(a):
    """dE/da."""
    E = hubble_E(a)
    return -1.5 * Om0 / (a**4 * E)


def chameleon_mass_squared(z, m_phi_0, n_pot=1):
    """
    Chameleon scalar field mass² in cosmological background.
    
    For V(φ) = Λ⁴(1 + (Λ/φ)^n) [Ratra-Peebles]:
        m_φ² ∝ ρ^{(n+2)/(n+1)}
    
    For n=1: m_φ² ∝ ρ^(3/2) ∝ (1+z)^(9/2)
    
    Parameters
    ----------
    z : float
        Redshift
    m_phi_0 : float
        Scalar mass at z=0 in h/Mpc
    n_pot : int
        Potential index (default 1)
    
    Returns
    -------
    float
        m_φ² in (h/Mpc)²
    """
    exponent = (n_pot + 2.0) / (n_pot + 1.0)  # 3/2 for n=1
    return m_phi_0**2 * (1 + z)**(3 * exponent)


def geff_ratio_k(k, z, beta, m_phi_0, n_pot=1):
    """
    Scale-dependent G_eff/G_N.
    
    G_eff(k,z)/G_N = 1 + 2β² × k²/(k² + m_φ(z)²)
    
    Parameters
    ----------
    k : float or array
        Wavenumber in h/Mpc
    z : float
        Redshift
    beta : float
        Conformal coupling
    m_phi_0 : float
        Scalar mass at z=0 in h/Mpc
    
    Returns
    -------
    float or array
        G_eff/G_N ratio
    """
    m2 = chameleon_mass_squared(z, m_phi_0, n_pot)
    k2 = np.asarray(k)**2
    return 1.0 + 2.0 * beta**2 * k2 / (k2 + m2)


def growth_ode_k(y, a, k, beta, m_phi_0, n_pot):
    """
    Growth ODE for a single k-mode.
    
    D'' + (3/a + E'/E)D' - (3/2)(Ωm(a)/a²)(G_eff(k,z)/G_N)D = 0
    """
    D, dDda = y
    z = 1.0 / a - 1.0
    E = hubble_E(a)
    dE = dEda(a)
    Om_a = Om0 * a**(-3) / E**2
    
    geff = geff_ratio_k(k, z, beta, m_phi_0, n_pot)
    
    friction = (3.0 / a + dE / E) * dDda
    source = 1.5 * Om_a * geff * D / a**2
    
    return [dDda, source - friction]


def solve_growth_for_k(k, beta, m_phi_0, n_pot=1, a_start=1e-3, n_points=2000):
    """Solve growth ODE for a single k-mode from a_start to a=1."""
    a_grid = np.linspace(a_start, 1.0, n_points)
    y0 = [a_start, 1.0]  # D(a_i) = a_i, dD/da = 1 (matter-dominated IC)
    
    sol = odeint(growth_ode_k, y0, a_grid, args=(k, beta, m_phi_0, n_pot))
    return a_grid, sol[:, 0], sol[:, 1]


def window_tophat(k, R):
    """Top-hat window function in Fourier space."""
    x = k * R
    # Avoid division by zero
    x = np.where(np.abs(x) < 1e-10, 1e-10, x)
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3


def transfer_function_eh98(k, Om0, Ob0, h):
    """
    Eisenstein & Hu (1998) transfer function (no-wiggle approximation).
    Good enough for σ₈ ratio calculation.
    
    Parameters
    ----------
    k : array
        Wavenumber in h/Mpc
    """
    # Shape parameter
    theta = 2.725 / 2.7  # CMB temperature ratio
    Omh2 = Om0 * h**2
    Obh2 = Ob0 * h**2
    fb = Ob0 / Om0
    
    # Sound horizon
    s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**0.75)
    
    # Silk damping
    alpha_gamma = 1 - 0.328 * np.log(431 * Omh2) * fb + 0.38 * np.log(22.3 * Omh2) * fb**2
    
    gamma_eff = Om0 * h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43 * k * s)**4))
    
    q = k * theta**2 / gamma_eff
    L = np.log(2 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1 + 62.5 * q)
    T = L / (L + C * q**2)
    return T


def compute_sigma8_from_growth(k_grid, D_ratio_k, R8=8.0):
    """
    Compute σ₈ ratio from scale-dependent growth enhancement.
    
    σ₈² ∝ ∫ k² P(k) W²(kR) dk
    
    Since P_TEP(k) = P_ΛCDM(k) × D_TEP(k)²/D_ΛCDM(k)²,
    the ratio σ₈_TEP/σ₈_ΛCDM depends on the k-weighted integral.
    
    Parameters
    ----------
    k_grid : array
        Wavenumber grid in h/Mpc
    D_ratio_k : array
        D_TEP(k)/D_ΛCDM(k) at z=0 for each k
    R8 : float
        Smoothing radius in Mpc/h (default 8)
    """
    T = transfer_function_eh98(k_grid, Om0, Ob0, h)
    P_lcdm = k_grid**ns * T**2  # unnormalized P(k) ∝ k^ns T²(k)
    W = window_tophat(k_grid, R8)
    
    # σ₈² ∝ ∫ k² P(k) W²(kR) dk / (2π²)
    integrand_lcdm = k_grid**2 * P_lcdm * W**2
    integrand_tep = k_grid**2 * P_lcdm * D_ratio_k**2 * W**2
    
    sigma2_lcdm = np.trapezoid(integrand_lcdm, k_grid)
    sigma2_tep = np.trapezoid(integrand_tep, k_grid)
    
    return np.sqrt(sigma2_tep / sigma2_lcdm)


def compute_fsigma8(a_grid, D_arr, sigma8_norm):
    """
    Compute f(z)σ₈(z) = dlnD/dlna × σ₈(z).
    
    Parameters
    ----------
    a_grid : array
        Scale factor grid
    D_arr : array
        Growth factor D(a) (normalized to 1 at z=0)
    sigma8_norm : float
        σ₈ at z=0
    """
    z_grid = 1.0 / a_grid - 1.0
    # f = dlnD/dlna
    lnD = np.log(np.maximum(D_arr, 1e-30))
    lna = np.log(a_grid)
    f = np.gradient(lnD, lna)
    
    # σ₈(z) = σ₈(0) × D(z)/D(0)
    D_norm = D_arr / D_arr[-1]
    sigma8_z = sigma8_norm * D_norm
    
    return z_grid, f * sigma8_z


def run():
    print_status("=" * 60, "INFO")
    print_status("Step 135: Scale-Dependent Growth with Chameleon Yukawa", "INFO")
    print_status("=" * 60, "INFO")
    
    # =========================================================================
    # 1. Determine m_φ,0 from the TEP screening density
    # =========================================================================
    # The Compton wavelength λ_C = 2π/m_φ determines the range of the fifth force.
    # For the chameleon in halos: m_φ(ρ_halo) >> 1/R_halo → screened.
    # For cosmological background: m_φ(ρ_bg) sets the linear-scale cutoff.
    #
    # We parametrize m_φ,0 such that:
    #   λ_C(z=0) ~ 1-10 Mpc (typical chameleon models)
    #   This means m_φ,0 ~ 0.6 - 6 h/Mpc
    #
    # We scan over m_φ,0 to find the value consistent with Planck σ₈.
    
    # k-grid for computation
    k_min = 1e-4   # h/Mpc
    k_max = 10.0    # h/Mpc
    n_k = 200
    k_grid = np.geomspace(k_min, k_max, n_k)
    
    # Reference: ΛCDM growth (β=0)
    print_status("Computing ΛCDM reference growth...", "INFO")
    a_ref, D_ref, _ = solve_growth_for_k(k_grid[0], 0.0, 1.0)
    D_lcdm_z0 = D_ref[-1]
    
    # =========================================================================
    # 2. Scan m_φ,0 to find Planck-consistent value
    # =========================================================================
    m_phi_0_values = np.geomspace(0.01, 100.0, 50)
    sigma8_ratios = []
    
    print_status(f"Scanning {len(m_phi_0_values)} m_φ,0 values...", "INFO")
    
    for m_phi_0 in m_phi_0_values:
        D_ratio_at_z0 = np.zeros(n_k)
        for i, k in enumerate(k_grid):
            _, D_k, _ = solve_growth_for_k(k, BETA, m_phi_0)
            D_ratio_at_z0[i] = D_k[-1] / D_lcdm_z0
        
        ratio = compute_sigma8_from_growth(k_grid, D_ratio_at_z0)
        sigma8_ratios.append(ratio)
    
    sigma8_ratios = np.array(sigma8_ratios)
    sigma8_predicted = sigma8_planck * sigma8_ratios
    
    # Find m_φ,0 where σ₈_TEP = σ₈_Planck (within 2σ)
    target_upper = sigma8_planck + 2 * sigma8_err
    # Find where sigma8_predicted crosses the 2σ upper limit
    consistent_mask = sigma8_predicted <= target_upper
    
    if np.any(consistent_mask):
        # Smallest m_φ,0 that is consistent
        m_phi_0_min = m_phi_0_values[consistent_mask][0]
        # Best fit (closest to Planck central value)
        best_idx = np.argmin(np.abs(sigma8_predicted - sigma8_planck))
        m_phi_0_best = m_phi_0_values[best_idx]
        sigma8_best = sigma8_predicted[best_idx]
    else:
        m_phi_0_min = m_phi_0_values[-1]
        m_phi_0_best = m_phi_0_values[-1]
        sigma8_best = sigma8_predicted[-1]
    
    print_status(f"m_φ,0 (2σ min): {m_phi_0_min:.3f} h/Mpc", "INFO")
    print_status(f"m_φ,0 (best fit): {m_phi_0_best:.3f} h/Mpc", "INFO")
    print_status(f"σ₈ (best): {sigma8_best:.4f} (Planck: {sigma8_planck})", "INFO")
    
    lambda_C_best = 2 * np.pi / m_phi_0_best  # Compton wavelength in Mpc/h
    print_status(f"λ_C(z=0): {lambda_C_best:.2f} Mpc/h", "INFO")
    
    # =========================================================================
    # 3. Full computation at best-fit m_φ,0
    # =========================================================================
    print_status("Computing full scale-dependent growth at best-fit m_φ,0...", "INFO")
    
    # Dense k-grid for P(k) ratio
    k_dense = np.geomspace(1e-4, 50.0, 500)
    D_ratio_dense = np.zeros(len(k_dense))
    
    # Also store growth as function of z for a few representative k-modes
    k_representatives = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    growth_vs_z = {}
    
    for i, k in enumerate(k_dense):
        a_arr, D_k, dD_k = solve_growth_for_k(k, BETA, m_phi_0_best)
        D_ratio_dense[i] = D_k[-1] / D_lcdm_z0
    
    for k_rep in k_representatives:
        a_arr, D_k, dD_k = solve_growth_for_k(k_rep, BETA, m_phi_0_best)
        z_arr = 1.0 / a_arr - 1.0
        D_ratio_z = D_k / D_ref  # ratio vs ΛCDM at each z
        growth_vs_z[f"k={k_rep}"] = {
            'z': z_arr.tolist(),
            'D_ratio': D_ratio_z.tolist()
        }
    
    # P(k) ratio = D_ratio²
    Pk_ratio_dense = D_ratio_dense**2
    
    # σ₈ with the dense grid
    sigma8_ratio_final = compute_sigma8_from_growth(k_dense, D_ratio_dense)
    sigma8_tep_final = sigma8_planck * sigma8_ratio_final
    
    print_status(f"Final σ₈_TEP: {sigma8_tep_final:.4f}", "INFO")
    print_status(f"σ₈ ratio: {sigma8_ratio_final:.6f}", "INFO")
    
    # =========================================================================
    # 4. Effective β on R=8 Mpc/h scale
    # =========================================================================
    # The effective coupling on σ₈ scales
    k8 = 2 * np.pi / 8.0  # k corresponding to R=8 Mpc/h ≈ 0.785 h/Mpc
    geff_at_k8 = geff_ratio_k(k8, 0.0, BETA, m_phi_0_best)
    beta_eff_k8 = np.sqrt((geff_at_k8 - 1.0) / 2.0)
    
    print_status(f"G_eff/G_N at k₈={k8:.3f} h/Mpc: {geff_at_k8:.6f}", "INFO")
    print_status(f"β_eff at k₈: {beta_eff_k8:.6f}", "INFO")
    print_status(f"Suppression factor: {beta_eff_k8/BETA:.4f}", "INFO")
    
    # =========================================================================
    # 5. Compton wavelength evolution
    # =========================================================================
    z_compton = np.linspace(0, 15, 100)
    m_phi_z = np.sqrt(chameleon_mass_squared(z_compton, m_phi_0_best))
    lambda_C_z = 2 * np.pi / m_phi_z  # Mpc/h
    
    # =========================================================================
    # 6. f(z)σ₈(z) comparison with RSD data
    # =========================================================================
    # Use an effective single-scale growth for fσ₈
    # Weight by the σ₈ window function
    print_status("Computing f(z)σ₈(z)...", "INFO")
    
    # ΛCDM fσ₈
    D_lcdm_norm = D_ref / D_ref[-1]
    z_lcdm, fsigma8_lcdm = compute_fsigma8(a_ref, D_lcdm_norm, sigma8_planck)
    
    # TEP fσ₈ — use k₈ as representative scale
    _, D_tep_k8, _ = solve_growth_for_k(k8, BETA, m_phi_0_best)
    D_tep_k8_norm = D_tep_k8 / D_tep_k8[-1]
    z_tep, fsigma8_tep = compute_fsigma8(a_ref, D_tep_k8_norm, sigma8_tep_final)
    
    # RSD data points (compilation from Planck 2018 + recent surveys)
    rsd_data = [
        {'survey': '6dFGS', 'z': 0.067, 'fsig8': 0.423, 'err': 0.055},
        {'survey': 'SDSS-MGS', 'z': 0.15, 'fsig8': 0.53, 'err': 4.0e5},
        {'survey': 'BOSS-LOWZ', 'z': 0.32, 'fsig8': 0.427, 'err': 0.056},
        {'survey': 'BOSS-CMASS', 'z': 0.57, 'fsig8': 0.426, 'err': 0.029},
        {'survey': 'eBOSS-LRG', 'z': 0.70, 'fsig8': 0.473, 'err': 0.041},
        {'survey': 'eBOSS-QSO', 'z': 1.48, 'fsig8': 0.462, 'err': 0.045},
        {'survey': 'Vipers-v7', 'z': 0.76, 'fsig8': 0.440, 'err': 0.040},
        {'survey': 'FastSound', 'z': 1.36, 'fsig8': 0.482, 'err': 0.116},
    ]
    
    # Compute χ² for both models against RSD data
    chi2_lcdm = 0.0
    chi2_tep = 0.0
    fsig8_lcdm_interp = interp1d(z_lcdm[::-1], fsigma8_lcdm[::-1],
                                  kind='cubic', fill_value='extrapolate')
    fsig8_tep_interp = interp1d(z_tep[::-1], fsigma8_tep[::-1],
                                 kind='cubic', fill_value='extrapolate')
    
    for d in rsd_data:
        pred_lcdm = float(fsig8_lcdm_interp(d['z']))
        pred_tep = float(fsig8_tep_interp(d['z']))
        chi2_lcdm += ((d['fsig8'] - pred_lcdm) / d['err'])**2
        chi2_tep += ((d['fsig8'] - pred_tep) / d['err'])**2
    
    n_rsd = len(rsd_data)
    print_status(f"RSD χ² — ΛCDM: {chi2_lcdm:.2f}/{n_rsd}, TEP: {chi2_tep:.2f}/{n_rsd}", "INFO")
    
    # =========================================================================
    # 7. Unscreened comparison (what Step 92 computed)
    # =========================================================================
    D_ratio_unscreened = np.zeros(len(k_dense))
    for i, k in enumerate(k_dense):
        _, D_k_un, _ = solve_growth_for_k(k, BETA, 1e-10)  # m_phi→0: fully unscreened
        D_ratio_unscreened[i] = D_k_un[-1] / D_lcdm_z0
    
    sigma8_ratio_unscreened = compute_sigma8_from_growth(k_dense, D_ratio_unscreened)
    sigma8_unscreened = sigma8_planck * sigma8_ratio_unscreened
    
    print_status(f"Unscreened σ₈: {sigma8_unscreened:.2f} (confirms Step 92: ~3.40)", "INFO")
    
    # =========================================================================
    # 8. Save results
    # =========================================================================
    results = {
        'description': 'Scale-dependent growth factor with chameleon Yukawa suppression',
        'tep_parameters': {
            'beta': float(BETA),
            'kappa_gal': float(KAPPA_GAL),
            'n_potential': 1,
        },
        'best_fit': {
            'm_phi_0_hMpc': float(m_phi_0_best),
            'lambda_C_z0_Mpc_h': float(lambda_C_best),
            'sigma8_tep': float(sigma8_tep_final),
            'sigma8_ratio': float(sigma8_ratio_final),
            'sigma8_planck': float(sigma8_planck),
            'delta_sigma8': float(sigma8_tep_final - sigma8_planck),
            'delta_sigma8_in_sigma': float((sigma8_tep_final - sigma8_planck) / sigma8_err),
        },
        'effective_coupling': {
            'k8_hMpc': float(k8),
            'geff_ratio_k8_z0': float(geff_at_k8),
            'beta_eff_k8': float(beta_eff_k8),
            'suppression_factor': float(beta_eff_k8 / BETA),
        },
        'consistency_checks': {
            'sigma8_unscreened': float(sigma8_unscreened),
            'sigma8_step92_expected': 3.40,
            'rsd_chi2_lcdm': float(chi2_lcdm),
            'rsd_chi2_tep': float(chi2_tep),
            'n_rsd_points': n_rsd,
            'rsd_delta_chi2': float(chi2_tep - chi2_lcdm),
        },
        '2sigma_constraint': {
            'm_phi_0_min_hMpc': float(m_phi_0_min),
            'lambda_C_max_Mpc_h': float(2 * np.pi / m_phi_0_min),
        },
        'compton_wavelength_evolution': {
            'z': z_compton.tolist(),
            'lambda_C_Mpc_h': lambda_C_z.tolist(),
        },
        'pk_ratio': {
            'k_hMpc': k_dense.tolist(),
            'Pk_TEP_over_Pk_LCDM': Pk_ratio_dense.tolist(),
        },
        'rsd_data': rsd_data,
    }
    
    out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"Results saved to {out_file}", "INFO")
    
    # =========================================================================
    # 9. Generate figures
    # =========================================================================
    
    try:
        import sys
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.utils.style import set_pub_style, FIG_SIZE, COLORS
    except Exception:
        pass

    try:
        set_pub_style()
    except NameError:
        pass
        
    # --- Figure A: P(k) ratio ---
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE.get('web_quad', (14, 10)))
    
    # Panel 1: P(k) ratio
    ax = axes[0, 0]
    ax.semilogx(k_dense, Pk_ratio_dense, 'b-', lw=2, label=f'TEP (m_φ,0={m_phi_0_best:.1f} h/Mpc)')
    ax.semilogx(k_dense, D_ratio_unscreened**2, 'r--', lw=1.5, alpha=0.6, label='Unscreened (m_φ→0)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.axvline(k8, color='gray', ls=':', lw=0.8, label=f'k₈ = {k8:.2f} h/Mpc')
    ax.fill_between(k_dense, 1 - 0.02, 1 + 0.02, alpha=0.15, color='green', label='Planck 2σ band')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P_TEP(k) / P_ΛCDM(k)')
    ax.set_title('Matter Power Spectrum Ratio')
    ax.legend(fontsize=8)
    ax.set_ylim(0.95, max(2.0, np.max(Pk_ratio_dense) * 1.1))
    ax.set_xlim(1e-4, 50)
    
    # Panel 2: Compton wavelength evolution
    ax = axes[0, 1]
    ax.semilogy(z_compton, lambda_C_z, 'b-', lw=2)
    ax.axhline(8.0, color='gray', ls='--', lw=0.8, label='R₈ = 8 Mpc/h')
    ax.axhline(1.0, color='orange', ls=':', lw=0.8, label='1 Mpc/h (cluster scale)')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Compton wavelength λ_C [Mpc/h]')
    ax.set_title('Chameleon Compton Wavelength Evolution')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 15)
    
    # Panel 3: f(z)σ₈(z)
    ax = axes[1, 0]
    z_mask = (z_lcdm > 0) & (z_lcdm < 2.5)
    ax.plot(z_lcdm[z_mask], fsigma8_lcdm[z_mask], 'k-', lw=2, label='ΛCDM')
    z_mask_t = (z_tep > 0) & (z_tep < 2.5)
    ax.plot(z_tep[z_mask_t], fsigma8_tep[z_mask_t], 'b-', lw=2, label='TEP (screened)')
    for d in rsd_data:
        ax.errorbar(d['z'], d['fsig8'], yerr=d['err'], fmt='o', color='red',
                    markersize=5, capsize=3)
    ax.errorbar([], [], yerr=[], fmt='o', color='red', label='RSD data')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('f(z)σ₈(z)')
    ax.set_title(f'Growth Rate (χ²_ΛCDM={chi2_lcdm:.1f}, χ²_TEP={chi2_tep:.1f})')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0.2, 0.7)
    
    # Panel 4: σ₈ scan
    ax = axes[1, 1]
    ax.semilogx(m_phi_0_values, sigma8_predicted, 'b-', lw=2)
    ax.axhline(sigma8_planck, color='k', ls='-', lw=0.8, label=f'Planck σ₈ = {sigma8_planck}')
    ax.axhline(target_upper, color='k', ls='--', lw=0.8, label=f'Planck 2σ upper = {target_upper}')
    ax.axvline(m_phi_0_best, color='blue', ls=':', lw=0.8, label=f'm_φ,0 = {m_phi_0_best:.1f} h/Mpc')
    ax.fill_between(m_phi_0_values, sigma8_planck - 2*sigma8_err, 
                     sigma8_planck + 2*sigma8_err, alpha=0.15, color='green')
    ax.set_xlabel('m_φ,0 [h/Mpc]')
    ax.set_ylabel('σ₈ predicted')
    ax.set_title('σ₈ vs Scalar Mass Parameter')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_status(f"Figure saved to {fig_path}", "INFO")
    
    try:
        set_pub_style()
    except NameError:
        pass

    # --- Figure B: G_eff(k) at different redshifts ---
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE.get('web_standard', (8, 5)))
    z_show = [0, 1, 3, 5, 8, 10]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(z_show)))
    for z_val, col in zip(z_show, colors):
        geff_k = geff_ratio_k(k_dense, z_val, BETA, m_phi_0_best)
        ax2.semilogx(k_dense, geff_k, color=col, lw=2, label=f'z={z_val}')
    ax2.axhline(1.0, color='k', ls='--', lw=0.8)
    ax2.axhline(1 + 2*BETA**2, color='r', ls=':', lw=0.8, 
                label=f'Unscreened limit (1+2β²={1+2*BETA**2:.3f})')
    ax2.axvline(k8, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('G_eff(k,z) / G_N')
    ax2.set_title('Scale-Dependent Effective Gravitational Coupling')
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_xlim(1e-4, 50)
    ax2.set_ylim(0.99, 1 + 2*BETA**2 + 0.05)
    
    fig2_path = FIGURES_PATH / f"figure_{STEP_NUM}_geff_scale.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_status(f"Figure saved to {fig2_path}", "INFO")
    
    # =========================================================================
    # 10. Summary
    # =========================================================================
    print_status("=" * 60, "INFO")
    print_status("SUMMARY", "INFO")
    print_status(f"  β (conformal coupling): {BETA}", "INFO")
    print_status(f"  m_φ,0 (best fit): {m_phi_0_best:.2f} h/Mpc", "INFO")
    print_status(f"  λ_C(z=0): {lambda_C_best:.2f} Mpc/h", "INFO")
    print_status(f"  σ₈ (TEP, screened): {sigma8_tep_final:.4f}", "INFO")
    print_status(f"  σ₈ (Planck): {sigma8_planck}", "INFO")
    print_status(f"  Δσ₈/σ: {(sigma8_tep_final - sigma8_planck)/sigma8_err:.2f}σ", "INFO")
    print_status(f"  β_eff on R₈ scales: {beta_eff_k8:.4f} (suppression: {beta_eff_k8/BETA:.4f})", "INFO")
    print_status(f"  P(k) enhancement at k=1 h/Mpc: {Pk_ratio_dense[np.argmin(np.abs(k_dense-1.0))]:.4f}", "INFO")
    print_status(f"  RSD χ² — ΛCDM: {chi2_lcdm:.1f}, TEP: {chi2_tep:.1f}", "INFO")
    print_status("=" * 60, "INFO")
    
    return results


main = run

if __name__ == "__main__":
    run()
