#!/usr/bin/env python3
"""
TEP-JWST Exploration: Finding the Right Model

The current analysis shows TEP explains only ~16% of the Red Monsters anomaly.
This script explores alternative models to understand what's needed.

Key questions:
1. Is the TEP-H0 logarithmic scaling correct for high-z?
2. Should we integrate proper time over the halo's formation history?
3. Are there second-order effects on distance/mass measurements?
4. What physical mechanism could give stronger enhancement?
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS
# =============================================================================

ALPHA_LOCAL = 0.58  # TEP-H0 calibration
RHO_C = 20.0  # g/cm³ - critical density from TEP-UCD

# Red Monsters data
RED_MONSTERS = {
    "S1": {"z": 5.85, "log_Mstar": 11.08, "log_Mh": 12.88, "SFE": 0.50},
    "S2": {"z": 5.30, "log_Mstar": 10.88, "log_Mh": 12.68, "SFE": 0.50},
    "S3": {"z": 5.55, "log_Mstar": 10.74, "log_Mh": 12.54, "SFE": 0.50},
}

# Reference
LOG_MH_REF = 12.0  # Typical FRESCO galaxy halo mass

# =============================================================================
# MODEL 1: Instantaneous TEP-H0 scaling
# =============================================================================

def model_1_instantaneous(log_Mh, log_Mh_ref, alpha):
    """
    Current model: Γ_t = 1 + α × (1/3) × Δlog(M_h)
    
    This assumes the enhancement is based on the current potential depth.
    """
    delta_log_Mh = log_Mh - log_Mh_ref
    return 1.0 + alpha * (1/3) * delta_log_Mh

# =============================================================================
# MODEL 2: Cumulative proper time integration
# =============================================================================

def halo_mass_history(z_obs, log_Mh_final, z_form=20):
    """
    Model halo mass growth history.
    
    Halos grow roughly exponentially in mass with cosmic time.
    M_h(z) ≈ M_h(z_obs) × exp(-β × (t(z_obs) - t(z)))
    
    where β ~ 1-2 Gyr^-1 for massive halos.
    """
    def M_h_at_z(z):
        if z > z_form:
            return 0
        t_obs = cosmo.age(z_obs).value  # Gyr
        t_z = cosmo.age(z).value  # Gyr
        beta = 1.5  # Gyr^-1, typical growth rate
        
        # Mass at redshift z
        log_Mh_z = log_Mh_final - beta * (t_obs - t_z) / np.log(10)
        return max(log_Mh_z, 8.0)  # Minimum halo mass
    
    return M_h_at_z

def model_2_cumulative(log_Mh_final, z_obs, log_Mh_ref, alpha, z_form=20):
    """
    Cumulative proper time model.
    
    The total proper time experienced is:
    τ = ∫ Γ_t(z) × dt
    
    where Γ_t(z) depends on the halo mass at each epoch.
    
    The enhancement factor is τ / t_cosmic.
    """
    M_h_history = halo_mass_history(z_obs, log_Mh_final, z_form)
    
    def integrand(z):
        if z > z_form:
            return 0
        log_Mh_z = M_h_history(z)
        gamma_t_z = model_1_instantaneous(log_Mh_z, log_Mh_ref, alpha)
        
        # dt/dz = -1 / ((1+z) × H(z))
        H_z = cosmo.H(z).value  # km/s/Mpc
        dt_dz = 1 / ((1 + z) * H_z) * 3.086e19 / 3.156e16  # Convert to Gyr
        
        return gamma_t_z * dt_dz
    
    # Integrate from z_form to z_obs
    tau, _ = quad(integrand, z_obs, z_form, limit=100)
    
    # Cosmic time at z_obs
    t_cosmic = cosmo.age(z_obs).value
    
    # Effective enhancement
    gamma_eff = tau / t_cosmic if t_cosmic > 0 else 1.0
    
    return gamma_eff

# =============================================================================
# MODEL 3: Potential-depth scaling (virial theorem)
# =============================================================================

def model_3_potential(log_Mh, z, log_Mh_ref, alpha):
    """
    Scale with gravitational potential depth.
    
    Φ ∝ M_h / R_vir ∝ M_h^(2/3) × (1+z)
    
    At high-z, halos are more compact, so potentials are deeper.
    """
    # Virial radius scales as R_vir ∝ M_h^(1/3) × (1+z)^(-1)
    # So Φ ∝ M_h / R_vir ∝ M_h^(2/3) × (1+z)
    
    delta_log_Mh = log_Mh - log_Mh_ref
    z_factor = (1 + z) / (1 + 5.5)  # Normalize to z=5.5
    
    # Enhancement scales with potential depth
    gamma_t = 1.0 + alpha * (2/3) * delta_log_Mh * z_factor
    
    return gamma_t

# =============================================================================
# MODEL 4: Formation time model
# =============================================================================

def model_4_formation_time(log_Mh, z_obs, log_Mh_ref, alpha):
    """
    Massive halos form earlier and spend more time in deep potentials.
    
    The enhancement depends on the integrated time spent at high mass.
    """
    # Formation redshift scales with mass (more massive = earlier)
    # z_form ≈ 10 + 2 × (log_Mh - 12)
    z_form = 10 + 2 * (log_Mh - 12)
    z_form = max(z_form, z_obs + 0.5)  # Must form before observation
    
    # Time spent at high mass
    t_obs = cosmo.age(z_obs).value
    t_form = cosmo.age(z_form).value
    t_massive = t_obs - t_form
    
    # Reference galaxy forms later
    z_form_ref = 10 + 2 * (log_Mh_ref - 12)
    t_form_ref = cosmo.age(z_form_ref).value
    t_massive_ref = t_obs - t_form_ref
    
    # Enhancement is ratio of time spent in deep potential
    if t_massive_ref > 0:
        time_ratio = t_massive / t_massive_ref
    else:
        time_ratio = 1.0
    
    # Combined with mass enhancement
    mass_enhancement = model_1_instantaneous(log_Mh, log_Mh_ref, alpha)
    
    return mass_enhancement * time_ratio

# =============================================================================
# MODEL 5: Self-consistent isochrony correction
# =============================================================================

def model_5_self_consistent(log_Mh_obs, z_obs, log_Mh_ref, alpha):
    """
    Self-consistent model where the observed halo mass is also biased.
    
    If stellar masses are overestimated, so are the inferred halo masses.
    The true halo mass is lower, but the TEP effect is still present.
    
    This creates a feedback loop:
    1. True M_h → Γ_t
    2. Γ_t → mass bias
    3. Observed M_h = True M_h × bias
    
    We solve for the true M_h that gives the observed M_h.
    """
    # Iterate to find self-consistent solution
    log_Mh_true = log_Mh_obs  # Initial guess
    
    for _ in range(10):
        gamma_t = model_1_instantaneous(log_Mh_true, log_Mh_ref, alpha)
        
        # Mass bias from isochrony (M/L ∝ t^0.7)
        mass_bias = 0.7 * np.log10(gamma_t) if gamma_t > 1 else 0
        
        # True mass is lower
        log_Mh_true_new = log_Mh_obs - mass_bias
        
        if abs(log_Mh_true_new - log_Mh_true) < 0.001:
            break
        log_Mh_true = log_Mh_true_new
    
    # Final enhancement based on true mass
    gamma_t_final = model_1_instantaneous(log_Mh_true, log_Mh_ref, alpha)
    
    return gamma_t_final, log_Mh_true, mass_bias

# =============================================================================
# MODEL 6: Redshift-dependent coupling
# =============================================================================

def alpha_at_z(z, alpha_0=ALPHA_LOCAL):
    """
    TEP coupling may be stronger at high-z.
    
    Possible physical reasons:
    1. Less screening (lower cosmic density)
    2. Scalar field not yet settled to minimum
    3. Different cosmological background
    
    Model: α(z) = α_0 × (1 + z)^n
    """
    n = 0.5  # Scaling exponent
    return alpha_0 * (1 + z) ** n

def model_6_z_dependent(log_Mh, z, log_Mh_ref):
    """
    Use redshift-dependent coupling.
    """
    alpha_z = alpha_at_z(z)
    return model_1_instantaneous(log_Mh, log_Mh_ref, alpha_z), alpha_z

# =============================================================================
# MODEL 7: Combined potential + z-dependent coupling
# =============================================================================

def model_7_combined(log_Mh, z, log_Mh_ref, alpha_0=ALPHA_LOCAL):
    """
    Combine potential depth scaling with z-dependent coupling.
    
    Physical motivation:
    1. At high-z, halos are more compact → deeper potentials
    2. At high-z, screening is weaker → stronger coupling
    3. Both effects compound
    
    Γ_t = 1 + α(z) × (2/3) × Δlog(M_h) × (1+z)/(1+z_ref)
    
    where α(z) = α_0 × (1+z)^0.5
    """
    z_ref = 5.5
    n = 0.5  # Redshift scaling exponent
    
    alpha_z = alpha_0 * (1 + z) ** n
    delta_log_Mh = log_Mh - log_Mh_ref
    z_factor = (1 + z) / (1 + z_ref)
    
    gamma_t = 1.0 + alpha_z * (2/3) * delta_log_Mh * z_factor
    
    return gamma_t, alpha_z

# =============================================================================
# MODEL 8: Proper time integral with halo assembly
# =============================================================================

def model_8_proper_time_integral(log_Mh_final, z_obs, log_Mh_ref, alpha_0=ALPHA_LOCAL):
    """
    Full proper time integral accounting for halo assembly history.
    
    Key insight: The TEP effect accumulates over the galaxy's lifetime.
    A galaxy that formed earlier in a massive halo has experienced
    more enhanced proper time than one that formed recently.
    
    τ_eff = ∫_{t_form}^{t_obs} Γ_t(t) dt
    
    The enhancement is τ_eff / (t_obs - t_form).
    """
    # Formation redshift (massive halos form earlier)
    z_form = min(15, 8 + 3 * (log_Mh_final - 12))
    
    # Time grid from formation to observation (increasing time)
    t_form = cosmo.age(z_form).value
    t_obs = cosmo.age(z_obs).value
    t_grid = np.linspace(t_form, t_obs, 100)
    
    # Halo mass history (linear growth in log space for simplicity)
    log_Mh_init = 10.0  # Initial halo mass at formation
    log_Mh_grid = log_Mh_init + (log_Mh_final - log_Mh_init) * (t_grid - t_form) / (t_obs - t_form)
    
    # Approximate redshift at each time
    # Use interpolation: z decreases as t increases
    z_grid = z_form + (z_obs - z_form) * (t_grid - t_form) / (t_obs - t_form)
    
    # Proper time enhancement at each epoch
    gamma_grid = np.zeros_like(t_grid)
    for i, (t, z, log_Mh) in enumerate(zip(t_grid, z_grid, log_Mh_grid)):
        # z-dependent coupling
        alpha_z = alpha_0 * (1 + z) ** 0.5
        
        # Potential depth factor
        z_factor = (1 + z) / (1 + z_obs)
        
        # Enhancement relative to reference
        delta_log_Mh = log_Mh - log_Mh_ref
        gamma_grid[i] = 1.0 + alpha_z * (2/3) * max(0, delta_log_Mh) * z_factor
    
    # Integrate proper time
    dt = np.diff(t_grid)
    gamma_avg = (gamma_grid[:-1] + gamma_grid[1:]) / 2
    tau_eff = np.sum(gamma_avg * dt)
    
    # Cosmic time elapsed
    t_elapsed = t_obs - t_form
    
    # Effective enhancement
    gamma_eff = tau_eff / t_elapsed if t_elapsed > 0 else 1.0
    
    return gamma_eff, z_form

# =============================================================================
# MODEL 9: Distance/luminosity correction cascade
# =============================================================================

def model_9_distance_cascade(log_Mh, z, log_Mh_ref, alpha_0=ALPHA_LOCAL):
    """
    Account for the cascade of isochrony effects on measurements.
    
    Under TEP, multiple measurements are affected:
    1. Stellar ages → M/L ratios → stellar masses
    2. Luminosity distances (if light propagation affected)
    3. SFR estimates (if UV/IR calibrations assume isochrony)
    
    The total bias compounds these effects.
    """
    # Base TEP enhancement
    gamma_t, alpha_z = model_7_combined(log_Mh, z, log_Mh_ref, alpha_0)
    
    # Stellar mass bias (M/L ∝ t^0.7)
    mass_bias_factor = gamma_t ** 0.7
    
    # SFR bias: UV luminosity per unit SFR depends on stellar age
    # Older populations have lower UV/SFR ratio
    # If we think stars are older, we infer higher SFR
    sfr_bias_factor = gamma_t ** 0.3  # Approximate
    
    # Combined effect on SFE = M* / (f_b × M_h)
    # If M* is overestimated and SFR is overestimated, SFE appears higher
    sfe_bias_factor = mass_bias_factor  # Dominated by mass
    
    return gamma_t, sfe_bias_factor, mass_bias_factor, sfr_bias_factor

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("TEP-JWST Exploration: Alternative Models")
    print("=" * 70)
    print()
    
    # Test all models on Red Monsters
    print("Testing models on Red Monsters (target: Γ_t ≈ 2.5 to explain anomaly)")
    print("-" * 70)
    
    for name, data in RED_MONSTERS.items():
        z = data["z"]
        log_Mh = data["log_Mh"]
        
        print(f"\n{name} (z={z:.2f}, log M_h={log_Mh:.2f}):")
        print("-" * 40)
        
        # Model 1: Instantaneous (current)
        g1 = model_1_instantaneous(log_Mh, LOG_MH_REF, ALPHA_LOCAL)
        print(f"  Model 1 (Instantaneous, α={ALPHA_LOCAL}):  Γ_t = {g1:.3f}")
        
        g1b = model_1_instantaneous(log_Mh, LOG_MH_REF, 1.2)
        print(f"  Model 1 (Instantaneous, α=1.2):    Γ_t = {g1b:.3f}")
        
        # Model 2: Cumulative
        g2 = model_2_cumulative(log_Mh, z, LOG_MH_REF, ALPHA_LOCAL)
        print(f"  Model 2 (Cumulative, α={ALPHA_LOCAL}):     Γ_t = {g2:.3f}")
        
        # Model 3: Potential depth
        g3 = model_3_potential(log_Mh, z, LOG_MH_REF, ALPHA_LOCAL)
        print(f"  Model 3 (Potential depth):         Γ_t = {g3:.3f}")
        
        # Model 4: Formation time
        g4 = model_4_formation_time(log_Mh, z, LOG_MH_REF, ALPHA_LOCAL)
        print(f"  Model 4 (Formation time):          Γ_t = {g4:.3f}")
        
        # Model 5: Self-consistent
        g5, log_Mh_true, bias = model_5_self_consistent(log_Mh, z, LOG_MH_REF, ALPHA_LOCAL)
        print(f"  Model 5 (Self-consistent):         Γ_t = {g5:.3f} (true log M_h = {log_Mh_true:.2f})")
        
        # Model 6: z-dependent α
        g6, alpha_z = model_6_z_dependent(log_Mh, z, LOG_MH_REF)
        print(f"  Model 6 (z-dependent α={alpha_z:.2f}):    Γ_t = {g6:.3f}")
        
        # Model 7: Combined
        g7, alpha_z7 = model_7_combined(log_Mh, z, LOG_MH_REF)
        print(f"  Model 7 (Combined, α={alpha_z7:.2f}):     Γ_t = {g7:.3f}")
        
        # Model 8: Proper time integral
        g8, z_form = model_8_proper_time_integral(log_Mh, z, LOG_MH_REF)
        print(f"  Model 8 (Proper time integral):    Γ_t = {g8:.3f} (z_form={z_form:.1f})")
        
        # Model 9: Distance cascade
        g9, sfe_bias, mass_bias, sfr_bias = model_9_distance_cascade(log_Mh, z, LOG_MH_REF)
        print(f"  Model 9 (Cascade): Γ_t={g9:.3f}, SFE_bias={sfe_bias:.3f}")
    
    # Find what's needed
    print()
    print("=" * 70)
    print("WHAT'S NEEDED TO EXPLAIN THE ANOMALY?")
    print("=" * 70)
    
    target_gamma = 2.5  # To explain 2.5x SFE via isochrony
    avg_delta_log_Mh = np.mean([d["log_Mh"] - LOG_MH_REF for d in RED_MONSTERS.values()])
    
    # For Model 1: Γ_t = 1 + α × (1/3) × Δlog(M_h)
    # 2.5 = 1 + α × (1/3) × 0.7
    # α = 1.5 / (0.7/3) = 6.4
    required_alpha_m1 = (target_gamma - 1) / ((1/3) * avg_delta_log_Mh)
    print(f"\nModel 1: Required α = {required_alpha_m1:.2f} ({required_alpha_m1/ALPHA_LOCAL:.1f}× local)")
    
    # For Model 3: Γ_t = 1 + α × (2/3) × Δlog(M_h) × z_factor
    # With z_factor ~ 1, need α ~ 3.2
    required_alpha_m3 = (target_gamma - 1) / ((2/3) * avg_delta_log_Mh)
    print(f"Model 3: Required α = {required_alpha_m3:.2f} ({required_alpha_m3/ALPHA_LOCAL:.1f}× local)")
    
    # For Model 6: α(z) = α_0 × (1+z)^n
    # At z=5.5, (1+z)^0.5 = 2.55
    # So α(5.5) = 0.58 × 2.55 = 1.48
    # Need: 2.5 = 1 + 1.48 × (1/3) × 0.7 → doesn't work
    # Need higher n
    # α(z) = α_0 × (1+z)^n = required_alpha_m1
    # (1+z)^n = required_alpha_m1 / α_0
    # n = log(required_alpha_m1 / α_0) / log(1+z)
    avg_z = np.mean([d["z"] for d in RED_MONSTERS.values()])
    required_n = np.log(required_alpha_m1 / ALPHA_LOCAL) / np.log(1 + avg_z)
    print(f"Model 6: Required n = {required_n:.2f} for α(z) = α_0 × (1+z)^n")
    
    print()
    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print()
    print("The required α ~ 6-7 is about 10× the local calibration.")
    print()
    print("Possible explanations:")
    print("1. TEP coupling IS stronger at high-z (unscreened regime)")
    print("2. The M^(1/3) scaling underestimates potential depth effects")
    print("3. Cumulative effects over formation history are important")
    print("4. The anomaly has multiple causes (TEP + astrophysics)")
    print()
    print("Most promising: Model 3 (potential depth) or Model 6 (z-dependent α)")
    print("These require α ~ 3-4× local, which is more plausible.")
    
    # Save a figure
    print()
    print("Generating comparison figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Γ_t vs α for different models
    ax1 = axes[0]
    alphas = np.linspace(0.5, 8, 50)
    avg_log_Mh = np.mean([d["log_Mh"] for d in RED_MONSTERS.values()])
    
    g1_vals = [model_1_instantaneous(avg_log_Mh, LOG_MH_REF, a) for a in alphas]
    g3_vals = [model_3_potential(avg_log_Mh, 5.5, LOG_MH_REF, a) for a in alphas]
    
    ax1.plot(alphas, g1_vals, 'b-', label='Model 1 (Instantaneous)', linewidth=2)
    ax1.plot(alphas, g3_vals, 'r-', label='Model 3 (Potential depth)', linewidth=2)
    ax1.axhline(2.5, color='k', linestyle='--', label='Target Γ_t = 2.5')
    ax1.axvline(ALPHA_LOCAL, color='g', linestyle=':', label=f'α_local = {ALPHA_LOCAL}')
    ax1.set_xlabel('α (TEP coupling)', fontsize=12)
    ax1.set_ylabel('Γ_t (enhancement factor)', fontsize=12)
    ax1.set_title('TEP Enhancement vs Coupling Strength', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(1, 4)
    
    # Right: Required α vs redshift for Model 6
    ax2 = axes[1]
    zs = np.linspace(0, 10, 50)
    for n in [0.5, 1.0, 1.5, 2.0]:
        alpha_z = ALPHA_LOCAL * (1 + zs) ** n
        ax2.plot(zs, alpha_z, label=f'n = {n}')
    
    ax2.axhline(required_alpha_m1, color='k', linestyle='--', label=f'Required α = {required_alpha_m1:.1f}')
    ax2.axvline(5.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('α(z)', fontsize=12)
    ax2.set_title('Redshift-Dependent Coupling: α(z) = α_0 × (1+z)^n', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 15)
    
    plt.tight_layout()
    plt.savefig('/Users/matthewsmawfield/www/TEP-JWST/results/outputs/tep_model_exploration.png', dpi=150)
    print("Saved: results/outputs/tep_model_exploration.png")
    
    return {
        "required_alpha_m1": required_alpha_m1,
        "required_alpha_m3": required_alpha_m3,
        "required_n_m6": required_n
    }

if __name__ == "__main__":
    main()
