#!/usr/bin/env python3
"""
TEP-JWST Analysis: Red Monsters and the "Too Many, Too Massive, Too Early" Problem

This script tests whether TEP's chronological enhancement can explain the anomalously
high star formation efficiency observed in early massive galaxies.

Data sources:
- Xiao et al. 2024, Nature: "Red Monsters" (3 galaxies at z~5-6)
- Literature values for comparison galaxies

TEP Prediction:
- Galaxies in deeper potential wells experience more proper time per cosmic time
- This manifests as higher effective star formation efficiency
- Enhancement factor: Γ_t ∝ 1 + α * log10(M_h / M_ref)
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
import json
from pathlib import Path

# =============================================================================
# RED MONSTERS DATA (Xiao et al. 2024, Nature)
# =============================================================================

# Precise data from Xiao et al. 2024, Nature
# DOI: 10.1038/s41586-024-08094-5
# arXiv: 2309.02492

RED_MONSTERS = {
    "S1": {
        "z": 5.85,  # From FRESCO spectroscopy
        "log_Mstar": 11.08,  # log10(M*/Msun), from SED fitting
        "log_Mstar_err": [0.13, 0.11],  # [lower, upper] uncertainty
        "SFR_IR": 795,  # Msun/yr, from far-IR SED fitting
        "SFR_IR_err": 40,
        "A_V": 3.5,  # mag, dust attenuation
        "SFE": 0.50,  # Baryon-to-star conversion efficiency
        "log_Mhalo_required": 12.88,  # If epsilon=0.2 were true
        "log_Mhalo_required_err": [0.13, 0.11],
        "notes": "Most extreme; SCUBA2 detected; spectroscopically confirmed"
    },
    "S2": {
        "z": 5.30,  # GN10, previously known
        "log_Mstar": 10.88,
        "log_Mstar_err": [0.17, 0.23],
        "SFR_IR": 1030,  # Msun/yr
        "SFR_IR_err": [150, 190],  # asymmetric
        "A_V": 3.2,
        "SFE": 0.50,
        "log_Mhalo_required": 12.68,
        "log_Mhalo_required_err": [0.17, 0.23],
        "notes": "Also known as GN10; Riechers+2020; spectroscopically confirmed"
    },
    "S3": {
        "z": 5.55,
        "log_Mstar": 10.74,
        "log_Mstar_err": [0.18, 0.17],
        "SFR_IR": 988,  # Msun/yr
        "SFR_IR_err": 49,
        "A_V": 3.0,
        "SFE": 0.50,
        "log_Mhalo_required": 12.54,
        "log_Mhalo_required_err": [0.18, 0.17],
        "notes": "SCUBA2 detected; spectroscopically confirmed"
    }
}

# Standard model expectation
EPSILON_STANDARD = 0.20  # Maximum observed efficiency at lower-z
F_BARYON = 0.158  # Cosmic baryon fraction (Planck 2020)

# Typical FRESCO galaxy for comparison (from same paper)
# Most of the 36 DSFGs in the sample have epsilon < 0.2
TYPICAL_GALAXY = {
    "z": 5.5,
    "log_Mstar": 10.0,  # log10(M*/Msun) ~ 10^10 - median of FRESCO sample
    "SFE": 0.20,  # 20% efficiency (standard model maximum)
    "notes": "Typical FRESCO DSFG at z~5-6"
}

# =============================================================================
# TEP PARAMETERS
# =============================================================================

# From TEP-H0 (Paper 12): Local calibration
ALPHA_LOCAL = 0.58  # TEP coupling constant (optimized in TEP-H0)

# From TEP-UCD (Paper 7): Universal critical density
RHO_C = 20.0  # g/cm³ - saturation density for soliton formation

# High-z enhancement: At early times, halos are denser and less screened
# The TEP effect should be STRONGER at high-z because:
# 1. Halos are more compact (higher density)
# 2. Less time for screening to develop
# 3. Cosmic mean density is higher, so relative enhancement is larger
ALPHA_HIGHZ = 1.2  # Enhanced coupling at z > 5 (unscreened regime)

# Reference mass: Use the typical FRESCO galaxy mass for comparison
# This ensures we're measuring the DIFFERENTIAL enhancement
LOG_MSTAR_REF = 10.0  # Typical FRESCO DSFG stellar mass (log10 Msun)
SHMR_OFFSET = 2.0  # log10(M_h / M*) at high-z (stellar-to-halo mass relation)

# Physical constants
MSUN_G = 1.989e33  # Solar mass in grams
PC_CM = 3.086e18  # Parsec in cm
KPC_CM = 3.086e21  # kpc in cm

# =============================================================================
# FUNCTIONS
# =============================================================================

def cosmic_age_at_z(z):
    """Return cosmic age at redshift z in Gyr."""
    return cosmo.age(z).value

def halo_mass_from_stellar(log_Mstar, shmr_offset=SHMR_OFFSET):
    """Estimate halo mass from stellar mass using SHMR."""
    return log_Mstar + shmr_offset

def halo_density(log_Mh, z):
    """
    Estimate mean halo density at redshift z.
    
    At high-z, halos are more compact. Using virial scaling:
    ρ_vir ∝ Δ_c(z) * ρ_crit(z) ∝ (1+z)^3
    
    For a 10^12 Msun halo at z=0: R_vir ~ 200 kpc, ρ ~ 10^-26 g/cm³
    At z=6: ρ ~ 10^-26 * (7)^3 ~ 3 × 10^-24 g/cm³
    """
    # Virial overdensity factor (approximate)
    delta_vir = 200  # times critical density
    
    # Critical density at z (Planck18)
    rho_crit_0 = 9.47e-30  # g/cm³ at z=0
    rho_crit_z = rho_crit_0 * (1 + z)**3 * (0.3 * (1+z)**3 + 0.7)**0.5  # Approximate
    
    # Mean halo density
    rho_halo = delta_vir * rho_crit_z
    
    return rho_halo

def screening_factor(rho, rho_c=RHO_C):
    """
    Calculate Vainshtein screening factor.
    
    From TEP-UCD: S ∝ ρ^{1/3}
    At low densities (galaxies): S << 1 (unscreened, full TEP effect)
    At high densities (neutron stars): S >> 1 (screened, GR recovered)
    
    For galactic densities (ρ ~ 10^-24 g/cm³), S ~ 0.01 (unscreened)
    """
    S = (rho / rho_c) ** (1/3)
    return S

def tep_enhancement_factor(log_Mh, log_Mh_ref, z, alpha=None):
    """
    Calculate TEP chronological enhancement factor.
    
    Physical model:
    - Deeper potential wells → faster proper time accumulation
    - Enhancement scales with potential depth: Φ ∝ M/R ∝ M^{2/3} (virial)
    - At high-z, halos are denser and less screened
    
    Γ_t = 1 + α_eff * log10(M_h / M_ref) * (1 - S)
    
    where S is the screening factor (small at galactic densities)
    """
    if alpha is None:
        # Use enhanced coupling at high-z
        alpha = ALPHA_HIGHZ if z > 3 else ALPHA_LOCAL
    
    delta_log_Mh = log_Mh - log_Mh_ref
    
    # Estimate halo density and screening
    rho = halo_density(log_Mh, z)
    S = screening_factor(rho)
    
    # Unscreened fraction (1 - S), but S is tiny at galactic densities
    # so this is essentially 1.0
    unscreened = max(0, 1 - S)
    
    # Enhancement factor
    # The 0.15 scaling converts from log(M) to fractional time enhancement
    # This is calibrated to give ~2x enhancement for 2 dex mass difference
    gamma_t = 1.0 + alpha * delta_log_Mh * 0.15 * unscreened
    
    return gamma_t

def predicted_sfe_enhancement(log_Mstar, z, log_Mstar_ref=LOG_MSTAR_REF):
    """
    Predict the star formation efficiency enhancement due to TEP.
    
    If proper time runs faster, stellar evolution and feedback cycles
    are accelerated, leading to higher effective SFE.
    
    The enhancement is compounded over the galaxy's lifetime:
    - Faster stellar evolution → earlier feedback
    - Earlier feedback → more efficient gas processing
    - Net effect: SFE_eff ≈ SFE_0 * Γ_t^n where n ~ 1-2
    """
    log_Mh = halo_mass_from_stellar(log_Mstar)
    log_Mh_ref = halo_mass_from_stellar(log_Mstar_ref)
    
    gamma_t = tep_enhancement_factor(log_Mh, log_Mh_ref, z)
    
    # Compounding factor: stellar evolution + feedback cycles
    # n = 1.5 gives reasonable match to observed 2.5x enhancement
    n = 1.5
    
    return gamma_t ** n

# =============================================================================
# ISOCHRONY CORRECTION MODEL
# =============================================================================

def isochrony_mass_correction(log_Mstar_observed, gamma_t):
    """
    Correct observed stellar mass for isochrony bias.
    
    Under TEP, SED fitting assumes stellar evolution follows cosmic time.
    If proper time runs faster (Γ_t > 1), stars appear more evolved than
    their cosmic age would suggest. This leads to:
    
    1. Overestimated ages (stars look older)
    2. Overestimated M/L ratios (evolved populations are dimmer per unit mass)
    3. Overestimated stellar masses
    
    The correction factor depends on how M/L scales with age.
    For a simple stellar population: M/L ∝ t^0.7 (Bruzual & Charlot 2003)
    
    If observed age = Γ_t * true_age, then:
    (M/L)_obs / (M/L)_true = Γ_t^0.7
    M_obs / M_true = Γ_t^0.7
    
    So: log(M_true) = log(M_obs) - 0.7 * log(Γ_t)
    """
    if gamma_t <= 1.0:
        return log_Mstar_observed
    
    # M/L scaling exponent (from stellar population models)
    ml_exponent = 0.7
    
    correction = ml_exponent * np.log10(gamma_t)
    log_Mstar_true = log_Mstar_observed - correction
    
    return log_Mstar_true

def isochrony_sfe_correction(sfe_observed, gamma_t):
    """
    Correct observed SFE for isochrony bias.
    
    SFE = M_star / (f_b * M_halo)
    
    If M_star is overestimated by factor Γ_t^0.7, then:
    SFE_true = SFE_obs / Γ_t^0.7
    
    This means part of the "anomalous" SFE is an artifact of
    assuming isochrony in the mass measurement.
    """
    if gamma_t <= 1.0:
        return sfe_observed
    
    ml_exponent = 0.7
    correction_factor = gamma_t ** ml_exponent
    sfe_true = sfe_observed / correction_factor
    
    return sfe_true

# =============================================================================
# LAYERED SCREENING MODEL
# =============================================================================

def layered_screening_factor(log_Mh, z):
    """
    Model layered screening in high-z halos.
    
    A massive halo has a density profile:
    - Outer regions (R > R_vir/2): low density, unscreened
    - Intermediate (R_vir/10 < R < R_vir/2): partially screened
    - Core (R < R_vir/10): high density, may be screened
    
    At high-z, halos are more compact but also denser.
    The net effect depends on the balance.
    
    From TEP-UCD: screening factor S ∝ (ρ/ρ_c)^{1/3}
    For galactic densities (ρ ~ 10^-24 g/cm³), S ~ 10^-8 (negligible)
    
    At z ~ 6, virial density is ~200 * ρ_crit(z) ~ 10^-25 g/cm³
    Still far below ρ_c = 20 g/cm³, so screening is negligible.
    
    However, the CORE of a massive galaxy may be denser.
    For a 10^11 Msun stellar core in 1 kpc: ρ ~ 0.1 Msun/pc³ ~ 10^-23 g/cm³
    Still S ~ 10^-8, negligible.
    
    Conclusion: At galactic scales, screening is negligible.
    TEP effects are fully active.
    """
    # Estimate core density
    Mstar = 10**log_Mh / 100  # Stellar mass (assuming SHMR)
    R_core_kpc = 1.0  # Typical core radius
    R_core_cm = R_core_kpc * KPC_CM
    
    # Core density in g/cm³
    rho_core = (Mstar * MSUN_G) / ((4/3) * np.pi * R_core_cm**3)
    
    # Screening factor
    S = (rho_core / RHO_C) ** (1/3)
    
    # Unscreened fraction
    f_unscreened = max(0, 1 - S)
    
    return f_unscreened, S, rho_core

def tep_h0_model(log_Mh, log_Mh_ref, alpha=ALPHA_HIGHZ):
    """
    TEP-H0 style logarithmic model.
    
    From TEP-H0: Δμ = α * log10(σ/σ_ref)
    Since σ ∝ M_h^(1/3), we have log(σ) ∝ (1/3) * log(M_h)
    
    So: Δτ/τ = α * (1/3) * log10(M_h/M_ref)
    
    This gives the fractional enhancement in proper time.
    """
    delta_log_Mh = log_Mh - log_Mh_ref
    # The 1/3 factor comes from σ ∝ M^(1/3)
    fractional_enhancement = alpha * (1/3) * delta_log_Mh
    return 1.0 + fractional_enhancement

def analyze_galaxy(name, data, reference_sfe=EPSILON_STANDARD):
    """
    Analyze a single galaxy with full TEP corrections.
    
    Key insight: The observed stellar mass and SFE are biased because
    they assume isochrony. Under TEP:
    
    1. Stars in deep potentials evolve faster
    2. SED fitting interprets this as older/more massive populations
    3. The "anomalous" SFE is partially an artifact
    
    We compute both:
    - The apparent anomaly (what observers see)
    - The TEP-corrected values (what's physically happening)
    """
    z = data["z"]
    log_Mstar_obs = data["log_Mstar"]  # Observed (biased) stellar mass
    observed_sfe = data["SFE"]
    
    # Cosmic age at this redshift
    t_cosmic = cosmic_age_at_z(z)
    
    # Use the paper's derived halo mass if available
    if "log_Mhalo_required" in data:
        log_Mh = data["log_Mhalo_required"]
    else:
        log_Mh = halo_mass_from_stellar(log_Mstar_obs)
    
    # Reference halo mass (typical FRESCO galaxy)
    log_Mh_ref = halo_mass_from_stellar(LOG_MSTAR_REF)
    
    # TEP enhancement factor
    gamma_t = tep_h0_model(log_Mh, log_Mh_ref)
    
    # Layered screening analysis
    f_unscreened, S, rho_core = layered_screening_factor(log_Mh, z)
    
    # ISOCHRONY CORRECTIONS
    # The observed mass is biased high because SED fitting assumes isochrony
    log_Mstar_true = isochrony_mass_correction(log_Mstar_obs, gamma_t)
    
    # The observed SFE is biased high for the same reason
    sfe_true = isochrony_sfe_correction(observed_sfe, gamma_t)
    
    # Observed vs standard ratio (the "apparent anomaly")
    sfe_ratio_apparent = observed_sfe / reference_sfe
    
    # True vs standard ratio (after isochrony correction)
    sfe_ratio_true = sfe_true / reference_sfe
    
    # How much of the apparent anomaly is explained by isochrony bias?
    # If observed SFE = 0.5 and true SFE = 0.25, then 50% is bias
    isochrony_explains = 1 - (sfe_true / observed_sfe) if observed_sfe > 0 else 0
    
    # Remaining anomaly after isochrony correction
    remaining_anomaly = sfe_ratio_true - 1.0 if sfe_ratio_true > 1 else 0
    
    return {
        "name": name,
        "z": z,
        "t_cosmic_Gyr": t_cosmic,
        "log_Mstar_observed": log_Mstar_obs,
        "log_Mstar_true": log_Mstar_true,
        "mass_bias_dex": log_Mstar_obs - log_Mstar_true,
        "log_Mh": log_Mh,
        "log_Mh_ref": log_Mh_ref,
        "observed_SFE": observed_sfe,
        "true_SFE": sfe_true,
        "reference_SFE": reference_sfe,
        "SFE_ratio_apparent": sfe_ratio_apparent,
        "SFE_ratio_true": sfe_ratio_true,
        "TEP_gamma_t": gamma_t,
        "isochrony_explains_fraction": isochrony_explains,
        "remaining_anomaly": remaining_anomaly,
        "screening_factor": S,
        "core_density_g_cm3": rho_core,
        "notes": data.get("notes", "")
    }

def main():
    """Run the TEP Red Monsters analysis."""
    print("=" * 70)
    print("TEP-JWST Analysis: Red Monsters with Isochrony Correction")
    print("=" * 70)
    print()
    
    # Analyze Red Monsters
    results = []
    
    print("RED MONSTERS (Xiao et al. 2024, Nature)")
    print("-" * 70)
    print(f"{'Galaxy':<8} {'z':<5} {'Γ_t':<6} {'M*_obs':<8} {'M*_true':<8} {'ΔM*':<6} {'SFE_obs':<8} {'SFE_true':<8}")
    print("-" * 70)
    
    for name, data in RED_MONSTERS.items():
        result = analyze_galaxy(name, data)
        results.append(result)
        print(f"{result['name']:<8} {result['z']:<5.2f} {result['TEP_gamma_t']:<6.3f} "
              f"{result['log_Mstar_observed']:<8.2f} {result['log_Mstar_true']:<8.2f} "
              f"{result['mass_bias_dex']:<6.2f} {result['observed_SFE']:<8.2f} {result['true_SFE']:<8.2f}")
    
    # Analyze typical galaxy for comparison
    print()
    print("COMPARISON: Typical FRESCO Galaxy")
    print("-" * 70)
    typical_result = analyze_galaxy("Typical", TYPICAL_GALAXY)
    print(f"{typical_result['name']:<8} {typical_result['z']:<5.2f} {typical_result['TEP_gamma_t']:<6.3f} "
          f"{typical_result['log_Mstar_observed']:<8.2f} {typical_result['log_Mstar_true']:<8.2f} "
          f"{typical_result['mass_bias_dex']:<6.2f} {typical_result['observed_SFE']:<8.2f} {typical_result['true_SFE']:<8.2f}")
    
    # Summary statistics
    print()
    print("=" * 70)
    print("ISOCHRONY CORRECTION SUMMARY")
    print("=" * 70)
    
    avg_gamma_t = np.mean([r["TEP_gamma_t"] for r in results])
    avg_mass_bias = np.mean([r["mass_bias_dex"] for r in results])
    avg_sfe_obs = np.mean([r["observed_SFE"] for r in results])
    avg_sfe_true = np.mean([r["true_SFE"] for r in results])
    avg_isochrony_explains = np.mean([r["isochrony_explains_fraction"] for r in results])
    avg_remaining = np.mean([r["remaining_anomaly"] for r in results])
    
    print(f"Average TEP enhancement (Γ_t):           {avg_gamma_t:.3f}")
    print(f"Average stellar mass bias:                {avg_mass_bias:.2f} dex (observed too high)")
    print(f"Average observed SFE:                     {avg_sfe_obs:.2f} (2.5x standard)")
    print(f"Average TRUE SFE (after correction):      {avg_sfe_true:.2f}")
    print(f"Fraction of anomaly from isochrony bias:  {avg_isochrony_explains*100:.1f}%")
    print(f"Remaining anomaly (true SFE / 0.2 - 1):   {avg_remaining:.2f}")
    print()
    
    # Interpretation
    print("INTERPRETATION")
    print("-" * 70)
    print("Under TEP, the 'anomalous' SFE has TWO components:")
    print()
    print("1. ISOCHRONY BIAS (measurement artifact):")
    print("   - SED fitting assumes stellar clocks follow cosmic time")
    print("   - In deep potentials, stars evolve faster (Γ_t > 1)")
    print("   - This makes populations appear older/more massive")
    print(f"   - Stellar masses are overestimated by ~{avg_mass_bias:.2f} dex")
    print(f"   - This explains {avg_isochrony_explains*100:.0f}% of the apparent anomaly")
    print()
    print("2. REMAINING PHYSICAL EFFECT:")
    if avg_sfe_true > 0.2:
        print(f"   - True SFE ({avg_sfe_true:.2f}) still exceeds standard ({EPSILON_STANDARD})")
        print("   - This may reflect genuinely enhanced star formation")
        print("   - Possible causes: higher gas densities, faster cooling at high-z")
    else:
        print(f"   - True SFE ({avg_sfe_true:.2f}) is consistent with standard ({EPSILON_STANDARD})")
        print("   - The entire 'anomaly' is explained by isochrony bias!")
        print("   - Red Monsters are not anomalous under TEP.")
    print()
    
    # Screening analysis
    print("=" * 70)
    print("LAYERED SCREENING ANALYSIS")
    print("=" * 70)
    avg_S = np.mean([r["screening_factor"] for r in results])
    avg_rho = np.mean([r["core_density_g_cm3"] for r in results])
    print(f"Average core density:     {avg_rho:.2e} g/cm³")
    print(f"Average screening factor: {avg_S:.2e}")
    print(f"Critical density (ρ_c):   {RHO_C:.0f} g/cm³")
    print()
    if avg_S < 0.01:
        print("Screening is NEGLIGIBLE at galactic scales.")
        print("TEP effects are fully active in these halos.")
    else:
        print(f"Partial screening detected (S = {avg_S:.2f}).")
        print("TEP effects are reduced in dense cores.")
    
    # Parameter sensitivity analysis
    print()
    print("=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    print("What α (high-z coupling) would fully explain the 2.5x SFE anomaly?")
    print("-" * 70)
    
    # For the average Red Monster: log_Mstar ~ 10.9, need gamma_t = 2.5
    # gamma_t = (1 + α * Δlog_Mh * 0.15)^1.5
    # 2.5 = (1 + α * 0.9 * 0.15)^1.5
    # 2.5^(1/1.5) = 1 + α * 0.135
    # α = (2.5^0.667 - 1) / 0.135
    
    target_gamma = 2.5
    n = 1.5
    delta_log_Mh = 0.9  # Average mass difference
    
    required_alpha = (target_gamma**(1/n) - 1) / (delta_log_Mh * 0.15)
    print(f"Average Δlog(M_h) for Red Monsters: {delta_log_Mh:.2f} dex")
    print(f"Required α to explain 100% of anomaly: {required_alpha:.2f}")
    print(f"Current α (high-z): {ALPHA_HIGHZ:.2f}")
    print(f"Ratio needed/current: {required_alpha/ALPHA_HIGHZ:.2f}x")
    print()
    
    # Physical interpretation
    print("PHYSICAL INTERPRETATION")
    print("-" * 70)
    print(f"TEP-H0 local calibration: α = {ALPHA_LOCAL:.2f}")
    print(f"Current high-z assumption: α = {ALPHA_HIGHZ:.2f} ({ALPHA_HIGHZ/ALPHA_LOCAL:.1f}x local)")
    print(f"Required for full explanation: α = {required_alpha:.2f} ({required_alpha/ALPHA_LOCAL:.1f}x local)")
    print()
    
    if required_alpha < 5.0:
        print("This is within plausible range for unscreened high-z halos.")
        print("At z > 5, halos are denser and screening is weaker, so")
        print("stronger TEP coupling is physically motivated.")
    else:
        print("This requires very strong coupling, which may indicate:")
        print("1. Additional astrophysical effects beyond TEP")
        print("2. Non-linear TEP scaling at extreme masses")
        print("3. The anomaly has multiple contributing factors")
    
    print()
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "analysis": "TEP Red Monsters",
        "date": "2026-01-15",
        "parameters": {
            "alpha_local": ALPHA_LOCAL,
            "alpha_highz": ALPHA_HIGHZ,
            "log_Mstar_ref": LOG_MSTAR_REF,
            "SHMR_offset": SHMR_OFFSET
        },
        "red_monsters": results,
        "typical_galaxy": typical_result,
        "summary": {
            "avg_TEP_gamma_t": avg_gamma_t,
            "avg_mass_bias_dex": avg_mass_bias,
            "avg_SFE_observed": avg_sfe_obs,
            "avg_SFE_true": avg_sfe_true,
            "avg_isochrony_explains_fraction": avg_isochrony_explains,
            "avg_remaining_anomaly": avg_remaining
        }
    }
    
    output_file = output_dir / "tep_red_monsters_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")
    
    return output

if __name__ == "__main__":
    main()
