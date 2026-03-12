#!/usr/bin/env python3
"""
TEP-JWST Step 147: Cosmic Star Formation Rate Density (SFRD) Correction

JWST measures cosmic SFRD at z>8 that is 3-10× higher than ΛCDM predictions.
If TEP's isochrony bias inflates apparent stellar masses, it also inflates
apparent SFRs from SED fitting (since SFR ∝ M* × sSFR, and M* is biased high).

This step:
  1. Computes observed SFRD in redshift bins from UNCOVER + CEERS data
  2. Applies TEP correction: SFR_true = SFR_obs / Γ_t^m (m ≈ 0.5, UV-based SFR bias)
  3. Compares to ΛCDM predictions (Madau & Dickinson 2014 + Mason+ 2015 extrapolation)
  4. Quantifies the reduction in SFRD excess per redshift bin

The SFR bias index m ≈ 0.5 is smaller than the mass bias index n ≈ 0.7 because:
- SFR from UV luminosity depends on recent star formation (< 100 Myr),
  which is less affected by long-term aging than the cumulative mass
- SFR from SED fitting is partially degenerate with M*, but not fully

Output:
  - results/outputs/step_147_cosmic_sfrd_correction.json
  - results/figures/figure_147_cosmic_sfrd_correction.png

Author: Matthew L. Smawfield
Date: February 2026
"""

import sys
import json
import numpy as np
np.random.seed(42)
import pandas as pd
from pathlib import Path
from scipy import stats

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import (
    compute_gamma_t, stellar_to_halo_mass, _cosmic_time_gyr,
    ALPHA_0
)

STEP_NUM = "147"
STEP_NAME = "cosmic_sfrd_correction"

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
RESULTS_INTERIM = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [OUTPUT_PATH, FIGURES_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# ΛCDM SFRD PREDICTIONS
# =============================================================================

def madau_dickinson_sfrd(z):
    """
    Madau & Dickinson (2014) cosmic SFRD fit.
    log10(SFRD) in M_sun/yr/Mpc^3.
    
    SFRD(z) = 0.015 * (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6)
    """
    sfrd = 0.015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)
    return sfrd


def lcdm_sfrd_highz(z):
    """
    ΛCDM SFRD prediction at z>8 from Mason+ (2015) / Bouwens+ (2022).
    Extrapolation of UV luminosity function integration.
    Returns SFRD in M_sun/yr/Mpc^3.
    
    At z>8, the Madau & Dickinson fit underpredicts relative to
    UV LF-based estimates. We use a compromise:
    - z=8: ~10^-2.2 M_sun/yr/Mpc^3
    - z=10: ~10^-2.8
    - z=12: ~10^-3.5
    Parametrized as log10(SFRD) = -1.5 - 0.35*(z-6) for z>6.
    """
    log_sfrd = -1.5 - 0.35 * (z - 6)
    return 10**log_sfrd


# =============================================================================
# SURVEY VOLUMES
# =============================================================================

def comoving_volume_shell(z_lo, z_hi, area_arcmin2, H0=67.4, Om=0.315):
    """
    Comoving volume of a survey field between z_lo and z_hi.
    area_arcmin2: survey area in arcmin^2.
    Returns volume in Mpc^3.
    """
    from astropy.cosmology import Planck18 as cosmo
    import numpy as np

    # Convert area from arcmin^2 to steradians
    area_sr = area_arcmin2 * (np.pi / (180 * 60))**2

    # Comoving volume
    vol_lo = cosmo.comoving_volume(z_lo).value * (area_sr / (4 * np.pi))
    vol_hi = cosmo.comoving_volume(z_hi).value * (area_sr / (4 * np.pi))

    return vol_hi - vol_lo


# Survey areas (approximate, from literature)
SURVEY_AREAS = {
    "UNCOVER": 45.0,      # ~45 arcmin^2 (Abell 2744 parallel field)
    "CEERS": 100.0,       # ~100 arcmin^2 (EGS field)
}


# =============================================================================
# LOAD DATA
# =============================================================================

def load_uncover():
    path = RESULTS_INTERIM / "step_002_uncover_full_sample_tep.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "z_phot"])
    # sfr100 is LINEAR SFR in M_sun/yr (100 Myr averaged)
    df = df.dropna(subset=["sfr100"])
    df = df[df["sfr100"] > 0].copy()
    df["log_sfr"] = np.log10(df["sfr100"])
    df["survey"] = "UNCOVER"
    return df


def load_ceers():
    path = DATA_INTERIM / "ceers_highz_sample.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "z_phot", "sfr"])
    # sfr is log10(SFR) in M_sun/yr — filter out -inf and extreme values
    df = df[np.isfinite(df["sfr"])].copy()
    df = df.rename(columns={"sfr": "log_sfr"})
    df["survey"] = "CEERS"
    return df


# =============================================================================
# COMPUTE SFRD
# =============================================================================

def compute_sfrd_bins(df, z_bins, survey_name, area_arcmin2):
    """
    Compute SFRD in redshift bins for a single survey.
    Returns list of dicts with observed and TEP-corrected SFRD.
    """
    results = []

    for z_lo, z_hi in z_bins:
        mask = (df["z_phot"] >= z_lo) & (df["z_phot"] < z_hi)
        sub = df[mask].copy()

        if len(sub) < 5:
            continue

        z_mid = (z_lo + z_hi) / 2.0

        # Comoving volume
        vol = comoving_volume_shell(z_lo, z_hi, area_arcmin2)

        # Observed SFR (linear) — clamp log_sfr to avoid overflow
        log_sfr_clamped = np.clip(sub["log_sfr"].values, -5, 5)
        sfr_linear = 10**log_sfr_clamped
        sfr_total_obs = np.sum(sfr_linear)
        sfrd_obs = sfr_total_obs / vol

        # TEP correction: SFR_true = SFR_obs / Γ_t^m
        # m = 0.5 for UV-based SFR (less affected than mass)
        m_sfr = 0.5
        log_Mh = stellar_to_halo_mass(sub["log_Mstar"].values, sub["z_phot"].values)
        gamma_t = compute_gamma_t(log_Mh, sub["z_phot"].values, ALPHA_0)
        sfr_correction = np.power(np.maximum(gamma_t, 0.01), m_sfr)
        sfr_corrected = sfr_linear / sfr_correction
        sfr_total_corr = np.sum(sfr_corrected)
        sfrd_corr = sfr_total_corr / vol

        # Also compute with mass-based correction (m = 0.7) for comparison
        m_mass = 0.7
        sfr_correction_mass = np.power(np.maximum(gamma_t, 0.01), m_mass)
        sfr_corrected_mass = sfr_linear / sfr_correction_mass
        sfrd_corr_mass = np.sum(sfr_corrected_mass) / vol

        # ΛCDM predictions
        sfrd_md14 = madau_dickinson_sfrd(z_mid)
        sfrd_lcdm = lcdm_sfrd_highz(z_mid)

        # Excess factors
        excess_obs = sfrd_obs / sfrd_lcdm if sfrd_lcdm > 0 else np.nan
        excess_corr = sfrd_corr / sfrd_lcdm if sfrd_lcdm > 0 else np.nan
        reduction_pct = (1 - sfrd_corr / sfrd_obs) * 100 if sfrd_obs > 0 else 0

        results.append({
            "z_lo": z_lo,
            "z_hi": z_hi,
            "z_mid": z_mid,
            "n": len(sub),
            "volume_Mpc3": float(vol),
            "sfrd_obs": float(sfrd_obs),
            "sfrd_corr_m05": float(sfrd_corr),
            "sfrd_corr_m07": float(sfrd_corr_mass),
            "sfrd_lcdm_extrapolation": float(sfrd_lcdm),
            "sfrd_madau_dickinson": float(sfrd_md14),
            "log_sfrd_obs": float(np.log10(max(sfrd_obs, 1e-10))),
            "log_sfrd_corr": float(np.log10(max(sfrd_corr, 1e-10))),
            "log_sfrd_corr_m07": float(np.log10(max(sfrd_corr_mass, 1e-10))),
            "log_sfrd_lcdm": float(np.log10(max(sfrd_lcdm, 1e-10))),
            "excess_factor_obs": float(excess_obs),
            "excess_factor_corr": float(excess_corr),
            "sfrd_reduction_pct": float(reduction_pct),
            "mean_gamma_t": float(np.mean(gamma_t)),
            "mean_sfr_correction": float(np.mean(sfr_correction)),
        })

    return results


# =============================================================================
# COMBINED SFRD (weighted by volume)
# =============================================================================

def combine_survey_sfrd(all_results, z_bins):
    """
    Combine SFRD measurements from multiple surveys using volume weighting.
    """
    combined = []
    for z_lo, z_hi in z_bins:
        z_mid = (z_lo + z_hi) / 2.0
        sfr_obs_total = 0
        sfr_corr_total = 0
        sfr_corr_m07_total = 0
        vol_total = 0
        n_total = 0

        for survey_results in all_results:
            for r in survey_results:
                if r["z_lo"] == z_lo and r["z_hi"] == z_hi:
                    sfr_obs_total += r["sfrd_obs"] * r["volume_Mpc3"]
                    sfr_corr_total += r["sfrd_corr_m05"] * r["volume_Mpc3"]
                    sfr_corr_m07_total += r["sfrd_corr_m07"] * r["volume_Mpc3"]
                    vol_total += r["volume_Mpc3"]
                    n_total += r["n"]

        if vol_total == 0:
            continue

        sfrd_obs = sfr_obs_total / vol_total
        sfrd_corr = sfr_corr_total / vol_total
        sfrd_corr_m07 = sfr_corr_m07_total / vol_total
        sfrd_lcdm = lcdm_sfrd_highz(z_mid)
        sfrd_md14 = madau_dickinson_sfrd(z_mid)

        excess_obs = sfrd_obs / sfrd_lcdm
        excess_corr = sfrd_corr / sfrd_lcdm
        reduction_pct = (1 - sfrd_corr / sfrd_obs) * 100

        combined.append({
            "z_lo": z_lo, "z_hi": z_hi, "z_mid": z_mid,
            "n": n_total,
            "log_sfrd_obs": float(np.log10(max(sfrd_obs, 1e-10))),
            "log_sfrd_corr": float(np.log10(max(sfrd_corr, 1e-10))),
            "log_sfrd_corr_m07": float(np.log10(max(sfrd_corr_m07, 1e-10))),
            "log_sfrd_lcdm": float(np.log10(max(sfrd_lcdm, 1e-10))),
            "log_sfrd_md14": float(np.log10(max(sfrd_md14, 1e-10))),
            "excess_factor_obs": float(excess_obs),
            "excess_factor_corr": float(excess_corr),
            "sfrd_reduction_pct": float(reduction_pct),
        })

    return combined


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure(combined, uncover_results, ceers_results, output_path):
    """Generate 3-panel SFRD correction figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE.get('web_three_panel', (16, 5.5)))

    # ---- Panel 1: SFRD vs redshift ----
    ax = axes[0]
    z_plot = np.linspace(4, 13, 200)
    ax.plot(z_plot, np.log10(madau_dickinson_sfrd(z_plot)),
            "k--", linewidth=1.5, label="Madau & Dickinson (2014)", alpha=0.6)
    ax.plot(z_plot[z_plot >= 6], np.log10(lcdm_sfrd_highz(z_plot[z_plot >= 6])),
            "k-", linewidth=2, label="ΛCDM extrapolation", alpha=0.8)

    z_mids = [c["z_mid"] for c in combined]
    log_obs = [c["log_sfrd_obs"] for c in combined]
    log_corr = [c["log_sfrd_corr"] for c in combined]

    ax.scatter(z_mids, log_obs, s=100, c="#e74c3c", zorder=5, edgecolors="k",
               label="Observed SFRD", marker="s")
    ax.scatter(z_mids, log_corr, s=100, c="#3498db", zorder=5, edgecolors="k",
               label="TEP-corrected (m=0.5)", marker="o")

    # Connect observed → corrected with arrows
    for zm, lo, lc in zip(z_mids, log_obs, log_corr):
        ax.annotate("", xy=(zm, lc), xytext=(zm, lo),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.set_xlabel("Redshift", fontsize=12)
    ax.set_ylabel("log₁₀(SFRD) [M☉/yr/Mpc³]", fontsize=12)
    ax.set_title("Cosmic SFRD: Observed vs TEP-Corrected", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(5.5, 11.5)
    ax.set_ylim(-4.5, -0.5)

    # ---- Panel 2: Excess factor ----
    ax2 = axes[1]
    excess_obs = [c["excess_factor_obs"] for c in combined]
    excess_corr = [c["excess_factor_corr"] for c in combined]

    x = np.arange(len(z_mids))
    width = 0.35
    bars1 = ax2.bar(x - width/2, excess_obs, width, color="#e74c3c",
                     label="Observed", edgecolor="k", alpha=0.8)
    bars2 = ax2.bar(x + width/2, excess_corr, width, color="#3498db",
                     label="TEP-corrected", edgecolor="k", alpha=0.8)

    ax2.axhline(y=1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"z={c['z_lo']:.0f}–{c['z_hi']:.0f}" for c in combined],
                         fontsize=9)
    ax2.set_ylabel("SFRD / SFRD_ΛCDM", fontsize=12)
    ax2.set_title("Excess Over ΛCDM Prediction", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_yscale("log")

    # ---- Panel 3: Reduction percentage ----
    ax3 = axes[2]
    reductions = [c["sfrd_reduction_pct"] for c in combined]
    colors = ["#2ecc71" if r > 20 else "#f39c12" for r in reductions]
    ax3.bar(x, reductions, color=colors, edgecolor="k", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"z={c['z_lo']:.0f}–{c['z_hi']:.0f}" for c in combined],
                         fontsize=9)
    ax3.set_ylabel("SFRD Reduction (%)", fontsize=12)
    ax3.set_title("TEP Correction Magnitude", fontsize=11)

    for i, (xi, r) in enumerate(zip(x, reductions)):
        ax3.text(xi, r + 1, f"{r:.0f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print_status(f"Figure saved to {output_path}", "INFO")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 65, "INFO")
    print_status("Step 147: Cosmic SFRD Correction at z > 6", "INFO")
    print_status("=" * 65, "INFO")

    # Load data
    print_status("Loading SFR data...", "INFO")
    df_uncover = load_uncover()
    df_ceers = load_ceers()
    if df_ceers is None:
        print_status("ceers_highz_sample.csv not found — run step_031/032 first. Aborting.", "ERROR")
        return {"status": "aborted", "reason": "missing ceers_highz_sample.csv"}
    print_status(f"  UNCOVER: {len(df_uncover)} galaxies with SFR", "INFO")
    print_status(f"  CEERS: {len(df_ceers)} galaxies with SFR", "INFO")

    # Redshift bins
    z_bins = [(6, 7), (7, 8), (8, 9), (9, 10), (10, 12)]

    # Compute per-survey
    print_status("\nComputing SFRD per survey...", "INFO")
    print_status("  UNCOVER:", "INFO")
    uncover_results = compute_sfrd_bins(df_uncover, z_bins, "UNCOVER", SURVEY_AREAS["UNCOVER"])
    for r in uncover_results:
        print_status(f"    z=[{r['z_lo']},{r['z_hi']}): N={r['n']}, "
                     f"log SFRD_obs={r['log_sfrd_obs']:.2f}, "
                     f"log SFRD_corr={r['log_sfrd_corr']:.2f}, "
                     f"reduction={r['sfrd_reduction_pct']:.0f}%", "INFO")

    print_status("  CEERS:", "INFO")
    ceers_results = compute_sfrd_bins(df_ceers, z_bins, "CEERS", SURVEY_AREAS["CEERS"])
    for r in ceers_results:
        print_status(f"    z=[{r['z_lo']},{r['z_hi']}): N={r['n']}, "
                     f"log SFRD_obs={r['log_sfrd_obs']:.2f}, "
                     f"log SFRD_corr={r['log_sfrd_corr']:.2f}, "
                     f"reduction={r['sfrd_reduction_pct']:.0f}%", "INFO")

    # Combined
    print_status("\nCombined SFRD (volume-weighted):", "INFO")
    combined = combine_survey_sfrd([uncover_results, ceers_results], z_bins)
    for c in combined:
        print_status(f"  z=[{c['z_lo']},{c['z_hi']}): N={c['n']}, "
                     f"log SFRD_obs={c['log_sfrd_obs']:.2f}, "
                     f"log SFRD_corr={c['log_sfrd_corr']:.2f}, "
                     f"excess_obs={c['excess_factor_obs']:.1f}×, "
                     f"excess_corr={c['excess_factor_corr']:.1f}×, "
                     f"reduction={c['sfrd_reduction_pct']:.0f}%", "INFO")

    # Summary statistics
    print_status("\n" + "=" * 65, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 65, "INFO")

    # z>8 combined
    z8_bins = [c for c in combined if c["z_lo"] >= 8]
    if z8_bins:
        mean_reduction_z8 = np.mean([c["sfrd_reduction_pct"] for c in z8_bins])
        mean_excess_obs_z8 = np.mean([c["excess_factor_obs"] for c in z8_bins])
        mean_excess_corr_z8 = np.mean([c["excess_factor_corr"] for c in z8_bins])
        max_excess_obs = max(c["excess_factor_obs"] for c in z8_bins)
        max_excess_corr = max(c["excess_factor_corr"] for c in z8_bins)

        print_status(f"  z>8 mean SFRD reduction: {mean_reduction_z8:.0f}%", "INFO")
        print_status(f"  z>8 mean excess (observed): {mean_excess_obs_z8:.1f}× ΛCDM", "INFO")
        print_status(f"  z>8 mean excess (TEP-corrected): {mean_excess_corr_z8:.1f}× ΛCDM", "INFO")
        print_status(f"  z>8 max excess: {max_excess_obs:.1f}× → {max_excess_corr:.1f}×", "INFO")

    # Overall
    all_reductions = [c["sfrd_reduction_pct"] for c in combined]
    mean_reduction_all = np.mean(all_reductions)
    print_status(f"  Overall mean SFRD reduction: {mean_reduction_all:.0f}%", "INFO")

    # Sensitivity: m=0.5 vs m=0.7
    print_status("\n  Sensitivity to SFR bias index m:", "INFO")
    for c in combined:
        if c["z_lo"] >= 8:
            r_m05 = c["log_sfrd_corr"]
            r_m07 = c["log_sfrd_corr_m07"]
            print_status(f"    z=[{c['z_lo']},{c['z_hi']}): "
                         f"m=0.5 → log SFRD={r_m05:.2f}, "
                         f"m=0.7 → log SFRD={r_m07:.2f}", "INFO")

    # Generate figure
    print_status("\nGenerating figure...", "INFO")
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    generate_figure(combined, uncover_results, ceers_results, fig_path)

    # Build results dict
    results = {
        "step": f"step_{STEP_NUM}",
        "name": "Cosmic SFRD Correction",
        "surveys": {
            "UNCOVER": uncover_results,
            "CEERS": ceers_results,
        },
        "combined": combined,
        "summary": {
            "z_gt_8_mean_reduction_pct": float(mean_reduction_z8) if z8_bins else None,
            "z_gt_8_mean_excess_obs": float(mean_excess_obs_z8) if z8_bins else None,
            "z_gt_8_mean_excess_corr": float(mean_excess_corr_z8) if z8_bins else None,
            "overall_mean_reduction_pct": float(mean_reduction_all),
            "sfr_bias_index_m": 0.5,
            "mass_bias_index_n": 0.7,
        },
        "methodology": {
            "sfr_correction": "SFR_true = SFR_obs / Gamma_t^m, m=0.5",
            "lcdm_extrapolation": "log10(SFRD) = -1.5 - 0.35*(z-6) for z>6",
            "volume_calculation": "flat LCDM comoving shell, H0=67.4, Om=0.315",
            "caveat": (
                "SFR bias index m=0.5 is approximate. UV-based SFRs are less "
                "affected by TEP than SED-based masses because UV traces recent "
                "star formation (< 100 Myr). Values m=0.3-0.7 bracket the "
                "plausible range. The correction is conservative at m=0.5."
            ),
        },
    }

    out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {out_file}", "INFO")

    print_status("\n" + "=" * 65, "INFO")
    if z8_bins:
        print_status(
            f"HEADLINE: TEP reduces cosmic SFRD at z>8 by {mean_reduction_z8:.0f}%, "
            f"cutting the ΛCDM excess from {mean_excess_obs_z8:.1f}× to {mean_excess_corr_z8:.1f}×.",
            "INFO"
        )
    print_status("=" * 65, "INFO")


if __name__ == "__main__":
    main()
