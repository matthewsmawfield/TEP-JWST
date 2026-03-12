#!/usr/bin/env python3
"""
TEP-JWST Step 173: TEP Concordance — Independent α₀ Recovery

The definitive statistical test: if TEP is a single-parameter theory with
coupling constant α₀, then fitting α₀ independently from each observable
should yield consistent values. This is the TEP analogue of the CMB/BAO/SNe
cosmic concordance diagram.

Observables (each fitted independently):
  1. Dust–Γ_t correlation (UNCOVER, z>8): maximize ρ(dust, Γ_t)
  2. AGB threshold F1 (UNCOVER, z>8): maximize F1(t_eff > 0.3 Gyr → dusty)
  3. Mass-sSFR inversion (UNCOVER): maximize Δρ(high_z − low_z)
  4. Cross-survey dust correlation (CEERS + COSMOS-Web, z>8): maximize mean ρ
  5. Anomalous galaxy resolution (all surveys, z>7): maximize resolution rate

Output:
  - results/outputs/step_173_tep_concordance.json
  - results/figures/figure_173_tep_concordance.png

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
    stellar_to_halo_mass_behroozi_like, LOG_MH_REF, Z_REF
)

STEP_NUM = "173"
STEP_NAME = "tep_concordance"

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
# α₀ GRID
# =============================================================================

ALPHA0_GRID = np.linspace(0.05, 1.5, 100)
N_BOOT = 200  # bootstrap iterations per observable


def _gamma_t_for_alpha(log_Mstar, z, alpha_0):
    """Compute Γ_t from stellar mass for a given α₀."""
    log_Mh = stellar_to_halo_mass(log_Mstar, z)
    return compute_gamma_t(log_Mh, z, alpha_0)


def _t_eff_for_alpha(log_Mstar, z, alpha_0):
    """Compute t_eff from stellar mass for a given α₀."""
    gamma_t = _gamma_t_for_alpha(log_Mstar, z, alpha_0)
    t_cosmic = _cosmic_time_gyr(z)
    return np.maximum(t_cosmic * gamma_t, 0.001)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_uncover():
    path = RESULTS_INTERIM / "step_02_uncover_full_sample_tep.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "dust", "z_phot"])
    return df


def load_ceers():
    path = DATA_INTERIM / "ceers_highz_sample.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "dust", "z_phot"])
    return df


def load_cosmosweb():
    path = DATA_INTERIM / "cosmosweb_highz_sample.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "dust", "z_phot"])
    return df


# =============================================================================
# OBSERVABLE 1: Dust–Γ_t Correlation (UNCOVER z>8)
# =============================================================================

def obs1_dust_teff_r2(df_uncover):
    """For each α₀, compute Pearson R²(dust, t_eff) at z>8.
    
    Unlike Spearman ρ, Pearson R² is sensitive to the actual scale of t_eff,
    which depends on α₀. This makes it a valid α₀ discriminator.
    """
    print_status("  Observable 1: Dust–t_eff R² (UNCOVER z>8)", "INFO")
    sub = df_uncover[df_uncover["z_phot"] > 8].copy()
    log_M = sub["log_Mstar"].values
    z = sub["z_phot"].values
    dust = sub["dust"].values
    n = len(sub)
    print_status(f"    N = {n}", "INFO")

    profile = np.zeros(len(ALPHA0_GRID))
    for i, a0 in enumerate(ALPHA0_GRID):
        t_eff = _t_eff_for_alpha(log_M, z, a0)
        r = np.corrcoef(dust, np.log10(np.maximum(t_eff, 1e-6)))[0, 1]
        profile[i] = r ** 2

    best_idx = np.argmax(profile)
    best_alpha = ALPHA0_GRID[best_idx]
    best_r2 = profile[best_idx]

    # Bootstrap CI on best α₀
    boot_alphas = []
    for _ in range(N_BOOT):
        idx = np.random.choice(n, n, replace=True)
        bprofile = np.zeros(len(ALPHA0_GRID))
        for i, a0 in enumerate(ALPHA0_GRID):
            t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
            r = np.corrcoef(dust[idx], np.log10(np.maximum(t_eff, 1e-6)))[0, 1]
            bprofile[i] = r ** 2
        boot_alphas.append(ALPHA0_GRID[np.argmax(bprofile)])

    ci = np.percentile(boot_alphas, [16, 84])
    print_status(f"    Best α₀ = {best_alpha:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], R² = {best_r2:.3f}", "INFO")

    return {
        "name": "Dust–t_eff R² (UNCOVER z>8)",
        "short": "Dust–t_eff",
        "alpha0_best": float(best_alpha),
        "alpha0_ci": [float(ci[0]), float(ci[1])],
        "metric_best": float(best_r2),
        "metric_name": "R²(dust, log t_eff)",
        "n": n,
        "profile": profile.tolist(),
    }


# =============================================================================
# OBSERVABLE 2: AGB Threshold F1 (UNCOVER z>8)
# =============================================================================

def obs2_agb_threshold_f1(df_uncover):
    """For each α₀, compute F1 of t_eff > 0.3 Gyr predicting dusty galaxies."""
    print_status("  Observable 2: AGB Threshold F1 (UNCOVER z>8)", "INFO")
    sub = df_uncover[df_uncover["z_phot"] > 8].copy()
    log_M = sub["log_Mstar"].values
    z = sub["z_phot"].values
    dust = sub["dust"].values

    # Binary dusty label: above-median dust
    dust_threshold = np.median(dust)
    y_true = (dust > dust_threshold).astype(int)
    n = len(sub)
    print_status(f"    N = {n}, dust threshold = {dust_threshold:.3f}", "INFO")

    def _f1(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-10)

    profile = np.zeros(len(ALPHA0_GRID))
    for i, a0 in enumerate(ALPHA0_GRID):
        t_eff = _t_eff_for_alpha(log_M, z, a0)
        y_pred = (t_eff > 0.3).astype(int)
        profile[i] = _f1(y_true, y_pred)

    best_idx = np.argmax(profile)
    best_alpha = ALPHA0_GRID[best_idx]
    best_f1 = profile[best_idx]

    # Bootstrap CI
    boot_alphas = []
    for _ in range(N_BOOT):
        idx = np.random.choice(n, n, replace=True)
        bprofile = np.zeros(len(ALPHA0_GRID))
        for i, a0 in enumerate(ALPHA0_GRID):
            t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
            y_pred = (t_eff > 0.3).astype(int)
            bprofile[i] = _f1(y_true[idx], y_pred)
        boot_alphas.append(ALPHA0_GRID[np.argmax(bprofile)])

    ci = np.percentile(boot_alphas, [16, 84])
    print_status(f"    Best α₀ = {best_alpha:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], F1 = {best_f1:.3f}", "INFO")

    return {
        "name": "AGB Threshold F1 (UNCOVER z>8)",
        "short": "AGB F1",
        "alpha0_best": float(best_alpha),
        "alpha0_ci": [float(ci[0]), float(ci[1])],
        "metric_best": float(best_f1),
        "metric_name": "F1",
        "n": n,
        "profile": profile.tolist(),
    }


# =============================================================================
# OBSERVABLE 3: Mass-sSFR Inversion (UNCOVER)
# =============================================================================

def obs3_ssfr_inversion(df_uncover):
    """For each α₀, compute Δρ = ρ(M*, sSFR, z>7) − ρ(M*, sSFR, z<6)."""
    print_status("  Observable 3: Mass-sSFR Inversion (UNCOVER)", "INFO")
    df = df_uncover.dropna(subset=["log_ssfr"]).copy()
    low = df[(df["z_phot"] >= 4) & (df["z_phot"] < 6)]
    high = df[(df["z_phot"] >= 7) & (df["z_phot"] < 10)]
    n_low, n_high = len(low), len(high)
    print_status(f"    N_low = {n_low}, N_high = {n_high}", "INFO")

    # The inversion itself doesn't depend on α₀ (it's an observed quantity),
    # but the TEP-PREDICTED inversion does. We test: for each α₀, correct
    # the observed sSFR by dividing by Γ_t, then see if the inversion
    # DISAPPEARS (which means α₀ successfully explains the inversion).
    # The best α₀ is the one that most completely removes the inversion.

    # Observed inversion
    rho_low_obs, _ = stats.spearmanr(low["log_Mstar"], low["log_ssfr"])
    rho_high_obs, _ = stats.spearmanr(high["log_Mstar"], high["log_ssfr"])
    delta_obs = rho_high_obs - rho_low_obs
    print_status(f"    Observed: ρ_low = {rho_low_obs:.3f}, ρ_high = {rho_high_obs:.3f}, Δρ = {delta_obs:.3f}", "INFO")

    log_M_high = high["log_Mstar"].values
    z_high = high["z_phot"].values
    ssfr_high = high["log_ssfr"].values

    # For each α₀: correct sSFR by TEP, measure how much the inversion is reduced
    # sSFR_true ≈ sSFR_obs / Γ_t^n → log_sSFR_true = log_sSFR_obs - n*log10(Γ_t)
    n_ml = 0.7
    profile = np.zeros(len(ALPHA0_GRID))
    for i, a0 in enumerate(ALPHA0_GRID):
        gt = _gamma_t_for_alpha(log_M_high, z_high, a0)
        ssfr_corr = ssfr_high - n_ml * np.log10(np.maximum(gt, 0.01))
        rho_corr, _ = stats.spearmanr(log_M_high, ssfr_corr)
        # Best α₀ is the one that brings corrected ρ closest to the low-z value
        profile[i] = -abs(rho_corr - rho_low_obs)  # negative distance → maximize

    best_idx = np.argmax(profile)
    best_alpha = ALPHA0_GRID[best_idx]
    best_residual = -profile[best_idx]

    # Bootstrap
    boot_alphas = []
    n_h = len(high)
    for _ in range(N_BOOT):
        idx = np.random.choice(n_h, n_h, replace=True)
        bprofile = np.zeros(len(ALPHA0_GRID))
        for i, a0 in enumerate(ALPHA0_GRID):
            gt = _gamma_t_for_alpha(log_M_high[idx], z_high[idx], a0)
            ssfr_corr = ssfr_high[idx] - n_ml * np.log10(np.maximum(gt, 0.01))
            rho_corr, _ = stats.spearmanr(log_M_high[idx], ssfr_corr)
            bprofile[i] = -abs(rho_corr - rho_low_obs)
        boot_alphas.append(ALPHA0_GRID[np.argmax(bprofile)])

    ci = np.percentile(boot_alphas, [16, 84])
    print_status(f"    Best α₀ = {best_alpha:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], residual = {best_residual:.4f}", "INFO")

    return {
        "name": "sSFR Inversion Correction (UNCOVER)",
        "short": "sSFR Inversion",
        "alpha0_best": float(best_alpha),
        "alpha0_ci": [float(ci[0]), float(ci[1])],
        "metric_best": float(best_residual),
        "metric_name": "|Δρ_corrected − ρ_low|",
        "n": n_high,
        "profile": (-np.array(profile)).tolist(),  # store as positive distance
    }


# =============================================================================
# OBSERVABLE 4: Cross-Survey Dust Correlation (CEERS + COSMOS-Web z>8)
# =============================================================================

def obs4_cross_survey_r2(df_ceers, df_cosmosweb):
    """For each α₀, compute mean Pearson R²(dust, log t_eff) across surveys at z>8.
    
    Pearson R² (unlike Spearman ρ) is sensitive to the actual t_eff scale.
    """
    print_status("  Observable 4: Cross-Survey Dust–t_eff R² (CEERS + COSMOS-Web z>8)", "INFO")

    surveys = []
    for name, df in [("CEERS", df_ceers), ("COSMOS-Web", df_cosmosweb)]:
        sub = df[df["z_phot"] > 8].copy()
        sub = sub.dropna(subset=["dust"])
        if len(sub) > 30:
            surveys.append((name, sub))
            print_status(f"    {name}: N = {len(sub)}", "INFO")

    if len(surveys) < 2:
        print_status("    Insufficient cross-survey data", "WARNING")
        return None

    profile = np.zeros(len(ALPHA0_GRID))
    for i, a0 in enumerate(ALPHA0_GRID):
        r2s = []
        for name, sub in surveys:
            t_eff = _t_eff_for_alpha(sub["log_Mstar"].values, sub["z_phot"].values, a0)
            r = np.corrcoef(sub["dust"].values, np.log10(np.maximum(t_eff, 1e-6)))[0, 1]
            r2s.append(r ** 2)
        profile[i] = np.mean(r2s)

    best_idx = np.argmax(profile)
    best_alpha = ALPHA0_GRID[best_idx]
    best_mean_r2 = profile[best_idx]

    # Bootstrap
    boot_alphas = []
    for _ in range(N_BOOT):
        bprofile = np.zeros(len(ALPHA0_GRID))
        for i, a0 in enumerate(ALPHA0_GRID):
            r2s = []
            for name, sub in surveys:
                n_s = len(sub)
                idx = np.random.choice(n_s, n_s, replace=True)
                t_eff = _t_eff_for_alpha(
                    sub["log_Mstar"].values[idx], sub["z_phot"].values[idx], a0
                )
                r = np.corrcoef(sub["dust"].values[idx], np.log10(np.maximum(t_eff, 1e-6)))[0, 1]
                r2s.append(r ** 2)
            bprofile[i] = np.mean(r2s)
        boot_alphas.append(ALPHA0_GRID[np.argmax(bprofile)])

    ci = np.percentile(boot_alphas, [16, 84])
    n_total = sum(len(s) for _, s in surveys)
    print_status(f"    Best α₀ = {best_alpha:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], mean R² = {best_mean_r2:.3f}", "INFO")

    return {
        "name": "Cross-Survey Dust–t_eff R² (CEERS + COSMOS-Web z>8)",
        "short": "Cross-Survey",
        "alpha0_best": float(best_alpha),
        "alpha0_ci": [float(ci[0]), float(ci[1])],
        "metric_best": float(best_mean_r2),
        "metric_name": "mean R²(dust, log t_eff)",
        "n": n_total,
        "profile": profile.tolist(),
    }


# =============================================================================
# OBSERVABLE 5: Anomalous Galaxy Resolution (all surveys z>7)
# =============================================================================

def obs5_impossible_resolution(df_uncover, df_ceers, df_cosmosweb):
    """For each α₀, compute fraction of anomalous galaxies resolved."""
    print_status("  Observable 5: Anomalous Galaxy Resolution (all surveys z>7)", "INFO")

    # Combine all surveys
    frames = []
    for name, df in [("UNCOVER", df_uncover), ("CEERS", df_ceers), ("COSMOS-Web", df_cosmosweb)]:
        sub = df[df["z_phot"] > 7].copy()
        sub = sub.dropna(subset=["log_Mstar"])
        if "z_phot" not in sub.columns:
            continue
        frames.append(sub[["log_Mstar", "z_phot"]])

    combined = pd.concat(frames, ignore_index=True)
    log_M = combined["log_Mstar"].values
    z = combined["z_phot"].values
    n = len(combined)
    print_status(f"    N = {n} galaxies at z>7", "INFO")

    # ΛCDM maximum stellar mass at each redshift
    # M*_max(z) ≈ f_b × ε_max × M_halo_max(z)
    # Using Sheth-Tormen HMF upper limit: log M_h_max ≈ 13.5 - 0.5*(z-6)
    # f_b = 0.157, ε_max = 0.3
    def _lcdm_max_logMstar(z_arr):
        log_Mh_max = 13.5 - 0.5 * (z_arr - 6)
        log_Mh_max = np.clip(log_Mh_max, 10.5, 14.0)
        fb = 0.157
        eps = 0.3
        return log_Mh_max + np.log10(fb * eps)

    max_logM = _lcdm_max_logMstar(z)
    impossible_mask = log_M > max_logM
    n_impossible = np.sum(impossible_mask)
    print_status(f"    N_impossible = {n_impossible}", "INFO")

    if n_impossible < 5:
        print_status("    Too few anomalous galaxies", "WARNING")
        return None

    # For each α₀, correct masses and count how many are resolved
    profile = np.zeros(len(ALPHA0_GRID))
    for i, a0 in enumerate(ALPHA0_GRID):
        gt = _gamma_t_for_alpha(log_M[impossible_mask], z[impossible_mask], a0)
        ml_bias = np.power(np.maximum(gt, 0.01), 0.7)
        log_M_corr = log_M[impossible_mask] - np.log10(ml_bias)
        max_logM_imp = max_logM[impossible_mask]
        n_resolved = np.sum(log_M_corr <= max_logM_imp)
        profile[i] = n_resolved / n_impossible

    best_idx = np.argmax(profile)
    best_alpha = ALPHA0_GRID[best_idx]
    best_frac = profile[best_idx]

    # Bootstrap
    imp_idx_all = np.where(impossible_mask)[0]
    boot_alphas = []
    for _ in range(N_BOOT):
        idx = np.random.choice(len(imp_idx_all), len(imp_idx_all), replace=True)
        sel = imp_idx_all[idx]
        bprofile = np.zeros(len(ALPHA0_GRID))
        for i, a0 in enumerate(ALPHA0_GRID):
            gt = _gamma_t_for_alpha(log_M[sel], z[sel], a0)
            ml_bias = np.power(np.maximum(gt, 0.01), 0.7)
            log_M_corr = log_M[sel] - np.log10(ml_bias)
            max_logM_sel = max_logM[sel]
            n_res = np.sum(log_M_corr <= max_logM_sel)
            bprofile[i] = n_res / len(sel)
        boot_alphas.append(ALPHA0_GRID[np.argmax(bprofile)])

    ci = np.percentile(boot_alphas, [16, 84])
    print_status(f"    Best α₀ = {best_alpha:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], resolved = {best_frac:.1%}", "INFO")

    return {
        "name": "Anomalous Galaxy Resolution (z>7)",
        "short": "Anomalous Gal.",
        "alpha0_best": float(best_alpha),
        "alpha0_ci": [float(ci[0]), float(ci[1])],
        "metric_best": float(best_frac),
        "metric_name": "fraction resolved",
        "n": int(n_impossible),
        "profile": profile.tolist(),
    }


# =============================================================================
# CONCORDANCE STATISTICS
# =============================================================================

def compute_concordance(observables):
    """Compute concordance χ² and Bayes factor."""
    alphas = [o["alpha0_best"] for o in observables]
    sigmas = [(o["alpha0_ci"][1] - o["alpha0_ci"][0]) / 2.0 for o in observables]

    # Weighted mean
    weights = [1.0 / max(s, 0.01)**2 for s in sigmas]
    w_sum = sum(weights)
    alpha_wm = sum(a * w for a, w in zip(alphas, weights)) / w_sum
    sigma_wm = 1.0 / np.sqrt(w_sum)

    # Concordance χ²
    chi2 = sum(((a - alpha_wm) / max(s, 0.01))**2 for a, s in zip(alphas, sigmas))
    ndof = len(observables) - 1
    p_concordance = float(1 - stats.chi2.cdf(chi2, ndof)) if ndof > 0 else 1.0

    # Maximum tension between any two observables
    max_tension = 0.0
    for i in range(len(observables)):
        for j in range(i + 1, len(observables)):
            sep = abs(alphas[i] - alphas[j])
            combined_sigma = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
            if combined_sigma > 0:
                tension = sep / combined_sigma
                max_tension = max(max_tension, tension)

    return {
        "alpha0_weighted_mean": float(alpha_wm),
        "alpha0_weighted_sigma": float(sigma_wm),
        "chi2_concordance": float(chi2),
        "ndof": ndof,
        "p_concordance": float(p_concordance),
        "max_pairwise_tension_sigma": float(max_tension),
        "concordant": bool(p_concordance > 0.05),
        "individual_alphas": alphas,
        "individual_sigmas": sigmas,
    }


# =============================================================================
# FIGURE
# =============================================================================

def generate_figure(observables, concordance, output_path):
    """Generate the TEP Concordance Diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 3]})

    # ---- Panel 1: Concordance whisker plot ----
    ax1 = axes[0]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    y_positions = list(range(len(observables)))

    for i, obs in enumerate(observables):
        ci = obs["alpha0_ci"]
        best = obs["alpha0_best"]
        color = colors[i % len(colors)]

        # Error bar (clamp to avoid negative xerr from bootstrap edge effects)
        xerr_lo = max(best - ci[0], 0.0)
        xerr_hi = max(ci[1] - best, 0.0)
        ax1.errorbar(
            best, i, xerr=[[xerr_lo], [xerr_hi]],
            fmt="o", color=color, markersize=8, capsize=5, linewidth=2,
            label=obs["short"]
        )

    # Weighted mean band
    wm = concordance["alpha0_weighted_mean"]
    ws = concordance["alpha0_weighted_sigma"]
    ax1.axvspan(wm - ws, wm + ws, alpha=0.15, color="gray", zorder=0)
    ax1.axvline(wm, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    # Cepheid α₀ = 0.548 ± 0.010
    ax1.axvspan(0.548 - 0.010, 0.548 + 0.010, alpha=0.08, color="orange", zorder=0)
    ax1.axvline(0.548, color="orange", linestyle=":", linewidth=1.5, alpha=0.7,
                label="Cepheid α₀")

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([o["short"] for o in observables], fontsize=10)
    ax1.set_xlabel("α₀", fontsize=12)
    ax1.set_title(
        f"TEP Concordance (χ² = {concordance['chi2_concordance']:.1f}, "
        f"p = {concordance['p_concordance']:.3f})",
        fontsize=11
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_xlim(0, 1.5)
    ax1.invert_yaxis()

    # ---- Panel 2: Likelihood profiles ----
    ax2 = axes[1]
    for i, obs in enumerate(observables):
        profile = np.array(obs["profile"])
        color = colors[i % len(colors)]

        # Normalize profile to [0, 1]
        if obs["short"] == "sSFR Inversion":
            # This is stored as positive distance → invert for likelihood
            profile_norm = 1 - profile / max(profile.max(), 1e-10)
        elif obs["short"] == "Anomalous Gal.":
            profile_norm = profile / max(profile.max(), 1e-10)
        else:
            pmin, pmax = profile.min(), profile.max()
            if pmax - pmin > 1e-10:
                profile_norm = (profile - pmin) / (pmax - pmin)
            else:
                profile_norm = np.ones_like(profile) * 0.5

        ax2.plot(ALPHA0_GRID, profile_norm, color=color, linewidth=2,
                 label=obs["short"], alpha=0.85)

    ax2.axvline(concordance["alpha0_weighted_mean"], color="gray",
                linestyle="--", linewidth=1.5, alpha=0.7, label="Weighted mean")
    ax2.axvline(0.548, color="orange", linestyle=":", linewidth=1.5, alpha=0.7,
                label="Cepheid α₀")

    ax2.set_xlabel("α₀", fontsize=12)
    ax2.set_ylabel("Normalized likelihood", fontsize=12)
    ax2.set_title("Independent Likelihood Profiles", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print_status(f"Figure saved to {output_path}", "INFO")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 65, "INFO")
    print_status("Step 173: TEP Concordance — Independent α₀ Recovery", "INFO")
    print_status("=" * 65, "INFO")

    # Load data
    print_status("Loading data...", "INFO")
    df_uncover = load_uncover()
    df_ceers = load_ceers()
    df_cosmosweb = load_cosmosweb()
    print_status(f"  UNCOVER: {len(df_uncover)}, CEERS: {len(df_ceers)}, COSMOS-Web: {len(df_cosmosweb)}", "INFO")

    # Run each observable
    print_status("\nScanning α₀ grid (N=100, bootstrap N=200)...", "INFO")
    observables = []

    o1 = obs1_dust_teff_r2(df_uncover)
    observables.append(o1)

    o2 = obs2_agb_threshold_f1(df_uncover)
    observables.append(o2)

    o3 = obs3_ssfr_inversion(df_uncover)
    observables.append(o3)

    o4 = obs4_cross_survey_r2(df_ceers, df_cosmosweb)
    if o4:
        observables.append(o4)

    o5 = obs5_impossible_resolution(df_uncover, df_ceers, df_cosmosweb)
    if o5:
        observables.append(o5)

    # Concordance statistics
    print_status("\n" + "=" * 65, "INFO")
    print_status("CONCORDANCE STATISTICS", "INFO")
    print_status("=" * 65, "INFO")

    concordance = compute_concordance(observables)

    print_status(f"  Weighted mean α₀ = {concordance['alpha0_weighted_mean']:.3f} ± {concordance['alpha0_weighted_sigma']:.3f}", "INFO")
    print_status(f"  Concordance χ² = {concordance['chi2_concordance']:.2f} (ndof = {concordance['ndof']})", "INFO")
    print_status(f"  p(concordance) = {concordance['p_concordance']:.4f}", "INFO")
    print_status(f"  Max pairwise tension = {concordance['max_pairwise_tension_sigma']:.2f}σ", "INFO")
    print_status(f"  Concordant (p > 0.05): {concordance['concordant']}", "INFO")

    # Comparison to Cepheid
    cepheid_tension = abs(concordance["alpha0_weighted_mean"] - 0.548) / max(
        np.sqrt(concordance["alpha0_weighted_sigma"]**2 + 0.010**2), 0.01
    )
    print_status(f"  Tension with Cepheid α₀ = 0.548 ± 0.010: {cepheid_tension:.2f}σ", "INFO")
    concordance["cepheid_tension_sigma"] = float(cepheid_tension)

    # Verdict
    if concordance["concordant"] and cepheid_tension < 2:
        verdict = "PASS: All observables recover consistent α₀, compatible with the live Cepheid reference"
    elif concordance["concordant"]:
        verdict = "PARTIAL: Observables are mutually concordant but in tension with the live Cepheid reference"
    else:
        verdict = "FAIL: Observables do not agree on α₀"
    concordance["verdict"] = verdict
    print_status(f"\n  VERDICT: {verdict}", "INFO")

    # Generate figure
    print_status("\nGenerating concordance figure...", "INFO")
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    generate_figure(observables, concordance, fig_path)

    # Save results
    results = {
        "step": f"step_{STEP_NUM}",
        "name": "TEP Concordance — Independent α₀ Recovery",
        "observables": [{k: v for k, v in o.items() if k != "profile"} for o in observables],
        "concordance": concordance,
        "alpha0_grid": {"min": float(ALPHA0_GRID[0]), "max": float(ALPHA0_GRID[-1]), "n": len(ALPHA0_GRID)},
    }

    out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {out_file}", "INFO")

    print_status("\n" + "=" * 65, "INFO")
    headline = (
        f"HEADLINE: {len(observables)} independent observables recover "
        f"α₀ = {concordance['alpha0_weighted_mean']:.2f} ± {concordance['alpha0_weighted_sigma']:.2f} "
        f"(χ² = {concordance['chi2_concordance']:.1f}, p = {concordance['p_concordance']:.3f}, "
        f"Cepheid tension = {cepheid_tension:.1f}σ)"
    )
    print_status(headline, "INFO")
    print_status("=" * 65, "INFO")


if __name__ == "__main__":
    main()
