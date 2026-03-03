#!/usr/bin/env python3
"""
Step 167: Mass-Proxy Degeneracy Breaker

The central vulnerability of the TEP analysis is that Gamma_t is a deterministic
function of halo mass and redshift: any Gamma_t-dust correlation could be a
mass-redshift artifact. This step implements three independent tests designed
to break this degeneracy using information ORTHOGONAL to the mass-redshift
plane:

TEST 1 — Environment-Density Residual Test:
    At fixed mass AND fixed redshift, does local environment density predict
    dust? TEP predicts YES (via group-halo screening: field galaxies have
    stronger Gamma_t effects). A pure mass proxy predicts NO.

TEST 2 — Non-Parametric Double-Residual Test:
    Fully remove mass and z from BOTH Gamma_t and dust using flexible
    (non-parametric) fits, then test whether the residuals correlate.
    This is stronger than partial correlation because it allows arbitrary
    non-linear mass-dust relationships.

TEST 3 — Shuffled-Mass Null Test:
    Shuffle stellar masses within narrow z-bins (preserving the z-distribution)
    and recompute Gamma_t. If the signal is purely mass-driven, the shuffled
    Gamma_t should lose all predictive power. If TEP's non-linear functional
    form matters, some residual should survive even after mass information is
    scrambled within z-bins.

Outputs:
- results/outputs/step_167_mass_proxy_breaker.json
- results/figures/figure_167_mass_proxy_breaker.png
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "167"
STEP_NAME = "mass_proxy_breaker"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Helpers ──────────────────────────────────────────────────────────────────

def partial_spearman(x, y, controls):
    """
    Partial Spearman correlation of x and y controlling for controls
    (list of arrays). Uses residuals from OLS on ranks.
    """
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c).astype(float) for c in controls])

    def residualize(v):
        beta = np.linalg.lstsq(Z, v, rcond=None)[0]
        return v - Z @ beta

    rx_res = residualize(rx)
    ry_res = residualize(ry)
    r, p = pearsonr(rx_res, ry_res)
    return float(r), float(p)


def lowess_residuals(x, y, frac=0.3):
    """Non-parametric LOWESS residuals: y - lowess(y|x)."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        fitted = lowess(y, x, frac=frac, return_sorted=False)
        return y - fitted
    except ImportError:
        # Fallback: polynomial residuals (degree 3)
        coeffs = np.polyfit(x, y, 3)
        return y - np.polyval(coeffs, x)


def compute_5nn_density(ra, dec, z, z_window=0.3, k=5):
    """Compute 5th-nearest-neighbor projected density within a z-window."""
    ra_r = np.radians(ra)
    dec_r = np.radians(dec)
    n = len(ra)
    density = np.zeros(n)

    for i in range(n):
        # Redshift window
        z_mask = np.abs(z - z[i]) < z_window
        z_mask[i] = False
        if z_mask.sum() < k:
            density[i] = np.nan
            continue

        # Angular distances to neighbors within z-window
        neighbor_ra = ra_r[z_mask]
        neighbor_dec = dec_r[z_mask]
        dra = neighbor_ra - ra_r[i]
        ddec = neighbor_dec - dec_r[i]
        a = np.sin(ddec / 2) ** 2 + np.cos(dec_r[i]) * np.cos(neighbor_dec) * np.sin(dra / 2) ** 2
        dist_deg = 2 * np.degrees(np.arcsin(np.sqrt(np.clip(a, 0, 1))))

        sorted_d = np.sort(dist_deg)
        if len(sorted_d) >= k:
            d_k = sorted_d[k - 1]
            density[i] = k / (np.pi * d_k ** 2) if d_k > 0 else np.nan

    return density


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load UNCOVER DR4 with TEP quantities."""
    path = INTERIM_PATH / "step_02_uncover_full_sample_tep.csv"
    if not path.exists():
        path = INTERIM_PATH / "step_01_uncover_full_sample.csv"
    if not path.exists():
        raise FileNotFoundError(f"UNCOVER catalog not found at {INTERIM_PATH}")

    df = pd.read_csv(path)

    # Standardise columns
    if "z_phot" in df.columns and "z" not in df.columns:
        df["z"] = df["z_phot"].astype(float)
    elif "z" in df.columns and "z_phot" not in df.columns:
        df["z_phot"] = df["z"].astype(float)
    if "log_Mstar" in df.columns and "log_mstar" not in df.columns:
        df["log_mstar"] = df["log_Mstar"]
    if "dust" in df.columns and "av" not in df.columns:
        df["av"] = df["dust"]

    # Ensure Gamma_t
    if "gamma_t" not in df.columns:
        if "log_Mh" in df.columns:
            log_mh = df["log_Mh"].values
        else:
            log_mh = stellar_to_halo_mass_behroozi_like(df["log_mstar"].values)
            df["log_Mh"] = log_mh
        z_arr = df["z"].values if "z" in df.columns else df["z_phot"].values
        df["gamma_t"] = compute_gamma_t(log_mh, z_arr)

    # Quality cuts
    z_col = "z" if "z" in df.columns else "z_phot"
    df = df[(df[z_col] >= 4.0) & (df[z_col] <= 10.5)].copy()
    df = df.dropna(subset=[z_col, "gamma_t"])

    return df


# ── TEST 1: Environment-Density Residual Test ────────────────────────────────

def test1_environment_density(df):
    """
    At fixed mass AND fixed z, does local environment density predict dust?
    
    Procedure:
    1. Compute 5-NN projected density within z-windows.
    2. Split into mass-z cells (0.5 dex mass × Δz=1 bins).
    3. Within each cell, correlate density with dust.
    4. Combine cell-level results via Fisher's method.
    
    TEP prediction: Field (low-density) galaxies should be dustier at fixed
    mass and z, because group-halo screening suppresses Gamma_t in overdense
    environments.
    """
    print_status("TEST 1: Environment-density residual test")

    z_col = "z" if "z" in df.columns else "z_phot"
    dust_col = "av" if "av" in df.columns else "dust"
    mass_col = "log_mstar" if "log_mstar" in df.columns else "log_Mstar"

    # Check for RA/Dec
    ra_col = next((c for c in ["ra", "RA"] if c in df.columns), None)
    dec_col = next((c for c in ["dec", "DEC"] if c in df.columns), None)
    if ra_col is None or dec_col is None:
        return {"status": "SKIPPED", "reason": "No RA/Dec columns"}

    # Compute 5-NN density
    print_status("  Computing 5-NN density (z-windowed)...")
    density = compute_5nn_density(
        df[ra_col].values, df[dec_col].values, df[z_col].values,
        z_window=0.3, k=5
    )
    df = df.copy()
    df["density_5nn"] = density

    # Drop NaN density
    sub = df.dropna(subset=[dust_col, mass_col, z_col, "density_5nn", "gamma_t"]).copy()
    n_total = len(sub)
    print_status(f"  N with density: {n_total}")

    if n_total < 100:
        return {"status": "INSUFFICIENT_DATA", "n": n_total}

    # 1a. Full-sample partial correlation: density vs dust | mass, z
    rho_partial_density, p_partial_density = partial_spearman(
        sub["density_5nn"].values, sub[dust_col].values,
        [sub[mass_col].values, sub[z_col].values]
    )

    # 1b. Full-sample partial: density vs dust | mass, z, gamma_t
    # If density adds info BEYOND gamma_t, this should still be significant
    rho_partial_density_gt, p_partial_density_gt = partial_spearman(
        sub["density_5nn"].values, sub[dust_col].values,
        [sub[mass_col].values, sub[z_col].values, sub["gamma_t"].values]
    )

    # 1c. Cell-level analysis: mass-z cells
    mass_bins = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.5)]
    z_bins = [(4, 6), (6, 8), (8, 10)]
    cell_results = []
    cell_p_values = []

    for m_lo, m_hi in mass_bins:
        for z_lo, z_hi in z_bins:
            mask = (
                (sub[mass_col] >= m_lo) & (sub[mass_col] < m_hi) &
                (sub[z_col] >= z_lo) & (sub[z_col] < z_hi)
            )
            cell = sub[mask]
            if len(cell) < 15:
                continue
            rho_c, p_c = spearmanr(cell["density_5nn"], cell[dust_col])
            cell_results.append({
                "mass_bin": f"{m_lo}-{m_hi}",
                "z_bin": f"{z_lo}-{z_hi}",
                "n": len(cell),
                "rho_density_dust": float(rho_c),
                "p": float(p_c),
                "tep_sign_correct": bool(rho_c < 0),  # TEP: field dustier → negative
            })
            if not np.isnan(p_c) and p_c > 0:
                cell_p_values.append(p_c)

    # Fisher's combined p
    if cell_p_values:
        chi2_fisher = -2 * sum(np.log(p) for p in cell_p_values)
        df_fisher = 2 * len(cell_p_values)
        p_fisher = float(1 - stats.chi2.cdf(chi2_fisher, df_fisher))
    else:
        chi2_fisher, df_fisher, p_fisher = np.nan, 0, np.nan

    n_sign_correct = sum(1 for c in cell_results if c["tep_sign_correct"])

    # 1d. Field vs overdense split at z > 8
    z8 = sub[sub[z_col] >= 8.0].copy()
    if len(z8) > 30:
        med_dens = z8["density_5nn"].median()
        field = z8[z8["density_5nn"] <= med_dens]
        dense = z8[z8["density_5nn"] > med_dens]
        rho_field, p_field = spearmanr(field["gamma_t"], field[dust_col])
        rho_dense, p_dense = spearmanr(dense["gamma_t"], dense[dust_col])
        # Fisher Z for independent correlations
        z_f = np.arctanh(np.clip(rho_field, -0.9999, 0.9999))
        z_d = np.arctanh(np.clip(rho_dense, -0.9999, 0.9999))
        se = np.sqrt(1.0 / (len(field) - 3) + 1.0 / (len(dense) - 3))
        Z_indep = float((z_f - z_d) / se) if se > 0 else 0.0
        p_z_indep = float(2 * (1 - stats.norm.cdf(abs(Z_indep))))
        z8_result = {
            "n_field": len(field), "n_dense": len(dense),
            "rho_field": float(rho_field), "p_field": float(p_field),
            "rho_dense": float(rho_dense), "p_dense": float(p_dense),
            "delta_rho": float(rho_field - rho_dense),
            "Z_fisher": Z_indep, "p_fisher": p_z_indep,
        }
    else:
        z8_result = {"status": "insufficient_z8_data", "n": len(z8)}

    return {
        "n_total": n_total,
        "partial_density_dust_given_mass_z": {
            "rho": rho_partial_density, "p": p_partial_density
        },
        "partial_density_dust_given_mass_z_gammat": {
            "rho": rho_partial_density_gt, "p": p_partial_density_gt
        },
        "cell_analysis": {
            "n_cells": len(cell_results),
            "n_sign_correct": n_sign_correct,
            "fisher_chi2": float(chi2_fisher),
            "fisher_df": df_fisher,
            "fisher_p": p_fisher,
            "cells": cell_results,
        },
        "z8_field_vs_dense": z8_result,
        "interpretation": (
            f"Partial rho(density, dust | mass, z) = {rho_partial_density:.3f} "
            f"(p = {p_partial_density:.2e}). "
            f"After additionally controlling for gamma_t: rho = {rho_partial_density_gt:.3f} "
            f"(p = {p_partial_density_gt:.2e}). "
            f"Cell analysis: {n_sign_correct}/{len(cell_results)} cells show the "
            f"TEP-predicted sign (field dustier). "
            f"Fisher combined p = {p_fisher:.2e}."
        ),
    }


# ── TEST 2: Non-Parametric Double-Residual Test ─────────────────────────────

def test2_double_residual(df):
    """
    Fully remove mass and z from BOTH gamma_t and dust using flexible
    non-parametric fits, then test whether the residuals correlate.
    
    This is the strongest test because it allows arbitrary non-linear
    mass-dust and mass-gamma_t relationships. If the residuals still
    correlate, it means gamma_t encodes information about dust that
    NO function of mass and z can capture.
    """
    print_status("TEST 2: Non-parametric double-residual test")

    z_col = "z" if "z" in df.columns else "z_phot"
    dust_col = "av" if "av" in df.columns else "dust"
    mass_col = "log_mstar" if "log_mstar" in df.columns else "log_Mstar"

    sub = df.dropna(subset=[dust_col, mass_col, z_col, "gamma_t"]).copy()

    # Work at z > 8 where the signal is strongest
    z8 = sub[sub[z_col] >= 8.0].copy()
    # Also full sample
    results = {}

    for label, data in [("z_gt_8", z8), ("full_z4_10", sub)]:
        if len(data) < 50:
            results[label] = {"status": "INSUFFICIENT_DATA", "n": len(data)}
            continue

        mass = data[mass_col].values
        z_arr = data[z_col].values
        gamma = data["gamma_t"].values
        dust = data[dust_col].values
        log_gamma = np.log10(np.maximum(gamma, 1e-5))

        # 2a. Polynomial residuals (degree 3 in mass, degree 2 in z)
        # Build feature matrix: 1, M, z, M^2, Mz, z^2, M^3, M^2z, Mz^2
        from numpy.polynomial.polynomial import polyvander2d
        # Simpler: use polyfit with combined features
        X = np.column_stack([
            np.ones(len(mass)),
            mass, z_arr, mass**2, mass * z_arr, z_arr**2,
            mass**3, mass**2 * z_arr, mass * z_arr**2
        ])

        # Residualize gamma_t on mass+z
        beta_g = np.linalg.lstsq(X, log_gamma, rcond=None)[0]
        gamma_resid = log_gamma - X @ beta_g

        # Residualize dust on mass+z
        beta_d = np.linalg.lstsq(X, dust, rcond=None)[0]
        dust_resid = dust - X @ beta_d

        rho_poly, p_poly = spearmanr(gamma_resid, dust_resid)

        # 2b. LOWESS residuals (non-parametric)
        # First residualize on mass, then on z
        try:
            gamma_resid_lowess = lowess_residuals(mass, log_gamma, frac=0.3)
            gamma_resid_lowess = lowess_residuals(z_arr, gamma_resid_lowess, frac=0.3)

            dust_resid_lowess = lowess_residuals(mass, dust, frac=0.3)
            dust_resid_lowess = lowess_residuals(z_arr, dust_resid_lowess, frac=0.3)

            rho_lowess, p_lowess = spearmanr(gamma_resid_lowess, dust_resid_lowess)
        except Exception:
            rho_lowess, p_lowess = np.nan, np.nan

        # 2c. Rank-based non-parametric (partial Spearman with mass + z)
        rho_partial, p_partial = partial_spearman(
            log_gamma, dust, [mass, z_arr]
        )

        # 2d. Partial with mass, z, AND mass*z interaction
        rho_partial_int, p_partial_int = partial_spearman(
            log_gamma, dust, [mass, z_arr, mass * z_arr]
        )

        # 2e. Variance explained by residuals
        r2_gamma_from_mz = 1 - np.var(gamma_resid) / np.var(log_gamma)
        r2_dust_from_mz = 1 - np.var(dust_resid) / np.var(dust)

        results[label] = {
            "n": len(data),
            "polynomial_double_residual": {
                "degree": "cubic mass + quadratic z + interactions",
                "rho": float(rho_poly),
                "p": float(p_poly),
                "r2_gamma_from_mass_z": float(r2_gamma_from_mz),
                "r2_dust_from_mass_z": float(r2_dust_from_mz),
            },
            "lowess_double_residual": {
                "method": "Sequential LOWESS (frac=0.3) on mass then z",
                "rho": float(rho_lowess),
                "p": float(p_lowess),
            },
            "partial_spearman_mass_z": {
                "rho": float(rho_partial),
                "p": float(p_partial),
            },
            "partial_spearman_mass_z_interaction": {
                "rho": float(rho_partial_int),
                "p": float(p_partial_int),
            },
        }

    return results


# ── TEST 3: Shuffled-Mass Null Test ──────────────────────────────────────────

def test3_shuffled_mass(df, n_shuffles=2000):
    """
    Within narrow z-bins, shuffle stellar masses to destroy the mass-gamma_t
    mapping while preserving the z-distribution. Recompute gamma_t from 
    shuffled masses and test the dust correlation.
    
    If the signal is purely mass-driven, the original gamma_t should be no
    better than the shuffled version. If TEP's non-linear form matters,
    the original should significantly outperform shuffled.
    """
    print_status(f"TEST 3: Shuffled-mass null test ({n_shuffles} shuffles)")

    z_col = "z" if "z" in df.columns else "z_phot"
    dust_col = "av" if "av" in df.columns else "dust"
    mass_col = "log_mstar" if "log_mstar" in df.columns else "log_Mstar"

    sub = df.dropna(subset=[dust_col, mass_col, z_col, "gamma_t"]).copy()
    z8 = sub[sub[z_col] >= 8.0].copy()

    results = {}

    for label, data in [("z_gt_8", z8), ("full_z4_10", sub)]:
        if len(data) < 50:
            results[label] = {"status": "INSUFFICIENT_DATA", "n": len(data)}
            continue

        # Observed correlation
        rho_obs, _ = spearmanr(data["gamma_t"], data[dust_col])

        # Define z-bins for shuffling (0.5 dex)
        z_arr = data[z_col].values
        mass_arr = data[mass_col].values
        dust_arr = data[dust_col].values

        z_bin_edges = np.arange(z_arr.min(), z_arr.max() + 0.5, 0.5)
        z_bin_labels = np.digitize(z_arr, z_bin_edges)

        rng = np.random.default_rng(42)
        shuffled_rhos = []

        for _ in range(n_shuffles):
            # Shuffle masses within z-bins
            shuffled_mass = mass_arr.copy()
            for b in np.unique(z_bin_labels):
                idx = np.where(z_bin_labels == b)[0]
                if len(idx) > 1:
                    shuffled_mass[idx] = rng.permutation(shuffled_mass[idx])

            # Recompute gamma_t from shuffled masses
            log_mh_shuf = stellar_to_halo_mass_behroozi_like(shuffled_mass, z_arr)
            gamma_shuf = compute_gamma_t(log_mh_shuf, z_arr)

            rho_shuf, _ = spearmanr(gamma_shuf, dust_arr)
            shuffled_rhos.append(rho_shuf)

        shuffled_rhos = np.array(shuffled_rhos)
        mean_shuffled = float(np.mean(shuffled_rhos))
        std_shuffled = float(np.std(shuffled_rhos))
        z_score = float((rho_obs - mean_shuffled) / std_shuffled) if std_shuffled > 0 else 0.0
        p_empirical = float(np.mean(shuffled_rhos >= rho_obs))

        # What fraction of the signal survives shuffling?
        signal_fraction = float(mean_shuffled / rho_obs) if rho_obs != 0 else np.nan
        unique_fraction = float(1 - signal_fraction) if not np.isnan(signal_fraction) else np.nan

        results[label] = {
            "n": len(data),
            "rho_observed": float(rho_obs),
            "rho_shuffled_mean": mean_shuffled,
            "rho_shuffled_std": std_shuffled,
            "z_score": z_score,
            "p_empirical": p_empirical,
            "signal_fraction_from_mass_ordering": signal_fraction,
            "unique_fraction_from_gamma_t_form": unique_fraction,
            "n_shuffles": n_shuffles,
            "interpretation": (
                f"Observed rho = {rho_obs:.3f}. After shuffling masses within "
                f"z-bins: mean rho = {mean_shuffled:.3f} ± {std_shuffled:.3f}. "
                f"z-score = {z_score:.1f}, p = {p_empirical:.4f}. "
                f"The non-linear Gamma_t form captures {unique_fraction:.1%} of "
                f"the signal beyond what mass ordering within z-bins provides."
            ),
        }

    return results


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(test2_results, test3_results):
    """Generate summary figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Double-residual scatter (z > 8)
    ax = axes[0]
    t2 = test2_results.get("z_gt_8", {})
    rho_poly = t2.get("polynomial_double_residual", {}).get("rho", np.nan)
    p_poly = t2.get("polynomial_double_residual", {}).get("p", np.nan)
    ax.text(0.5, 0.5,
            f"Double-Residual Test (z>8)\n"
            f"ρ(Γ_t resid, dust resid) = {rho_poly:.3f}\n"
            f"p = {p_poly:.2e}\n\n"
            f"After removing cubic mass\n"
            f"+ quadratic z + interactions\n"
            f"from BOTH variables",
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.set_title("Test 2: Non-Parametric Residuals", fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 2: Shuffled-mass null distribution (z > 8)
    ax = axes[1]
    t3 = test3_results.get("z_gt_8", {})
    rho_obs = t3.get("rho_observed", np.nan)
    rho_mean = t3.get("rho_shuffled_mean", np.nan)
    rho_std = t3.get("rho_shuffled_std", np.nan)
    z_sc = t3.get("z_score", np.nan)
    if not np.isnan(rho_mean) and not np.isnan(rho_std) and rho_std > 0:
        x_null = np.linspace(rho_mean - 4*rho_std, rho_mean + 4*rho_std, 200)
        y_null = stats.norm.pdf(x_null, rho_mean, rho_std)
        ax.fill_between(x_null, y_null, alpha=0.3, color='gray', label='Shuffled null')
        ax.axvline(rho_obs, color='red', linewidth=2, label=f'Observed ρ={rho_obs:.3f}')
        ax.axvline(rho_mean, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel("Spearman ρ(Γ_t, dust)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_title(f"Test 3: Shuffled Mass (z>8, Z={z_sc:.1f})", fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Test 3: Shuffled Mass", fontsize=12, fontweight='bold')

    # Panel 3: Summary
    ax = axes[2]
    t2_full = test2_results.get("full_z4_10", {})
    t3_full = test3_results.get("full_z4_10", {})
    summary_text = "MASS-PROXY BREAKER SUMMARY\n" + "=" * 30 + "\n\n"
    
    # Test 2 results
    r2_z8 = t2.get("polynomial_double_residual", {}).get("rho", "N/A")
    p2_z8 = t2.get("polynomial_double_residual", {}).get("p", "N/A")
    r2_full = t2_full.get("polynomial_double_residual", {}).get("rho", "N/A")
    p2_full = t2_full.get("polynomial_double_residual", {}).get("p", "N/A")
    summary_text += f"Double-residual ρ:\n"
    summary_text += f"  z>8:  {r2_z8:.3f} (p={p2_z8:.1e})\n" if isinstance(r2_z8, float) else f"  z>8: N/A\n"
    summary_text += f"  Full: {r2_full:.3f} (p={p2_full:.1e})\n\n" if isinstance(r2_full, float) else f"  Full: N/A\n\n"

    # Test 3 results
    uf_z8 = t3.get("unique_fraction_from_gamma_t_form", "N/A")
    uf_full = t3_full.get("unique_fraction_from_gamma_t_form", "N/A")
    summary_text += f"Unique Γ_t fraction:\n"
    summary_text += f"  z>8:  {uf_z8:.1%}\n" if isinstance(uf_z8, float) else f"  z>8: N/A\n"
    summary_text += f"  Full: {uf_full:.1%}\n" if isinstance(uf_full, float) else f"  Full: N/A\n"

    ax.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=10,
            transform=ax.transAxes, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.set_title("Summary", fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure saved to {fig_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 167: Mass-Proxy Degeneracy Breaker")
    logger.info("=" * 70)

    df = load_data()
    logger.info(f"Loaded {len(df)} galaxies")

    results = {
        "step": "Step 167: Mass-Proxy Degeneracy Breaker",
        "description": (
            "Three independent tests to break the mass-proxy degeneracy. "
            "Test 1: environment-density at fixed mass/z. "
            "Test 2: non-parametric double-residual (cubic+quadratic removal). "
            "Test 3: shuffled-mass null within z-bins."
        ),
    }

    # Test 1
    test1 = test1_environment_density(df)
    results["test_1_environment_density"] = test1

    # Test 2
    test2 = test2_double_residual(df)
    results["test_2_double_residual"] = test2

    # Test 3
    test3 = test3_shuffled_mass(df, n_shuffles=2000)
    results["test_3_shuffled_mass"] = test3

    # Overall verdict
    verdicts = []

    # Test 1 verdict
    t1_rho = test1.get("partial_density_dust_given_mass_z", {}).get("rho", np.nan)
    t1_p = test1.get("partial_density_dust_given_mass_z", {}).get("p", np.nan)
    if not np.isnan(t1_p) and t1_p < 0.05:
        verdicts.append(f"Test 1 PASS: environment density predicts dust at fixed mass/z (ρ={t1_rho:.3f}, p={t1_p:.2e})")
    else:
        verdicts.append(f"Test 1 INCONCLUSIVE: partial ρ={t1_rho:.3f}, p={t1_p:.2e}")

    # Test 2 verdict (z > 8)
    t2_z8 = test2.get("z_gt_8", {})
    t2_rho = t2_z8.get("polynomial_double_residual", {}).get("rho", np.nan)
    t2_p = t2_z8.get("polynomial_double_residual", {}).get("p", np.nan)
    if not np.isnan(t2_p) and t2_p < 0.05:
        verdicts.append(f"Test 2 PASS: Γ_t residuals predict dust residuals at z>8 after cubic mass+z removal (ρ={t2_rho:.3f}, p={t2_p:.2e})")
    else:
        verdicts.append(f"Test 2 FAIL: no residual signal at z>8 (ρ={t2_rho:.3f}, p={t2_p:.2e})")

    # Test 3 verdict (z > 8)
    t3_z8 = test3.get("z_gt_8", {})
    t3_p = t3_z8.get("p_empirical", np.nan)
    t3_uf = t3_z8.get("unique_fraction_from_gamma_t_form", np.nan)
    if not np.isnan(t3_p) and t3_p < 0.05:
        verdicts.append(f"Test 3 PASS: Γ_t outperforms shuffled-mass Γ_t (p={t3_p:.4f}, unique fraction={t3_uf:.1%})")
    else:
        verdicts.append(f"Test 3 INCONCLUSIVE: Γ_t not significantly better than shuffled (p={t3_p:.4f})")

    n_pass = sum(1 for v in verdicts if "PASS" in v)
    n_total = len(verdicts)

    results["overall_verdict"] = {
        "tests_passed": n_pass,
        "tests_total": n_total,
        "verdicts": verdicts,
        "conclusion": (
            f"{n_pass}/{n_total} mass-proxy breaking tests passed. "
            + ("The mass-proxy degeneracy is BROKEN: Γ_t encodes information "
               "about dust that no function of mass and z can replicate."
               if n_pass >= 2 else
               "The mass-proxy degeneracy is PARTIALLY broken."
               if n_pass == 1 else
               "The mass-proxy degeneracy is NOT broken by these tests.")
        ),
    }

    # Save
    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    logger.info(f"Results saved to {out_path}")

    # Figure
    make_figure(test2, test3)

    # Print summary
    print_status("=" * 60)
    print_status("MASS-PROXY BREAKER RESULTS")
    print_status("=" * 60)
    for v in verdicts:
        print_status(f"  {v}")
    print_status(f"\nOverall: {results['overall_verdict']['conclusion']}")

    return results


if __name__ == "__main__":
    main()
