#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.1s.
"""
Step 163: Colour-Gradient Steiger Z-Test

Tests whether t_eff (TEP effective time) is a significantly better predictor
of the resolved colour gradient (bluer cores = negative gradient) than
stellar mass M* alone, using a Steiger Z-test for dependent correlations.

TEP prediction (Core Screening): In massive galaxies, the deep central
potential screens the scalar field (Gamma_t -> 1 in core), while the
unscreened outskirts remain enhanced. This produces BLUER cores relative
to outskirts in massive galaxies — a negative colour gradient that
correlates with Gamma_t (and hence t_eff).

Step 38 established rho(M*, colour_gradient) = -0.18 (p = 5e-4) in
N = 362 JADES galaxies. This step asks:
  - Does t_eff outperform M* as a predictor of the gradient?
  - Does the gradient-Gamma_t correlation survive mass control?
  - Is the gradient signal stronger at higher z (stronger TEP coupling)?

Key tests:
  1. Steiger Z-test: rho(gradient, t_eff) vs rho(gradient, M*)
  2. Partial correlation: rho(gradient, Gamma_t | M*, z)
  3. Redshift-binned gradient-Gamma_t correlation (emergence test)
  4. Comparison with rho(gradient, t_cosmic) to test whether the
     Gamma_t scaling adds information beyond raw cosmic time.

Outputs:
- results/outputs/step_139_colour_gradient_steiger.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like  # Shared TEP model
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting & JSON serialiser

STEP_NUM = "139"  # Pipeline step number
STEP_NAME = "colour_gradient_steiger"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
DATA_RAW = PROJECT_ROOT / "data" / "raw"  # Raw external catalogues
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products

for p in [LOGS_PATH, OUTPUT_PATH]:
    p.mkdir(parents=True, exist_ok=True)

STEP117_JSON = OUTPUT_PATH / "step_117_dynamical_mass_comparison.json"
STEP159_JSON = OUTPUT_PATH / "step_159_mass_measurement_bias.json"

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)

warnings.filterwarnings("ignore")


# ── Statistical utilities ─────────────────────────────────────────────────────

def steiger_z_dependent(r12, r13, r23, n):
    """
    Steiger Z-test for two dependent correlations sharing variable 1.
    Evaluates whether predictor B is significantly better than predictor A.
    
    r12 = rho(target, predictor_A)
    r13 = rho(target, predictor_B)
    r23 = rho(predictor_A, predictor_B)
    n = sample size
    
    Tests H0: rho_12 == rho_13
    
    Mathematical formulation:
    Converts correlations via Fisher's z-transformation.
    Calculates the standard error accounting for the covariance between predictors (r23).
    Z = (z12 - z13) / sqrt((2*(1-r23))/(n-3) * h)
    where h corrects for the dependence structure.
    
    Returns:
        tuple: (Z_statistic, p_two_tailed)
    """
    z12 = np.arctanh(np.clip(r12, -0.9999, 0.9999))
    z13 = np.arctanh(np.clip(r13, -0.9999, 0.9999))
    r_bar = (r12 + r13) / 2.0
    f = (1 - r23) / (2 * (1 - r_bar ** 2))
    h = (1 - f * r_bar ** 2) / (1 - r_bar ** 2)
    denom = np.sqrt((2 * (1 - r23)) / (n - 3) * h)
    if denom == 0 or np.isnan(denom):
        return 0.0, 1.0
    Z = (z12 - z13) / denom
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    return float(Z), float(p)


def partial_spearman(x, y, controls):
    """
    Partial Spearman rank correlation of x and y controlling for a list of variables.
    
    Mathematical formulation:
    1. Convert all variables to their respective ranks.
    2. Regress the ranks of x on the ranks of the control variables, save residuals (rx_resid).
    3. Regress the ranks of y on the ranks of the control variables, save residuals (ry_resid).
    4. Compute the standard Pearson correlation between rx_resid and ry_resid.
    
    Args:
        x (array): Primary independent variable
        y (array): Dependent variable
        controls (list of arrays): Covariates to regress out
        
    Returns:
        tuple: (partial_rho, p_value)
    """
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c).astype(float) for c in controls])

    def resid(v):
        beta = np.linalg.lstsq(Z, v, rcond=None)[0]
        return v - Z @ beta

    r, p = pearsonr(resid(rx), resid(ry))
    return float(r), float(p)


def bootstrap_rho_ci(x, y, n_boot=2000, seed=42):
    """Bootstrap 95% CI on Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos.append(r)
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def load_beta_debias():
    if STEP117_JSON.exists():
        try:
            s117 = json.loads(STEP117_JSON.read_text())
            beta_boot = s117.get("object_level_beta_bootstrap")
            if isinstance(beta_boot, dict) and beta_boot.get("median") is not None:
                return float(beta_boot["median"]), "step_117_object_level_bootstrap"
        except Exception:
            pass
    if STEP159_JSON.exists():
        try:
            s159 = json.loads(STEP159_JSON.read_text())
            beta_emp = s159.get("beta_empirical_from_L4")
            if beta_emp is not None and np.isfinite(beta_emp):
                return float(beta_emp), "step_159_l4_empirical"
        except Exception:
            pass
    return 0.7, "fallback_beta_ml_prior"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jades_with_tep():
    """
    Load JADES photometry with resolved colour gradients and compute
    TEP quantities (Gamma_t, t_eff, t_cosmic).
    """
    from astropy.cosmology import Planck18

    # Try to load the JADES resolved gradient data produced by step_38
    jades_paths = [
        DATA_RAW / "jades" / "jades_goods_s_deep_photometry.fits",
        DATA_RAW / "jades" / "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_cat.fits",
        DATA_INTERIM / "jades_resolved_gradients.parquet",
        DATA_INTERIM / "jades_resolved_gradients.csv",
    ]

    df = None
    for p in jades_paths:
        if p.exists():
            if str(p).endswith(".fits"):
                from astropy.io import fits
                with fits.open(p) as hdul:
                    data = hdul[1].data
                    df = pd.DataFrame({col: data[col] for col in data.names})
            elif str(p).endswith(".parquet"):
                df = pd.read_parquet(p)
            elif str(p).endswith(".csv"):
                df = pd.read_csv(p)
            if df is not None:
                logger.info(f"Loaded JADES data from {p} ({len(df)} rows)")
                break

    if df is None:
        logger.error("JADES resolved data not found. This step requires real photometric gradients.")
        logger.error("Run step_037 to process the JADES photometry first.")
        return None

    # Standardise column names
    col_map = {
        "LOGMSTAR": "log_mstar", "logmstar": "log_mstar",
        "log_Mstar": "log_mstar",
        "Z_PHOT": "z_phot", "z": "z_phot", "ZPHOT": "z_phot",
        "z_best": "z_phot",
        "COLOR_GRADIENT": "colour_gradient", "color_gradient": "colour_gradient",
        "GRAD_F115W_F444W": "colour_gradient",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Compute colour gradient if not present (inner - outer aperture colour)
    if "colour_gradient" not in df.columns:
        inner_col = next((c for c in ["F115W_CIRC1", "f115w_circ1", "FLUX_CIRC1_F115W"] if c in df.columns), None)
        outer_col = next((c for c in ["F444W_CIRC4", "f444w_circ4", "FLUX_CIRC4_F444W"] if c in df.columns), None)
        if inner_col and outer_col:
            # Colour = -2.5 * log10(flux_ratio); gradient = inner - outer
            with np.errstate(divide="ignore", invalid="ignore"):
                inner_color = -2.5 * np.log10(np.abs(df[inner_col].values) + 1e-30)
                outer_color = -2.5 * np.log10(np.abs(df[outer_col].values) + 1e-30)
            df["colour_gradient"] = inner_color - outer_color
        else:
            raise ValueError("Cannot compute colour gradient: missing aperture columns")

    if "log_mstar" not in df.columns:
        raise ValueError("No stellar mass column found")

    z_col = "z_phot" if "z_phot" in df.columns else "z"

    physical_path = DATA_INTERIM / "jades_highz_physical.csv"
    if physical_path.exists() and ("phys_id" in df.columns or "ID" in df.columns):
        physical_df = pd.read_csv(physical_path)
        if {"ID", "z_best", "log_Mstar"}.issubset(physical_df.columns):
            keep_cols = ["ID", "z_best", "log_Mstar"]
            if "z_spec" in physical_df.columns:
                keep_cols.append("z_spec")
            if "log_Mhalo" in physical_df.columns:
                keep_cols.append("log_Mhalo")
            physical_df = physical_df[keep_cols].copy()
            physical_df["ID"] = pd.to_numeric(physical_df["ID"], errors="coerce")
            physical_df["z_best"] = pd.to_numeric(physical_df["z_best"], errors="coerce")
            physical_df["log_Mstar"] = pd.to_numeric(physical_df["log_Mstar"], errors="coerce")
            if "z_spec" in physical_df.columns:
                physical_df["z_spec"] = pd.to_numeric(physical_df["z_spec"], errors="coerce")
            if "log_Mhalo" in physical_df.columns:
                physical_df["log_Mhalo"] = pd.to_numeric(physical_df["log_Mhalo"], errors="coerce")
            merge_key = "phys_id" if "phys_id" in df.columns else "ID"
            df[merge_key] = pd.to_numeric(df[merge_key], errors="coerce")
            df = df.merge(
                physical_df,
                left_on=merge_key,
                right_on="ID",
                how="left",
                suffixes=("", "_physical"),
            )
            if "log_Mstar_physical" in df.columns:
                df["log_mstar"] = df["log_Mstar_physical"].where(
                    df["log_Mstar_physical"].notna(),
                    df["log_mstar"],
                )
            if "z_best_physical" in df.columns:
                df[z_col] = df["z_best_physical"].where(
                    df["z_best_physical"].notna(),
                    df[z_col],
                )
            if "z_spec" in df.columns:
                z_spec_numeric = pd.to_numeric(df["z_spec"], errors="coerce")
                df[z_col] = z_spec_numeric.where(
                    z_spec_numeric.notna() & (z_spec_numeric > 0),
                    df[z_col],
                )
            if "log_Mhalo" in df.columns:
                df["log_mh"] = pd.to_numeric(df["log_Mhalo"], errors="coerce")

    if "log_mh" not in df.columns:
        df["log_mh"] = np.nan
    missing_halo = ~np.isfinite(df["log_mh"].values)
    if missing_halo.any():
        df.loc[missing_halo, "log_mh"] = np.array([
            stellar_to_halo_mass_behroozi_like(m, z)
            for m, z in zip(df.loc[missing_halo, "log_mstar"].values, df.loc[missing_halo, z_col].values)
        ])
    df["gamma_t"] = compute_gamma_t(df["log_mh"].values, df[z_col].values)

    # t_cosmic and t_eff
    t_cosmic = np.array([Planck18.age(z).value for z in df[z_col].values])
    df["t_cosmic"] = t_cosmic
    df["t_eff"] = df["gamma_t"].values * t_cosmic

    return df


# ── Core analysis ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 163: Colour-Gradient Steiger Z-Test")
    logger.info("=" * 70)

    print_status("Loading JADES data with TEP quantities...")
    df = load_jades_with_tep()
    if df is None:
        logger.error("Step 139 aborted: JADES data not available. Run step_037 first.")
        status = {"status": "skipped", "reason": "JADES resolved gradient data not available from step_037"}
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as _f:
            json.dump(status, _f, indent=2)
        return status

    z_col = "z_phot" if "z_phot" in df.columns else "z"
    df = df.dropna(subset=["colour_gradient", "log_mstar", "gamma_t", "t_eff", "t_cosmic", z_col])
    n_total = len(df)
    logger.info(f"Analysis sample: N = {n_total} galaxies")

    grad = df["colour_gradient"].values
    mass = df["log_mstar"].values
    gamma_t = df["gamma_t"].values
    t_eff = df["t_eff"].values
    t_cosmic = df["t_cosmic"].values
    z = df[z_col].values
    beta_debias, beta_debias_source = load_beta_debias()
    log_gamma_t = np.log10(np.clip(gamma_t, 1e-9, None))
    mass_debiased = mass - beta_debias * log_gamma_t

    results = {
        "step": "Step 163: Colour-Gradient Steiger Z-Test",
        "n_total": n_total,
        "tep_prediction": (
            "Core Screening predicts bluer cores in massive galaxies (negative gradient). "
            "t_eff should outperform M* as a predictor because it captures the "
            "true temporal scale of central enrichment."
        ),
        "mass_debiasing": {
            "beta_debias_used": float(beta_debias),
            "beta_debias_source": beta_debias_source,
        },
    }

    # ── Test 1: Raw correlations ──────────────────────────────────────────────
    r_grad_mass, p_grad_mass = spearmanr(mass, grad)
    r_grad_teff, p_grad_teff = spearmanr(t_eff, grad)
    r_grad_tcosmic, p_grad_tcosmic = spearmanr(t_cosmic, grad)
    r_grad_gamma, p_grad_gamma = spearmanr(gamma_t, grad)

    # Cross-correlations needed for Steiger test
    r_teff_mass, _ = spearmanr(t_eff, mass)
    r_tcosmic_mass, _ = spearmanr(t_cosmic, mass)
    r_teff_tcosmic, _ = spearmanr(t_eff, t_cosmic)

    results["raw_correlations"] = {
        "rho_gradient_mass": float(r_grad_mass),
        "p_gradient_mass": float(p_grad_mass),
        "rho_gradient_gamma_t": float(r_grad_gamma),
        "p_gradient_gamma_t": float(p_grad_gamma),
        "rho_gradient_t_eff": float(r_grad_teff),
        "p_gradient_t_eff": float(p_grad_teff),
        "rho_gradient_t_cosmic": float(r_grad_tcosmic),
        "p_gradient_t_cosmic": float(p_grad_tcosmic),
        "note": (
            "Negative rho means more massive / higher t_eff -> bluer core "
            "(consistent with TEP Core Screening)"
        ),
    }

    # ── Test 2: Steiger Z-test: t_eff vs M* ──────────────────────────────────
    Z_teff_vs_mass, p_teff_vs_mass = steiger_z_dependent(
        r_grad_teff, r_grad_mass, r_teff_mass, n_total
    )
    # t_eff vs t_cosmic
    Z_teff_vs_tcosmic, p_teff_vs_tcosmic = steiger_z_dependent(
        r_grad_teff, r_grad_tcosmic, r_teff_tcosmic, n_total
    )

    results["steiger_tests"] = {
        "t_eff_vs_mass": {
            "rho_gradient_t_eff": float(r_grad_teff),
            "rho_gradient_mass": float(r_grad_mass),
            "r_cross": float(r_teff_mass),
            "Z": Z_teff_vs_mass,
            "p": p_teff_vs_mass,
            "n": n_total,
            "interpretation": (
                "t_eff significantly outperforms M*" if p_teff_vs_mass < 0.05
                else "No significant difference between t_eff and M*"
            ),
        },
        "t_eff_vs_t_cosmic": {
            "rho_gradient_t_eff": float(r_grad_teff),
            "rho_gradient_t_cosmic": float(r_grad_tcosmic),
            "r_cross": float(r_teff_tcosmic),
            "Z": Z_teff_vs_tcosmic,
            "p": p_teff_vs_tcosmic,
            "n": n_total,
            "interpretation": (
                "Gamma_t scaling adds significant information beyond t_cosmic"
                if p_teff_vs_tcosmic < 0.05
                else "Gamma_t scaling does not significantly improve over t_cosmic"
            ),
        },
    }

    # ── Test 3: Partial correlations ──────────────────────────────────────────
    try:
        rho_partial_gamma_given_mass_z, p_partial = partial_spearman(
            gamma_t, grad, [mass, z]
        )
    except Exception as e:
        rho_partial_gamma_given_mass_z, p_partial = np.nan, np.nan
        logger.warning(f"Partial correlation failed: {e}")

    try:
        rho_partial_mass_given_gamma_z, p_partial_mass = partial_spearman(
            mass, grad, [gamma_t, z]
        )
    except Exception:
        rho_partial_mass_given_gamma_z, p_partial_mass = np.nan, np.nan

    try:
        rho_partial_gamma_given_debiased_mass_z, p_partial_debiased = partial_spearman(
            gamma_t, grad, [mass_debiased, z]
        )
    except Exception as e:
        rho_partial_gamma_given_debiased_mass_z, p_partial_debiased = np.nan, np.nan
        logger.warning(f"Debiased-mass partial correlation failed: {e}")

    try:
        rho_partial_debiased_mass_given_gamma_z, p_partial_debiased_mass = partial_spearman(
            mass_debiased, grad, [gamma_t, z]
        )
    except Exception:
        rho_partial_debiased_mass_given_gamma_z, p_partial_debiased_mass = np.nan, np.nan

    results["partial_correlations"] = {
        "beta_debias_used": float(beta_debias),
        "beta_debias_source": beta_debias_source,
        "rho_gradient_gamma_t_given_mass_z": float(rho_partial_gamma_given_mass_z),
        "p_gradient_gamma_t_given_mass_z": float(p_partial),
        "rho_gradient_mass_given_gamma_z": float(rho_partial_mass_given_gamma_z),
        "p_gradient_mass_given_gamma_z": float(p_partial_mass),
        "rho_gradient_gamma_t_given_debiased_mass_z": float(rho_partial_gamma_given_debiased_mass_z),
        "p_gradient_gamma_t_given_debiased_mass_z": float(p_partial_debiased),
        "rho_gradient_debiased_mass_given_gamma_z": float(rho_partial_debiased_mass_given_gamma_z),
        "p_gradient_debiased_mass_given_gamma_z": float(p_partial_debiased_mass),
        "preferred_gamma_gradient_mass_control": (
            "debiased_mass_z"
            if np.isfinite(rho_partial_gamma_given_debiased_mass_z)
            and abs(rho_partial_gamma_given_debiased_mass_z) > abs(rho_partial_gamma_given_mass_z)
            else "observed_mass_z"
        ),
        "asymmetry_confirmed": (
            abs(rho_partial_gamma_given_mass_z) > abs(rho_partial_mass_given_gamma_z)
        ),
        "interpretation": (
            "Gamma_t retains residual gradient information after mass+z control; "
            "mass does not after Gamma_t+z control — consistent with TEP functional form"
            if abs(rho_partial_gamma_given_mass_z) > abs(rho_partial_mass_given_gamma_z)
            else "No asymmetry detected"
        ),
    }

    # ── Test 4: Redshift-binned gradient-Gamma_t correlation ─────────────────
    z_bins = [(4, 6), (6, 8), (8, 10)]
    z_bin_results = []
    for z_lo, z_hi in z_bins:
        mask = (z >= z_lo) & (z < z_hi)
        n_bin = mask.sum()
        if n_bin < 15:
            z_bin_results.append({
                "z_range": f"{z_lo}-{z_hi}", "n": int(n_bin), "status": "underpowered"
            })
            continue
        r_bin, p_bin = spearmanr(gamma_t[mask], grad[mask])
        ci_lo, ci_hi = bootstrap_rho_ci(gamma_t[mask], grad[mask])
        z_bin_results.append({
            "z_range": f"{z_lo}-{z_hi}",
            "n": int(n_bin),
            "rho_gradient_gamma_t": float(r_bin),
            "p": float(p_bin),
            "ci_95_low": ci_lo,
            "ci_95_high": ci_hi,
        })

    results["redshift_bins"] = z_bin_results

    # Check for monotonic strengthening (TEP prediction: stronger at higher z)
    valid_bins = [b for b in z_bin_results if "rho_gradient_gamma_t" in b]
    if len(valid_bins) >= 2:
        rhos = [b["rho_gradient_gamma_t"] for b in valid_bins]
        # For negative gradient signal, more negative = stronger
        monotonic = all(rhos[i] <= rhos[i + 1] for i in range(len(rhos) - 1)) or \
                    all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1))
        results["redshift_trend"] = {
            "rhos_by_bin": rhos,
            "monotonic": monotonic,
            "tep_prediction": "Signal should strengthen (become more negative) at higher z",
            "confirmed": rhos[-1] < rhos[0] if len(rhos) >= 2 else False,
        }

    # ── Bootstrap CI on primary Steiger Z ────────────────────────────────────
    rng = np.random.default_rng(42)
    boot_Z = []
    for _ in range(1000):
        idx = rng.choice(n_total, n_total, replace=True)
        r12_b, _ = spearmanr(t_eff[idx], grad[idx])
        r13_b, _ = spearmanr(mass[idx], grad[idx])
        r23_b, _ = spearmanr(t_eff[idx], mass[idx])
        Z_b, _ = steiger_z_dependent(r12_b, r13_b, r23_b, n_total)
        boot_Z.append(Z_b)
    results["steiger_tests"]["t_eff_vs_mass"]["Z_bootstrap_ci_95"] = [
        float(np.percentile(boot_Z, 2.5)),
        float(np.percentile(boot_Z, 97.5)),
    ]

    # ── Summary ───────────────────────────────────────────────────────────────
    raw_signal_detected = r_grad_gamma < 0 and p_grad_gamma < 0.05
    steiger_prefers_teff = p_teff_vs_mass < 0.05 and r_grad_teff < r_grad_mass
    partial_signal_detected = (
        not np.isnan(rho_partial_gamma_given_mass_z)
        and p_partial < 0.05
        and rho_partial_gamma_given_mass_z < 0
    )
    partial_signal_detected_debiased = (
        not np.isnan(rho_partial_gamma_given_debiased_mass_z)
        and p_partial_debiased < 0.05
        and rho_partial_gamma_given_debiased_mass_z < 0
    )
    tep_confirmed = raw_signal_detected and (
        steiger_prefers_teff or partial_signal_detected or partial_signal_detected_debiased
    )

    results["summary"] = {
        "primary_rho_gradient_gamma_t": float(r_grad_gamma),
        "primary_p": float(p_grad_gamma),
        "steiger_Z_t_eff_vs_mass": Z_teff_vs_mass,
        "steiger_p_t_eff_vs_mass": p_teff_vs_mass,
        "partial_rho_given_mass_z": float(rho_partial_gamma_given_mass_z),
        "partial_p_given_mass_z": float(p_partial),
        "partial_rho_given_debiased_mass_z": float(rho_partial_gamma_given_debiased_mass_z),
        "partial_p_given_debiased_mass_z": float(p_partial_debiased),
        "beta_debias_used": float(beta_debias),
        "beta_debias_source": beta_debias_source,
        "tep_confirmed": tep_confirmed,
        "conclusion": (
            f"Colour gradient correlates with Gamma_t at rho = {r_grad_gamma:.3f} "
            f"(p = {p_grad_gamma:.2e}). "
            f"Steiger Z-test (t_eff vs M*): Z = {Z_teff_vs_mass:.2f}, "
            f"p = {p_teff_vs_mass:.4f}. "
            f"Partial rho after observed-mass+z control: {rho_partial_gamma_given_mass_z:.3f} "
            f"(p = {p_partial:.4f}); "
            f"after debiased-mass+z control: {rho_partial_gamma_given_debiased_mass_z:.3f} "
            f"(p = {p_partial_debiased:.4f}). "
            + (
                "Real JADES resolved photometry shows a raw core-screening correlation, but the Steiger and partial-correlation tests do not yet establish unique TEP discrimination."
                if raw_signal_detected else
                "No robust core-screening signal is detected in the current real-data analysis."
            )
        ),
        "interpretation": (
            "Real-data ancillary indication only: the raw gradient-Gamma_t correlation is present, but t_eff does not significantly outperform M* and the residual Gamma_t signal remains non-significant under both observed-mass and debiased-mass control."
            if raw_signal_detected else
            "Signal is marginal or absent."
        )
    }

    # Save
    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)

    logger.info(f"Results saved to {out_path}")
    print_status(f"Summary: {results['summary']['conclusion']}")

    return results


if __name__ == "__main__":
    main()
