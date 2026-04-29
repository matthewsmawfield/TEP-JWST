#!/usr/bin/env python3
"""
Step 171: Sigma-Based Kinematic Expansion — Mass-Circularity-Breaking TEP Test

This step merges ALL available kinematic datasets (SUSPENSE, Esdaile+Tanaka,
de Graaff, Saldana-Lopez, Danhaive gold) into a single combined sample and
runs sigma-derived (SED-independent) tests that break the photometric
mass-proxy circularity.

Key design choice: Gamma_t_sigma is computed from sigma ALONE (via the
sigma-Mhalo relation), NOT from Mdyn = 5 sigma^2 R_e / G. This avoids
shared-variable artifacts when correlating with M*/Mdyn excess.

Key tests:
  T1  M*-sigma zero-point evolution: does the M*-sigma residual
      increase with z as TEP predicts?
  T2  TEP-corrected fundamental plane scatter: does the TEP-corrected M*
      follow a tighter M*-sigma-Re relation?
  T3  Sigma-only Gamma_t as M* predictor: does log(Gamma_t_sigma_only)
      add predictive power for M*_obs beyond sigma alone?
  T4  Cross-survey consistency.
  T5  High-z (z>=4) focused subset.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import print_status
from scripts.utils.rank_stats import bootstrap_partial_rank_ci, partial_rank_correlation
from scripts.utils.tep_model import KAPPA_GAL, compute_gamma_t

STEP_NUM = 171
STEP_NAME = "sigma_kinematic_expansion"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"

# Data sources
SUSPENSE_JSON = PROJECT_ROOT / "data" / "interim" / "suspense_kinematics_ages.json"
LITERATURE_JSON = PROJECT_ROOT / "data" / "interim" / "literature_kinematic_sample.json"
SAME_REGIME_JSON = PROJECT_ROOT / "data" / "interim" / "same_regime_literature_kinematic_sample.json"
DANHAIVE_FULL_JSON = INTERIM_PATH / "danhaive_2025_gold_extracted_v2.json"

N_ML = 0.5  # M/L power-law index for high-z
N_BOOT = 2000
RNG_SEED = 171

# ---------------------------------------------------------------------------
# Sigma-only halo-mass mapping (no dependence on R_e or Mdyn)
# ---------------------------------------------------------------------------
# We use a simplified sigma-Mhalo relation calibrated from local weak-lensing
# and abundance-matching results (e.g. Zahid+ 2016, Bogdan+ 2015):
#   log(M_h) ~ 5 * log10(sigma / 200 km/s) + 12.5
# This is monotonic in sigma and completely independent of both SED-based M*
# and size-based Mdyn.
SIGMA_REF = 200.0  # km/s
LOG_MH_AT_SIGMA_REF = 12.5
SIGMA_SLOPE = 5.0


def _sigma_to_log_mhalo(sigma_kms):
    """Map velocity dispersion to halo mass (sigma-only, no R_e)."""
    sigma = np.asarray(sigma_kms, dtype=float)
    return SIGMA_SLOPE * np.log10(np.maximum(sigma, 1.0) / SIGMA_REF) + LOG_MH_AT_SIGMA_REF


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_suspense():
    """Load SUSPENSE sample (z~1-3.5, with ages)."""
    if not SUSPENSE_JSON.exists():
        return pd.DataFrame()
    rows = json.loads(SUSPENSE_JSON.read_text())
    df = pd.DataFrame(rows)
    required = {"object_id", "z", "log_Mstar_obs", "log_Mdyn", "sigma_kms"}
    if not required <= set(df.columns):
        return pd.DataFrame()
    df["source_survey"] = "SUSPENSE"
    df["source_paper"] = "Slob et al. 2025"
    df["is_upper_limit_mdyn"] = False
    return df


def _load_literature():
    """Load Esdaile+Tanaka sample (z~3.2-4.0)."""
    if not LITERATURE_JSON.exists():
        return pd.DataFrame()
    payload = json.loads(LITERATURE_JSON.read_text())
    objects = payload.get("objects", []) if isinstance(payload, dict) else payload
    df = pd.DataFrame(objects)
    if len(df) == 0:
        return pd.DataFrame()

    df["source_survey"] = "Literature"
    df["source_paper"] = df.get("source", "Unknown")
    df["is_upper_limit_mdyn"] = df.get("log_Mdyn", pd.Series(dtype=float)).isna()

    if "log_Mdyn_upper" in df.columns:
        mask = df["log_Mdyn"].isna() & df["log_Mdyn_upper"].notna()
        df.loc[mask, "log_Mdyn"] = df.loc[mask, "log_Mdyn_upper"]
        df.loc[mask, "is_upper_limit_mdyn"] = True
    if "sigma_kms_upper" in df.columns and "sigma_kms" in df.columns:
        mask = df["sigma_kms"].isna() & df["sigma_kms_upper"].notna()
        df.loc[mask, "sigma_kms"] = df.loc[mask, "sigma_kms_upper"]
    return df


def _load_same_regime():
    """Load same-regime literature kinematic sample (z>4)."""
    if not SAME_REGIME_JSON.exists():
        return pd.DataFrame()
    payload = json.loads(SAME_REGIME_JSON.read_text())
    objects = payload.get("objects", []) if isinstance(payload, dict) else payload
    df = pd.DataFrame(objects)
    if len(df) == 0:
        return pd.DataFrame()

    df["source_survey"] = df.get("sample_group", "unknown")
    df["source_paper"] = df.get("source", "Unknown")
    df["is_upper_limit_mdyn"] = False

    if "log_Mdyn_upper" in df.columns:
        mask_upper = (
            (df.get("log_Mdyn", pd.Series(dtype=float)).isna())
            & df["log_Mdyn_upper"].notna()
        )
        if "log_Mdyn" not in df.columns:
            df["log_Mdyn"] = np.nan
        df.loc[mask_upper, "log_Mdyn"] = df.loc[mask_upper, "log_Mdyn_upper"]
        df.loc[mask_upper, "is_upper_limit_mdyn"] = True

    if "sigma_kms_upper" in df.columns:
        if "sigma_kms" not in df.columns:
            df["sigma_kms"] = np.nan
        mask = df["sigma_kms"].isna() & df["sigma_kms_upper"].notna()
        df.loc[mask, "sigma_kms"] = df.loc[mask, "sigma_kms_upper"]
    return df


def _load_danhaive_sub4():
    """Load Danhaive gold objects at z<4.0 not in same-regime sample."""
    if not DANHAIVE_FULL_JSON.exists():
        return pd.DataFrame()
    payload = json.loads(DANHAIVE_FULL_JSON.read_text())
    objects = payload.get("objects", [])
    df = pd.DataFrame(objects)
    if len(df) == 0:
        return pd.DataFrame()
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df = df[df["z"] < 4.0].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["source_survey"] = "danhaive_2025_gold"
    df["source_paper"] = "Danhaive et al. 2025"
    df["is_upper_limit_mdyn"] = False

    if "log_Mdyn" not in df.columns:
        df["log_Mdyn"] = np.nan
    if "sigma_kms" not in df.columns:
        df["sigma_kms"] = np.nan

    if "log_Mdyn_upper" in df.columns:
        mask = df["log_Mdyn"].isna() & df["log_Mdyn_upper"].notna()
        df.loc[mask, "log_Mdyn"] = df.loc[mask, "log_Mdyn_upper"]
        df.loc[mask, "is_upper_limit_mdyn"] = True
    if "sigma_kms_upper" in df.columns:
        mask = df["sigma_kms"].isna() & df["sigma_kms_upper"].notna()
        df.loc[mask, "sigma_kms"] = df.loc[mask, "sigma_kms_upper"]
    return df


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------

def _build_combined_sample():
    """Merge all kinematic datasets, deduplicate, and prepare for analysis."""
    dfs = []
    sources = {}

    suspense = _load_suspense()
    if len(suspense):
        dfs.append(suspense)
        sources["SUSPENSE (Slob+25)"] = len(suspense)

    literature = _load_literature()
    if len(literature):
        dfs.append(literature)
        sources["Esdaile+21 / Tanaka+19"] = len(literature)

    same_regime = _load_same_regime()
    if len(same_regime):
        dfs.append(same_regime)
        for paper in same_regime["source_paper"].unique():
            sources[str(paper)] = int((same_regime["source_paper"] == paper).sum())

    danhaive_sub4 = _load_danhaive_sub4()
    if len(danhaive_sub4):
        dfs.append(danhaive_sub4)
        key = "Danhaive+25 (z<4 gold)"
        sources[key] = len(danhaive_sub4)

    if not dfs:
        return None, None

    keep_cols = [
        "object_id", "z", "log_Mstar_obs", "log_Mdyn", "sigma_kms",
        "re_kpc", "source_survey", "source_paper", "is_upper_limit_mdyn",
    ]
    frames = []
    for df in dfs:
        for col in keep_cols:
            if col not in df.columns:
                df[col] = np.nan if col not in ("source_survey", "source_paper", "is_upper_limit_mdyn") else "unknown"
        frames.append(df[keep_cols].copy())

    combined = pd.concat(frames, ignore_index=True)

    for col in ["z", "log_Mstar_obs", "log_Mdyn", "sigma_kms", "re_kpc"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["is_upper_limit_mdyn"] = combined["is_upper_limit_mdyn"].fillna(False).astype(bool)

    combined["object_id"] = combined["object_id"].astype(str)
    combined = combined.drop_duplicates(subset=["object_id"], keep="first").reset_index(drop=True)

    valid = (
        combined["z"].notna()
        & combined["log_Mstar_obs"].notna()
        & combined["sigma_kms"].notna()
        & (combined["z"] > 0)
        & (combined["z"] < 15)
        & (combined["log_Mstar_obs"] > 5)
        & (combined["log_Mstar_obs"] < 13)
        & (combined["sigma_kms"] > 0)
    )
    combined = combined[valid].reset_index(drop=True)

    metadata = {
        "n_total": int(len(combined)),
        "n_with_mdyn": int(combined["log_Mdyn"].notna().sum()),
        "n_exact_mdyn": int(combined["log_Mdyn"].notna().sum() - combined["is_upper_limit_mdyn"].sum()),
        "n_upper_limit_mdyn": int(combined["is_upper_limit_mdyn"].sum()),
        "z_min": float(combined["z"].min()),
        "z_max": float(combined["z"].max()),
        "z_median": float(combined["z"].median()),
        "sigma_min_kms": float(combined["sigma_kms"].min()),
        "sigma_max_kms": float(combined["sigma_kms"].max()),
        "source_counts": sources,
        "source_paper_breakdown": {
            str(k): int(v)
            for k, v in combined["source_paper"].value_counts().items()
        },
    }
    return combined, metadata


# ---------------------------------------------------------------------------
# Gamma_t computations
# ---------------------------------------------------------------------------

def _compute_gamma_t_sigma_only(df):
    """Gamma_t from sigma alone (no Mdyn, no M*, no R_e)."""
    sigma = df["sigma_kms"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    log_mh = _sigma_to_log_mhalo(sigma)
    return compute_gamma_t(log_mh, z)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return None, None, int(mask.sum())
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def _ols_fit(x, y):
    """Return (slope, intercept, r2, residuals) for a simple OLS fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) < 4:
        return None, None, None, np.full(len(x), np.nan)
    A = np.column_stack([xm, np.ones(len(xm))])
    beta, residuals, _, _ = np.linalg.lstsq(A, ym, rcond=None)
    fitted = A @ beta
    ss_res = float(np.sum((ym - fitted) ** 2))
    ss_tot = float(np.sum((ym - np.mean(ym)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    full_resid = np.full(len(x), np.nan)
    full_resid[mask] = ym - fitted
    return float(beta[0]), float(beta[1]), float(r2), full_resid


def _ols_fit_2d(x1, x2, y):
    """OLS: y = a*x1 + b*x2 + c.  Returns (a, b, c, r2, residuals)."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    if int(mask.sum()) < 6:
        return None, None, None, None, np.full(len(y), np.nan)
    A = np.column_stack([x1[mask], x2[mask], np.ones(int(mask.sum()))])
    beta, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
    fitted = A @ beta
    ss_res = float(np.sum((y[mask] - fitted) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    full_resid = np.full(len(y), np.nan)
    full_resid[mask] = y[mask] - fitted
    return float(beta[0]), float(beta[1]), float(beta[2]), float(r2), full_resid


# ---------------------------------------------------------------------------
# Test T1: M*-sigma zero-point evolution
# ---------------------------------------------------------------------------

def _test_mstar_sigma_evolution(df):
    """
    T1 — Does the M*-sigma residual increase with z as TEP predicts?

    We fit log(M*_obs) = a * log(sigma) + b on the full sample, then ask
    whether the residual correlates with z.  Under TEP the residual should
    be positive at high z because M*_obs is inflated by Gamma_t^{n_ML}.

    After applying the sigma-only TEP correction, the residual-z correlation
    should be reduced.
    """
    log_sigma = np.log10(df["sigma_kms"].to_numpy(dtype=float))
    log_mstar = df["log_Mstar_obs"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    gamma_t = df["gamma_t_sigma_only"].to_numpy(dtype=float)

    # Fit M*-sigma relation (observed)
    slope_obs, intercept_obs, r2_obs, resid_obs = _ols_fit(log_sigma, log_mstar)

    # Corrected M*
    tep_correction = N_ML * np.log10(np.maximum(gamma_t, 1.001))
    log_mstar_corr = log_mstar - tep_correction
    slope_corr, intercept_corr, r2_corr, resid_corr = _ols_fit(log_sigma, log_mstar_corr)

    # Residual-z partial correlation (controlling for sigma to avoid mass-z selection)
    rho_resid_z_obs, p_resid_z_obs, n_obs = partial_rank_correlation(z, resid_obs, log_sigma)
    rho_resid_z_corr, p_resid_z_corr, n_corr = partial_rank_correlation(z, resid_corr, log_sigma)
    ci_obs = bootstrap_partial_rank_ci(z, resid_obs, log_sigma, n_boot=N_BOOT, seed=RNG_SEED)
    ci_corr = bootstrap_partial_rank_ci(z, resid_corr, log_sigma, n_boot=N_BOOT, seed=RNG_SEED)

    # Scatter comparison
    std_resid_obs = float(np.nanstd(resid_obs, ddof=1))
    std_resid_corr = float(np.nanstd(resid_corr, ddof=1))

    z_trend_improved = bool(abs(rho_resid_z_corr) < abs(rho_resid_z_obs))
    scatter_improved = bool(std_resid_corr < std_resid_obs)

    return {
        "n": int(len(df)),
        "description": (
            "M*-sigma zero-point evolution test. Under TEP, the M*-sigma "
            "residual should increase with z (positive rho). After sigma-only "
            "TEP correction, this trend should weaken."
        ),
        "observed_mstar_sigma_fit": {
            "slope": slope_obs,
            "intercept": intercept_obs,
            "r2": r2_obs,
            "std_residual_dex": std_resid_obs,
        },
        "corrected_mstar_sigma_fit": {
            "slope": slope_corr,
            "intercept": intercept_corr,
            "r2": r2_corr,
            "std_residual_dex": std_resid_corr,
        },
        "observed_residual_z_trend": {
            "partial_rho_resid_z_given_sigma": float(rho_resid_z_obs),
            "p": float(p_resid_z_obs),
            "n": int(n_obs),
            "ci_95": [float(ci_obs[0]), float(ci_obs[1])],
        },
        "corrected_residual_z_trend": {
            "partial_rho_resid_z_given_sigma": float(rho_resid_z_corr),
            "p": float(p_resid_z_corr),
            "n": int(n_corr),
            "ci_95": [float(ci_corr[0]), float(ci_corr[1])],
        },
        "z_trend_improved": z_trend_improved,
        "scatter_improved": scatter_improved,
        "scatter_change_fraction": float(1.0 - std_resid_corr / std_resid_obs) if std_resid_obs > 0 else 0.0,
        "mean_tep_correction_dex": float(np.nanmean(tep_correction)),
    }


# ---------------------------------------------------------------------------
# Test T2: TEP-corrected fundamental-plane scatter
# ---------------------------------------------------------------------------

def _test_fundamental_plane(df):
    """
    T2 — Does TEP correction tighten the fundamental plane?

    Fit log(M*) = a * log(sigma) + b * log(R_e) + c.
    Compare R^2 and scatter for observed vs TEP-corrected M*.
    """
    has_re = df["re_kpc"].notna() & (df["re_kpc"] > 0)
    sub = df[has_re].copy().reset_index(drop=True)
    if len(sub) < 10:
        return {"n": int(len(sub)), "note": "too few objects with R_e"}

    log_sigma = np.log10(sub["sigma_kms"].to_numpy(dtype=float))
    log_re = np.log10(sub["re_kpc"].to_numpy(dtype=float))
    log_mstar = sub["log_Mstar_obs"].to_numpy(dtype=float)
    gamma_t = sub["gamma_t_sigma_only"].to_numpy(dtype=float)
    z = sub["z"].to_numpy(dtype=float)
    tep_correction = N_ML * np.log10(np.maximum(gamma_t, 1.001))
    log_mstar_corr = log_mstar - tep_correction

    a_obs, b_obs, c_obs, r2_obs, resid_obs = _ols_fit_2d(log_sigma, log_re, log_mstar)
    a_corr, b_corr, c_corr, r2_corr, resid_corr = _ols_fit_2d(log_sigma, log_re, log_mstar_corr)

    std_obs = float(np.nanstd(resid_obs, ddof=1))
    std_corr = float(np.nanstd(resid_corr, ddof=1))

    # Residual-z trend
    controls = np.column_stack([log_sigma, log_re])
    rho_obs, p_obs, n_obs = partial_rank_correlation(z, resid_obs, controls)
    rho_corr, p_corr, n_corr = partial_rank_correlation(z, resid_corr, controls)

    return {
        "n": int(len(sub)),
        "description": "Fundamental plane scatter comparison (observed vs TEP-corrected M*).",
        "observed_fp": {
            "coeff_log_sigma": a_obs, "coeff_log_re": b_obs, "intercept": c_obs,
            "r2": r2_obs, "std_residual_dex": std_obs,
        },
        "corrected_fp": {
            "coeff_log_sigma": a_corr, "coeff_log_re": b_corr, "intercept": c_corr,
            "r2": r2_corr, "std_residual_dex": std_corr,
        },
        "r2_improved": bool(r2_corr is not None and r2_obs is not None and r2_corr > r2_obs),
        "scatter_improved": bool(std_corr < std_obs),
        "scatter_change_fraction": float(1.0 - std_corr / std_obs) if std_obs > 0 else 0.0,
        "observed_resid_z_trend": {"partial_rho": float(rho_obs), "p": float(p_obs), "n": int(n_obs)},
        "corrected_resid_z_trend": {"partial_rho": float(rho_corr), "p": float(p_corr), "n": int(n_corr)},
        "z_trend_improved": bool(abs(rho_corr) < abs(rho_obs)),
    }


# ---------------------------------------------------------------------------
# Test T3: Gamma_t adds predictive power for M* beyond sigma alone
# ---------------------------------------------------------------------------

def _test_gamma_sigma_only_prediction(df):
    """
    T3 — Sigma-only Gamma_t as additional M*_obs predictor.

    Test: partial rho(M*_obs, log Gamma_t_sigma_only | log sigma, z).
    Under TEP this should be positive, under LCDM zero.

    Gamma_t_sigma_only is computed from sigma alone, so it adds information
    only through its z-dependent functional form.  A positive partial means
    the specific TEP z-scaling improves the prediction of M* beyond what
    sigma and z individually provide.
    """
    log_mstar = df["log_Mstar_obs"].to_numpy(dtype=float)
    log_sigma = np.log10(df["sigma_kms"].to_numpy(dtype=float))
    z = df["z"].to_numpy(dtype=float)
    log_gamma = np.log10(np.maximum(df["gamma_t_sigma_only"].to_numpy(dtype=float), 1.001))

    # partial rho(M*_obs, log_Gamma_t | log_sigma, z)
    controls = np.column_stack([log_sigma, z])
    rho_gamma, p_gamma, n_gamma = partial_rank_correlation(log_gamma, log_mstar, controls)
    ci_gamma = bootstrap_partial_rank_ci(log_gamma, log_mstar, controls, n_boot=N_BOOT, seed=RNG_SEED)

    # For comparison: partial rho(M*_obs, z | log_sigma)
    rho_z, p_z, n_z = partial_rank_correlation(z, log_mstar, log_sigma)

    # Raw M*-sigma partial controlling for z only
    rho_sigma, p_sigma, n_sigma = partial_rank_correlation(log_sigma, log_mstar, z)

    return {
        "n": int(len(df)),
        "description": (
            "Does sigma-only Gamma_t add predictive power for M*_obs beyond "
            "log(sigma) and z? Positive partial = TEP-consistent z-dependent "
            "mass enhancement."
        ),
        "partial_rho_gamma_mstar_given_sigma_z": float(rho_gamma),
        "p_partial_gamma_mstar_given_sigma_z": float(p_gamma),
        "n_partial": int(n_gamma),
        "ci_95_partial_gamma_mstar_given_sigma_z": [float(ci_gamma[0]), float(ci_gamma[1])],
        "partial_rho_z_mstar_given_sigma": float(rho_z),
        "p_partial_z_mstar_given_sigma": float(p_z),
        "partial_rho_sigma_mstar_given_z": float(rho_sigma),
        "p_partial_sigma_mstar_given_z": float(p_sigma),
    }


# ---------------------------------------------------------------------------
# Test T4: Cross-survey consistency
# ---------------------------------------------------------------------------

def _test_cross_survey(df):
    """
    T4 — Per-survey M*-sigma residual vs z trend.
    """
    log_sigma = np.log10(df["sigma_kms"].to_numpy(dtype=float))
    log_mstar = df["log_Mstar_obs"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    gamma_t = df["gamma_t_sigma_only"].to_numpy(dtype=float)
    log_gamma = np.log10(np.maximum(gamma_t, 1.001))

    # Global M*-sigma fit
    _, _, _, resid_obs = _ols_fit(log_sigma, log_mstar)

    results = {}
    papers = df["source_paper"].unique()
    for paper in sorted(papers):
        mask = (df["source_paper"] == paper).to_numpy()
        n_paper = int(mask.sum())
        if n_paper < 6:
            results[str(paper)] = {"n": n_paper, "note": "too few objects for 2-control partial"}
            continue
        # Test: rho(M*-sigma residual, z | sigma) per-survey
        rho, p, n = partial_rank_correlation(z[mask], resid_obs[mask], log_sigma[mask])
        # Also: rho(Gamma_t, M* | sigma, z) per-survey
        controls_sub = np.column_stack([log_sigma[mask], z[mask]])
        rho_g, p_g, n_g = partial_rank_correlation(log_gamma[mask], log_mstar[mask], controls_sub)
        results[str(paper)] = {
            "n": n_paper,
            "partial_rho_resid_z_given_sigma": float(rho),
            "p_resid_z": float(p),
            "partial_rho_gamma_mstar_given_sigma_z": float(rho_g),
            "p_gamma_mstar": float(p_g),
        }

    n_surveys = sum(1 for v in results.values() if v.get("partial_rho_resid_z_given_sigma") is not None)
    n_positive = sum(
        1 for v in results.values()
        if v.get("partial_rho_resid_z_given_sigma") is not None
        and v["partial_rho_resid_z_given_sigma"] > 0
    )
    return {
        "description": "Per-survey M*-sigma residual vs z trend and Gamma_t prediction test.",
        "n_surveys_tested": n_surveys,
        "n_surveys_positive_resid_z": n_positive,
        "per_survey": results,
    }


# ---------------------------------------------------------------------------
# Test T5: High-z subset
# ---------------------------------------------------------------------------

def _test_highz_subset(df, z_min=4.0):
    """
    T5 — Restrict to z >= z_min and repeat T1 + T3 core tests.
    """
    highz = df[df["z"] >= z_min].copy().reset_index(drop=True)
    if len(highz) < 6:
        return {"n": int(len(highz)), "z_min": z_min, "note": "insufficient objects"}

    log_sigma = np.log10(highz["sigma_kms"].to_numpy(dtype=float))
    log_mstar = highz["log_Mstar_obs"].to_numpy(dtype=float)
    z = highz["z"].to_numpy(dtype=float)
    gamma_t = highz["gamma_t_sigma_only"].to_numpy(dtype=float)
    log_gamma = np.log10(np.maximum(gamma_t, 1.001))

    # T1 core: M*-sigma residual vs z
    _, _, _, resid_obs = _ols_fit(log_sigma, log_mstar)
    rho_resid, p_resid, n_resid = partial_rank_correlation(z, resid_obs, log_sigma)
    ci_resid = bootstrap_partial_rank_ci(z, resid_obs, log_sigma, n_boot=N_BOOT, seed=RNG_SEED)

    # T3 core: Gamma_t adds power for M* beyond sigma + z
    controls = np.column_stack([log_sigma, z])
    rho_g, p_g, n_g = partial_rank_correlation(log_gamma, log_mstar, controls)
    ci_g = bootstrap_partial_rank_ci(log_gamma, log_mstar, controls, n_boot=N_BOOT, seed=RNG_SEED)

    return {
        "n": int(len(highz)),
        "z_min": z_min,
        "partial_rho_resid_z_given_sigma": float(rho_resid),
        "p_resid_z": float(p_resid),
        "ci_95_resid_z": [float(ci_resid[0]), float(ci_resid[1])],
        "partial_rho_gamma_mstar_given_sigma_z": float(rho_g),
        "p_gamma_mstar": float(p_g),
        "ci_95_gamma_mstar": [float(ci_g[0]), float(ci_g[1])],
    }


# ---------------------------------------------------------------------------
# Assessment classification
# ---------------------------------------------------------------------------

def _classify_sigma_test(t1, t2, t3, t5):
    """
    Classify the overall result.  Focus on two key indicators:
      (a) T1: M*-sigma residual increases with z (positive rho_resid_z).
      (b) T1: TEP correction reduces or removes the z trend.
      (c) T2: Fundamental-plane scatter decreases after correction.
      (d) T3: Gamma_t adds predictive power for M* beyond sigma + z.

    Opposite-sign T1 evidence is never labelled supportive.  A significant
    Gamma_t correlation can still be useful context, but it is mixed if the
    primary sign test points the wrong way.
    """
    resid_rho = t1.get("observed_residual_z_trend", {}).get("partial_rho_resid_z_given_sigma")
    resid_p = t1.get("observed_residual_z_trend", {}).get("p")
    z_trend_improved = t1.get("z_trend_improved", False)
    scatter_improved = t1.get("scatter_improved", False)
    fp_scatter_improved = t2.get("scatter_improved", False) if isinstance(t2, dict) else False
    gamma_rho = t3.get("partial_rho_gamma_mstar_given_sigma_z")
    gamma_p = t3.get("p_partial_gamma_mstar_given_sigma_z")
    highz_gamma_rho = t5.get("partial_rho_gamma_mstar_given_sigma_z") if isinstance(t5, dict) else None
    highz_gamma_p = t5.get("p_gamma_mstar") if isinstance(t5, dict) else None

    t1_positive = resid_rho is not None and resid_rho > 0
    t1_significant = t1_positive and resid_p is not None and resid_p < 0.05
    gamma_significant = gamma_rho is not None and gamma_rho > 0 and gamma_p is not None and gamma_p < 0.05
    gamma_directional = gamma_rho is not None and gamma_rho > 0 and gamma_p is not None and gamma_p < 0.10
    highz_gamma_significant = (
        highz_gamma_rho is not None and highz_gamma_rho > 0
        and highz_gamma_p is not None and highz_gamma_p < 0.05
    )

    strong = (
        t1_significant
        and z_trend_improved
        and gamma_significant
    )
    moderate = (
        t1_positive and resid_p is not None and resid_p < 0.10
        and (z_trend_improved or scatter_improved or fp_scatter_improved)
    )
    secondary_only = gamma_significant or highz_gamma_significant or gamma_directional
    directional = t1_positive or (gamma_rho is not None and gamma_rho > 0)

    if strong:
        return "sigma_based_tep_support", True
    elif moderate:
        return "directionally_supportive_sigma_test", True
    elif not t1_positive and secondary_only:
        return "mixed_sigma_test_opposite_primary_sign", False
    elif directional:
        return "weakly_directional_sigma_test", False
    else:
        return "inconclusive_sigma_test", False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print_status(f"STEP {STEP_NUM}: Sigma-Based Kinematic Expansion (Mass-Circularity-Breaking Test)", "INFO")

    combined, metadata = _build_combined_sample()
    if combined is None or len(combined) < 10:
        result = {
            "step": STEP_NUM, "name": STEP_NAME, "status": "FAILED",
            "reason": "Insufficient combined kinematic data.",
        }
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    print_status(f"  Combined kinematic sample: N={len(combined)}, z={metadata['z_min']:.2f}-{metadata['z_max']:.2f}", "INFO")
    print_status(f"  Sources: {metadata['source_paper_breakdown']}", "INFO")

    # Compute sigma-only Gamma_t (SED-independent, R_e-independent)
    combined["gamma_t_sigma_only"] = _compute_gamma_t_sigma_only(combined)

    # Run all tests
    t1 = _test_mstar_sigma_evolution(combined)
    t2 = _test_fundamental_plane(combined)
    t3 = _test_gamma_sigma_only_prediction(combined)
    t4 = _test_cross_survey(combined)
    t5 = _test_highz_subset(combined, z_min=4.0)

    assessment, supportive = _classify_sigma_test(t1, t2, t3, t5)

    # Print key results
    print_status("  T1 — M*-sigma zero-point evolution:", "INFO")
    obs_rho = t1["observed_residual_z_trend"]["partial_rho_resid_z_given_sigma"]
    obs_p = t1["observed_residual_z_trend"]["p"]
    corr_rho = t1["corrected_residual_z_trend"]["partial_rho_resid_z_given_sigma"]
    corr_p = t1["corrected_residual_z_trend"]["p"]
    print_status(f"    Observed  rho(resid, z | sigma) = {obs_rho:.4f} (p={obs_p:.4e})", "INFO")
    print_status(f"    Corrected rho(resid, z | sigma) = {corr_rho:.4f} (p={corr_p:.4e})", "INFO")
    print_status(f"    z-trend improved: {t1['z_trend_improved']}, scatter improved: {t1['scatter_improved']}", "INFO")

    if isinstance(t2, dict) and "observed_fp" in t2:
        print_status(f"  T2 — Fundamental plane (N={t2['n']}):", "INFO")
        print_status(f"    R2 observed={t2['observed_fp']['r2']:.4f}, corrected={t2['corrected_fp']['r2']:.4f}", "INFO")
        print_status(f"    Scatter: {t2['observed_fp']['std_residual_dex']:.3f} → {t2['corrected_fp']['std_residual_dex']:.3f} dex", "INFO")

    print_status("  T3 — Gamma_t adds predictive power:", "INFO")
    print_status(f"    partial rho(Gamma_t, M* | sigma, z) = {t3['partial_rho_gamma_mstar_given_sigma_z']:.4f} "
                 f"(p={t3['p_partial_gamma_mstar_given_sigma_z']:.4e})", "INFO")

    print_status(f"  T4 — Cross-survey: {t4['n_surveys_positive_resid_z']}/{t4['n_surveys_tested']} positive resid-z", "INFO")

    if t5.get("partial_rho_resid_z_given_sigma") is not None:
        print_status(f"  T5 — High-z (z>={t5['z_min']}, N={t5['n']}):", "INFO")
        print_status(f"    rho(resid, z | sigma) = {t5['partial_rho_resid_z_given_sigma']:.4f} "
                     f"(p={t5['p_resid_z']:.4e})", "INFO")
        print_status(f"    rho(Gamma_t, M* | sigma, z) = {t5['partial_rho_gamma_mstar_given_sigma_z']:.4f} "
                     f"(p={t5['p_gamma_mstar']:.4e})", "INFO")

    print_status(f"  Assessment: {assessment} (supportive={supportive})", "INFO")

    # Save combined sample CSV
    combined_csv = INTERIM_PATH / f"step_{STEP_NUM}_combined_kinematic_sample.csv"
    combined.to_csv(combined_csv, index=False)

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "assessment": assessment,
        "supportive": supportive,
        "sample": metadata,
        "tests": {
            "T1_mstar_sigma_evolution": t1,
            "T2_fundamental_plane": t2,
            "T3_gamma_sigma_only_prediction": t3,
            "T4_cross_survey": t4,
            "T5_highz_z_ge_4": t5,
        },
        "methodology": {
            "description": (
                "Sigma-based mass-circularity-breaking test suite. Gamma_t is computed "
                "from velocity dispersion alone (sigma -> M_halo -> Gamma_t), with zero "
                "dependence on SED-fitted M* or size-based Mdyn. This reduces mass-proxy "
                "circularity, while still depending on the sigma-to-halo calibration and "
                "sample mix. The primary test (T1) asks whether the M*-sigma "
                "residual increases with z as TEP predicts. The secondary test (T3) asks "
                "whether the sigma-only Gamma_t adds predictive power for M*_obs beyond "
                "sigma and z individually."
            ),
            "gamma_t_derivation": (
                f"log(M_h) = {SIGMA_SLOPE} * log10(sigma / {SIGMA_REF} km/s) + {LOG_MH_AT_SIGMA_REF}; "
                "Gamma_t = compute_gamma_t(log_Mh, z)"
            ),
            "mass_correction_model": f"n_ML = {N_ML}",
            "kappa_gal": KAPPA_GAL,
            "combined_sample_sources": list(metadata["source_counts"].keys()),
            "mass_proxy_independence": (
                "Gamma_t_sigma_only is derived entirely from the velocity dispersion "
                "sigma (km/s) via a sigma–M_halo mapping. It has zero dependence on "
                "SED-fitted M*, half-light radius R_e, or dynamical mass M_dyn. This "
                "reduces the mass-proxy circularity but does not remove sensitivity to "
                "the adopted sigma–M_halo mapping or sample composition."
            ),
        },
    }

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print_status(f"  Output: {out_json}", "INFO")
    return result


if __name__ == "__main__":
    run()
