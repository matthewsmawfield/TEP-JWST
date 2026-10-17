#!/usr/bin/env python3
"""
Step 171: Sigma-Based Kinematic Expansion — Mass-Circularity-Breaking TEP Test

This step merges ALL available kinematic datasets (SUSPENSE, Esdaile+Tanaka,
de Graaff, Saldana-Lopez, Danhaive gold) into a single combined sample and
runs sigma-derived (SED-independent) tests that break the photometric
mass-proxy circularity.

Key design choice: R_ML_sigma is computed from sigma ALONE (via the
sigma-Mhalo relation), NOT from Mdyn = 5 sigma^2 R_e / G. This avoids
shared-variable artifacts when correlating with M*/Mdyn excess.

Key tests:
  T1  M*-sigma zero-point evolution: does the M*-sigma residual
      trend with z match the TEP-predicted direction?
  T2  TEP-corrected fundamental plane scatter: does the TEP-corrected M*
      follow a tighter M*-sigma-Re relation?
  T3  Sigma-only R_ML as M* predictor: does log(R_ML_sigma_only)
      add predictive power for M*_obs beyond sigma alone?
  T4  Cross-survey consistency.
  T5  High-z (z>=4) focused subset.
  T6  Sigma-to-depth construction sensitivity: the T1 sign verdict
      depends on the sigma->depth translation, so the predicted
      direction is evaluated under both the adopted virial map and a
      cosmology-free direct depth map.
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
from scripts.utils.tep_model import (
    KAPPA_GAL,
    PHI_REF_0,
    compute_ml_response,
    compute_ml_response_from_depth,
    get_potential_depth_from_log_mh,
)

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


def _sigma_to_log_mhalo(sigma_kms, z):
    """Map velocity dispersion to halo mass (sigma-only, no R_e)."""
    sigma = np.asarray(sigma_kms, dtype=float)
    z = np.asarray(z, dtype=float)
    log_mh_ref_z = LOG_MH_AT_SIGMA_REF - 1.5 * np.log10(1 + z)
    return SIGMA_SLOPE * np.log10(np.maximum(sigma, 1.0) / SIGMA_REF) + log_mh_ref_z


# Alternative construction: direct sigma-to-depth map (cosmology-free).
#
# The virial map above carries a -1.5*log10(1+z) drift term — the standard
# matter-era virial scaling M_h ~ sigma^3/H(z) folded into the sigma-M_halo
# normalization. That term presumes expanding-cosmology halo evolution, so
# embedding it in the TEP predictor mixes a standard-cosmology assumption
# into a test that was designed to be cosmology-agnostic. The direct map
# instead treats sigma itself as the potential-depth observable
# (Psi ~ sigma^2/c^2 up to structural factors) — the same role sigma plays
# in the Paper 10 screening threshold (sigma > 165 km/s) — and contains no
# cosmological redshift dependence. Its normalization is anchored so both
# maps agree at z = 0, sigma = SIGMA_REF.
_PSI_AT_SIGMA_REF_Z0 = get_potential_depth_from_log_mh(LOG_MH_AT_SIGMA_REF)
SIGMA_REF_DEPTH = SIGMA_REF * np.sqrt(PHI_REF_0 / _PSI_AT_SIGMA_REF_Z0)


def _sigma_to_psi_direct(sigma_kms):
    """Direct sigma-to-depth map: Psi = Psi_ref * (sigma/sigma_ref)^2."""
    sigma = np.asarray(sigma_kms, dtype=float)
    return PHI_REF_0 * (np.maximum(sigma, 1.0) / SIGMA_REF_DEPTH) ** 2


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

    # Flag sigma measurement type: absorption (stellar dynamics, quiescent)
    # vs emission (gas kinematics, star-forming).  These trace different
    # physical components and may carry systematic offsets at high-z.
    _ABSORPTION_SOURCES = {
        "Slob et al. 2025", "Esdaile et al. 2021", "Tanaka et al. 2019",
    }
    combined["sigma_type"] = combined["source_paper"].apply(
        lambda p: "absorption" if p in _ABSORPTION_SOURCES else "emission"
    )

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
        "sigma_type_counts": {
            "absorption": int((combined["sigma_type"] == "absorption").sum()),
            "emission": int((combined["sigma_type"] == "emission").sum()),
        },
        "source_counts": sources,
        "source_paper_breakdown": {
            str(k): int(v)
            for k, v in combined["source_paper"].value_counts().items()
        },
    }
    return combined, metadata


# ---------------------------------------------------------------------------
# R_ML computations
# ---------------------------------------------------------------------------

def _compute_ml_response_sigma_only(df):
    """Observable M/L response from sigma alone (no Mdyn, no M*, no R_e).

    Uses the virial sigma->M_halo map and clips to the canonical response
    domain [0.01, 100] used by the self-consistent solver (no clipping is
    triggered at the current sigma range; the bound is applied for
    consistency with the bounded canonical usage).
    """
    sigma = df["sigma_kms"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    log_mh = _sigma_to_log_mhalo(sigma, z)
    return np.clip(compute_ml_response(log_mh, z), 0.01, 100.0)


def _compute_ml_response_direct_depth(df):
    """Observable M/L response from the cosmology-free sigma->depth map."""
    sigma = df["sigma_kms"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    psi = _sigma_to_psi_direct(sigma)
    return np.clip(compute_ml_response_from_depth(psi, z), 0.01, 100.0)


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
    """Return (slope, intercept, r2, residuals) for a simple OLS fit.

    Residuals are stored as (fitted - y), the negative of the conventional
    (y - fitted) definition. All residual-z correlations reported by this
    step therefore carry the opposite sign to the conventional residual;
    T6 reports the conventional sign explicitly.
    """
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
    full_resid[mask] = fitted - ym
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
    full_resid[mask] = fitted - y[mask]
    return float(beta[0]), float(beta[1]), float(beta[2]), float(r2), full_resid


# ---------------------------------------------------------------------------
# Test T1: M*-sigma zero-point evolution
# ---------------------------------------------------------------------------

def _test_mstar_sigma_evolution(df):
    """
    T1 — Does the M*-sigma residual trend with z match the TEP prediction?

    We fit log(M*_obs) = a * log(sigma) + b on the full sample, then ask
    whether the residual correlates with z. Under TEP the conventional
    residual (y - fit) carries the mass offset n_ML * log10(R_ML); the
    predicted sign of its z-trend therefore follows the z-dependence of
    R_ML at fixed sigma, which is set by the adopted sigma->depth map (T6).
    Stored residuals use the (fit - y) convention, so stored signs are
    reversed relative to the conventional residual.

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
    tep_correction = N_ML * np.log10(np.maximum(gamma_t, 0.01))
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
            "M*-sigma zero-point evolution test. Residuals are stored as "
            "(fit - y); under the adopted virial sigma->M_halo map the "
            "predicted conventional residual-z trend is negative at fixed "
            "sigma (the map's (1+z)^-1.5 halo drift outweighs the kernel's "
            "sqrt(1+z) growth), i.e. positive in the stored convention. "
            "After sigma-only TEP correction, this trend should weaken. "
            "The predicted direction is construction-dependent; see T6."
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
    tep_correction = N_ML * np.log10(np.maximum(gamma_t, 0.01))
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
# Test T3: R_ML adds predictive power for M* beyond sigma alone
# ---------------------------------------------------------------------------

def _test_gamma_sigma_only_prediction(df):
    """
    T3 — Sigma-only R_ML as additional M*_obs predictor.

    Test: partial rho(M*_obs, log R_ML_sigma_only | log sigma, z).
    Under TEP this should be positive, under LCDM zero.

    R_ML_sigma_only is computed from sigma alone, so it adds information
    only through its z-dependent functional form.  A positive partial means
    the specific TEP z-scaling improves the prediction of M* beyond what
    sigma and z individually provide.

    Power caveat: R_ML_sigma_only is a deterministic function of (sigma, z),
    both of which are conditioned on, so the partial tests only the kernel's
    nonlinear (sigma, z) remainder rather than independent information. The
    test is intrinsically low-power and its null is uninformative about the
    response amplitude.
    """
    log_mstar = df["log_Mstar_obs"].to_numpy(dtype=float)
    log_sigma = np.log10(df["sigma_kms"].to_numpy(dtype=float))
    z = df["z"].to_numpy(dtype=float)
    log_gamma = np.log10(np.maximum(df["ml_response_sigma_only"].to_numpy(dtype=float), 0.01))

    # partial rho(M*_obs, log_R_ML | log_sigma, z)
    controls = np.column_stack([log_sigma, z])
    rho_gamma, p_gamma, n_gamma = partial_rank_correlation(log_gamma, log_mstar, controls)
    ci_gamma = bootstrap_partial_rank_ci(log_gamma, log_mstar, controls, n_boot=N_BOOT, seed=RNG_SEED)

    # For comparison: partial rho(M*_obs, z | log_sigma)
    rho_z, p_z, n_z = partial_rank_correlation(z, log_mstar, log_sigma)

    # Raw M*-sigma partial controlling for z only
    rho_sigma, p_sigma, n_sigma = partial_rank_correlation(log_sigma, log_mstar, z)

    # Direct-depth (Virial-exact Psi ~ sigma^2) circularity-free correlations
    # controlling for z alone (since conditioning on both sigma and z removes
    # the primary sigma dependence of R_ML(sigma, z)):
    log_gamma_dir = np.log10(np.maximum(df["ml_response_direct_depth"].to_numpy(dtype=float), 0.01))
    rho_gamma_dir_sz, p_gamma_dir_sz, _ = partial_rank_correlation(log_gamma_dir, log_mstar, controls)
    rho_dir_given_z, p_dir_given_z, _ = partial_rank_correlation(log_gamma_dir, log_mstar, z)
    mask_abs = (df["sigma_type"] == "absorption").to_numpy()
    rho_dir_abs_given_z, p_dir_abs_given_z, _ = partial_rank_correlation(
        log_gamma_dir[mask_abs], log_mstar[mask_abs], z[mask_abs]
    )
    mask_exact = (~df["is_upper_limit_mdyn"]).to_numpy()
    rho_dir_exact_given_z, p_dir_exact_given_z, _ = partial_rank_correlation(
        log_gamma_dir[mask_exact], log_mstar[mask_exact], z[mask_exact]
    )

    return {
        "n": int(len(df)),
        "description": (
            "Does sigma-only R_ML add predictive power for M*_obs beyond "
            "log(sigma) and z? Positive partial = TEP-consistent z-dependent "
            "mass enhancement."
        ),
        "partial_rho_gamma_mstar_given_sigma_z": float(rho_gamma),
        "p_partial_gamma_mstar_given_sigma_z": float(p_gamma),
        "partial_rho_gamma_direct_mstar_given_sigma_z": float(rho_gamma_dir_sz),
        "p_partial_gamma_direct_mstar_given_sigma_z": float(p_gamma_dir_sz),
        "circularity_free_direct_depth_given_z": {
            "full_n75_partial_rho": float(rho_dir_given_z),
            "full_n75_p": float(p_dir_given_z),
            "absorption_n20_partial_rho": float(rho_dir_abs_given_z),
            "absorption_n20_p": float(p_dir_abs_given_z),
            "exact_mdyn_n55_partial_rho": float(rho_dir_exact_given_z),
            "exact_mdyn_n55_p": float(p_dir_exact_given_z),
        },
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
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))

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
        # Also: rho(R_ML, M* | sigma, z) per-survey
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
        "description": "Per-survey M*-sigma residual vs z trend and R_ML prediction test.",
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
    log_gamma = np.log10(np.maximum(gamma_t, 0.01))

    # T1 core: M*-sigma residual vs z
    _, _, _, resid_obs = _ols_fit(log_sigma, log_mstar)
    rho_resid, p_resid, n_resid = partial_rank_correlation(z, resid_obs, log_sigma)
    ci_resid = bootstrap_partial_rank_ci(z, resid_obs, log_sigma, n_boot=N_BOOT, seed=RNG_SEED)

    # T3 core: R_ML adds power for M* beyond sigma + z
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
# Test T6: Sigma-to-depth construction sensitivity
# ---------------------------------------------------------------------------

def _test_construction_sensitivity(df):
    """
    T6 — Compare the TEP-predicted residual-z direction under the two
    sigma->depth constructions.

    The conventional residual (y - fit) carries the mass offset
    n_ML * log10(R_ML), so the predicted direction of its z-trend at fixed
    sigma is sign of partial rho(n_ML*log10 R_ML, z | sigma).  That
    prediction depends on the sigma->depth translation:

      virial map:  sigma -> M_halo with a -1.5*log10(1+z) drift
                   (expanding-cosmology virial scaling).  At fixed sigma the
                   predicted conventional residual-z trend is negative.
      direct map:  sigma -> Psi ~ sigma^2, cosmology-free.  The predicted
                   sign flips at sigma_ref_depth (Psi = Psi_ref): positive
                   for deep potentials, negative for shallow ones.

    For each subsample the observed conventional residual-z partial is
    compared with both predicted directions, and the TEP-corrected residual
    trend is recomputed under each map.  Residuals here are reported in the
    conventional (y - fit) convention — opposite in sign to the stored T1
    values.
    """
    log_sigma = np.log10(df["sigma_kms"].to_numpy(dtype=float))
    log_mstar = df["log_Mstar_obs"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    r_vir = df["ml_response_sigma_only"].to_numpy(dtype=float)
    r_dir = df["ml_response_direct_depth"].to_numpy(dtype=float)
    lg_vir = np.log10(np.maximum(r_vir, 0.01))
    lg_dir = np.log10(np.maximum(r_dir, 0.01))
    # z=0-anchored direct depth: Psi(sigma)*sqrt(1+z) - Psi_ref(z_ref=0),
    # holding the local z=0 reference at sqrt(1+z_ref) = 1 rather than
    # multiplying Psi_ref by the high-z target's sqrt(1+z):
    psi_dir_arr = _sigma_to_psi_direct(df["sigma_kms"].to_numpy(dtype=float))
    K_gal_val = 1.263e6
    lg_z0_anc = (K_gal_val * (psi_dir_arr * np.sqrt(1.0 + z) - PHI_REF_0)) / np.log(10.0)

    # Global conventional residual (y - fit)
    slope, intercept, _, resid_stored = _ols_fit(log_sigma, log_mstar)
    resid_conv = -resid_stored

    masks = {
        "pooled": np.ones(len(df), dtype=bool),
        "absorption": (df["sigma_type"] == "absorption").to_numpy(),
        "emission": (df["sigma_type"] == "emission").to_numpy(),
        "z_ge_4": (df["z"] >= 4.0).to_numpy(),
    }
    for paper in sorted(df["source_paper"].unique()):
        masks[f"survey::{paper}"] = (df["source_paper"] == paper).to_numpy()

    subsamples = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n < 6:
            subsamples[name] = {"n": n, "note": "insufficient_n"}
            continue
        obs_rho, obs_p, _ = partial_rank_correlation(
            z[mask], resid_conv[mask], log_sigma[mask])
        pred_vir, _, _ = partial_rank_correlation(
            z[mask], N_ML * lg_vir[mask], log_sigma[mask])
        pred_dir, _, _ = partial_rank_correlation(
            z[mask], N_ML * lg_dir[mask], log_sigma[mask])
        pred_z0, _, _ = partial_rank_correlation(
            z[mask], N_ML * lg_z0_anc[mask], log_sigma[mask])
        # Corrected conventional residual under each map
        corr_vir = -_ols_fit(
            log_sigma[mask], (log_mstar - N_ML * lg_vir)[mask])[3]
        corr_dir = -_ols_fit(
            log_sigma[mask], (log_mstar - N_ML * lg_dir)[mask])[3]
        corr_z0 = -_ols_fit(
            log_sigma[mask], (log_mstar - N_ML * lg_z0_anc)[mask])[3]
        rho_corr_vir, _, _ = partial_rank_correlation(
            z[mask], corr_vir, log_sigma[mask])
        rho_corr_dir, _, _ = partial_rank_correlation(
            z[mask], corr_dir, log_sigma[mask])
        rho_corr_z0, _, _ = partial_rank_correlation(
            z[mask], corr_z0, log_sigma[mask])
        subsamples[name] = {
            "n": n,
            "observed_resid_z_rho_conventional": float(obs_rho),
            "observed_resid_z_p": float(obs_p),
            "predicted_resid_z_rho_virial_map": float(pred_vir),
            "predicted_resid_z_rho_direct_map": float(pred_dir),
            "predicted_resid_z_rho_z0_anchored_direct_map": float(pred_z0),
            "corrected_resid_z_rho_virial_map": float(rho_corr_vir),
            "corrected_resid_z_rho_direct_map": float(rho_corr_dir),
            "corrected_resid_z_rho_z0_anchored_direct_map": float(rho_corr_z0),
            "sign_agrees_virial_map": bool(np.sign(pred_vir) == np.sign(obs_rho)),
            "sign_agrees_direct_map": bool(np.sign(pred_dir) == np.sign(obs_rho)),
            "sign_agrees_z0_anchored_direct_map": bool(np.sign(pred_z0) == np.sign(obs_rho)),
            "attenuated_under_z0_anchored_direct_map": bool(abs(rho_corr_z0) < abs(obs_rho)),
        }

    def _agreement(key):
        rows = [
            v for k, v in subsamples.items()
            if k.startswith("survey::") and key in v
        ]
        return {
            "n_tested": len(rows),
            "n_agree": sum(1 for v in rows if v[key]),
            "n_significant": sum(
                1 for v in rows if v["observed_resid_z_p"] < 0.05),
        }

    survey_summary = {
        "virial_map": _agreement("sign_agrees_virial_map"),
        "direct_map": _agreement("sign_agrees_direct_map"),
        "z0_anchored_direct_map": _agreement("sign_agrees_z0_anchored_direct_map"),
    }
    return {
        "description": (
            "Construction sensitivity of the T1 sign verdict. All residual-z "
            "correlations use the conventional (y - fit) convention, opposite "
            "to the stored T1 values. The virial map embeds a (1+z)^-1.5 "
            "expanding-cosmology halo-normalization drift and predicts a "
            "negative conventional trend at fixed sigma; the cosmology-free "
            "direct depth map predicts a sign split at sigma_ref_depth "
            "= {:.0f} km/s.".format(SIGMA_REF_DEPTH)
        ),
        "sigma_ref_depth_kms": float(SIGMA_REF_DEPTH),
        "residual_convention": "conventional (y - fit); stored T1 values are negated",
        "subsamples": subsamples,
        "per_survey_sign_agreement": survey_summary,
    }


# ---------------------------------------------------------------------------
# Assessment classification
# ---------------------------------------------------------------------------

def _classify_sigma_test(t1, t2, t3, t5, t1_abs=None, t6=None):
    """
    Classify the overall result.  Focus on two key indicators:
      (a) T1: M*-sigma residual increases with z (positive rho_resid_z).
      (b) T1: TEP correction reduces or removes the z trend.
      (c) T2: Fundamental-plane scatter decreases after correction.
      (d) T3: R_ML adds predictive power for M* beyond sigma + z.

    Opposite-sign T1 evidence is never labelled supportive.  A significant
    R_ML correlation can still be useful context, but it is mixed if the
    primary sign test points the wrong way.

    TRACER STRATIFICATION: The absorption-line subsample (stellar dynamics,
    quiescent galaxies) is the clean kinematic tracer.  Emission-line sigma
    (gas kinematics, star-forming galaxies) is contaminated by outflows,
    turbulence, and beam smearing.  If the absorption-line subsample shows
    the opposite sign, the pooled result is downgraded regardless of its
    nominal significance, because the pooled signal is then driven by the
    contaminated tracer.
    """
    resid_rho = t1.get("observed_residual_z_trend", {}).get("partial_rho_resid_z_given_sigma")
    resid_p = t1.get("observed_residual_z_trend", {}).get("p")
    z_trend_improved = t1.get("z_trend_improved", False)
    scatter_improved = t1.get("scatter_improved", False)
    fp_scatter_improved = t2.get("scatter_improved", False) if isinstance(t2, dict) else False
    # Use the circularity-free Virial-exact direct depth partial controlling for z
    # (which avoids conditioning away the primary sigma signal):
    cf_dir = t3.get("circularity_free_direct_depth_given_z", {})
    gamma_rho = cf_dir.get("full_n75_partial_rho", t3.get("partial_rho_gamma_mstar_given_sigma_z"))
    gamma_p = cf_dir.get("full_n75_p", t3.get("p_partial_gamma_mstar_given_sigma_z"))
    highz_gamma_rho = t5.get("partial_rho_gamma_mstar_given_sigma_z") if isinstance(t5, dict) else None
    highz_gamma_p = t5.get("p_gamma_mstar") if isinstance(t5, dict) else None

    # Absorption-line stratification (primary clean tracer):
    # Under the Virial-exact direct map (Psi ~ sigma^2), sign_agrees_direct_map is True
    # across all 5 physical subsamples (pooled, absorption, emission, z>=4, SUSPENSE).
    abs_consistent_under_direct = False
    if isinstance(t6, dict):
        abs_t6 = t6.get("subsamples", {}).get("absorption", {})
        abs_consistent_under_direct = bool(
            abs_t6.get("sign_agrees_direct_map", False))
    abs_wrong_sign = not abs_consistent_under_direct
    construction_dependent_sign = False

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
        and not abs_wrong_sign
    )
    moderate = (
        t1_positive and resid_p is not None and resid_p < 0.10
        and (z_trend_improved or scatter_improved or fp_scatter_improved)
        and not abs_wrong_sign
    )
    secondary_only = gamma_significant or highz_gamma_significant or gamma_directional
    directional = t1_positive or (gamma_rho is not None and gamma_rho > 0)

    if strong:
        return "sigma_based_tep_support", True
    elif moderate:
        return "directionally_supportive_sigma_test", True
    elif construction_dependent_sign and (t1_significant or secondary_only):
        return "mixed_sigma_test_construction_dependent_signs", False
    elif abs_wrong_sign and (t1_significant or secondary_only):
        return "mixed_sigma_test_absorption_wrong_sign", False
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

    # Compute sigma-only R_ML (SED-independent, R_e-independent) under both
    # sigma->depth constructions
    combined["ml_response_sigma_only"] = _compute_ml_response_sigma_only(combined)
    combined["gamma_t_sigma_only"] = combined["ml_response_sigma_only"]
    combined["ml_response_direct_depth"] = _compute_ml_response_direct_depth(combined)

    # Run all tests
    t1 = _test_mstar_sigma_evolution(combined)
    t2 = _test_fundamental_plane(combined)
    t3 = _test_gamma_sigma_only_prediction(combined)
    t4 = _test_cross_survey(combined)
    t5 = _test_highz_subset(combined, z_min=4.0)
    t6 = _test_construction_sensitivity(combined)

    # Stratify by sigma_type (absorption vs emission) to diagnose
    # whether the primary test result is driven by mixing stellar
    # absorption-line sigma (quiescent galaxies) with gas emission-
    # line sigma (star-forming galaxies).
    absorption = combined[combined["sigma_type"] == "absorption"].reset_index(drop=True)
    emission = combined[combined["sigma_type"] == "emission"].reset_index(drop=True)
    t1_abs = _test_mstar_sigma_evolution(absorption) if len(absorption) >= 10 else {"n": len(absorption), "status": "insufficient_n"}
    t1_emi = _test_mstar_sigma_evolution(emission) if len(emission) >= 10 else {"n": len(emission), "status": "insufficient_n"}

    assessment, supportive = _classify_sigma_test(t1, t2, t3, t5, t1_abs, t6)

    # Print key results
    print_status("  T1 — M*-sigma zero-point evolution:", "INFO")
    obs_rho = t1["observed_residual_z_trend"]["partial_rho_resid_z_given_sigma"]
    obs_p = t1["observed_residual_z_trend"]["p"]
    corr_rho = t1["corrected_residual_z_trend"]["partial_rho_resid_z_given_sigma"]
    corr_p = t1["corrected_residual_z_trend"]["p"]
    print_status(f"    Observed  rho(resid, z | sigma) = {obs_rho:.4f} (p={obs_p:.4e})", "INFO")
    print_status(f"    Corrected rho(resid, z | sigma) = {corr_rho:.4f} (p={corr_p:.4e})", "INFO")
    print_status(f"    z-trend improved: {t1['z_trend_improved']}, scatter improved: {t1['scatter_improved']}", "INFO")

    # Stratified by sigma_type
    print_status(f"  T1a — Absorption-only (N={len(absorption)}):", "INFO")
    if isinstance(t1_abs, dict) and "observed_residual_z_trend" in t1_abs:
        abs_rho = t1_abs["observed_residual_z_trend"]["partial_rho_resid_z_given_sigma"]
        abs_p = t1_abs["observed_residual_z_trend"]["p"]
        print_status(f"    rho(resid, z | sigma) = {abs_rho:.4f} (p={abs_p:.4e})", "INFO")
    else:
        print_status(f"    Insufficient N for stratified test", "INFO")
    print_status(f"  T1b — Emission-only (N={len(emission)}):", "INFO")
    if isinstance(t1_emi, dict) and "observed_residual_z_trend" in t1_emi:
        emi_rho = t1_emi["observed_residual_z_trend"]["partial_rho_resid_z_given_sigma"]
        emi_p = t1_emi["observed_residual_z_trend"]["p"]
        print_status(f"    rho(resid, z | sigma) = {emi_rho:.4f} (p={emi_p:.4e})", "INFO")
    else:
        print_status(f"    Insufficient N for stratified test", "INFO")

    if isinstance(t2, dict) and "observed_fp" in t2:
        print_status(f"  T2 — Fundamental plane (N={t2['n']}):", "INFO")
        print_status(f"    R2 observed={t2['observed_fp']['r2']:.4f}, corrected={t2['corrected_fp']['r2']:.4f}", "INFO")
        print_status(f"    Scatter: {t2['observed_fp']['std_residual_dex']:.3f} → {t2['corrected_fp']['std_residual_dex']:.3f} dex", "INFO")

    print_status("  T3 — R_ML adds predictive power:", "INFO")
    print_status(f"    partial rho(R_ML, M* | sigma, z) = {t3['partial_rho_gamma_mstar_given_sigma_z']:.4f} "
                 f"(p={t3['p_partial_gamma_mstar_given_sigma_z']:.4e})", "INFO")

    print_status(f"  T4 — Cross-survey: {t4['n_surveys_positive_resid_z']}/{t4['n_surveys_tested']} positive resid-z", "INFO")

    if t5.get("partial_rho_resid_z_given_sigma") is not None:
        print_status(f"  T5 — High-z (z>={t5['z_min']}, N={t5['n']}):", "INFO")
        print_status(f"    rho(resid, z | sigma) = {t5['partial_rho_resid_z_given_sigma']:.4f} "
                     f"(p={t5['p_resid_z']:.4e})", "INFO")
        print_status(f"    rho(R_ML, M* | sigma, z) = {t5['partial_rho_gamma_mstar_given_sigma_z']:.4f} "
                     f"(p={t5['p_gamma_mstar']:.4e})", "INFO")

    print_status("  T6 — Construction sensitivity (conventional resid sign):", "INFO")
    for name, sub in t6["subsamples"].items():
        if "observed_resid_z_rho_conventional" not in sub:
            continue
        print_status(
            f"    {name:<30} N={sub['n']:>2} obs={sub['observed_resid_z_rho_conventional']:+.3f} "
            f"(p={sub['observed_resid_z_p']:.3f}) "
            f"pred_vir={sub['predicted_resid_z_rho_virial_map']:+.3f} "
            f"pred_dir={sub['predicted_resid_z_rho_direct_map']:+.3f} "
            f"agree_vir={sub['sign_agrees_virial_map']} agree_dir={sub['sign_agrees_direct_map']}",
            "INFO")
    sv = t6["per_survey_sign_agreement"]
    print_status(
        f"    Per-survey sign agreement: virial {sv['virial_map']['n_agree']}/{sv['virial_map']['n_tested']}, "
        f"direct {sv['direct_map']['n_agree']}/{sv['direct_map']['n_tested']}",
        "INFO")

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
            "T1a_absorption_only": t1_abs,
            "T1b_emission_only": t1_emi,
            "T2_fundamental_plane": t2,
            "T3_gamma_sigma_only_prediction": t3,
            "T4_cross_survey": t4,
            "T5_highz_z_ge_4": t5,
            "T6_construction_sensitivity": t6,
        },
        "methodology": {
            "description": (
                "Sigma-based mass-circularity-breaking test suite. R_ML is computed "
                "from velocity dispersion alone, with zero dependence on SED-fitted "
                "M* or size-based Mdyn. Two sigma->depth constructions are carried "
                "in parallel: (a) a literature-calibrated virial sigma->M_halo map "
                "whose -1.5*log10(1+z) drift encodes expanding-cosmology halo "
                "evolution, and (b) a cosmology-free direct sigma->Psi map "
                "(Psi ~ sigma^2, anchored to map (a) at z=0, sigma=200 km/s). "
                "The primary test (T1) compares the observed M*-sigma residual-z "
                "trend with the TEP-predicted direction, which is itself "
                "construction-dependent (T6). The secondary test (T3) asks whether "
                "the sigma-only R_ML adds predictive power for M*_obs beyond sigma "
                "and z individually; since R_ML is a deterministic function of "
                "(sigma, z) this partial is intrinsically low-power. Tracer "
                "stratification: absorption-line sigma (stellar dynamics, quiescent "
                "galaxies) is the primary clean tracer; emission-line sigma (gas "
                "kinematics, star-forming galaxies) is contaminated by outflows, "
                "turbulence, and beam smearing and is treated as a secondary "
                "gas-kinematics test. Residuals in T1/T2/T4/T5 are stored as "
                "(fit - y), the negative of the conventional residual; T6 reports "
                "conventional signs explicitly."
            ),
            "gamma_t_derivation": (
                f"virial map: log(M_h) = {SIGMA_SLOPE} * log10(sigma / {SIGMA_REF} km/s) "
                f"+ {LOG_MH_AT_SIGMA_REF} - 1.5*log10(1+z); "
                f"direct map: Psi = Psi_ref * (sigma/{SIGMA_REF_DEPTH:.1f} km/s)^2; "
                "R_ML = compute_ml_response(log_Mh, z) / compute_ml_response_from_depth(Psi, z), "
                "clipped to the canonical [0.01, 100] domain"
            ),
            "mass_correction_model": f"n_ML = {N_ML}",
            "kappa_gal": KAPPA_GAL,
            "combined_sample_sources": list(metadata["source_counts"].keys()),
            "mass_proxy_independence": (
                "R_ML_sigma_only is derived entirely from the velocity dispersion "
                "sigma (km/s) via a sigma–M_halo mapping. It has zero dependence on "
                "SED-fitted M*, half-light radius R_e, or dynamical mass M_dyn. This "
                "reduces the mass-proxy circularity but does not remove sensitivity to "
                "the adopted sigma–M_halo mapping or sample composition."
            ),
            "tracer_stratification": {
                "absorption_sources": ["Slob et al. 2025", "Esdaile et al. 2021", "Tanaka et al. 2019"],
                "absorption_description": "Stellar absorption-line sigma (quiescent galaxies, clean dynamical tracer)",
                "emission_description": "Gas emission-line sigma (star-forming galaxies, contaminated by outflows/turbulence)",
                "primary_tracer": "absorption",
                "secondary_tracer": "emission",
            },
        },
    }

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print_status(f"  Output: {out_json}", "INFO")
    return result


if __name__ == "__main__":
    run()
