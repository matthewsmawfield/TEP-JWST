#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 42.1s.
"""
TEP-JWST Step 157: COSMOS2025 sSFR inversion (Table B4b)

Mass-sSFR inversion analysis in the COSMOS-Web blank field (Shuntov+2025).
Tests whether the partial correlation rho(Gamma_t, log sSFR | M*, z) flips
sign at z>7, replicating the L3 sSFR inversion in an independent 0.54 deg^2
field. Produces Table B4b statistics (sSFR rows) and the Steiger Z-test
comparing low-z vs high-z partial correlations.

Data: COSMOSWeb_mastercatalog_v1_lephare.fits (Shuntov et al. 2025)
Requires: data/raw/COSMOSWeb_mastercatalog_v1_lephare.fits

Author: Matthew L. Smawfield
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from numpy.linalg import lstsq

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import (
    compute_gamma_t,
    stellar_to_halo_mass_behroozi_like,  # Shared TEP model
)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting
from scripts.utils.rank_stats import partial_rank_correlation  # Partial Spearman helper

STEP_NUM  = "157"  # Pipeline step number
STEP_NAME = "cosmos2025_ssfr_inversion"  # Used in log / output filenames

DATA_PATH   = PROJECT_ROOT / "data" / "raw"  # Raw external catalogues
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH   = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

COSMOSWEB_FITS = DATA_PATH / "COSMOSWeb_mastercatalog_v1_lephare.fits"
STEP117_JSON = OUTPUT_PATH / "step_117_dynamical_mass_comparison.json"
STEP159_JSON = OUTPUT_PATH / "step_159_mass_measurement_bias.json"

# sSFR inversion redshift bins (Table B4b rows)
SSFR_BINS = [
    (4.0, 7.0,  "z=4-7 (low-z)"),
    (7.0, 8.0,  "z=7-8"),
    (8.0, 9.0,  "z=8-9"),
    (9.0, 13.0, "z=9-13 (high-z)"),
]


def partial_spearman(x, y, controls):
    """Spearman partial correlation of x and y after regressing out controls."""
    rho, p, _ = partial_rank_correlation(x, y, controls)
    return float(rho), float(p)


def _weighted_corr(x, y, weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-9, None)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = np.average(x, weights=weights)
    my = np.average(y, weights=weights)
    xc = x - mx
    yc = y - my
    cov = np.sum(weights * xc * yc)
    vx = np.sum(weights * xc * xc)
    vy = np.sum(weights * yc * yc)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))


def weighted_partial_spearman(x, y, controls, weights):
    x_rank = stats.rankdata(np.asarray(x, dtype=float))
    y_rank = stats.rankdata(np.asarray(y, dtype=float))
    ctrl_rank = [stats.rankdata(np.asarray(ctrl, dtype=float)) for ctrl in controls]
    X = np.column_stack(ctrl_rank + [np.ones(len(x_rank))])
    w = np.clip(np.asarray(weights, dtype=float), 1e-9, None)
    sw = np.sqrt(w)[:, None]
    beta_x, _, _, _ = lstsq(X * sw, x_rank * sw[:, 0], rcond=None)
    beta_y, _, _, _ = lstsq(X * sw, y_rank * sw[:, 0], rcond=None)
    resid_x = x_rank - X @ beta_x
    resid_y = y_rank - X @ beta_y
    rho = _weighted_corr(resid_x, resid_y, w)
    n_eff = float((w.sum() ** 2) / np.sum(np.square(w)))
    if not np.isfinite(rho) or n_eff <= 3:
        return float("nan"), float("nan"), n_eff
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999)) * np.sqrt(max(n_eff - 3.0, 1e-6))
    p = 2 * stats.norm.sf(abs(z))
    return float(rho), max(float(p), 1e-300), n_eff


def reference_mass_reweight(target_mass, reference_mass, bin_width=0.25, max_weight=10.0):
    target_mass = np.asarray(target_mass, dtype=float)
    reference_mass = np.asarray(reference_mass, dtype=float)
    valid_target = np.isfinite(target_mass)
    valid_reference = np.isfinite(reference_mass)
    if valid_target.sum() < 20 or valid_reference.sum() < 20:
        return None
    lo = float(np.floor(min(np.nanmin(target_mass[valid_target]), np.nanmin(reference_mass[valid_reference])) / bin_width) * bin_width)
    hi = float(np.ceil(max(np.nanmax(target_mass[valid_target]), np.nanmax(reference_mass[valid_reference])) / bin_width) * bin_width + bin_width)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    bins = np.arange(lo, hi + 0.5 * bin_width, bin_width)
    if len(bins) < 3:
        return None
    ref_hist, _ = np.histogram(reference_mass[valid_reference], bins=bins, density=True)
    tgt_hist, _ = np.histogram(target_mass[valid_target], bins=bins, density=True)
    idx = np.clip(np.digitize(target_mass, bins) - 1, 0, len(bins) - 2)
    weights = np.zeros_like(target_mass, dtype=float)
    positive_bins = 0
    for i in range(len(ref_hist)):
        mask = valid_target & (idx == i)
        if not np.any(mask):
            continue
        if tgt_hist[i] > 0 and ref_hist[i] > 0:
            weights[mask] = ref_hist[i] / tgt_hist[i]
            positive_bins += 1
    if positive_bins < 2 or np.count_nonzero(weights > 0) < 20:
        return None
    return np.clip(weights, 0.0, max_weight)


def bootstrap_ci_partial(df_sub, mass_col="log_Mstar", n_boot=200, seed=42):
    """Bootstrap 95% CI for partial Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(df_sub)
    boot_rhos = []
    x = df_sub["gamma_t"].values
    y = df_sub["log_ssfr"].values
    ctrl_m = df_sub[mass_col].values
    ctrl_z = df_sub["z_phot"].values
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            r, _ = partial_spearman(x[idx], y[idx], [ctrl_m[idx], ctrl_z[idx]])
            boot_rhos.append(r)
        except Exception:
            pass
    if len(boot_rhos) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def steiger_z_test(r_ab, r_ac, r_bc, n):
    """
    Meng-Rosenthal-Rubin (1992) Steiger Z-test comparing two dependent
    correlations r_ab and r_ac that share variable a.
    r_bc = correlation between b and c.
    """
    # Fisher z-transform difference
    z_ab = np.arctanh(r_ab)
    z_ac = np.arctanh(r_ac)
    r_bar = (r_ab ** 2 + r_ac ** 2) / 2
    f = (1 - r_bc) / (2 * (1 - r_bar))
    h = (1 - f * r_bar) / (1 - r_bar)
    Z = (z_ab - z_ac) * np.sqrt((n - 3) / (2 * (1 - r_bc) * h))
    p = 2 * stats.norm.sf(abs(Z))
    return float(Z), max(float(p), 1e-300)


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


def load_cosmos_catalog():
    """Load and filter COSMOS-Web LePHARE catalog."""
    from astropy.io import fits
    from astropy.cosmology import Planck18 as cosmo

    if not COSMOSWEB_FITS.exists():
        logger.error(f"COSMOS-Web FITS not found: {COSMOSWEB_FITS}")
        return None

    logger.info(f"Loading {COSMOSWEB_FITS.name} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(COSMOSWEB_FITS) as hdul:
            t = hdul[1].data
            df = pd.DataFrame({
                "z_phot":      t["zfinal"].astype(float),
                "z_l68":       t["zpdf_l68"].astype(float),
                "z_u68":       t["zpdf_u68"].astype(float),
                "log_Mstar":   t["mass_med"].astype(float),
                "log_Mstar_l68": t["mass_l68"].astype(float),
                "log_Mstar_u68": t["mass_u68"].astype(float),
                "ssfr_med":    t["ssfr_med"].astype(float),
                "ebv":         t["ebv_minchi2"].astype(float),
                "chi2_best":   t["chi2_best"].astype(float),
                "galaxy_type": t["type"].astype(int),
            })

    # Filter: galaxies only (type=0), valid SED parameters
    df = df[
        (df["galaxy_type"] == 0) &
        (df["z_phot"] > 0) & (df["z_phot"] < 15) &
        (df["log_Mstar"] > -90) &
        (df["ssfr_med"] > -90) &
        (df["log_Mstar"] > 6.0)
    ].copy()

    logger.info(f"Valid galaxy SED fits: {len(df):,}")

    # log_ssfr is already log10 from LePHARE
    df["log_ssfr"] = df["ssfr_med"]

    # Compute TEP quantities
    df["log_Mh"] = stellar_to_halo_mass_behroozi_like(
        df["log_Mstar"].values, df["z_phot"].values
    )
    df["gamma_t"] = compute_gamma_t(df["log_Mh"].values, df["z_phot"].values)
    df["t_cosmic"] = cosmo.age(df["z_phot"].values).value
    df["t_eff"] = np.clip(df["t_cosmic"] * df["gamma_t"], 1e-4, None)
    df["log_gamma_t"] = np.log10(np.clip(df["gamma_t"].values, 1e-9, None))
    df["sigma_log_Mstar"] = 0.5 * np.abs(df["log_Mstar_u68"] - df["log_Mstar_l68"])
    df["sigma_z_rel"] = 0.5 * np.abs(df["z_u68"] - df["z_l68"]) / np.clip(1.0 + df["z_phot"], 1e-6, None)
    denom = 0.05 + np.square(np.clip(df["sigma_log_Mstar"], 0, None)) + np.square(np.clip(df["sigma_z_rel"], 0, None))
    df["quality_weight"] = 1.0 / np.clip(denom, 1e-6, None)

    return df


def analyze_ssfr_bin(sub, z_lo, z_hi, label, beta_debias, reference_mass_sample=None):
    """Partial rho(Gamma_t, log_sSFR | M*, z) for one redshift bin."""
    n = len(sub)
    if n < 10:
        return {"label": label, "n": n, "note": "insufficient_sample"}

    x = sub["gamma_t"].values
    y = sub["log_ssfr"].values

    # Raw Spearman
    raw_rho, raw_p = spearmanr(x, y)
    raw_p = max(float(raw_p), 1e-300)

    # Partial rho controlling for M* and z
    try:
        p_rho, p_p = partial_spearman(
            x, y,
            controls=[sub["log_Mstar"].values, sub["z_phot"].values]
        )
        p_p = max(float(p_p), 1e-300)
    except Exception:
        p_rho, p_p = float("nan"), float("nan")

    # Bootstrap CI on partial rho
    ci_lo, ci_hi = bootstrap_ci_partial(sub, n_boot=200)

    log_mstar_debiased = sub["log_Mstar"].values - beta_debias * sub["log_gamma_t"].values
    try:
        p_rho_debiased, p_p_debiased = partial_spearman(
            x,
            y,
            controls=[log_mstar_debiased, sub["z_phot"].values],
        )
        p_p_debiased = max(float(p_p_debiased), 1e-300)
    except Exception:
        p_rho_debiased, p_p_debiased = float("nan"), float("nan")
    ci_lo_debiased, ci_hi_debiased = bootstrap_ci_partial(
        sub.assign(log_Mstar_debiased=log_mstar_debiased),
        mass_col="log_Mstar_debiased",
        n_boot=200,
    )

    try:
        w_rho_debiased, w_p_debiased, n_eff = weighted_partial_spearman(
            x,
            y,
            controls=[log_mstar_debiased, sub["z_phot"].values],
            weights=sub["quality_weight"].values,
        )
    except Exception:
        w_rho_debiased, w_p_debiased, n_eff = float("nan"), float("nan"), float("nan")

    ref_mass_weights = None
    ref_mass_rho = float("nan")
    ref_mass_p = float("nan")
    ref_mass_n_eff = float("nan")
    if reference_mass_sample is not None and len(reference_mass_sample) >= 20:
        ref_mass_weights = reference_mass_reweight(
            sub["log_Mstar"].values,
            np.asarray(reference_mass_sample["log_Mstar"].values, dtype=float),
        )
        if ref_mass_weights is not None:
            try:
                ref_mass_rho, ref_mass_p, ref_mass_n_eff = weighted_partial_spearman(
                    x,
                    y,
                    controls=[log_mstar_debiased, sub["z_phot"].values],
                    weights=ref_mass_weights,
                )
            except Exception:
                ref_mass_rho, ref_mass_p, ref_mass_n_eff = float("nan"), float("nan"), float("nan")

    return {
        "label":       label,
        "z_lo":        z_lo,
        "z_hi":        z_hi,
        "n":           n,
        "raw_rho":     float(raw_rho),
        "raw_p":       raw_p,
        "partial_rho": p_rho,
        "partial_p":   p_p,
        "ci_lo":       ci_lo,
        "ci_hi":       ci_hi,
        "partial_rho_debiased_mass": p_rho_debiased,
        "partial_p_debiased_mass": p_p_debiased,
        "ci_lo_debiased_mass": ci_lo_debiased,
        "ci_hi_debiased_mass": ci_hi_debiased,
        "weighted_partial_rho_debiased_mass": w_rho_debiased,
        "weighted_partial_p_debiased_mass": w_p_debiased,
        "weighted_partial_n_eff": float(n_eff),
        "reference_mass_reweighted_partial_rho_debiased_mass": (
            float(ref_mass_rho) if np.isfinite(ref_mass_rho) else None
        ),
        "reference_mass_reweighted_partial_p_debiased_mass": (
            max(float(ref_mass_p), 1e-300) if np.isfinite(ref_mass_p) else None
        ),
        "reference_mass_reweighted_n_eff": (
            float(ref_mass_n_eff) if np.isfinite(ref_mass_n_eff) else None
        ),
        "reference_mass_reweight_source": (
            "z=8-9 log_Mstar distribution"
            if ref_mass_weights is not None
            else None
        ),
        "median_sigma_log_mstar": float(np.nanmedian(sub["sigma_log_Mstar"].values)),
        "median_sigma_z_rel": float(np.nanmedian(sub["sigma_z_rel"].values)),
        "median_quality_weight": float(np.nanmedian(sub["quality_weight"].values)),
    }


def run():
    print_status(f"STEP {STEP_NUM}: COSMOS2025 sSFR inversion (Table B4b)", "INFO")

    beta_debias, beta_source = load_beta_debias()
    df = load_cosmos_catalog()
    if df is None or len(df) == 0:
        result = {
            "step": STEP_NUM, "name": STEP_NAME,
            "status": "FAILED_NO_DATA",
            "note": f"COSMOS-Web FITS not found at {COSMOSWEB_FITS}. "
                    "Run step_033_cosmosweb_download.py first.",
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    # Redshift-binned partial correlations
    bins_results = []
    reference_mass_sample = df[(df["z_phot"] >= 8.0) & (df["z_phot"] < 9.0)].copy()
    for z_lo, z_hi, label in SSFR_BINS:
        sub = df[(df["z_phot"] >= z_lo) & (df["z_phot"] < z_hi)].copy()
        res = analyze_ssfr_bin(
            sub,
            z_lo,
            z_hi,
            label,
            beta_debias=beta_debias,
            reference_mass_sample=reference_mass_sample if z_lo >= 9.0 else None,
        )
        bins_results.append(res)
        logger.info(
            f"  {label}: N={res.get('n',0):,}, "
            f"partial_rho={res.get('partial_rho', float('nan')):.3f}, "
            f"debiased={res.get('partial_rho_debiased_mass', float('nan')):.3f}, "
            f"weighted_debiased={res.get('weighted_partial_rho_debiased_mass', float('nan')):.3f}"
        )

    selection_sensitivity = {
        "reference_mass_reweight_note": "z>=9 bins are additionally reweighted to the z=8-9 stellar-mass distribution as a selection/completeness sensitivity test; these are companion diagnostics, not the primary headline.",
        "z=9-10": analyze_ssfr_bin(
            df[(df["z_phot"] >= 9.0) & (df["z_phot"] < 10.0)].copy(),
            9.0,
            10.0,
            "z=9-10",
            beta_debias=beta_debias,
            reference_mass_sample=reference_mass_sample,
        ),
        "z=10-13": analyze_ssfr_bin(
            df[(df["z_phot"] >= 10.0) & (df["z_phot"] < 13.0)].copy(),
            10.0,
            13.0,
            "z=10-13",
            beta_debias=beta_debias,
            reference_mass_sample=reference_mass_sample,
        ),
        "z=9-13_logMstar_ge_8.5": analyze_ssfr_bin(
            df[(df["z_phot"] >= 9.0) & (df["z_phot"] < 13.0) & (df["log_Mstar"] >= 8.5)].copy(),
            9.0,
            13.0,
            "z=9-13_logMstar_ge_8.5",
            beta_debias=beta_debias,
            reference_mass_sample=reference_mass_sample[reference_mass_sample["log_Mstar"] >= 8.5].copy(),
        ),
        "z=9-13_logMstar_ge_9.0": analyze_ssfr_bin(
            df[(df["z_phot"] >= 9.0) & (df["z_phot"] < 13.0) & (df["log_Mstar"] >= 9.0)].copy(),
            9.0,
            13.0,
            "z=9-13_logMstar_ge_9.0",
            beta_debias=beta_debias,
            reference_mass_sample=reference_mass_sample[reference_mass_sample["log_Mstar"] >= 9.0].copy(),
        ),
    }

    # Extract low-z and high-z partial rhos for Steiger Z-test
    lowz_res  = bins_results[0]  # z=4-7
    highz_res = bins_results[-1] # z=9-13

    rho_low  = lowz_res.get("partial_rho", 0.0) or 0.0
    rho_high = highz_res.get("partial_rho", 0.0) or 0.0

    # Independent-sample Fisher-z difference for low-z vs high-z partial correlations.
    # These are disjoint redshift bins, so this is not a classical Steiger test.
    n_low  = lowz_res.get("n", 1)
    n_high = highz_res.get("n", 1)
    z_low  = np.arctanh(np.clip(rho_low,  -0.999, 0.999))
    z_high = np.arctanh(np.clip(rho_high, -0.999, 0.999))
    se = np.sqrt(1.0 / (n_low - 3) + 1.0 / (n_high - 3))
    steiger_Z_val = (z_high - z_low) / se
    steiger_p = 2 * stats.norm.sf(abs(steiger_Z_val))
    steiger_p = max(float(steiger_p), 1e-300)

    logger.info(
        f"Independent-sample Fisher z-difference (z=9-13 vs z=4-7): "
        f"Z={steiger_Z_val:.2f}, p={steiger_p:.2e}"
    )
    logger.info(
        f"sSFR inversion: rho(low-z)={rho_low:.3f} -> rho(high-z)={rho_high:.3f}, "
        f"delta_rho={rho_high - rho_low:.3f}"
    )

    supportive_bins = [
        row for row in bins_results
        if row.get("z_lo", 0) >= 7.0 and np.isfinite(row.get("partial_rho_debiased_mass", np.nan))
        and row.get("partial_rho_debiased_mass", 0.0) > 0
    ]
    primary_external_bin = next((row for row in bins_results if row.get("label") == "z=8-9"), None)
    ultrahighz_bin = next((row for row in bins_results if row.get("label") == "z=9-13 (high-z)"), None)
    ultrahighz_selection_sensitive = False
    ultrahighz_candidates = [
        selection_sensitivity["z=10-13"],
        selection_sensitivity["z=9-13_logMstar_ge_8.5"],
        selection_sensitivity["z=9-13_logMstar_ge_9.0"],
    ]
    for row in ultrahighz_candidates:
        if row is None:
            continue
        debiased = row.get("partial_rho_debiased_mass")
        ref_reweighted = row.get("reference_mass_reweighted_partial_rho_debiased_mass")
        if (np.isfinite(debiased) and debiased > 0) or (ref_reweighted is not None and np.isfinite(ref_reweighted) and ref_reweighted >= 0):
            ultrahighz_selection_sensitive = True
            break

    activation_scan = [
        {
            "label": row["label"],
            "n": row["n"],
            "partial_rho": row.get("partial_rho"),
            "partial_rho_debiased_mass": row.get("partial_rho_debiased_mass"),
            "weighted_partial_rho_debiased_mass": row.get("weighted_partial_rho_debiased_mass"),
            "reference_mass_reweighted_partial_rho_debiased_mass": row.get("reference_mass_reweighted_partial_rho_debiased_mass"),
            "sign_supportive_after_debias": bool(
                np.isfinite(row.get("partial_rho_debiased_mass", np.nan))
                and row.get("partial_rho_debiased_mass", 0.0) > 0
            ),
        }
        for row in bins_results
    ]

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "COSMOS2025 sSFR inversion — Table B4b",
        "ssfr_bins": bins_results,
        "beta_debias_used": float(beta_debias),
        "beta_debias_source": beta_source,
        "comparison_test": "independent_sample_fisher_z_difference",
        "steiger_Z":  float(steiger_Z_val),
        "steiger_p":  steiger_p,
        "rho_lowz":   rho_low,
        "rho_highz":  rho_high,
        "delta_rho":  float(rho_high - rho_low),
        "rho_lowz_debiased": lowz_res.get("partial_rho_debiased_mass"),
        "rho_highz_debiased": highz_res.get("partial_rho_debiased_mass"),
        "delta_rho_debiased": (
            float(highz_res.get("partial_rho_debiased_mass") - lowz_res.get("partial_rho_debiased_mass"))
            if np.isfinite(lowz_res.get("partial_rho_debiased_mass", np.nan))
            and np.isfinite(highz_res.get("partial_rho_debiased_mass", np.nan))
            else None
        ),
        "activation_scan": activation_scan,
        "selection_sensitivity": selection_sensitivity,
        "external_replication_summary": {
            "n_supportive_bins_after_debias": int(len(supportive_bins)),
            "n_highz_bins_tested": int(sum(row.get("z_lo", 0) >= 7.0 for row in bins_results)),
            "primary_matched_bin": primary_external_bin,
            "ultrahighz_sensitivity_bin": ultrahighz_bin,
            "assessment": (
                "matched_z8_9_supportive_but_ultrahighz_selection_sensitive"
                if primary_external_bin
                and primary_external_bin.get("partial_rho_debiased_mass", 0.0) > 0
                and ultrahighz_selection_sensitive
                else (
                    "matched_z8_9_supportive_but_ultrahighz_mixed"
                    if primary_external_bin and primary_external_bin.get("partial_rho_debiased_mass", 0.0) > 0
                    else "no_supportive_matched_highz_replication"
                )
            ),
        },
        "data_source": "COSMOSWeb_mastercatalog_v1_lephare.fits (Shuntov et al. 2025)",
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(
        f"Step {STEP_NUM} complete. Fisher z-difference={steiger_Z_val:.2f}, p={steiger_p:.2e}", "INFO"
    )
    return result


main = run

if __name__ == "__main__":
    run()
