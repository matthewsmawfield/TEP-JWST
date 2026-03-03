#!/usr/bin/env python3
"""
Step 136: Functional Form Discrimination Test

Tests whether the specific TEP functional form (t_eff = Gamma_t * t_cosmic)
outperforms simpler predictors of dust at z > 8.

Five competing models for predicting dust attenuation:
  Model 1 (Mass-only):     dust ~ M*
  Model 2 (Redshift-only): dust ~ z
  Model 3 (Additive):      dust ~ M* + z
  Model 4 (TEP):           dust ~ t_eff
  Model 5 (TEP threshold): dust_detected = (t_eff > 0.3 Gyr)

Evaluation metrics:
  - Spearman rho (rank correlation with dust)
  - AUC for classifying dust-detected vs non-detected
  - Cross-validated R^2 (5-fold)
  - AIC/BIC for OLS regression models
  - Steiger's Z-test comparing dependent correlations

The key TEP-specific prediction: across the FULL z = 4-10 range, t_eff
should outperform M* or z alone because it captures the z-dependent
activation of the scalar coupling. Within a single z-bin, t_eff and M*
are rank-equivalent (since Gamma_t is monotonic in M*), but across z-bins
the TEP combination provides unique information.

Outputs:
- results/outputs/step_136_functional_form_test.json
- results/figures/figure_160_functional_form_test.png
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "160"
STEP_NAME = "functional_form_test"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Utility functions ────────────────────────────────────────────────────────

def steiger_z_test(rho_jk, rho_jh, rho_kh, n):
    """
    Steiger's Z-test for comparing two dependent correlations sharing a
    common variable (j). Tests H0: rho_jk == rho_jh.

    Parameters
    ----------
    rho_jk : float  — correlation(dust, predictor_1)
    rho_jh : float  — correlation(dust, predictor_2)
    rho_kh : float  — correlation(predictor_1, predictor_2)
    n : int          — sample size

    Returns
    -------
    dict with z_stat and p_value (two-tailed)
    """
    if n < 4:
        return {"z_stat": np.nan, "p_value": np.nan}

    # Fisher z-transform
    z_jk = np.arctanh(np.clip(rho_jk, -0.999, 0.999))
    z_jh = np.arctanh(np.clip(rho_jh, -0.999, 0.999))

    # Meng, Rosenthal & Rubin (1992) variance formula
    r_bar = (rho_jk + rho_jh) / 2.0
    f = (1 - rho_kh) / (2 * max(1 - r_bar**2, 1e-10))
    h = (1 - f * r_bar**2) / max(1 - r_bar**2, 1e-10)

    denom = np.sqrt((2 * (1 - rho_kh)) / max(n - 3, 1) * h)
    z_stat = (z_jk - z_jh) / max(denom, 1e-10)
    p_value = 2 * stats.norm.sf(abs(z_stat))

    return {"z_stat": float(z_stat), "p_value": float(p_value)}


def compute_auc(labels, scores):
    """Compute AUC using the Mann-Whitney U statistic."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u_stat = 0.0
    for p_val in pos:
        u_stat += np.sum(p_val > neg) + 0.5 * np.sum(p_val == neg)
    return float(u_stat / (len(pos) * len(neg)))


def cross_validated_r2(X, y, k=5, seed=42):
    """
    k-fold cross-validated R^2 for OLS regression.
    X: (n, p) design matrix, y: (n,) target.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    ss_res_total = 0.0
    ss_tot_total = 0.0

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # OLS fit with intercept
        X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
        X_test_i = np.column_stack([np.ones(len(X_test)), X_test])

        try:
            beta = np.linalg.lstsq(X_train_i, y_train, rcond=None)[0]
            y_pred = X_test_i @ beta
        except np.linalg.LinAlgError:
            return np.nan

        ss_res_total += np.sum((y_test - y_pred) ** 2)
        ss_tot_total += np.sum((y_test - y_test.mean()) ** 2)

    if ss_tot_total == 0:
        return np.nan
    return float(1 - ss_res_total / ss_tot_total)


def ols_aic_bic(X, y):
    """AIC and BIC for OLS regression. X: (n, p) design matrix."""
    n = len(y)
    X_i = np.column_stack([np.ones(n), X])
    k = X_i.shape[1]

    try:
        beta = np.linalg.lstsq(X_i, y, rcond=None)[0]
        residuals = y - X_i @ beta
        rss = np.sum(residuals ** 2)
        sigma2 = rss / n
        if sigma2 <= 0:
            return np.nan, np.nan, np.nan
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)
        r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
        return float(aic), float(bic), float(r2)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_survey(name, path, z_col, mass_col, dust_col, z_min=4.0, z_max=10.5, mass_min=8.0):
    """Load a survey dataset and compute TEP quantities."""
    df = pd.read_csv(path)

    # Redshift
    if z_col in df.columns:
        df["z"] = df[z_col].astype(float)
    else:
        raise ValueError(f"Column {z_col} not found in {path}")

    # Mass
    if mass_col in df.columns:
        df["log_Mstar"] = df[mass_col].astype(float)
    else:
        raise ValueError(f"Column {mass_col} not found in {path}")

    # Dust
    if dust_col in df.columns:
        df["dust"] = df[dust_col].astype(float)
    else:
        raise ValueError(f"Column {dust_col} not found in {path}")

    # Quality cuts
    df = df[(df["z"] >= z_min) & (df["z"] <= z_max) & (df["log_Mstar"] >= mass_min)].copy()
    df = df.dropna(subset=["z", "log_Mstar", "dust"])

    # Compute TEP quantities
    from astropy.cosmology import Planck18 as cosmo

    df["log_Mh"] = df.apply(
        lambda r: stellar_to_halo_mass_behroozi_like(r["log_Mstar"], r["z"]), axis=1
    )
    df["gamma_t"] = df.apply(
        lambda r: compute_gamma_t(r["log_Mh"], r["z"]), axis=1
    )
    df["t_cosmic"] = df["z"].apply(lambda zz: cosmo.age(zz).value)
    df["t_eff"] = df["gamma_t"] * df["t_cosmic"]
    df["dust_detected"] = (df["dust"] > 0).astype(int)

    df["survey"] = name
    logger.info(f"  {name}: loaded {len(df)} galaxies (z={z_min}-{z_max})")
    return df


def load_all_surveys():
    """Load UNCOVER, CEERS, and COSMOS-Web."""
    surveys = []

    # UNCOVER — has pre-computed TEP columns
    uncover_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        df_u = pd.read_csv(uncover_path)
        df_u["z"] = df_u["z_phot"].astype(float)
        df_u = df_u[(df_u["z"] >= 4.0) & (df_u["z"] <= 10.5) & (df_u["log_Mstar"] >= 8.0)].copy()
        df_u = df_u.dropna(subset=["z", "log_Mstar", "dust"])
        df_u["dust_detected"] = (df_u["dust"] > 0).astype(int)
        df_u["survey"] = "UNCOVER"
        logger.info(f"  UNCOVER: loaded {len(df_u)} galaxies")
        surveys.append(df_u)

    # CEERS
    ceers_path = DATA_INTERIM_PATH / "ceers_z8_sample.csv"
    if ceers_path.exists():
        df_c = load_survey("CEERS", ceers_path, "z_phot", "log_Mstar", "dust",
                           z_min=8.0, z_max=10.5)
        surveys.append(df_c)

    # COSMOS-Web
    cosmo_path = DATA_INTERIM_PATH / "cosmosweb_z8_sample.csv"
    if cosmo_path.exists():
        df_cw = load_survey("COSMOS-Web", cosmo_path, "z_phot", "log_Mstar", "dust",
                            z_min=8.0, z_max=10.5)
        surveys.append(df_cw)

    return pd.concat(surveys, ignore_index=True)


# ── Core Analysis ────────────────────────────────────────────────────────────

def analyze_subsample(df, label):
    """Run the full model comparison on a subsample."""
    n = len(df)
    if n < 20:
        return {"label": label, "n": n, "skipped": True, "reason": "n < 20"}

    dust = df["dust"].values
    log_mstar = df["log_Mstar"].values
    z = df["z"].values
    t_eff = df["t_eff"].values
    t_cosmic = df["t_cosmic"].values
    gamma_t = df["gamma_t"].values
    dust_detected = df["dust_detected"].values

    result = {"label": label, "n": n}

    # ── Spearman correlations ────────────────────────────────────────────
    predictors = {
        "M*": log_mstar,
        "z": z,
        "t_cosmic": t_cosmic,
        "gamma_t": gamma_t,
        "t_eff": t_eff,
    }

    corrs = {}
    for pname, pvals in predictors.items():
        rho, p = spearmanr(dust, pvals)
        corrs[pname] = {"rho": float(rho), "p": max(float(p), 1e-300)}
    result["spearman"] = corrs

    # ── Steiger's Z-tests (dust,t_eff) vs (dust,M*) and vs (dust,t_cosmic) ─
    rho_dust_teff = corrs["t_eff"]["rho"]
    rho_dust_mstar = corrs["M*"]["rho"]
    rho_dust_tcosmic = corrs["t_cosmic"]["rho"]

    # Correlation between predictors
    rho_teff_mstar, _ = spearmanr(t_eff, log_mstar)
    rho_teff_tcosmic, _ = spearmanr(t_eff, t_cosmic)
    rho_mstar_tcosmic, _ = spearmanr(log_mstar, t_cosmic)

    result["steiger_teff_vs_mstar"] = steiger_z_test(
        rho_dust_teff, rho_dust_mstar, rho_teff_mstar, n
    )
    result["steiger_teff_vs_tcosmic"] = steiger_z_test(
        rho_dust_teff, rho_dust_tcosmic, rho_teff_tcosmic, n
    )

    # ── AUC for dust detection ───────────────────────────────────────────
    if dust_detected.sum() > 0 and dust_detected.sum() < n:
        aucs = {}
        for pname, pvals in predictors.items():
            aucs[pname] = compute_auc(dust_detected, pvals)
        # TEP threshold model: predict detected if t_eff > 0.3 Gyr
        teff_thresh_scores = (t_eff > 0.3).astype(float)
        aucs["t_eff_threshold_0.3"] = compute_auc(dust_detected, teff_thresh_scores)
        result["auc"] = aucs
    else:
        result["auc"] = {"note": "all dust values identical (detected or not)"}

    # ── Regression model comparison (AIC/BIC/R²) ────────────────────────
    models = {
        "M*_only": log_mstar.reshape(-1, 1),
        "z_only": z.reshape(-1, 1),
        "M*_plus_z": np.column_stack([log_mstar, z]),
        "M*_times_z": np.column_stack([log_mstar, z, log_mstar * z]),
        "t_eff": t_eff.reshape(-1, 1),
        "gamma_t": gamma_t.reshape(-1, 1),
        "t_cosmic": t_cosmic.reshape(-1, 1),
    }

    regression = {}
    for mname, X in models.items():
        aic, bic, r2 = ols_aic_bic(X, dust)
        cv_r2 = cross_validated_r2(X, dust, k=5)
        regression[mname] = {
            "aic": aic, "bic": bic, "r2_train": r2, "r2_cv": cv_r2,
            "n_params": X.shape[1] + 1,  # +1 for intercept
        }
    result["regression"] = regression

    # ── Rank the models ──────────────────────────────────────────────────
    model_ranking = sorted(
        [(k, v["aic"]) for k, v in regression.items() if not np.isnan(v["aic"])],
        key=lambda x: x[1]
    )
    result["model_ranking_aic"] = [
        {"model": k, "aic": v, "delta_aic": v - model_ranking[0][1]}
        for k, v in model_ranking
    ]

    return result


def analyze_redshift_evolution(df):
    """
    The KEY test: does the mass-dust correlation evolve with z in the way
    TEP predicts (proportional to alpha(z) ~ sqrt(1+z))?
    """
    z_bins = [(4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0)]
    results = []

    for z_lo, z_hi in z_bins:
        sub = df[(df["z"] >= z_lo) & (df["z"] < z_hi)]
        n = len(sub)
        if n < 10:
            results.append({
                "z_range": [z_lo, z_hi], "n": n, "skipped": True
            })
            continue

        rho_mstar, p_mstar = spearmanr(sub["dust"], sub["log_Mstar"])
        rho_teff, p_teff = spearmanr(sub["dust"], sub["t_eff"])
        rho_gammat, p_gammat = spearmanr(sub["dust"], sub["gamma_t"])
        rho_tcosmic, p_tcosmic = spearmanr(sub["dust"], sub["t_cosmic"])

        # Within a z-bin, gamma_t is monotonic in M*, so rho should be similar
        # The test is: does the MAGNITUDE change with z as TEP predicts?
        z_mid = (z_lo + z_hi) / 2
        alpha_predicted = 0.58 * np.sqrt(1 + z_mid)

        results.append({
            "z_range": [z_lo, z_hi],
            "z_mid": z_mid,
            "n": n,
            "alpha_predicted": float(alpha_predicted),
            "rho_dust_mstar": float(rho_mstar),
            "p_dust_mstar": float(p_mstar),
            "rho_dust_teff": float(rho_teff),
            "p_dust_teff": float(p_teff),
            "rho_dust_gammat": float(rho_gammat),
            "p_dust_gammat": float(p_gammat),
            "rho_dust_tcosmic": float(rho_tcosmic),
            "p_dust_tcosmic": float(p_tcosmic),
        })

    # Test: does |rho_dust_mstar| correlate with alpha_predicted?
    valid = [r for r in results if not r.get("skipped", False)]
    if len(valid) >= 4:
        alphas = [r["alpha_predicted"] for r in valid]
        rhos = [r["rho_dust_mstar"] for r in valid]
        rho_evolution, p_evolution = spearmanr(alphas, rhos)
        evolution_test = {
            "rho_alpha_vs_rho_dust_mstar": float(rho_evolution),
            "p_value": float(p_evolution),
            "interpretation": (
                "TEP predicts the mass-dust correlation strengthens with "
                "alpha(z) ~ sqrt(1+z). A positive correlation here supports TEP."
            )
        }
    else:
        evolution_test = {"skipped": True, "reason": "insufficient z-bins"}

    return {"z_bins": results, "evolution_test": evolution_test}


def analyze_agb_threshold(df, threshold_gyr=0.3):
    """
    Test the TEP-specific AGB threshold prediction:
    galaxies with t_eff > threshold should be dusty; others should not.
    This is a NON-LINEAR prediction a smooth mass proxy cannot replicate.
    """
    sub = df[df["z"] >= 8.0].copy()
    n = len(sub)
    if n < 20:
        return {"n": n, "skipped": True}

    above = sub[sub["t_eff"] > threshold_gyr]
    below = sub[sub["t_eff"] <= threshold_gyr]

    n_above = len(above)
    n_below = len(below)

    if n_above < 5 or n_below < 5:
        return {"n": n, "n_above": n_above, "n_below": n_below, "skipped": True}

    det_above = (above["dust"] > 0).sum()
    det_below = (below["dust"] > 0).sum()
    frac_above = det_above / n_above
    frac_below = det_below / n_below

    mean_dust_above = above["dust"].mean()
    mean_dust_below = below["dust"].mean()

    # Fisher exact test
    table = np.array([[det_above, n_above - det_above],
                      [det_below, n_below - det_below]])
    odds_ratio, p_fisher = stats.fisher_exact(table, alternative="greater")

    # Mann-Whitney U
    u_stat, p_mw = stats.mannwhitneyu(above["dust"], below["dust"], alternative="greater")

    # Compare with M*-based threshold at same quantile
    teff_quantile = (sub["t_eff"] <= threshold_gyr).mean()
    mstar_threshold = np.quantile(sub["log_Mstar"], teff_quantile)

    above_m = sub[sub["log_Mstar"] > mstar_threshold]
    below_m = sub[sub["log_Mstar"] <= mstar_threshold]

    det_above_m = (above_m["dust"] > 0).sum()
    det_below_m = (below_m["dust"] > 0).sum()
    frac_above_m = det_above_m / max(len(above_m), 1)
    frac_below_m = det_below_m / max(len(below_m), 1)
    mean_dust_above_m = above_m["dust"].mean()
    mean_dust_below_m = below_m["dust"].mean()

    table_m = np.array([[det_above_m, len(above_m) - det_above_m],
                        [det_below_m, len(below_m) - det_below_m]])
    try:
        odds_ratio_m, p_fisher_m = stats.fisher_exact(table_m, alternative="greater")
    except ValueError:
        odds_ratio_m, p_fisher_m = np.nan, np.nan
    u_stat_m, p_mw_m = stats.mannwhitneyu(above_m["dust"], below_m["dust"], alternative="greater")

    return {
        "threshold_gyr": threshold_gyr,
        "n": n,
        "teff_threshold": {
            "n_above": int(n_above), "n_below": int(n_below),
            "det_frac_above": float(frac_above), "det_frac_below": float(frac_below),
            "delta_frac": float(frac_above - frac_below),
            "mean_dust_above": float(mean_dust_above),
            "mean_dust_below": float(mean_dust_below),
            "dust_ratio": float(mean_dust_above / max(mean_dust_below, 1e-10)),
            "odds_ratio": float(odds_ratio) if not np.isnan(odds_ratio) else None,
            "p_fisher": float(p_fisher),
            "p_mannwhitney": float(p_mw),
        },
        "mstar_threshold_comparison": {
            "mstar_threshold": float(mstar_threshold),
            "quantile_matched": float(teff_quantile),
            "n_above": int(len(above_m)), "n_below": int(len(below_m)),
            "det_frac_above": float(frac_above_m), "det_frac_below": float(frac_below_m),
            "delta_frac": float(frac_above_m - frac_below_m),
            "mean_dust_above": float(mean_dust_above_m),
            "mean_dust_below": float(mean_dust_below_m),
            "dust_ratio": float(mean_dust_above_m / max(mean_dust_below_m, 1e-10)),
            "odds_ratio": float(odds_ratio_m) if not np.isnan(odds_ratio_m) else None,
            "p_fisher": float(p_fisher_m) if not np.isnan(p_fisher_m) else None,
            "p_mannwhitney": float(p_mw_m),
        },
        "teff_vs_mstar_comparison": {
            "teff_delta_frac": float(frac_above - frac_below),
            "mstar_delta_frac": float(frac_above_m - frac_below_m),
            "teff_better": bool((frac_above - frac_below) > (frac_above_m - frac_below_m)),
            "teff_dust_ratio": float(mean_dust_above / max(mean_dust_below, 1e-10)),
            "mstar_dust_ratio": float(mean_dust_above_m / max(mean_dust_below_m, 1e-10)),
        }
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(results, output_path):
    """Generate summary figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figure")
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Step 136: Functional Form Discrimination Test", fontsize=14, fontweight="bold")

    # Panel A: Spearman rho comparison across subsamples
    ax = axes[0, 0]
    subsample_labels = []
    rho_mstar_vals = []
    rho_teff_vals = []
    rho_tcosmic_vals = []
    for sub in results.get("subsamples", []):
        if sub.get("skipped"):
            continue
        subsample_labels.append(sub["label"])
        rho_mstar_vals.append(sub["spearman"]["M*"]["rho"])
        rho_teff_vals.append(sub["spearman"]["t_eff"]["rho"])
        rho_tcosmic_vals.append(sub["spearman"]["t_cosmic"]["rho"])

    x = np.arange(len(subsample_labels))
    w = 0.25
    ax.bar(x - w, rho_mstar_vals, w, label="M*", color="#4ECDC4", alpha=0.8)
    ax.bar(x, rho_teff_vals, w, label="t_eff (TEP)", color="#FF6B6B", alpha=0.8)
    ax.bar(x + w, rho_tcosmic_vals, w, label="t_cosmic", color="#95E1D3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(subsample_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ(dust, predictor)")
    ax.set_title("A: Correlation Comparison")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.5)

    # Panel B: AUC comparison
    ax = axes[0, 1]
    auc_labels = []
    auc_mstar = []
    auc_teff = []
    auc_tcosmic = []
    for sub in results.get("subsamples", []):
        if sub.get("skipped") or isinstance(sub.get("auc"), dict) and "note" in sub.get("auc", {}):
            continue
        auc_data = sub.get("auc", {})
        if "M*" in auc_data and "t_eff" in auc_data and "t_cosmic" in auc_data:
            auc_labels.append(sub["label"])
            auc_mstar.append(auc_data["M*"])
            auc_teff.append(auc_data["t_eff"])
            auc_tcosmic.append(auc_data["t_cosmic"])

    if auc_labels:
        x = np.arange(len(auc_labels))
        ax.bar(x - w, auc_mstar, w, label="M*", color="#4ECDC4", alpha=0.8)
        ax.bar(x, auc_teff, w, label="t_eff (TEP)", color="#FF6B6B", alpha=0.8)
        ax.bar(x + w, auc_tcosmic, w, label="t_cosmic", color="#95E1D3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(auc_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("AUC (dust detected)")
        ax.axhline(0.5, color="k", lw=0.5, ls="--", label="chance")
        ax.set_title("B: Classification AUC")
        ax.legend(fontsize=8)

    # Panel C: Redshift evolution of mass-dust correlation
    ax = axes[1, 0]
    z_evo = results.get("redshift_evolution", {})
    z_bins = z_evo.get("z_bins", [])
    valid_bins = [b for b in z_bins if not b.get("skipped", False)]
    if valid_bins:
        z_mids = [b["z_mid"] for b in valid_bins]
        rhos_mstar = [b["rho_dust_mstar"] for b in valid_bins]
        rhos_gammat = [b["rho_dust_gammat"] for b in valid_bins]
        ax.plot(z_mids, rhos_mstar, "o-", color="#4ECDC4", label="ρ(dust, M*)", markersize=8)
        ax.plot(z_mids, rhos_gammat, "s--", color="#FF6B6B", label="ρ(dust, Γt)", markersize=8)
        ax.axhline(0, color="k", lw=0.5)

        # Overlay TEP prediction shape (normalized)
        z_plot = np.linspace(4, 10, 50)
        alpha_norm = np.sqrt(1 + z_plot) / np.sqrt(1 + 4)
        scale = max(rhos_mstar) / max(alpha_norm) if max(alpha_norm) > 0 else 1
        ax.plot(z_plot, alpha_norm * scale * 0.5, ":", color="gray", alpha=0.5,
                label="TEP α(z) shape")
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Spearman ρ(dust, predictor)")
        ax.set_title("C: Redshift Evolution of Mass–Dust Correlation")
        ax.legend(fontsize=8)

    # Panel D: AGB threshold comparison
    ax = axes[1, 1]
    agb = results.get("agb_threshold", {})
    if not agb.get("skipped", False):
        teff_data = agb.get("teff_threshold", {})
        mstar_data = agb.get("mstar_threshold_comparison", {})

        categories = ["t_eff threshold\n(TEP: 0.3 Gyr)", "M* threshold\n(quantile-matched)"]
        above_fracs = [teff_data.get("det_frac_above", 0), mstar_data.get("det_frac_above", 0)]
        below_fracs = [teff_data.get("det_frac_below", 0), mstar_data.get("det_frac_below", 0)]

        x = np.arange(2)
        ax.bar(x - 0.15, above_fracs, 0.3, label="Above threshold", color="#FF6B6B", alpha=0.8)
        ax.bar(x + 0.15, below_fracs, 0.3, label="Below threshold", color="#4ECDC4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Dust detection fraction")
        ax.set_title("D: AGB Threshold vs Mass Threshold (z > 8)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.section("Step 136: Functional Form Discrimination Test")

    # Load data
    logger.info("Loading surveys...")
    df = load_all_surveys()
    logger.info(f"Total: {len(df)} galaxies across {df['survey'].nunique()} surveys")

    results = {
        "step": "Step 136: Functional Form Discrimination Test",
        "total_n": len(df),
        "surveys": list(df["survey"].unique()),
    }

    # ── Subsample analyses ───────────────────────────────────────────────
    subsamples = [
        ("Full (z=4-10)", df),
        ("z > 8 (all)", df[df["z"] >= 8.0]),
        ("z > 8 (dust>0)", df[(df["z"] >= 8.0) & (df["dust"] > 0)]),
        ("z = 4-7", df[(df["z"] >= 4.0) & (df["z"] < 7.0)]),
        ("UNCOVER z>8", df[(df["survey"] == "UNCOVER") & (df["z"] >= 8.0)]),
        ("CEERS z>8", df[(df["survey"] == "CEERS") & (df["z"] >= 8.0)]),
        ("COSMOS-Web z>8", df[(df["survey"] == "COSMOS-Web") & (df["z"] >= 8.0)]),
    ]

    subsample_results = []
    for label, sub_df in subsamples:
        logger.info(f"\nAnalyzing: {label} (N={len(sub_df)})")
        res = analyze_subsample(sub_df, label)
        subsample_results.append(res)

        if not res.get("skipped"):
            # Log key comparisons
            sp = res["spearman"]
            logger.info(f"  ρ(dust, M*) = {sp['M*']['rho']:.3f}")
            logger.info(f"  ρ(dust, t_eff) = {sp['t_eff']['rho']:.3f}")
            logger.info(f"  ρ(dust, t_cosmic) = {sp['t_cosmic']['rho']:.3f}")

            steiger = res.get("steiger_teff_vs_mstar", {})
            if not np.isnan(steiger.get("z_stat", np.nan)):
                logger.info(f"  Steiger Z (t_eff vs M*): z={steiger['z_stat']:.2f}, p={steiger['p_value']:.4f}")

    results["subsamples"] = subsample_results

    # ── Redshift evolution ───────────────────────────────────────────────
    logger.section("Redshift Evolution Analysis")
    # Use only UNCOVER for evolution (it spans z=4-10)
    uncover = df[df["survey"] == "UNCOVER"]
    results["redshift_evolution"] = analyze_redshift_evolution(uncover)

    evo = results["redshift_evolution"]["evolution_test"]
    if not evo.get("skipped"):
        logger.info(f"  ρ(α_predicted, ρ_dust_mstar) = {evo['rho_alpha_vs_rho_dust_mstar']:.3f}")
        logger.info(f"  p = {evo['p_value']:.4f}")

    # ── AGB threshold test ───────────────────────────────────────────────
    logger.section("AGB Threshold Discrimination")
    results["agb_threshold"] = analyze_agb_threshold(df, threshold_gyr=0.3)

    agb = results["agb_threshold"]
    if not agb.get("skipped"):
        teff_d = agb["teff_threshold"]
        mstar_d = agb["mstar_threshold_comparison"]
        logger.info(f"  t_eff threshold: det_frac above={teff_d['det_frac_above']:.3f}, below={teff_d['det_frac_below']:.3f}")
        logger.info(f"  M* threshold:    det_frac above={mstar_d['det_frac_above']:.3f}, below={mstar_d['det_frac_below']:.3f}")
        logger.info(f"  t_eff dust ratio: {teff_d['dust_ratio']:.1f}x")
        logger.info(f"  M* dust ratio:   {mstar_d['dust_ratio']:.1f}x")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.section("Summary")

    # Extract the key z>8 (all) result
    z8_all = next((s for s in subsample_results if s["label"] == "z > 8 (all)"), None)
    if z8_all and not z8_all.get("skipped"):
        ranking = z8_all.get("model_ranking_aic", [])
        best_model = ranking[0]["model"] if ranking else "unknown"
        logger.info(f"  Best model (AIC) at z>8: {best_model}")

        for r in ranking[:4]:
            logger.info(f"    {r['model']}: ΔAIC = {r['delta_aic']:.1f}")

        results["headline"] = {
            "best_model_z8_aic": best_model,
            "rho_dust_teff_z8": z8_all["spearman"]["t_eff"]["rho"],
            "rho_dust_mstar_z8": z8_all["spearman"]["M*"]["rho"],
            "rho_dust_tcosmic_z8": z8_all["spearman"]["t_cosmic"]["rho"],
            "auc_teff_z8": z8_all.get("auc", {}).get("t_eff"),
            "auc_mstar_z8": z8_all.get("auc", {}).get("M*"),
            "auc_tcosmic_z8": z8_all.get("auc", {}).get("t_cosmic"),
            "steiger_teff_vs_mstar": z8_all.get("steiger_teff_vs_mstar"),
        }

    # ── Save results ─────────────────────────────────────────────────────
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    logger.info(f"\nResults saved to {output_file}")

    # ── Generate figure ──────────────────────────────────────────────────
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    if make_figure(results, fig_path):
        logger.info(f"Figure saved to {fig_path}")
        results["figure"] = str(fig_path)

    print_status(f"Step {STEP_NUM} ({STEP_NAME}): PASS")
    return results


if __name__ == "__main__":
    main()
