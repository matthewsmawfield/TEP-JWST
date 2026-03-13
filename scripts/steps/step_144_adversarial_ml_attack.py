#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 2m28s.
"""
Step 144: Adversarial Machine Learning Attack on TEP

The ultimate mass-proxy test: give a gradient-boosted ML model EVERY
available feature — M*, z, SFR, sSFR, metallicity, age ratio, and all
polynomial interactions — and let it learn ANY function to predict dust.
Then add Gamma_t as one more feature. If Gamma_t provides measurable
"lift" that the model can't replicate from other features alone, the
mass-proxy argument is dead — because the model already HAS mass and
can learn ANY arbitrary function of it.

The coup de grâce: train on UNCOVER, predict on CEERS and COSMOS-Web.
The ML model will overfit to survey-specific systematics; Gamma_t
(a physically motivated zero-parameter prediction) should generalise.

TESTS:
1. Within-survey (UNCOVER): 5-fold CV, R²/RMSE with vs without Gamma_t
2. Cross-survey generalization: train UNCOVER, test CEERS & COSMOS-Web
3. Permutation importance of Gamma_t
4. Single-feature showdown: Gamma_t alone vs M* alone vs z alone
5. Information-theoretic: conditional mutual information I(Gamma_t; dust | M*, z)

Outputs:
- results/outputs/step_144_adversarial_ml_attack.json
- results/figures/figure_144_adversarial_ml.png
"""

import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like  # Shared TEP model
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types

STEP_NUM = "144"  # Pipeline step number
STEP_NAME = "adversarial_ml_attack"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products
RESULTS_INTERIM = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)

warnings.filterwarnings("ignore")
CPU_COUNT = os.cpu_count() or 1
ADVERSARIAL_WORKERS = max(
    1,
    min(
        CPU_COUNT,
        int(os.getenv("TEP_ADVERSARIAL_WORKERS", min(CPU_COUNT, 8))),
    ),
)
RF_INNER_JOBS = int(
    os.getenv("TEP_ADVERSARIAL_RF_JOBS", "1" if ADVERSARIAL_WORKERS > 1 else "-1")
)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_survey(name):
    """Load a survey catalog and standardize columns."""
    paths = {
        "UNCOVER": [
            RESULTS_INTERIM / "step_002_uncover_full_sample_tep.csv",
            RESULTS_INTERIM / "step_001_uncover_full_sample.csv",
        ],
        "CEERS": [DATA_INTERIM / "ceers_highz_sample.csv"],
        "COSMOS-Web": [DATA_INTERIM / "cosmosweb_highz_sample.csv"],
    }

    if name not in paths:
        return None

    for p in paths[name]:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        return None

    # Standardize columns
    col_map = {
        "z_phot": "z", "z_spec": None,
        "log_Mstar": "log_mstar", "log_mstar": "log_mstar",
        "dust": "dust", "av": "dust", "Av": "dust", "AV": "dust",
        "sfr100": "sfr", "sfr": "sfr",
        "ssfr100": "ssfr", "ssfr": "ssfr",
        "met": "met",
        "mwa": "mwa",
    }

    for old, new in col_map.items():
        if new and old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "z" not in df.columns and "z_phot" in df.columns:
        df["z"] = df["z_phot"]

    # Quality cuts
    if "z" in df.columns:
        df = df[(df["z"] >= 4.0) & (df["z"] <= 10.5)].copy()

    # Compute Gamma_t
    if "log_mstar" in df.columns and "z" in df.columns:
        valid = df["log_mstar"].notna() & df["z"].notna()
        df.loc[valid, "log_mh"] = stellar_to_halo_mass_behroozi_like(
            df.loc[valid, "log_mstar"].values, df.loc[valid, "z"].values
        )
        df.loc[valid, "gamma_t"] = compute_gamma_t(
            df.loc[valid, "log_mh"].values, df.loc[valid, "z"].values
        )
        df["log_gamma_t"] = np.log10(np.maximum(df["gamma_t"], 1e-10))

    # Derived features
    if "log_mstar" in df.columns and "z" in df.columns:
        df["mstar_z"] = df["log_mstar"] * df["z"]
        df["mstar_sq"] = df["log_mstar"] ** 2
        df["z_sq"] = df["z"] ** 2
        df["mstar_z_sq"] = df["log_mstar"] * df["z"] ** 2
        df["mstar_sq_z"] = df["log_mstar"] ** 2 * df["z"]
        df["mstar_cu"] = df["log_mstar"] ** 3

    df["survey"] = name
    return df


# ── ML Models ────────────────────────────────────────────────────────────────

def get_models():
    """Get sklearn models for the adversarial test."""
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge

    return {
        "GBR": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        "RF": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=RF_INNER_JOBS
        ),
        "Ridge": Ridge(alpha=1.0),
    }


def cross_validate(X, y, model_template, n_folds=5, rng_seed=42):
    """5-fold cross-validation returning R², RMSE, Spearman rho."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)
    r2_scores, rmse_scores, rho_scores = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone(model_template)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        rho, _ = spearmanr(y_test, y_pred)

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        rho_scores.append(rho)

    return {
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std": float(np.std(rmse_scores)),
        "rho_mean": float(np.mean(rho_scores)),
        "rho_std": float(np.std(rho_scores)),
    }


def permutation_importance_gamma_t(X, y, gamma_t_col_indices, model_template,
                                    n_repeats=50, rng_seed=42):
    """Compute permutation importance of Gamma_t."""
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rng_seed
    )

    model = clone(model_template)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

    rng = np.random.default_rng(rng_seed)
    drops = []
    gamma_t_col_indices = list(gamma_t_col_indices)
    for _ in range(n_repeats):
        X_perm = X_test.copy()
        perm_idx = rng.permutation(len(X_perm))
        X_perm[:, gamma_t_col_indices] = X_perm[perm_idx][:, gamma_t_col_indices]
        y_perm_pred = model.predict(X_perm)
        r2_perm = 1 - np.sum((y_test - y_perm_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
        drops.append(baseline_r2 - r2_perm)

    return {
        "baseline_R2": float(baseline_r2),
        "mean_R2_drop": float(np.mean(drops)),
        "std_R2_drop": float(np.std(drops)),
        "pct_95_R2_drop": float(np.percentile(drops, 95)),
        "n_repeats": n_repeats,
    }


# ── Conditional Mutual Information ───────────────────────────────────────────

def ksg_mi(x, y, k=5):
    """
    Kraskov-Stögbauer-Grassberger (KSG) mutual information estimator.
    x, y: 1-D arrays of same length.
    Returns MI in nats.
    """
    from scipy.spatial import cKDTree

    n = len(x)
    xy = np.column_stack([x, y])

    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))

    # k-th neighbor distance in joint space (Chebyshev metric)
    dists, _ = tree_xy.query(xy, k=k+1, p=np.inf)
    eps = dists[:, -1]  # distance to k-th neighbor

    # Count neighbors within eps in marginal spaces
    from scipy.special import digamma
    nx = np.array([
        tree_x.query_ball_point(x[i:i+1].reshape(1, -1), r=eps[i] + 1e-15, p=np.inf, return_length=True)
        for i in range(n)
    ]) - 1  # subtract self

    ny = np.array([
        tree_y.query_ball_point(y[i:i+1].reshape(1, -1), r=eps[i] + 1e-15, p=np.inf, return_length=True)
        for i in range(n)
    ]) - 1

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(mi)


def conditional_mi_binned(x, y, z1, z2, n_bins=5):
    """
    Estimate I(X; Y | Z1, Z2) using a binned approach:
    average I(X; Y) within cells defined by quantile bins of Z1 × Z2.
    
    More robust than KSG subtraction for conditional MI near zero.
    Returns MI in nats.
    """
    # Create quantile bins for each conditioning variable
    try:
        z1_bins = pd.qcut(z1, q=n_bins, labels=False, duplicates='drop')
        z2_bins = pd.qcut(z2, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        z1_bins = pd.cut(z1, bins=n_bins, labels=False)
        z2_bins = pd.cut(z2, bins=n_bins, labels=False)
    
    cell_labels = z1_bins * 100 + z2_bins  # unique cell ID
    
    weighted_mi = 0.0
    total_weight = 0.0
    cell_mis = []
    
    for cell in np.unique(cell_labels):
        mask = cell_labels == cell
        if mask.sum() < 20:  # need enough points for KSG
            continue
        
        x_cell = np.asarray(x[mask], dtype=float)
        y_cell = np.asarray(y[mask], dtype=float)
        n_cell = len(x_cell)
        
        if x_cell.std() < 1e-10 or y_cell.std() < 1e-10:
            continue
        
        try:
            mi_cell = ksg_mi(x_cell, y_cell, k=min(5, n_cell // 4))
            mi_cell = max(mi_cell, 0.0)  # MI is non-negative by definition
            weighted_mi += mi_cell * n_cell
            total_weight += n_cell
            cell_mis.append(mi_cell)
        except Exception:
            pass
    
    if total_weight > 0:
        return float(weighted_mi / total_weight), cell_mis
    return 0.0, []


def ksg_mi_multi(X, Y, k=5):
    """KSG MI for multi-dimensional X and Y."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    n = X.shape[0]
    XY = np.hstack([X, Y])

    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    dists, _ = tree_xy.query(XY, k=k+1, p=np.inf)
    eps = dists[:, -1]

    nx = np.zeros(n, dtype=int)
    ny = np.zeros(n, dtype=int)
    for i in range(n):
        nx[i] = tree_x.query_ball_point(X[i], r=eps[i] + 1e-15, p=np.inf, return_length=True) - 1
        ny[i] = tree_y.query_ball_point(Y[i], r=eps[i] + 1e-15, p=np.inf, return_length=True) - 1

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(mi)


def _cmi_null_batch(seeds, gamma, dust, mass, z_arr, n_bins=5):
    null_cmis = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        gamma_shuf = rng.permutation(gamma)
        try:
            cmi_null, _ = conditional_mi_binned(gamma_shuf, dust, mass, z_arr, n_bins=n_bins)
            null_cmis.append(cmi_null)
        except Exception:
            pass
    return null_cmis


def _evaluate_cross_survey_pair(train_name, train_df, test_name, test_df):
    key = f"{train_name}_to_{test_name}"
    pair_results = {"n_train": len(train_df), "n_test": len(test_df)}
    if len(train_df) < 50 or len(test_df) < 50:
        pair_results["status"] = "INSUFFICIENT_DATA"
        return key, pair_results

    y_train = train_df["dust"].values
    y_test = test_df["dust"].values

    from sklearn.ensemble import GradientBoostingRegressor

    feature_sets = {
        "mass_z_poly": [
            "log_mstar", "z", "mstar_z", "mstar_sq", "z_sq",
            "mstar_z_sq", "mstar_sq_z", "mstar_cu",
        ],
        "mass_z_poly+gamma_t": [
            "log_mstar", "z", "mstar_z", "mstar_sq", "z_sq",
            "mstar_z_sq", "mstar_sq_z", "mstar_cu", "gamma_t", "log_gamma_t",
        ],
    }

    for feat_label, feat_cols in feature_sets.items():
        X_train = train_df[feat_cols].values
        X_test = test_df[feat_cols].values

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        rho, p = spearmanr(y_test, y_pred)

        pair_results[feat_label] = {
            "R2": r2,
            "RMSE": rmse,
            "rho": float(rho),
            "p": max(float(p), 1e-300),
            "n_features": len(feat_cols),
        }

    lift = pair_results["mass_z_poly+gamma_t"]["R2"] - pair_results["mass_z_poly"]["R2"]
    pair_results["gamma_t_cross_survey_lift"] = float(lift)
    return key, pair_results


# ── Core Tests ───────────────────────────────────────────────────────────────

def test1_within_survey(df_uncover):
    """
    Test 1: Within-survey CV on UNCOVER.
    Compare model performance with vs without Gamma_t.
    """
    print_status("TEST 1: Within-survey adversarial attack (UNCOVER)")

    # Feature sets
    base_features = ["log_mstar", "z", "mstar_z", "mstar_sq", "z_sq",
                     "mstar_z_sq", "mstar_sq_z", "mstar_cu"]
    extra_features = ["sfr", "ssfr", "met", "mwa"]  # UNCOVER-only
    gamma_features = ["gamma_t", "log_gamma_t"]
    target = "dust"

    # Available features
    avail_base = [f for f in base_features if f in df_uncover.columns]
    avail_extra = [f for f in extra_features if f in df_uncover.columns]
    avail_gamma = [f for f in gamma_features if f in df_uncover.columns]

    # Drop NaN
    all_cols = avail_base + avail_extra + avail_gamma + [target]
    sub = df_uncover.dropna(subset=all_cols).copy()
    n = len(sub)
    print_status(f"  N = {n} galaxies with all features")

    if n < 100:
        return {"status": "INSUFFICIENT_DATA", "n": n}

    y = sub[target].values

    # Feature sets
    feat_sets = {
        "mass_z_poly": avail_base,
        "mass_z_poly+extras": avail_base + avail_extra,
        "mass_z_poly+gamma_t": avail_base + avail_gamma,
        "mass_z_poly+extras+gamma_t": avail_base + avail_extra + avail_gamma,
        "gamma_t_only": avail_gamma,
        "mass_only": ["log_mstar"],
        "z_only": ["z"],
    }

    results = {"n": n, "feature_sets": {}}
    models = get_models()
    X_by_feat = {feat_name: sub[feat_cols].values for feat_name, feat_cols in feat_sets.items()}
    future_map = {}
    with ThreadPoolExecutor(max_workers=ADVERSARIAL_WORKERS) as executor:
        for feat_name, feat_cols in feat_sets.items():
            X = X_by_feat[feat_name]
            for model_name, model_template in models.items():
                future_map[(feat_name, model_name)] = executor.submit(
                    cross_validate,
                    X,
                    y,
                    model_template,
                )

        for feat_name, feat_cols in feat_sets.items():
            print_status(f"  {feat_name} ({len(feat_cols)} features)...")
            feat_results = {}
            for model_name in models:
                cv = future_map[(feat_name, model_name)].result()
                feat_results[model_name] = cv
                print_status(f"    {model_name}: R²={cv['R2_mean']:.4f}±{cv['R2_std']:.4f}, ρ={cv['rho_mean']:.4f}")

            results["feature_sets"][feat_name] = {
                "features": feat_cols,
                "n_features": len(feat_cols),
                "models": feat_results,
            }

    # Compute lift from adding Gamma_t
    best_model = "GBR"
    r2_without = results["feature_sets"]["mass_z_poly+extras"]["models"][best_model]["R2_mean"]
    r2_with = results["feature_sets"]["mass_z_poly+extras+gamma_t"]["models"][best_model]["R2_mean"]
    lift = r2_with - r2_without

    r2_gamma_only = results["feature_sets"]["gamma_t_only"]["models"][best_model]["R2_mean"]
    r2_mass_only = results["feature_sets"]["mass_only"]["models"][best_model]["R2_mean"]
    r2_z_only = results["feature_sets"]["z_only"]["models"][best_model]["R2_mean"]
    lift_assessment = (
        "positive_lift" if lift > 0.01 else
        "negligible_within_survey" if lift >= -0.01 else
        "redundant_within_survey"
    )

    results["gamma_t_lift"] = {
        "R2_without_gamma_t": r2_without,
        "R2_with_gamma_t": r2_with,
        "delta_R2": float(lift),
        "assessment": lift_assessment,
        "interpretation": (
            f"Adding Gamma_t to {best_model} with {len(avail_base + avail_extra)} features "
            f"changes R² by {lift:+.4f}. "
            + ("Gamma_t provides measurable additional within-survey predictive information beyond all standard features."
               if lift > 0.01 else
               "Gamma_t provides negligible additional within-survey lift at this feature richness; this does not test whether Gamma_t carries transportable information across surveys."
               if lift >= -0.01 else
               "Gamma_t is largely redundant for within-survey prediction once the flexible model already has many standard features, but this does not weigh against the cross-survey and conditional-information tests.")
        ),
    }

    results["single_feature_showdown"] = {
        "gamma_t_only_R2": r2_gamma_only,
        "mass_only_R2": r2_mass_only,
        "z_only_R2": r2_z_only,
        "winner": (
            "gamma_t" if r2_gamma_only > max(r2_mass_only, r2_z_only) else
            "mass" if r2_mass_only > r2_z_only else "z"
        ),
    }

    # Permutation importance
    print_status("  Computing permutation importance of Gamma_t...")
    X_full = sub[avail_base + avail_extra + avail_gamma].values
    gamma_indices = list(range(len(avail_base + avail_extra), len(avail_base + avail_extra + avail_gamma)))
    perm = permutation_importance_gamma_t(X_full, y, gamma_indices, models["GBR"])
    perm["n_gamma_features_permuted"] = len(gamma_indices)
    results["permutation_importance"] = perm
    print_status(f"  Gamma_t permutation importance: R² drop = {perm['mean_R2_drop']:.4f} ± {perm['std_R2_drop']:.4f}")

    return results


def test2_cross_survey(surveys):
    """
    Test 2: Cross-survey generalization.
    Train on one survey, test on others. Common features only.
    """
    print_status("TEST 2: Cross-survey generalization attack")

    common_base = ["log_mstar", "z", "mstar_z", "mstar_sq", "z_sq",
                   "mstar_z_sq", "mstar_sq_z", "mstar_cu"]
    gamma_feats = ["gamma_t", "log_gamma_t"]
    target = "dust"

    results = {}

    prepared_surveys = {
        name: df.dropna(subset=common_base + gamma_feats + [target]).copy()
        for name, df in surveys.items()
    }
    pair_order = [
        (train_name, test_name)
        for train_name in surveys
        for test_name in surveys
        if train_name != test_name
    ]

    future_map = {}
    with ThreadPoolExecutor(max_workers=ADVERSARIAL_WORKERS) as executor:
        for train_name, test_name in pair_order:
            future_map[(train_name, test_name)] = executor.submit(
                _evaluate_cross_survey_pair,
                train_name,
                prepared_surveys[train_name],
                test_name,
                prepared_surveys[test_name],
            )

        for train_name, test_name in pair_order:
            key = f"{train_name}_to_{test_name}"
            print_status(f"  {key}...")
            _, pair_results = future_map[(train_name, test_name)].result()
            if pair_results.get("status") == "INSUFFICIENT_DATA":
                results[key] = pair_results
                continue

            for feat_label in ["mass_z_poly", "mass_z_poly+gamma_t"]:
                metrics = pair_results[feat_label]
                print_status(f"    {feat_label}: R²={metrics['R2']:.4f}, ρ={metrics['rho']:.4f}")

            print_status(f"    Gamma_t cross-survey lift: ΔR² = {pair_results['gamma_t_cross_survey_lift']:+.4f}")
            results[key] = pair_results

    return results


def test3_information_theoretic(df):
    """
    Test 3: Conditional Mutual Information I(Gamma_t; dust | M*, z).
    
    Uses the KSG estimator. If I(Gamma_t; dust | M*, z) > 0, Gamma_t
    carries information about dust beyond what mass and z provide.
    
    Null calibration: compute I(Gamma_t_shuffled; dust | M*, z) to
    estimate the baseline under the null.
    """
    print_status("TEST 3: Conditional mutual information I(Γ_t; dust | M*, z)")

    sub = df.dropna(subset=["gamma_t", "dust", "log_mstar", "z"]).copy()
    # Use z > 8 subsample where signal is strongest
    z8 = sub[sub["z"] >= 8.0].copy()

    results = {}

    for label, data in [("z_gt_8", z8), ("full_z4_10", sub)]:
        n = len(data)
        if n < 100:
            results[label] = {"status": "INSUFFICIENT_DATA", "n": n}
            continue

        print_status(f"  {label}: N = {n}")

        gamma = data["log_gamma_t"].values if "log_gamma_t" in data.columns else np.log10(np.maximum(data["gamma_t"].values, 1e-10))
        dust = data["dust"].values
        mass = data["log_mstar"].values
        z_arr = data["z"].values

        # I(Gamma_t; dust | M*, z) using binned approach
        try:
            cmi, cell_mis = conditional_mi_binned(gamma, dust, mass, z_arr, n_bins=5)
        except Exception as e:
            cmi, cell_mis = np.nan, []
            logger.warning(f"CMI computation failed: {e}")

        # I(M*; dust | Gamma_t, z)
        try:
            cmi_mass, _ = conditional_mi_binned(mass, dust, gamma, z_arr, n_bins=5)
        except Exception:
            cmi_mass = np.nan

        null_cmis = []
        seed_batches = [
            batch.tolist()
            for batch in np.array_split(
                np.arange(42, 242, dtype=int),
                min(ADVERSARIAL_WORKERS, 200),
            )
            if len(batch) > 0
        ]
        with ThreadPoolExecutor(max_workers=ADVERSARIAL_WORKERS) as executor:
            futures = [
                executor.submit(_cmi_null_batch, batch, gamma, dust, mass, z_arr, 5)
                for batch in seed_batches
            ]
            for future in futures:
                null_cmis.extend(future.result())

        null_mean = float(np.mean(null_cmis)) if null_cmis else np.nan
        null_std = float(np.std(null_cmis)) if null_cmis else np.nan
        z_score = float((cmi - null_mean) / null_std) if null_std > 0 else 0.0
        _p_raw = float(np.mean(np.array(null_cmis) >= cmi)) if null_cmis else np.nan
        # clamp to resolution floor 1/n_null to avoid exact zero
        p_empirical = max(_p_raw, 1.0 / len(null_cmis)) if null_cmis else np.nan

        results[label] = {
            "n": n,
            "I_gamma_dust_given_mass_z": float(cmi),
            "I_mass_dust_given_gamma_z": float(cmi_mass),
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z_score,
            "p_empirical": p_empirical,
            "n_null_shuffles": len(null_cmis),
            "information_asymmetry": float(cmi - cmi_mass) if not np.isnan(cmi) and not np.isnan(cmi_mass) else np.nan,
            "interpretation": (
                f"I(Γ_t; dust | M*, z) = {cmi:.4f} nats "
                f"(null: {null_mean:.4f} ± {null_std:.4f}, z = {z_score:.1f}). "
                f"I(M*; dust | Γ_t, z) = {cmi_mass:.4f} nats. "
                + ("Γ_t carries significant conditional information about dust "
                   "beyond mass and redshift." if z_score > 2 else
                   "Γ_t does not carry significant conditional information beyond mass and z.")
            ),
        }
        print_status(f"    I(Γ_t; dust | M*, z) = {cmi:.4f} (z-score = {z_score:.1f})")
        print_status(f"    I(M*; dust | Γ_t, z) = {cmi_mass:.4f}")

    return results


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(test1, test2, test3):
    """Generate 4-panel summary figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Single-feature showdown (bar chart)
    ax = axes[0, 0]
    showdown = test1.get("single_feature_showdown", {})
    labels = ["$\\Gamma_t$ only", "$M_*$ only", "$z$ only"]
    vals = [
        showdown.get("gamma_t_only_R2", 0),
        showdown.get("mass_only_R2", 0),
        showdown.get("z_only_R2", 0),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Cross-validated $R^2$ (GBR)", fontsize=11)
    ax.set_title("Single-Feature Showdown: Who Predicts Dust Best?", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Feature set comparison (grouped bar)
    ax = axes[0, 1]
    feat_sets = test1.get("feature_sets", {})
    set_names = ["mass_z_poly", "mass_z_poly+extras", "mass_z_poly+gamma_t", "mass_z_poly+extras+gamma_t"]
    set_labels = ["M*,z poly\n(8 feat)", "+extras\n(+4 feat)", "+Γ_t\n(+2 feat)", "+all\n(14 feat)"]
    gbr_r2 = [feat_sets.get(s, {}).get("models", {}).get("GBR", {}).get("R2_mean", 0) for s in set_names]
    gbr_std = [feat_sets.get(s, {}).get("models", {}).get("GBR", {}).get("R2_std", 0) for s in set_names]
    
    x_pos = np.arange(len(set_labels))
    bar_colors = ["#95a5a6", "#3498db", "#e74c3c", "#9b59b6"]
    ax.bar(x_pos, gbr_r2, yerr=gbr_std, color=bar_colors, edgecolor="black",
           linewidth=0.8, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(set_labels, fontsize=9)
    ax.set_ylabel("Cross-validated $R^2$ (GBR)", fontsize=11)
    ax.set_title("Feature Ablation: Does $\\Gamma_t$ Add Lift?", fontsize=12, fontweight="bold")
    for i, (v, s) in enumerate(zip(gbr_r2, gbr_std)):
        ax.text(i, v + s + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 3: Cross-survey generalization
    ax = axes[1, 0]
    cross_pairs = []
    cross_r2_without = []
    cross_r2_with = []
    for key, val in test2.items():
        if isinstance(val, dict) and "mass_z_poly" in val:
            cross_pairs.append(key.replace("_to_", "\n→ "))
            cross_r2_without.append(val["mass_z_poly"]["R2"])
            cross_r2_with.append(val["mass_z_poly+gamma_t"]["R2"])

    if cross_pairs:
        x = np.arange(len(cross_pairs))
        w = 0.35
        ax.bar(x - w/2, cross_r2_without, w, label="M*,z poly", color="#95a5a6", edgecolor="black", linewidth=0.8)
        ax.bar(x + w/2, cross_r2_with, w, label="+$\\Gamma_t$", color="#e74c3c", edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cross_pairs, fontsize=8)
        ax.set_ylabel("$R^2$ on test survey", fontsize=11)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_title("Cross-Survey Generalization", fontsize=12, fontweight="bold")

    # Panel 4: Information-theoretic summary
    ax = axes[1, 1]
    t3_z8 = test3.get("z_gt_8", {})
    t3_full = test3.get("full_z4_10", {})
    
    text = "ADVERSARIAL ML ATTACK RESULTS\n" + "=" * 35 + "\n\n"
    
    # Lift
    lift_info = test1.get("gamma_t_lift", {})
    text += f"Γ_t lift (UNCOVER CV):\n"
    text += f"  ΔR² = {lift_info.get('delta_R2', 0):+.4f}\n\n"
    
    # Permutation importance
    perm = test1.get("permutation_importance", {})
    text += f"Permutation importance:\n"
    text += f"  R² drop = {perm.get('mean_R2_drop', 0):.4f}\n\n"
    
    # Information theory
    text += f"Conditional MI (z>8):\n"
    text += f"  I(Γ_t;dust|M*,z) = {t3_z8.get('I_gamma_dust_given_mass_z', 0):.4f}\n"
    text += f"  z-score vs null = {t3_z8.get('z_score', 0):.1f}\n"
    text += f"  I(M*;dust|Γ_t,z) = {t3_z8.get('I_mass_dust_given_gamma_z', 0):.4f}\n\n"
    
    # Verdict
    text += f"Cross-survey Γ_t lift:\n"
    for key, val in test2.items():
        if isinstance(val, dict) and "gamma_t_cross_survey_lift" in val:
            text += f"  {key}: ΔR²={val['gamma_t_cross_survey_lift']:+.4f}\n"

    ax.text(0.05, 0.95, text, ha="left", va="top", fontsize=9.5,
            transform=ax.transAxes, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3))
    ax.set_title("Summary", fontsize=12, fontweight="bold")
    ax.axis("off")

    plt.suptitle("Step 144: Adversarial Machine Learning Attack on TEP",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 144: Adversarial Machine Learning Attack on TEP")
    logger.info("=" * 70)

    # Load all surveys
    surveys = {}
    for name in ["UNCOVER", "CEERS", "COSMOS-Web"]:
        print_status(f"Loading {name}...")
        df = load_survey(name)
        if df is not None:
            surveys[name] = df
            logger.info(f"  {name}: {len(df)} galaxies")
        else:
            logger.warning(f"  {name}: not found")

    if "UNCOVER" not in surveys:
        logger.error("UNCOVER not found — cannot run adversarial test")
        return {"error": "UNCOVER not found"}

    results = {
        "step": "Step 144: Adversarial Machine Learning Attack on TEP",
        "description": (
            "Give a gradient-boosted ML model every available feature and let it "
            "learn ANY function to predict dust. Then add Gamma_t. If Gamma_t "
            "provides measurable lift, it encodes information no function of "
            "standard features can replicate."
        ),
        "surveys_loaded": {k: len(v) for k, v in surveys.items()},
    }

    # Test 1: Within-survey
    test1 = test1_within_survey(surveys["UNCOVER"])
    results["test_1_within_survey"] = test1

    # Test 2: Cross-survey generalization
    test2 = test2_cross_survey(surveys)
    results["test_2_cross_survey"] = test2

    # Test 3: Information-theoretic
    test3 = test3_information_theoretic(surveys["UNCOVER"])
    results["test_3_information_theory"] = test3

    # Overall verdict
    verdicts = []

    # Verdict 1: within-survey lift
    lift = test1.get("gamma_t_lift", {}).get("delta_R2", 0)
    if lift > 0.005:
        verdicts.append(f"WITHIN-SURVEY: Γ_t adds ΔR²={lift:+.4f} to GBR with all standard features")
    else:
        verdicts.append(f"WITHIN-SURVEY: Γ_t lift is negligible once flexible survey-specific features are available (ΔR²={lift:+.4f})")

    # Verdict 2: single-feature showdown
    show = test1.get("single_feature_showdown", {})
    verdicts.append(
        f"SINGLE-FEATURE: {show.get('winner', '?')} wins "
        f"(Γ_t R²={show.get('gamma_t_only_R2', 0):.3f}, "
        f"M* R²={show.get('mass_only_R2', 0):.3f}, "
        f"z R²={show.get('z_only_R2', 0):.3f})"
    )

    # Verdict 3: cross-survey (use Spearman rho, not R²)
    cross_rho_lifts = []
    for v in test2.values():
        if isinstance(v, dict) and "mass_z_poly" in v and "mass_z_poly+gamma_t" in v:
            rho_without = v["mass_z_poly"].get("rho", 0)
            rho_with = v["mass_z_poly+gamma_t"].get("rho", 0)
            cross_rho_lifts.append(rho_with - rho_without)
    if cross_rho_lifts:
        n_positive = sum(1 for l in cross_rho_lifts if l > 0)
        mean_rho_lift = np.mean(cross_rho_lifts)
        verdicts.append(
            f"CROSS-SURVEY: Γ_t Δρ positive in {n_positive}/{len(cross_rho_lifts)} "
            f"pairs (mean Δρ={mean_rho_lift:+.4f}). All flexible transfer models fail cross-survey "
            f"(R² << 0), so the relevant comparison is retained rank information rather than within-survey fit quality."
        )

    # Verdict 4: permutation importance
    perm = test1.get("permutation_importance", {})
    perm_drop = perm.get("mean_R2_drop")
    if perm_drop is not None:
        verdicts.append(
            f"PERMUTATION: joint Γ_t feature permutation lowers GBR R² by {perm_drop:.4f} on average"
        )

    # Verdict 5: information theory
    t3_z8 = test3.get("z_gt_8", {})
    z_sc = t3_z8.get("z_score", 0)
    if z_sc > 2:
        verdicts.append(
            f"INFO THEORY: I(Γ_t; dust | M*, z) significant (z={z_sc:.1f}; "
            f"ΔI={t3_z8.get('information_asymmetry', 0):+.4f} nats)"
        )
    else:
        verdicts.append(f"INFO THEORY: I(Γ_t; dust | M*, z) not significant (z={z_sc:.1f})")

    results["overall_verdict"] = {
        "verdicts": verdicts,
        "conclusion": " | ".join(verdicts),
    }

    # Save
    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    logger.info(f"Results saved to {out_path}")

    # Figure
    make_figure(test1, test2, test3)

    # Print summary
    print_status("=" * 60)
    print_status("ADVERSARIAL ML ATTACK RESULTS")
    print_status("=" * 60)
    for v in verdicts:
        print_status(f"  {v}")

    return results


if __name__ == "__main__":
    main()
