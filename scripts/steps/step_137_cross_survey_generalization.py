#!/usr/bin/env python3
"""
Step 137: Cross-Survey Generalization Test

Tests whether t_eff (zero parameters) generalizes better across surveys
than a fitted polynomial (M*, z, M*×z) which may overfit to
survey-specific SED systematics.

Two approaches:
  1. Continuous: Leave-one-survey-out OLS R² for predicting dust (A_V)
  2. Spearman: Leave-one-survey-out Spearman ρ(predicted_dust, actual_dust)

For each leave-one-out split:
  - Train polynomial OLS on 2 surveys → predict dust on held-out survey
  - Use t_eff directly on held-out survey (no training)
  - Compare R² and Spearman ρ

Outputs:
- results/outputs/step_137_cross_survey_generalization.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger  # Centralised logging
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like  # Shared TEP model
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types

STEP_NUM = "137"  # Pipeline step number
STEP_NAME = "cross_survey_generalization"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products

for p in [LOGS_PATH, OUTPUT_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)  # Step-specific logger
set_step_logger(logger)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Utility functions ────────────────────────────────────────────────────────

def ols_fit(X, y):
    """Fit OLS with intercept. Returns beta."""
    X_i = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_i, y, rcond=None)[0]
    return beta


def ols_predict(X, beta):
    """Predict using OLS beta."""
    X_i = np.column_stack([np.ones(len(X)), X])
    return X_i @ beta


def r2_score(y_true, y_pred):
    """R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def cv_r2(X, y, k=5, seed=42):
    """k-fold cross-validated R² for OLS."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    ss_res, ss_tot = 0.0, 0.0
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        beta = ols_fit(X[train_idx], y[train_idx])
        pred = ols_predict(X[test_idx], beta)
        ss_res += np.sum((y[test_idx] - pred) ** 2)
        ss_tot += np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_survey_generic(name, path, z_col, mass_col, dust_col, z_min=8.0, z_max=10.5, mass_min=8.0):
    """Load a survey dataset and compute TEP quantities."""
    df = pd.read_csv(path)
    df["z"] = df[z_col].astype(float)
    df["log_Mstar"] = df[mass_col].astype(float)
    df["dust"] = df[dust_col].astype(float)
    df = df[(df["z"] >= z_min) & (df["z"] <= z_max) & (df["log_Mstar"] >= mass_min)].copy()
    df = df.dropna(subset=["z", "log_Mstar", "dust"])

    from astropy.cosmology import Planck18 as cosmo
    df["log_Mh"] = df.apply(
        lambda r: stellar_to_halo_mass_behroozi_like(r["log_Mstar"], r["z"]), axis=1
    )
    df["gamma_t"] = df.apply(
        lambda r: compute_gamma_t(r["log_Mh"], r["z"]), axis=1
    )
    df["t_cosmic"] = df["z"].apply(lambda zz: cosmo.age(zz).value)
    df["t_eff"] = df["gamma_t"] * df["t_cosmic"]
    df["survey"] = name
    logger.info(f"  {name}: loaded {len(df)} galaxies (z>={z_min})")
    return df


def load_all_surveys():
    """Load UNCOVER (z>8), CEERS, and COSMOS-Web."""
    surveys = {}

    uncover_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        df_u = pd.read_csv(uncover_path)
        df_u["z"] = df_u["z_phot"].astype(float)
        df_u = df_u[(df_u["z"] >= 8.0) & (df_u["z"] <= 10.5) & (df_u["log_Mstar"] >= 8.0)].copy()
        df_u = df_u.dropna(subset=["z", "log_Mstar", "dust"])
        df_u["survey"] = "UNCOVER"
        logger.info(f"  UNCOVER z>8: loaded {len(df_u)} galaxies")
        surveys["UNCOVER"] = df_u

    ceers_path = DATA_INTERIM_PATH / "ceers_z8_sample.csv"
    if ceers_path.exists():
        surveys["CEERS"] = load_survey_generic("CEERS", ceers_path, "z_phot", "log_Mstar", "dust")

    cosmo_path = DATA_INTERIM_PATH / "cosmosweb_z8_sample.csv"
    if cosmo_path.exists():
        surveys["COSMOS-Web"] = load_survey_generic("COSMOS-Web", cosmo_path, "z_phot", "log_Mstar", "dust")

    return surveys


# ── Main Analysis ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 137: Cross-Survey Generalization Test")
    logger.info("=" * 70)

    surveys = load_all_surveys()
    survey_names = list(surveys.keys())
    logger.info(f"Loaded {len(survey_names)} surveys: {survey_names}")

    results = {
        "step": "Step 137: Cross-Survey Generalization Test",
        "description": (
            "Tests whether t_eff (0 parameters) generalizes better across surveys "
            "than a fitted polynomial (M*, z, M*×z; 4 parameters). Uses continuous "
            "dust prediction (Spearman ρ and OLS R²) for leave-one-survey-out."
        ),
        "surveys": {},
        "within_survey": [],
        "leave_one_out": [],
        "summary": {},
    }

    # ── Within-survey Spearman ρ and CV R² ────────────────────────────────
    logger.info("\n--- Within-Survey Statistics ---")
    for sname, df in surveys.items():
        n = len(df)
        dust = df["dust"].values
        t_eff = df["t_eff"].values
        X_poly = np.column_stack([
            df["log_Mstar"].values,
            df["z"].values,
            df["log_Mstar"].values * df["z"].values,
        ])

        rho_teff, p_teff = spearmanr(dust, t_eff)
        rho_mstar, p_mstar = spearmanr(dust, df["log_Mstar"].values)
        
        p_teff = max(float(p_teff), 1e-300)
        p_mstar = max(float(p_mstar), 1e-300)

        poly_cv_r2 = cv_r2(X_poly, dust)
        teff_cv_r2 = cv_r2(t_eff.reshape(-1, 1), dust)

        entry = {
            "survey": sname,
            "n": int(n),
            "rho_teff": float(rho_teff),
            "p_teff": float(p_teff),
            "rho_mstar": float(rho_mstar),
            "p_mstar": float(p_mstar),
            "poly_cv_r2": float(poly_cv_r2),
            "teff_cv_r2": float(teff_cv_r2),
        }
        results["within_survey"].append(entry)
        results["surveys"][sname] = {"n": int(n)}
        logger.info(
            f"  {sname} (N={n}): ρ(dust,t_eff)={rho_teff:.3f}, "
            f"ρ(dust,M*)={rho_mstar:.3f}, poly_CV_R²={poly_cv_r2:.3f}"
        )

    # ── Leave-one-survey-out ─────────────────────────────────────────────
    logger.info("\n--- Leave-One-Survey-Out (Continuous Dust Prediction) ---")
    for test_name in survey_names:
        train_names = [s for s in survey_names if s != test_name]
        train_dfs = [surveys[s] for s in train_names]
        test_df = surveys[test_name]

        if not train_dfs:
            continue

        train_df = pd.concat(train_dfs, ignore_index=True)

        # Features
        X_train = np.column_stack([
            train_df["log_Mstar"].values,
            train_df["z"].values,
            train_df["log_Mstar"].values * train_df["z"].values,
        ])
        X_test = np.column_stack([
            test_df["log_Mstar"].values,
            test_df["z"].values,
            test_df["log_Mstar"].values * test_df["z"].values,
        ])
        y_train = train_df["dust"].values
        y_test = test_df["dust"].values
        teff_test = test_df["t_eff"].values

        # Polynomial: train on other surveys, predict on held-out
        beta = ols_fit(X_train, y_train)
        poly_pred = ols_predict(X_test, beta)

        # Metrics on held-out survey
        poly_r2_cross = r2_score(y_test, poly_pred)
        rho_poly_cross, p_poly_cross = spearmanr(y_test, poly_pred)
        rho_teff_cross, p_teff_cross = spearmanr(y_test, teff_test)
        
        p_poly_cross = max(float(p_poly_cross), 1e-300)
        p_teff_cross = max(float(p_teff_cross), 1e-300)

        # Within-survey poly R² for comparison
        within_entry = next(
            (e for e in results["within_survey"] if e["survey"] == test_name), None
        )
        poly_cv_within = within_entry["poly_cv_r2"] if within_entry else np.nan

        entry = {
            "test_survey": test_name,
            "train_surveys": train_names,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "poly_r2_cross": float(poly_r2_cross),
            "poly_r2_within": float(poly_cv_within) if not np.isnan(poly_cv_within) else None,
            "poly_r2_drop": float(poly_cv_within - poly_r2_cross) if not np.isnan(poly_cv_within) else None,
            "rho_poly_cross": float(rho_poly_cross),
            "p_poly_cross": float(p_poly_cross),
            "rho_teff_cross": float(rho_teff_cross),
            "p_teff_cross": float(p_teff_cross),
            "teff_rho_advantage": float(rho_teff_cross - rho_poly_cross),
        }
        results["leave_one_out"].append(entry)
        logger.info(
            f"  Test={test_name} (N={len(test_df)}): "
            f"poly cross-ρ={rho_poly_cross:.3f}, t_eff ρ={rho_teff_cross:.3f}, "
            f"Δρ(t_eff-poly)={rho_teff_cross - rho_poly_cross:+.3f}, "
            f"poly R²_cross={poly_r2_cross:.3f}"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    loo = results["leave_one_out"]
    if loo:
        rho_advantages = [e["teff_rho_advantage"] for e in loo]
        poly_r2_drops = [e["poly_r2_drop"] for e in loo if e["poly_r2_drop"] is not None]

        results["summary"] = {
            "n_tests": len(loo),
            "mean_teff_rho_advantage": float(np.mean(rho_advantages)),
            "teff_rho_wins": int(sum(1 for d in rho_advantages if d > 0)),
            "mean_poly_r2_drop": float(np.mean(poly_r2_drops)) if poly_r2_drops else None,
            "individual_advantages": {
                e["test_survey"]: float(e["teff_rho_advantage"]) for e in loo
            },
            "interpretation": (
                "t_eff generalizes better cross-survey than polynomial "
                "(positive mean ρ advantage = t_eff wins)."
                if np.mean(rho_advantages) > 0
                else "Polynomial generalizes comparably or better than t_eff."
            ),
        }
        logger.info(f"\n--- Summary ---")
        logger.info(f"  Mean t_eff ρ advantage: {np.mean(rho_advantages):+.3f}")
        logger.info(f"  t_eff wins: {sum(1 for d in rho_advantages if d > 0)}/{len(rho_advantages)}")
        if poly_r2_drops:
            logger.info(f"  Mean poly R² drop (within→cross): {np.mean(poly_r2_drops):+.3f}")

    # Save
    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    logger.info(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
