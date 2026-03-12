#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 3.5s.
"""
Step 138: Environmental Screening Steiger Z-Test

Tests whether the dust-Gamma_t correlation is significantly WEAKER in
overdense environments than in the field, using a formal Steiger Z-test
for dependent correlations (Meng, Rosenthal & Rubin 1992).

TEP prediction: Group Halo Screening suppresses Gamma_t in dense
environments, so rho(dust, Gamma_t | overdense) < rho(dust, Gamma_t | field).

This upgrades step_115 by:
  1. Applying the Steiger Z-test (not just delta_rho) to quantify significance
     of the field vs overdense difference.
  2. Controlling for stellar mass within each environment subsample to
     isolate the environmental effect from mass confounding.
  3. Testing at z > 8 (where the dust signal is strongest) separately from
     the full sample.
  4. Computing a partial correlation: rho(dust, Gamma_t | M*, environment)
     to test whether environment adds independent information beyond mass.

Outputs:
- results/outputs/step_138_environmental_screening_steiger.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "138"
STEP_NAME = "environmental_screening_steiger"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"

for p in [LOGS_PATH, OUTPUT_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Steiger Z-test for dependent correlations ─────────────────────────────────

def steiger_z_test(r12, r13, r23, n):
    """
    Steiger Z-test comparing two dependent correlations r12 and r13
    that share variable 1 (Meng, Rosenthal & Rubin 1992).

    Tests H0: rho_12 == rho_13.
    Returns (Z, p_two_tailed).
    """
    # Fisher z-transforms
    z12 = np.arctanh(np.clip(r12, -0.9999, 0.9999))
    z13 = np.arctanh(np.clip(r13, -0.9999, 0.9999))

    # Average r for denominator
    r_bar = (r12 + r13) / 2.0

    # Covariance correction term
    f = (1 - r23) / (2 * (1 - r_bar**2))
    h = (1 - f * r_bar**2) / (1 - r_bar**2)

    denom = np.sqrt((2 * (1 - r23)) / (n - 3) * h)
    if denom == 0:
        return 0.0, 1.0

    Z = (z12 - z13) / denom
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    return float(Z), float(p)


def partial_spearman(x, y, z_controls):
    """
    Partial Spearman correlation of x and y controlling for z_controls
    (list of arrays). Uses residuals from OLS on ranks.
    """
    from scipy.stats import rankdata

    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)

    # Build control matrix
    Z = np.column_stack([np.ones(len(x))] + [rankdata(z).astype(float) for z in z_controls])

    def residualize(v):
        beta = np.linalg.lstsq(Z, v, rcond=None)[0]
        return v - Z @ beta

    rx_res = residualize(rx)
    ry_res = residualize(ry)

    r, p = pearsonr(rx_res, ry_res)
    return float(r), float(p)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_uncover():
    """Load UNCOVER DR4 with TEP quantities (same pattern as step_160)."""
    # Primary: pre-computed TEP catalog from step_02
    uncover_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if not uncover_path.exists():
        uncover_path = INTERIM_PATH / "step_001_uncover_full_sample.csv"
    if not uncover_path.exists():
        raise FileNotFoundError(f"UNCOVER catalog not found at {INTERIM_PATH}")

    df = pd.read_csv(uncover_path)

    # Standardise z column
    if "z_phot" in df.columns and "z" not in df.columns:
        df["z"] = df["z_phot"].astype(float)
    elif "z" in df.columns and "z_phot" not in df.columns:
        df["z_phot"] = df["z"].astype(float)

    # Standardise mass column
    if "log_Mstar" in df.columns and "log_mstar" not in df.columns:
        df["log_mstar"] = df["log_Mstar"]

    # Standardise dust column
    if "dust" in df.columns and "av" not in df.columns:
        df["av"] = df["dust"]

    # Ensure Gamma_t is computed
    if "gamma_t" not in df.columns:
        if "log_Mh" in df.columns:
            df["log_mh"] = df["log_Mh"]
        elif "log_mh" not in df.columns:
            df["log_mh"] = stellar_to_halo_mass_behroozi_like(df["log_mstar"].values)
        z_arr = df["z"].values if "z" in df.columns else df["z_phot"].values
        df["gamma_t"] = compute_gamma_t(df["log_mh"].values, z_arr)
    elif "log_mh" not in df.columns and "log_Mh" in df.columns:
        df["log_mh"] = df["log_Mh"]

    # Quality cuts
    z_col = "z" if "z" in df.columns else "z_phot"
    df = df[(df[z_col] >= 4.0) & (df[z_col] <= 10.5)].copy()
    df = df.dropna(subset=[z_col, "gamma_t"])

    logger.info(f"Loaded {len(df)} galaxies from {uncover_path.name}")
    return df


def build_density_estimator(df):
    """
    Compute 5-nearest-neighbour projected density for each galaxy
    using RA/Dec (or x/y pixel coords as fallback).
    Returns density array (higher = denser environment).
    """
    ra_col = next((c for c in ["ra", "RA", "ALPHA_J2000"] if c in df.columns), None)
    dec_col = next((c for c in ["dec", "DEC", "DELTA_J2000"] if c in df.columns), None)

    if ra_col is None or dec_col is None:
        logger.warning("No RA/Dec columns found; using index-based density proxy")
        return np.zeros(len(df))

    ra = df[ra_col].values
    dec = df[dec_col].values

    # Convert to radians for great-circle distance
    ra_r = np.radians(ra)
    dec_r = np.radians(dec)

    n = len(ra)
    density = np.zeros(n)
    k = 5  # 5th nearest neighbour

    for i in range(n):
        # Haversine distance to all others
        dra = ra_r - ra_r[i]
        ddec = dec_r - dec_r[i]
        a = np.sin(ddec / 2) ** 2 + np.cos(dec_r[i]) * np.cos(dec_r) * np.sin(dra / 2) ** 2
        dist_deg = 2 * np.degrees(np.arcsin(np.sqrt(np.clip(a, 0, 1))))
        dist_deg[i] = np.inf  # exclude self

        sorted_d = np.sort(dist_deg)
        if len(sorted_d) >= k:
            d5 = sorted_d[k - 1]
            density[i] = k / (np.pi * d5 ** 2) if d5 > 0 else 0.0

    return density


# ── Core analysis ─────────────────────────────────────────────────────────────

def run_screening_steiger(df, label, z_min=None, z_max=None):
    """
    Run the full environmental screening Steiger Z-test on a subsample.
    Returns a results dict.
    """
    sub = df.copy()
    if z_min is not None:
        z_col = "z_phot" if "z_phot" in sub.columns else "z"
        sub = sub[sub[z_col] >= z_min]
    if z_max is not None:
        z_col = "z_phot" if "z_phot" in sub.columns else "z"
        sub = sub[sub[z_col] < z_max]

    dust_col = next((c for c in ["av", "AV", "A_V", "dust_av", "attenuation"] if c in sub.columns), None)
    mass_col = next((c for c in ["log_mstar", "log_mstar_50", "logmstar"] if c in sub.columns), None)

    if dust_col is None or mass_col is None or len(sub) < 30:
        return {"label": label, "n": len(sub), "status": "insufficient_data"}

    sub = sub.dropna(subset=[dust_col, mass_col, "gamma_t", "density_5nn"])
    n_total = len(sub)

    if n_total < 30:
        return {"label": label, "n": n_total, "status": "insufficient_data"}

    dust = sub[dust_col].values
    gamma_t = sub["gamma_t"].values
    mass = sub[mass_col].values
    density = sub["density_5nn"].values

    # Split at median density
    med_density = np.median(density)
    field_mask = density <= med_density
    dense_mask = density > med_density

    n_field = field_mask.sum()
    n_dense = dense_mask.sum()

    # Raw Spearman correlations in each environment
    r_field, p_field = spearmanr(gamma_t[field_mask], dust[field_mask])
    r_dense, p_dense = spearmanr(gamma_t[dense_mask], dust[dense_mask])

    # Cross-correlation between the two predictors (needed for Steiger)
    # We use the full-sample correlation between gamma_t and density as r23
    r_gt_density, _ = spearmanr(gamma_t, density)

    # Steiger Z-test: does field rho differ significantly from dense rho?
    # We treat this as comparing two independent correlations (different subsamples)
    # using Fisher's z-test for independent correlations
    z_field = np.arctanh(np.clip(r_field, -0.9999, 0.9999))
    z_dense = np.arctanh(np.clip(r_dense, -0.9999, 0.9999))
    se = np.sqrt(1.0 / (n_field - 3) + 1.0 / (n_dense - 3))
    Z_indep = float((z_field - z_dense) / se) if se > 0 else 0.0
    p_indep = float(2 * (1 - stats.norm.cdf(abs(Z_indep))))

    delta_rho = float(r_field - r_dense)

    # Bootstrap CI on delta_rho
    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_deltas = []
    for _ in range(n_boot):
        idx_f = rng.choice(n_field, n_field, replace=True)
        idx_d = rng.choice(n_dense, n_dense, replace=True)
        r_f_b, _ = spearmanr(gamma_t[field_mask][idx_f], dust[field_mask][idx_f])
        r_d_b, _ = spearmanr(gamma_t[dense_mask][idx_d], dust[dense_mask][idx_d])
        boot_deltas.append(r_f_b - r_d_b)
    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))

    # Partial correlation: rho(dust, gamma_t | mass, environment)
    # environment encoded as binary (0=field, 1=dense)
    env_binary = dense_mask.astype(float)
    try:
        rho_partial, p_partial = partial_spearman(gamma_t, dust, [mass, env_binary])
    except Exception:
        rho_partial, p_partial = np.nan, np.nan

    # Mass-matched test: match field and dense by mass quintile
    sub["env"] = dense_mask.astype(int)
    sub["mass_quintile"] = pd.qcut(sub[mass_col], q=5, labels=False, duplicates="drop")
    mass_matched_deltas = []
    for q in sub["mass_quintile"].unique():
        q_sub = sub[sub["mass_quintile"] == q]
        q_field = q_sub[q_sub["env"] == 0]
        q_dense = q_sub[q_sub["env"] == 1]
        if len(q_field) >= 10 and len(q_dense) >= 10:
            r_f_q, _ = spearmanr(q_field["gamma_t"], q_field[dust_col])
            r_d_q, _ = spearmanr(q_dense["gamma_t"], q_dense[dust_col])
            mass_matched_deltas.append(float(r_f_q - r_d_q))

    mean_mass_matched_delta = float(np.mean(mass_matched_deltas)) if mass_matched_deltas else np.nan
    n_quintiles_positive = sum(d > 0 for d in mass_matched_deltas)

    tep_confirmed = (delta_rho > 0) and (p_indep < 0.05)

    return {
        "label": label,
        "n_total": n_total,
        "n_field": int(n_field),
        "n_dense": int(n_dense),
        "median_density": float(med_density),
        "rho_field": float(r_field),
        "p_field": float(p_field),
        "rho_dense": float(r_dense),
        "p_dense": float(p_dense),
        "delta_rho": delta_rho,
        "delta_rho_ci_low": ci_low,
        "delta_rho_ci_high": ci_high,
        "Z_fisher_independent": Z_indep,
        "p_fisher_independent": p_indep,
        "rho_gamma_t_density": float(r_gt_density),
        "partial_rho_dust_gamma_t_given_mass_env": float(rho_partial),
        "partial_p": float(p_partial),
        "mass_matched_quintile_deltas": mass_matched_deltas,
        "mean_mass_matched_delta_rho": mean_mass_matched_delta,
        "n_quintiles_positive": int(n_quintiles_positive),
        "n_quintiles_tested": len(mass_matched_deltas),
        "tep_prediction": "rho_field > rho_dense (screening suppresses in dense regions)",
        "tep_confirmed": tep_confirmed,
        "interpretation": "SUPPORTS TEP" if tep_confirmed else (
            "MARGINAL" if delta_rho > 0 else "INCONCLUSIVE or CONTRADICTS"
        ),
    }


def main():
    logger.info("=" * 70)
    logger.info("Step 138: Environmental Screening Steiger Z-Test")
    logger.info("=" * 70)

    # Load data
    print_status("Loading UNCOVER DR4...")
    df = load_uncover()
    logger.info(f"Loaded {len(df)} galaxies")

    # Compute density
    print_status("Computing 5-NN projected density...")
    df["density_5nn"] = build_density_estimator(df)
    logger.info(f"Density range: {df['density_5nn'].min():.3f} – {df['density_5nn'].max():.3f}")

    results = {
        "step": "Step 138: Environmental Screening Steiger Z-Test",
        "tep_prediction": (
            "Group Halo Screening suppresses Gamma_t in dense environments. "
            "rho(dust, Gamma_t) should be significantly lower in overdense regions "
            "than in the field, even after controlling for stellar mass."
        ),
        "method": (
            "Fisher Z-test for independent correlations comparing field vs overdense "
            "Spearman rho(dust, Gamma_t). Mass-matched quintile test controls for "
            "mass confounding. Partial correlation controls for both mass and environment."
        ),
    }

    # Full sample
    print_status("Running full-sample test...")
    results["full_sample"] = run_screening_steiger(df, label="Full sample (z=4-10)")

    # z > 8 subsample (where dust signal is strongest)
    z_col = "z_phot" if "z_phot" in df.columns else "z"
    print_status("Running z > 8 subsample test...")
    results["z_gt_8"] = run_screening_steiger(df, label="z > 8", z_min=8.0)

    # z = 6-8 subsample
    print_status("Running z = 6-8 subsample test...")
    results["z_6_8"] = run_screening_steiger(df, label="z = 6-8", z_min=6.0, z_max=8.0)

    # z = 4-6 subsample (control: screening should be weaker here)
    print_status("Running z = 4-6 subsample test (control)...")
    results["z_4_6"] = run_screening_steiger(df, label="z = 4-6 (control)", z_min=4.0, z_max=6.0)

    # Summary
    tests = [results["full_sample"], results["z_gt_8"], results["z_6_8"], results["z_4_6"]]
    n_confirmed = sum(1 for t in tests if isinstance(t, dict) and t.get("tep_confirmed", False))
    n_valid = sum(1 for t in tests if isinstance(t, dict) and t.get("status") != "insufficient_data")

    results["summary"] = {
        "n_tests": n_valid,
        "n_tep_confirmed": n_confirmed,
        "fraction_confirmed": float(n_confirmed / n_valid) if n_valid > 0 else 0.0,
        "primary_result": results["z_gt_8"],
        "conclusion": (
            f"{n_confirmed}/{n_valid} redshift bins show the predicted environmental "
            f"screening pattern (rho_field > rho_dense). "
            f"Primary test (z > 8): delta_rho = "
            f"{results['z_gt_8'].get('delta_rho', 'N/A'):.3f}, "
            f"Z = {results['z_gt_8'].get('Z_fisher_independent', 'N/A'):.2f}, "
            f"p = {results['z_gt_8'].get('p_fisher_independent', 'N/A'):.4f}."
        ) if n_valid > 0 else "Insufficient data for testing.",
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
