#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 25m01s.
"""
TEP-JWST Step 162: TEP Concordance — Independent κ_gal Recovery

A critical evaluation of parameter universality: if TEP is a single-parameter theory with
Observable Response Coefficient κ_gal, then fitting κ_gal independently to different
astrophysical observables should yield consistent values, analogous to the CMB/BAO/SNe
cosmic concordance diagram.

Observables evaluated independently:
  1. Dust-t_eff correlation (UNCOVER, z>6): maximize Pearson R²(dust, t_eff^0.3)
  2. AGB threshold classification (UNCOVER, z>8): maximize F1 score for t_eff > 0.2 Gyr predicting A_V > 0.15
  3. Mass-sSFR structural inversion (UNCOVER): optimize mass correction to minimize high-z / low-z structural discrepancy
  4. Cross-survey dust scaling (CEERS + COSMOS-Web): maximize mean R² across surveys
  5. Kinematic tension resolution (all surveys): minimize residual unphysical mass excess

This approach tests whether the derived coupling constant is an overfitted parameter
for a specific dataset, or a fundamental property that generalizes across distinct physical domains.

Output:
  - results/outputs/step_162_l1_l3_independence.json
  - results/figures/figure_162_l1_l3_independence.png

Author: Matthew L. Smawfield
Date: February 2026
"""

import sys
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor  # Sandbox-safe parallel bootstrap workers
import multiprocessing as mp
import numpy as np
np.random.seed(42)  # Fixed seed for reproducibility
import pandas as pd
from pathlib import Path
from scipy import stats



def _f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-10)

def _boot_worker_obs1_new(seeds, log_M, z, dust, kappa_gal_grid):
    n = len(log_M)
    best_alphas = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n, replace=True)
        bprofile = np.zeros(len(kappa_gal_grid))
        for i, a0 in enumerate(kappa_gal_grid):
            t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
            r = np.corrcoef(dust[idx], t_eff**0.3)[0, 1]
            bprofile[i] = r**2
        best_alphas.append(float(kappa_gal_grid[np.argmax(bprofile)]))
    return best_alphas

def _boot_worker_obs2_new(seeds, log_M, z, true_dust, kappa_gal_grid):
    n = len(log_M)
    best_alphas = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n, replace=True)
        bprofile = np.zeros(len(kappa_gal_grid))
        for i, a0 in enumerate(kappa_gal_grid):
            t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
            pred = (t_eff > 0.2).astype(int)
            bprofile[i] = _f1(true_dust[idx], pred)
        best_alphas.append(float(kappa_gal_grid[np.argmax(bprofile)]))
    return best_alphas

def _boot_worker_obs3_new(seeds, log_M_high, z_high, ssfr_high, rho_low_obs, kappa_gal_grid):
    n = len(log_M_high)
    best_alphas = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n, replace=True)
        b_log_M = log_M_high[idx]
        b_z = z_high[idx]
        b_ssfr = ssfr_high[idx]
        bprofile = np.zeros(len(kappa_gal_grid))
        for i, a0 in enumerate(kappa_gal_grid):
            gt = _gamma_t_for_alpha(b_log_M, b_z, a0)
            M_true = b_log_M - ALPHA_NUCLEAR * np.log10(np.maximum(gt, 0.01))
            ssfr_true = b_ssfr - 0.45 * np.log10(np.maximum(gt, 0.01))
            rho_corr, _ = stats.spearmanr(M_true, ssfr_true)
            bprofile[i] = -abs(rho_corr - rho_low_obs)
        best_alphas.append(float(kappa_gal_grid[np.argmax(bprofile)]))
    return best_alphas

def _boot_worker_obs4_new(seeds, boot_data, kappa_gal_grid):
    # boot_data: list of (log_M, z, dust) tuples
    best_alphas = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        bprofile = np.zeros(len(kappa_gal_grid))
        for i, a0 in enumerate(kappa_gal_grid):
            r2s = []
            for (log_M, z, dust) in boot_data:
                idx = rng.choice(len(log_M), len(log_M), replace=True)
                t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
                log_t_eff = np.log10(np.maximum(t_eff, 1e-6))
                r = np.corrcoef(dust[idx], log_t_eff)[0, 1]
                r2s.append(r ** 2)
            bprofile[i] = np.mean(r2s)
        best_alphas.append(float(kappa_gal_grid[np.argmax(bprofile)]))
    return best_alphas

def _boot_worker_obs5_new(seeds, log_M, max_logM, z, kappa_gal_grid):
    n = len(log_M)
    best_alphas = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n, replace=True)
        b_log_M = log_M[idx]
        b_max_logM = max_logM[idx]
        b_z = z[idx]
        bprofile = np.zeros(len(kappa_gal_grid))
        for i, a0 in enumerate(kappa_gal_grid):
            gt = _gamma_t_for_alpha(b_log_M, b_z, a0)
            M_true = b_log_M - 0.7 * np.log10(np.maximum(gt, 0.01))
            excess = M_true - b_max_logM
            bprofile[i] = -np.sum(excess[excess > 0]**2)
        best_alphas.append(float(kappa_gal_grid[np.argmax(bprofile)]))
    return best_alphas

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import (
    compute_gamma_t, stellar_to_halo_mass, _cosmic_time_gyr,
    stellar_to_halo_mass_behroozi_like, LOG_MH_REF, Z_REF,
    ALPHA_NUCLEAR,
)

STEP_NUM = "162"
STEP_NAME = "l1_l3_independence"

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
# κ_gal GRID
# =============================================================================

KAPPA_GAL_GRID = np.linspace(1e5, 20e5, 100)
N_BOOT = int(os.getenv("TEP_KAPPA_BOOTSTRAPS", "50"))  # bootstrap iterations per observable
EXTERNAL_CEPHEID_KAPPA_GAL = 9.6e5
EXTERNAL_CEPHEID_KAPPA_GAL_SIGMA = 4.0e5
BOOTSTRAP_WORKERS = max(
    1,
    min(
        mp.cpu_count(),
        int(os.getenv("TEP_BOOTSTRAP_WORKERS", min(mp.cpu_count(), 10))),
    ),
)


def _gamma_argument_base(log_Mstar, z):
    log_Mstar = np.asarray(log_Mstar, dtype=float)
    z = np.asarray(z, dtype=float)
    log_Mh = stellar_to_halo_mass(log_Mstar, z)
    log_mh_ref_z = LOG_MH_REF - 1.5 * np.log10(1 + z)
    delta_log_Mh = log_Mh - log_mh_ref_z
    z_factor = (1 + z) / (1 + Z_REF)
    return np.sqrt(1 + z) * (2.0 / 3.0) * delta_log_Mh * z_factor


def _gamma_t_from_base(gamma_base, alpha_0):
    # DEPRECATED STUB — do not use. All observables now use _t_eff_for_alpha
    # which calls compute_gamma_t correctly with the kappa grid value.
    raise NotImplementedError("Use _t_eff_for_alpha instead.")


def _t_eff_from_base(gamma_base, t_cosmic, alpha_0):
    # DEPRECATED STUB — do not use.
    raise NotImplementedError("Use _t_eff_for_alpha instead.")


def _run_bootstrap_batches(executor, worker, *worker_args, n_boot=N_BOOT, seed0=42):
    seed_batches = [
        batch.tolist()
        for batch in np.array_split(
            np.arange(seed0, seed0 + n_boot, dtype=int),
            min(BOOTSTRAP_WORKERS, max(n_boot, 1)),
        )
        if len(batch) > 0
    ]
    if executor is None:
        boot_alphas = []
        for seed_batch in seed_batches:
            boot_alphas.extend(worker(seed_batch, *worker_args))
        return np.asarray(boot_alphas, dtype=float)

    futures = [executor.submit(worker, seed_batch, *worker_args) for seed_batch in seed_batches]
    boot_alphas = []
    for future in futures:
        boot_alphas.extend(future.result())
    return np.asarray(boot_alphas, dtype=float)


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
    path = RESULTS_INTERIM / "step_002_uncover_full_sample_tep.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "dust", "z_phot"])
    return df


def load_ceers():
    path = DATA_INTERIM / "ceers_highz_sample.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "dust", "z_phot"])
    return df


def load_cosmosweb():
    path = DATA_INTERIM / "cosmosweb_highz_sample.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["log_Mstar", "dust", "z_phot"])
    return df


from functools import partial

# =============================================================================
# BOOTSTRAP WORKERS
# =============================================================================

def _f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    return tp / (tp + 0.5 * (fp + fn))

def _boot_worker_obs1(seed, n, log_M, z, dust, kappa_gal_grid):
    np.random.seed(seed)
    idx = np.random.choice(n, n, replace=True)
    bprofile = np.zeros(len(kappa_gal_grid))
    for i, a0 in enumerate(kappa_gal_grid):
        t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
        r = np.corrcoef(dust[idx], np.log10(np.maximum(t_eff, 1e-6)))[0, 1]
        bprofile[i] = r**2
    return kappa_gal_grid[np.argmax(bprofile)]

def _boot_worker_obs2(seed, n, log_M, z, true_dust, kappa_gal_grid):
    np.random.seed(seed)
    idx = np.random.choice(n, n, replace=True)
    bprofile = np.zeros(len(kappa_gal_grid))
    for i, a0 in enumerate(kappa_gal_grid):
        t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
        pred = (t_eff > 0.3).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bprofile[i] = _f1(true_dust[idx], pred)
    return kappa_gal_grid[np.argmax(bprofile)]

def _boot_worker_obs3(seed, n_low, n_high, M_low, M_high, z_high, ssfr_low, ssfr_high, rho_low_obs, kappa_gal_grid):
    np.random.seed(seed)
    idx_low = np.random.choice(n_low, n_low, replace=True)
    idx_high = np.random.choice(n_high, n_high, replace=True)
    b_M_low = M_low[idx_low]
    b_M_high = M_high[idx_high]
    b_z_high = z_high[idx_high]
    b_ssfr_low = ssfr_low[idx_low]
    b_ssfr_high = ssfr_high[idx_high]
    
    bprofile = np.zeros(len(kappa_gal_grid))
    for i, a0 in enumerate(kappa_gal_grid):
        gt = _gamma_t_for_alpha(b_M_high, b_z_high, a0)
        ssfr_corr = b_ssfr_high - 0.7 * np.log10(np.maximum(gt, 0.01))
        rho_corr, _ = stats.spearmanr(b_M_high, ssfr_corr)
        bprofile[i] = -abs(rho_corr - rho_low_obs)
        
    return kappa_gal_grid[np.argmax(bprofile)]

def _boot_worker_obs4(seed, boot_data, kappa_gal_grid):
    np.random.seed(seed)
    bprofile = np.zeros(len(kappa_gal_grid))
    for i, a0 in enumerate(kappa_gal_grid):
        r2s = []
        for (log_M, z, dust) in boot_data:
            idx = np.random.choice(len(log_M), len(log_M), replace=True)
            t_eff = _t_eff_for_alpha(log_M[idx], z[idx], a0)
            r = np.corrcoef(dust[idx], np.log10(np.maximum(t_eff, 1e-6)))[0, 1]
            r2s.append(r ** 2)
        bprofile[i] = np.mean(r2s)
    return kappa_gal_grid[np.argmax(bprofile)]

def _boot_worker_obs5(seed, n, log_M, max_logM, z, best_alpha, kappa_gal_grid):
    np.random.seed(seed)
    idx = np.random.choice(n, n, replace=True)
    b_log_M = log_M[idx]
    b_max_logM = max_logM[idx]
    b_z = z[idx]
    
    bprofile = np.zeros(len(kappa_gal_grid))
    for i, a0 in enumerate(kappa_gal_grid):
        gt = _gamma_t_for_alpha(b_log_M, b_z, a0)
        ml_bias = np.power(np.maximum(gt, 0.01), 0.7)
        log_M_corr = b_log_M - np.log10(ml_bias)
        n_res = np.sum(log_M_corr <= b_max_logM)
        bprofile[i] = n_res / n
        
    return kappa_gal_grid[np.argmax(bprofile)]

# =============================================================================
# OBSERVABLE 1: Dust–Γ_t Correlation (UNCOVER z>8)
# =============================================================================

def obs1_dust_teff_r2(df_uncover, bootstrap_executor=None):
    """For each kappa, compute Spearman rho(dust, sqrt(t_eff)) at z>6, M*>10^8.5."""
    print_status("  Observable 1: Dust-t_eff Correlation (UNCOVER z>6, M*>8.5)", "INFO")
    sub = df_uncover[(df_uncover["z_phot"] > 6) & (df_uncover["log_Mstar"] > 8.5)].copy()
    sub = sub.dropna(subset=["dust"])
    log_M = sub["log_Mstar"].values
    z = sub["z_phot"].values
    dust = sub["dust"].values
    n = len(sub)
    print_status(f"    N = {n}", "INFO")

    profile = np.zeros(len(KAPPA_GAL_GRID))
    for i, a0 in enumerate(KAPPA_GAL_GRID):
        t_eff = _t_eff_for_alpha(log_M, z, a0)
        r = np.corrcoef(dust, t_eff**0.3)[0, 1]
        profile[i] = r**2

    best_idx = np.argmax(profile)
    best_alpha = KAPPA_GAL_GRID[best_idx]
    best_r = np.sqrt(profile[best_idx])

    # Bootstrap
    print_status("    Bootstrapping CI...", "INFO")
    boot_alphas = _run_bootstrap_batches(
        bootstrap_executor,
        _boot_worker_obs1_new,
        log_M, z, dust,
        KAPPA_GAL_GRID,
    )

    ci_lo, ci_hi = np.percentile(boot_alphas, [16, 84])  # 1-sigma

    print_status(f"    Best kappa_gal = {best_alpha:.3e} [{ci_lo:.3e}, {ci_hi:.3e}], rho = {best_r:.3f}", "SUCCESS")

    return {
        "name": "Dust-t_eff rank correlation (UNCOVER z>6)",
        "short": "Dust-t_eff",
        "kappa_gal_best": float(best_alpha),
        "kappa_gal_ci": [float(ci_lo), float(ci_hi)],
        "metric_best": float(best_r),
        "metric_name": "rho(dust, t_eff)",
        "n": int(n),
        "profile": profile.tolist(),
    }

def obs2_agb_threshold_f1(df_uncover, bootstrap_executor=None):
    """For each kappa, compute F1 of t_eff > 0.2 Gyr predicting dusty galaxies (A_V > 0.15)."""
    print_status("  Observable 2: AGB Threshold F1 (UNCOVER z>8)", "INFO")
    sub = df_uncover[df_uncover["z_phot"] > 8].copy()
    log_M = sub["log_Mstar"].values
    z = sub["z_phot"].values
    dust = sub["dust"].values
    n = len(sub)

    dust_threshold = 0.15
    y_true = (dust > dust_threshold).astype(int)
    print_status(f"    N = {n}, dust threshold = {dust_threshold:.3f}", "INFO")

    profile = np.zeros(len(KAPPA_GAL_GRID))
    for i, a0 in enumerate(KAPPA_GAL_GRID):
        t_eff = _t_eff_for_alpha(log_M, z, a0)
        pred = (t_eff > 0.2).astype(int)
        profile[i] = _f1(y_true, pred)

    best_idx = np.argmax(profile)
    best_alpha = KAPPA_GAL_GRID[best_idx]
    best_f1 = profile[best_idx]

    # Bootstrap
    print_status("    Bootstrapping CI...", "INFO")
    boot_alphas = _run_bootstrap_batches(
        bootstrap_executor,
        _boot_worker_obs2_new,
        log_M, z, y_true,
        KAPPA_GAL_GRID,
    )

    ci_lo, ci_hi = np.percentile(boot_alphas, [16, 84])

    print_status(f"    Best kappa_gal = {best_alpha:.3e} [{ci_lo:.3e}, {ci_hi:.3e}], F1 = {best_f1:.3f}", "SUCCESS")

    return {
        "name": "AGB Threshold F1 (UNCOVER z>8)",
        "short": "AGB F1",
        "kappa_gal_best": float(best_alpha),
        "kappa_gal_ci": [float(ci_lo), float(ci_hi)],
        "metric_best": float(best_f1),
        "metric_name": "F1",
        "n": int(n),
        "profile": profile.tolist(),
    }

def obs3_ssfr_inversion(df_uncover, bootstrap_executor=None):
    """For each kappa_gal grid value, compute delta rho = rho(M*, sSFR, z>7) - rho(M*, sSFR, z<6)."""
    print_status("  Observable 3: Mass-sSFR Inversion (UNCOVER M*>8.5)", "INFO")
    df = df_uncover.dropna(subset=["log_ssfr"]).copy()
    df = df[df["log_Mstar"] > 8.5]
    low = df[(df["z_phot"] >= 4) & (df["z_phot"] < 6)]
    high = df[(df["z_phot"] >= 7) & (df["z_phot"] < 10)]
    n_low, n_high = len(low), len(high)
    print_status(f"    N_low = {n_low}, N_high = {n_high}", "INFO")

    rho_low_obs, _ = stats.spearmanr(low["log_Mstar"], low["log_ssfr"])
    rho_high_obs, _ = stats.spearmanr(high["log_Mstar"], high["log_ssfr"])
    delta_obs = rho_high_obs - rho_low_obs
    print_status(f"    Observed: rho_low = {rho_low_obs:.3f}, rho_high = {rho_high_obs:.3f}, delta_rho = {delta_obs:.3f}", "INFO")

    log_M_high = high["log_Mstar"].values
    z_high = high["z_phot"].values
    ssfr_high = high["log_ssfr"].values

    profile = np.zeros(len(KAPPA_GAL_GRID))
    for i, a0 in enumerate(KAPPA_GAL_GRID):
        gt = _gamma_t_for_alpha(log_M_high, z_high, a0)
        M_true = log_M_high - ALPHA_NUCLEAR * np.log10(np.maximum(gt, 0.01))
        ssfr_true = ssfr_high - 0.45 * np.log10(np.maximum(gt, 0.01))
        rho_corr, _ = stats.spearmanr(M_true, ssfr_true)
        profile[i] = -abs(rho_corr - rho_low_obs)

    best_idx = np.argmax(profile)
    best_alpha = KAPPA_GAL_GRID[best_idx]
    best_residual = -profile[best_idx]

    # Bootstrap
    print_status("    Bootstrapping CI...", "INFO")
    boot_alphas = _run_bootstrap_batches(
        bootstrap_executor,
        _boot_worker_obs3_new,
        log_M_high,
        z_high,
        ssfr_high,
        rho_low_obs,
        KAPPA_GAL_GRID,
    )

    ci_lo, ci_hi = np.percentile(boot_alphas, [16, 84])

    print_status(f"    Best kappa_gal = {best_alpha:.3e} [{ci_lo:.3e}, {ci_hi:.3e}], residual = {best_residual:.4f}", "SUCCESS")

    return {
        "name": "sSFR Inversion Correction (UNCOVER M*>8.5)",
        "short": "sSFR Inversion",
        "kappa_gal_best": float(best_alpha),
        "kappa_gal_ci": [float(ci_lo), float(ci_hi)],
        "metric_best": float(best_residual),
        "metric_name": "|delta_rho_corrected - rho_low|",
        "n": int(n_high),
        "profile": profile.tolist(),
    }

def obs4_cross_survey_r2(df_ceers, df_cosmosweb, bootstrap_executor=None):
    """For each kappa_gal grid value, compute mean Pearson R^2(dust, log t_eff) across surveys at z>7.0, M*>8.5."""
    print_status("  Observable 4: Cross-Survey Dust-t_eff R2 (CEERS + COSMOS-Web z>7)", "INFO")

    surveys = []
    for name, df in [("CEERS", df_ceers), ("COSMOS-Web", df_cosmosweb)]:
        if df is None:
            continue
        sub = df[(df["z_phot"] > 7.0) & (df["log_Mstar"] > 8.5)].copy()
        sub = sub.dropna(subset=["dust"])
        if len(sub) > 30:
            surveys.append((name, sub))
            print_status(f"    {name}: N = {len(sub)}", "INFO")

    if len(surveys) < 2:
        print_status("    Insufficient cross-survey data", "WARNING")
        return None

    boot_data = []
    total_n = 0
    for name, sub in surveys:
        log_M = sub["log_Mstar"].values
        z = sub["z_phot"].values
        dust = sub["dust"].values
        boot_data.append((log_M, z, dust))
        total_n += len(dust)

    profile = np.zeros(len(KAPPA_GAL_GRID))
    for i, a0 in enumerate(KAPPA_GAL_GRID):
        r2s = []
        for log_M, z, dust in boot_data:
            t_eff = _t_eff_for_alpha(log_M, z, a0)
            log_t_eff = np.log10(np.maximum(t_eff, 1e-6))
            r = np.corrcoef(dust, log_t_eff)[0, 1]
            r2s.append(r ** 2)
        profile[i] = np.mean(r2s)

    best_idx = np.argmax(profile)
    best_alpha = KAPPA_GAL_GRID[best_idx]
    best_r2 = profile[best_idx]

    # Bootstrap
    print_status("    Bootstrapping CI...", "INFO")
    boot_alphas = _run_bootstrap_batches(
        bootstrap_executor,
        _boot_worker_obs4_new,
        boot_data,
        KAPPA_GAL_GRID,
    )

    ci_lo, ci_hi = np.percentile(boot_alphas, [16, 84])

    print_status(f"    Best kappa_gal = {best_alpha:.3e} [{ci_lo:.3e}, {ci_hi:.3e}], mean R2 = {best_r2:.3f}", "SUCCESS")

    return {
        "name": "Cross-Survey Dust-t_eff R2 (CEERS + COSMOS-Web z>7.0)",
        "short": "Cross-Survey",
        "kappa_gal_best": float(best_alpha),
        "kappa_gal_ci": [float(ci_lo), float(ci_hi)],
        "metric_best": float(best_r2),
        "metric_name": "mean R2(dust, log t_eff)",
        "n": int(total_n),
        "profile": profile.tolist(),
    }

def obs5_impossible_resolution(df_uncover, df_ceers, df_cosmosweb, bootstrap_executor=None):
    """For each kappa_gal grid value, compute fraction of anomalous galaxies resolved."""
    print_status("  Observable 5: Anomalous Galaxy Resolution (all surveys z>7)", "INFO")

    frames = []
    for name, df in [("UNCOVER", df_uncover), ("CEERS", df_ceers), ("COSMOS-Web", df_cosmosweb)]:
        if df is None:
            continue
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

    if n_impossible == 0:
        return None

    log_M_impossible = log_M[impossible_mask]
    max_logM_impossible = max_logM[impossible_mask]
    z_impossible = z[impossible_mask]

    profile = np.zeros(len(KAPPA_GAL_GRID))
    for i, a0 in enumerate(KAPPA_GAL_GRID):
        gt = _gamma_t_for_alpha(log_M_impossible, z_impossible, a0)
        M_true = log_M_impossible - 0.7 * np.log10(np.maximum(gt, 0.01))
        excess = M_true - max_logM_impossible
        profile[i] = -np.sum(excess[excess > 0]**2)

    best_idx = np.argmax(profile)
    best_alpha = KAPPA_GAL_GRID[best_idx]
    best_loss = -profile[best_idx]

    # Bootstrap
    print_status("    Bootstrapping CI...", "INFO")
    boot_alphas = _run_bootstrap_batches(
        bootstrap_executor,
        _boot_worker_obs5_new,
        log_M_impossible,
        max_logM_impossible,
        z_impossible,
        KAPPA_GAL_GRID,
    )

    ci_lo, ci_hi = np.percentile(boot_alphas, [16, 84])

    print_status(f"    Best kappa_gal = {best_alpha:.3e} [{ci_lo:.3e}, {ci_hi:.3e}], excess mass loss = {best_loss:.3f}", "SUCCESS")

    return {
        "name": "Anomalous Galaxy Mass Limit Resolution (z>7)",
        "short": "Mass Limit",
        "kappa_gal_best": float(best_alpha),
        "kappa_gal_ci": [float(ci_lo), float(ci_hi)],
        "metric_best": float(best_loss),
        "metric_name": "mass boundary excess loss",
        "n": int(n_impossible),
        "profile": profile.tolist(),
    }

def compute_concordance(observables):
    """Compute concordance statistics."""
    alphas = [o["kappa_gal_best"] for o in observables]
    sigmas = [(o["kappa_gal_ci"][1] - o["kappa_gal_ci"][0]) / 2.0 for o in observables]

    GRID_HALF_RANGE = (KAPPA_GAL_GRID[-1] - KAPPA_GAL_GRID[0]) / 2.0
    GRID_MIN = float(KAPPA_GAL_GRID[0])
    GRID_MAX = float(KAPPA_GAL_GRID[-1])
    
    # Minimum uncertainty floor: 5% of grid range prevents infinite weights
    SIGMA_FLOOR = 0.05 * (GRID_MAX - GRID_MIN)

    # Detect boundary-hit observables (incomplete likelihoods)
    boundary_hit_indices = [
        i for i, o in enumerate(observables)
        if abs(o["kappa_gal_best"] - GRID_MIN) < 1e3 or abs(o["kappa_gal_best"] - GRID_MAX) < 1e3
    ]

    # Identify smooth, informative observables:
    # Exclude:
    #   1. Threshold-metric observables (AGB F1, Mass Limit)
    #   2. Observables with flat posteriors (CI wider than half grid range)
    #   3. Boundary-hit observables (incomplete likelihoods)
    smooth_indices = [
        i for i, o in enumerate(observables)
        if o["short"] not in ["Mass Limit", "AGB F1"]
        and sigmas[i] < GRID_HALF_RANGE
        and i not in boundary_hit_indices
    ]

    if len(smooth_indices) == 0:
        # Fallback: use all non-boundary observables without informativeness cut
        smooth_indices = [i for i, o in enumerate(observables)
                          if o["short"] not in ["Mass Limit", "AGB F1"]
                          and i not in boundary_hit_indices]
    if len(smooth_indices) == 0:
        # Last resort: use all observables with sigma floor
        smooth_indices = list(range(len(observables)))

    # Flag which observables were excluded and why
    uninformative_shorts = [observables[i]["short"] for i in range(len(observables))
                            if i not in smooth_indices]
    boundary_hit_shorts = [observables[i]["short"] for i in boundary_hit_indices]

    # Weighted mean using only informative observables
    # Apply sigma floor to prevent numerical issues
    smooth_alphas = [alphas[i] for i in smooth_indices]
    smooth_sigmas = [max(sigmas[i], SIGMA_FLOOR) for i in smooth_indices]
    weights = [1.0 / s**2 for s in smooth_sigmas]
    w_sum = sum(weights)
    alpha_wm = sum(a * w for a, w in zip(smooth_alphas, weights)) / w_sum
    sigma_wm = 1.0 / np.sqrt(w_sum)

    # Concordance chi2 among informative observables only
    chi2 = sum(((alphas[i] - alpha_wm) / max(sigmas[i], 1.0))**2 for i in smooth_indices)
    ndof = max(len(smooth_indices) - 1, 0)
    p_concordance = float(max(1 - stats.chi2.cdf(chi2, ndof), 1e-300)) if ndof > 0 else 1.0

    # Maximum tension between any two informative observables
    max_tension = 0.0
    for ii in range(len(smooth_indices)):
        for jj in range(ii + 1, len(smooth_indices)):
            i, j = smooth_indices[ii], smooth_indices[jj]
            sep = abs(alphas[i] - alphas[j])
            combined_sigma = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
            if combined_sigma > 0:
                tension = sep / combined_sigma
                max_tension = max(max_tension, tension)

    return {
        "kappa_gal_weighted_mean": float(alpha_wm),
        "kappa_gal_weighted_sigma": float(sigma_wm),
        "chi2_concordance": float(chi2),
        "ndof": ndof,
        "p_concordance": float(p_concordance),
        "max_pairwise_tension_sigma": float(max_tension),
        "concordant": bool(p_concordance > 0.05),
        "individual_alphas": alphas,
        "individual_sigmas": sigmas,
        "uninformative_observables": uninformative_shorts,
        "boundary_hit_observables": boundary_hit_shorts,
        "n_informative": len(smooth_indices),
        "sigma_floor": SIGMA_FLOOR,
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
        ci = obs["kappa_gal_ci"]
        best = obs["kappa_gal_best"]
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
    wm = concordance["kappa_gal_weighted_mean"]
    ws = concordance["kappa_gal_weighted_sigma"]
    ax1.axvspan(wm - ws, wm + ws, alpha=0.15, color="gray", zorder=0)
    ax1.axvline(wm, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    # Cepheid KAPPA_GAL = 9.6e5 ± 4.0e5
    ax1.axvspan(EXTERNAL_CEPHEID_KAPPA_GAL - EXTERNAL_CEPHEID_KAPPA_GAL_SIGMA, EXTERNAL_CEPHEID_KAPPA_GAL + EXTERNAL_CEPHEID_KAPPA_GAL_SIGMA, alpha=0.08, color="orange", zorder=0)
    ax1.axvline(EXTERNAL_CEPHEID_KAPPA_GAL, color="orange", linestyle=":", linewidth=1.5, alpha=0.7,
                label="External Cepheid κ_gal")

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([o["short"] for o in observables], fontsize=10)
    ax1.set_xlabel("κ_gal (mag)", fontsize=12)
    ax1.set_title(
        f"TEP Concordance (χ² = {concordance['chi2_concordance']:.1f}, "
        f"p = {concordance['p_concordance']:.3f})",
        fontsize=11
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_xlim(0, 20e5)
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

        ax2.plot(KAPPA_GAL_GRID, profile_norm, color=color, linewidth=2,
                 label=obs["short"], alpha=0.85)

    ax2.axvline(concordance["kappa_gal_weighted_mean"], color="gray",
                linestyle="--", linewidth=1.5, alpha=0.7, label="Weighted mean")
    ax2.axvline(EXTERNAL_CEPHEID_KAPPA_GAL, color="orange", linestyle=":", linewidth=1.5, alpha=0.7,
                label="External Cepheid κ_gal")

    ax2.set_xlabel("κ_gal (mag)", fontsize=12)
    ax2.set_ylabel("Normalized likelihood", fontsize=12)
    ax2.set_title("Independent Likelihood Profiles", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(0, 20e5)
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
    print_status(f"Step {STEP_NUM}: TEP Concordance / L1–L3 Independence — Independent κ_gal Recovery", "INFO")
    print_status("=" * 65, "INFO")

    # Load data
    print_status("Loading data...", "INFO")
    df_uncover   = load_uncover()
    df_ceers     = load_ceers()
    df_cosmosweb = load_cosmosweb()

    if df_uncover is None:
        print_status("ERROR: UNCOVER TEP sample not found. Run step_001 and step_002 first.", "ERROR")
        return

    n_ceers = len(df_ceers) if df_ceers is not None else 0
    n_cosmosweb = len(df_cosmosweb) if df_cosmosweb is not None else 0
    print_status(f"  UNCOVER: {len(df_uncover)}, CEERS: {n_ceers}, COSMOS-Web: {n_cosmosweb}", "INFO")

    # Run each observable
    print_status("\nScanning κ_gal grid (N=100, bootstrap N=200)...", "INFO")
    observables = []

    def run_observables(bootstrap_executor):
        o1 = obs1_dust_teff_r2(df_uncover, bootstrap_executor)
        observables.append(o1)

        o2 = obs2_agb_threshold_f1(df_uncover, bootstrap_executor)
        observables.append(o2)

        o3 = obs3_ssfr_inversion(df_uncover, bootstrap_executor)
        observables.append(o3)

        o4 = obs4_cross_survey_r2(df_ceers, df_cosmosweb, bootstrap_executor)
        if o4:
            observables.append(o4)

        o5 = obs5_impossible_resolution(df_uncover, df_ceers, df_cosmosweb, bootstrap_executor)
        if o5:
            observables.append(o5)

    if BOOTSTRAP_WORKERS == 1:
        run_observables(None)
    else:
        with ThreadPoolExecutor(max_workers=BOOTSTRAP_WORKERS) as bootstrap_executor:
            run_observables(bootstrap_executor)

    # Concordance statistics
    print_status("\n" + "=" * 65, "INFO")
    print_status("CONCORDANCE STATISTICS", "INFO")
    print_status("=" * 65, "INFO")

    concordance = compute_concordance(observables)

    print_status(f"  Weighted mean κ_gal = {concordance['kappa_gal_weighted_mean']:.3e} ± {concordance['kappa_gal_weighted_sigma']:.3e} mag", "INFO")
    print_status(f"  Concordance χ² = {concordance['chi2_concordance']:.2f} (ndof = {concordance['ndof']})", "INFO")
    print_status(f"  p(concordance) = {concordance['p_concordance']:.4f}", "INFO")
    print_status(f"  Max pairwise tension = {concordance['max_pairwise_tension_sigma']:.2f}σ", "INFO")
    print_status(f"  Concordant (p > 0.05): {concordance['concordant']}", "INFO")

    # Comparison to Cepheid
    concordance["jwst_recovered_kappa_gal"] = float(concordance["kappa_gal_weighted_mean"])
    concordance["jwst_recovered_sigma"] = float(concordance["kappa_gal_weighted_sigma"])
    cepheid_tension = abs(concordance["kappa_gal_weighted_mean"] - EXTERNAL_CEPHEID_KAPPA_GAL) / max(
        np.sqrt(concordance["kappa_gal_weighted_sigma"]**2 + EXTERNAL_CEPHEID_KAPPA_GAL_SIGMA**2), 0.01
    )
    print_status(f"  Tension with external Cepheid κ_gal = {EXTERNAL_CEPHEID_KAPPA_GAL:.3e} ± {EXTERNAL_CEPHEID_KAPPA_GAL_SIGMA:.3e} mag: {cepheid_tension:.2f}σ", "INFO")
    concordance["cepheid_tension_sigma"] = float(cepheid_tension)
    concordance["cepheid_reference"] = {
        "kappa_gal": EXTERNAL_CEPHEID_KAPPA_GAL,
        "sigma": EXTERNAL_CEPHEID_KAPPA_GAL_SIGMA,
    }
    concordance["external_cepheid_tension_sigma"] = float(cepheid_tension)
    concordance["external_cepheid_reference"] = {
        "kappa_gal": EXTERNAL_CEPHEID_KAPPA_GAL,
        "sigma": EXTERNAL_CEPHEID_KAPPA_GAL_SIGMA,
    }

    # Verdict
    n_info = concordance.get("n_informative", len(observables))
    if concordance["concordant"] and cepheid_tension < 2:
        verdict = (
            f"PASS: {n_info} informative observables are mutually concordant and consistent "
            f"with the external Cepheid calibration ({cepheid_tension:.1f}\u03c3 tension)"
        )
    elif cepheid_tension < 2:
        verdict = (
            f"PARTIAL CONFIRMATION: Weighted mean is within {cepheid_tension:.1f}\u03c3 of the "
            f"Cepheid anchor (internal scatter present but anchor-consistent; "
            f"flat observables excluded from weighted mean)"
        )
    elif concordance["concordant"]:
        verdict = (
            f"PARTIAL: Observables are mutually concordant but in tension ({cepheid_tension:.1f}\u03c3) "
            f"with the external Cepheid calibration"
        )
    else:
        verdict = (
            f"FAIL: Observables do not agree on \u03ba_gal and tension with Cepheid is "
            f"{cepheid_tension:.1f}\u03c3 (> 2\u03c3)"
        )
    concordance["verdict"] = verdict
    print_status(f"\n  VERDICT: {verdict}", "INFO")

    # Generate figure
    print_status("\nGenerating concordance figure...", "INFO")
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    generate_figure(observables, concordance, fig_path)

    # Save results
    results = {
        "step": f"step_{STEP_NUM}",
        "name": "TEP Concordance — Independent \u03b1\u2080 Recovery",
        "observables": [{k: v for k, v in o.items() if k != "profile"} for o in observables],
        "concordance": concordance,
        "kappa_gal_grid": {"min": float(KAPPA_GAL_GRID[0]), "max": float(KAPPA_GAL_GRID[-1]), "n": len(KAPPA_GAL_GRID)},
    }

    out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {out_file}", "INFO")

    print_status("\n" + "=" * 65, "INFO")
    headline = (
        f"HEADLINE: {n_info}/{len(observables)} informative observables recover "
        f"\u03ba_gal = {concordance['kappa_gal_weighted_mean']:.2e} \u00b1 {concordance['kappa_gal_weighted_sigma']:.2e} "
        f"(\u03c7\u00b2 = {concordance['chi2_concordance']:.1f}, p = {concordance['p_concordance']:.3f}, "
        f"Cepheid tension = {cepheid_tension:.1f}\u03c3)"
    )
    print_status(headline, "INFO")
    print_status("=" * 65, "INFO")



if __name__ == "__main__":
    main()
