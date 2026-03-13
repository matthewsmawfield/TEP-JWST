#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import print_status  # Centralised logging
from scripts.utils.rank_stats import bootstrap_partial_rank_ci, partial_rank_correlation  # Partial Spearman helpers
from scripts.utils.tep_model import ALPHA_0 as TEP_ALPHA_0  # Shared TEP coupling constant

STEP_NUM = 170  # Pipeline step number
STEP_NAME = "kinematic_decisive_test"  # Used in log / output filenames
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
INPUT_JSON = PROJECT_ROOT / "data" / "interim" / "suspense_kinematics_ages.json"  # SUSPENSE ingested data
KINEMATICS_TEX = PROJECT_ROOT / "data" / "raw" / "literature_kinematics" / "suspense" / "aa55812-25corr.tex"  # Raw SUSPENSE LaTeX
AGES_TEX = PROJECT_ROOT / "data" / "raw" / "literature_kinematics" / "suspense_ages" / "table_quiescent.tex"  # SUSPENSE ages table
STEP117_JSON = OUTPUT_PATH / "step_117_dynamical_mass_comparison.json"  # L4 dynamical mass results
STEP169_JSON = OUTPUT_PATH / "step_169_dja_sigma_pilot.json"  # DJA sigma pilot results
STEP169_CSV = INTERIM_PATH / "step_169_dja_sigma_pilot.csv"  # DJA sigma pilot data
SAME_REGIME_JSON = PROJECT_ROOT / "data" / "interim" / "same_regime_literature_kinematic_sample.json"  # Same-regime kinematic sample
STEP171_JSON = OUTPUT_PATH / "step_171_sigma_kinematic_expansion.json"  # Sigma-based kinematic expansion

H0 = 70.0
OMEGA_M = 0.3
OMEGA_L = 0.7
ALPHA_0 = TEP_ALPHA_0  # from shared tep_model
M_REF = 1e10  # power-law reference mass for dynamical-mass Gamma_t proxy
# NOTE: This step uses a power-law form Gamma_t = (M_dyn / M_REF)^alpha_0 rather
# than the full exponential model because the SUSPENSE inputs are dynamical masses
# (not halo masses). Converting Mdyn -> Mhalo would introduce additional uncertainty.
# For the ranking-based Steiger comparison, only the monotonic ordering matters,
# and the z-dependent effects are absorbed by the partial correlations' z-control.
N_BOOT = 2000
N_MONTE_CARLO = 2000
RNG_SEED = 170
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
MIN_DJA_BALMER_N_FOR_SUPPORTIVE_TALLY = 8


def get_age_universe(z):
    def e_inv(z_prime):
        return 1.0 / np.sqrt(OMEGA_M * (1.0 + z_prime) ** 3 + OMEGA_L)
    integral, _ = quad(e_inv, float(z), np.inf)
    return (9.778 / H0) * integral


def _clean_object_id(text):
    return (
        str(text)
        .replace(r"\tablefootmark{c}", "")
        .replace(r"\tablefootmark{d}", "")
        .replace(r"\textsuperscript{\textdagger}", "")
        .replace("*", "")
        .strip()
    )


def _parse_tex_value(text):
    clean = str(text).replace("$", "").strip()
    pm_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*\\pm\s*([-+]?\d+(?:\.\d+)?)", clean)
    if pm_match:
        value = float(pm_match.group(1))
        err = float(pm_match.group(2))
        return value, err, err
    value_match = FLOAT_RE.search(clean.replace(">", ""))
    if value_match is None:
        return None, None, None
    value = float(value_match.group())
    plus_match = re.search(r"\^\{\+?([-+]?\d+(?:\.\d+)?)\}", clean)
    minus_match = re.search(r"_\{\-?([-+]?\d+(?:\.\d+)?)\}", clean)
    err_lo = float(minus_match.group(1)) if minus_match else None
    err_hi = float(plus_match.group(1)) if plus_match else None
    return value, err_lo, err_hi


def _load_published_uncertainties():
    kin = {}
    ages = {}
    if KINEMATICS_TEX.exists():
        in_table = False
        for raw in KINEMATICS_TEX.read_text().splitlines():
            if r"\begin{tabular}{l l l l l l l l l l l l}" in raw:
                in_table = True
                continue
            if not in_table:
                continue
            if r"\end{tabular}" in raw:
                break
            line = raw.strip()
            if (
                not line
                or "&" not in line
                or line.startswith(r"\hline")
                or line.startswith(r"\multicolumn")
                or line.startswith(r"\cmidrule")
                or line.startswith("ID &")
                or line.startswith("& & (kpc)")
            ):
                continue
            parts = [part.strip() for part in line.split("&")]
            if len(parts) < 12:
                continue
            object_id = f"SUSPENSE-{_clean_object_id(parts[0])}"
            _, mdyn_lo, mdyn_hi = _parse_tex_value(parts[10])
            _, mstar_lo, mstar_hi = _parse_tex_value(parts[11])
            kin[object_id] = {
                "log_Mdyn_err_lo": mdyn_lo,
                "log_Mdyn_err_hi": mdyn_hi,
                "log_Mstar_err_lo": mstar_lo,
                "log_Mstar_err_hi": mstar_hi,
            }
    if AGES_TEX.exists():
        for raw in AGES_TEX.read_text().splitlines():
            line = raw.strip()
            if not line or "&" not in line:
                continue
            parts = [part.strip() for part in line.split("&")]
            if len(parts) < 10:
                continue
            object_id = f"SUSPENSE-{_clean_object_id(parts[0])}"
            _, age_lo, age_hi = _parse_tex_value(parts[9])
            ages[object_id] = {
                "age_gyr_err_lo": age_lo,
                "age_gyr_err_hi": age_hi,
            }
    return kin, ages


def _attach_published_uncertainties(df):
    df = df.copy()
    kin, ages = _load_published_uncertainties()
    for col in ["log_Mdyn_err_lo", "log_Mdyn_err_hi", "log_Mstar_err_lo", "log_Mstar_err_hi", "age_gyr_err_lo", "age_gyr_err_hi"]:
        df[col] = np.nan
    for idx, object_id in enumerate(df["object_id"].astype(str)):
        if object_id in kin:
            for key, value in kin[object_id].items():
                df.at[idx, key] = value
        if object_id in ages:
            for key, value in ages[object_id].items():
                df.at[idx, key] = value
    coverage = {
        "n_with_log_mdyn_uncertainty": int(df["log_Mdyn_err_lo"].notna().sum()),
        "n_with_log_mstar_uncertainty": int(df["log_Mstar_err_lo"].notna().sum()),
        "n_with_age_uncertainty": int(df["age_gyr_err_lo"].notna().sum()),
        "n_with_all_three_uncertainties": int((df["log_Mdyn_err_lo"].notna() & df["log_Mstar_err_lo"].notna() & df["age_gyr_err_lo"].notna()).sum()),
    }
    return df, coverage


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return None, None, int(mask.sum())
    rho, p_value = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p_value), int(mask.sum())


def _rank_residuals(values, controls):
    values = np.asarray(values, dtype=float)
    controls = [np.asarray(ctrl, dtype=float) for ctrl in controls]
    mask = np.isfinite(values)
    for ctrl in controls:
        mask &= np.isfinite(ctrl)
    resid = np.full(values.shape, np.nan, dtype=float)
    if int(mask.sum()) < 4:
        return resid, mask
    y_rank = stats.rankdata(values[mask]).astype(float)
    x_rank = np.column_stack([stats.rankdata(ctrl[mask]).astype(float) for ctrl in controls])
    design = np.column_stack([x_rank, np.ones(int(mask.sum()), dtype=float)])
    beta = np.linalg.lstsq(design, y_rank, rcond=None)[0]
    resid[mask] = y_rank - design @ beta
    return resid, mask


def steiger_z_dependent(r12, r13, r23, n):
    if n < 5 or any(v is None or not np.isfinite(v) for v in [r12, r13, r23]):
        return None, None
    z12 = np.arctanh(np.clip(r12, -0.9999, 0.9999))
    z13 = np.arctanh(np.clip(r13, -0.9999, 0.9999))
    r_bar = 0.5 * (r12 + r13)
    one_minus_rbar_sq = 1.0 - r_bar**2
    if not np.isfinite(one_minus_rbar_sq) or abs(one_minus_rbar_sq) < 1e-12:
        return None, None
    f_term = (1.0 - r23) / (2.0 * one_minus_rbar_sq)
    h_term = (1.0 - f_term * r_bar**2) / (1.0 - r_bar**2)
    denom = np.sqrt((2.0 * (1.0 - r23)) / (n - 3.0) * h_term)
    if not np.isfinite(denom) or denom <= 0:
        return None, None
    z_stat = (z13 - z12) / denom
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_value)


def _steiger_given_z(age, log_mstar, gamma_dyn, z):
    age_resid, m_age = _rank_residuals(age, [z])
    mass_resid, m_mass = _rank_residuals(log_mstar, [z])
    gamma_resid, m_gamma = _rank_residuals(gamma_dyn, [z])
    mask = m_age & m_mass & m_gamma & np.isfinite(age_resid) & np.isfinite(mass_resid) & np.isfinite(gamma_resid)
    n = int(mask.sum())
    if n < 4:
        return {
            "n": n,
            "rho_outcome_mass_given_z": None,
            "rho_outcome_direct_given_z": None,
            "rho_mass_direct_given_z": None,
            "delta_rho_direct_minus_mass_given_z": None,
            "z_stat_direct_better_than_mass": None,
            "p_value_direct_better_than_mass": None,
        }
    r_age_mass = float(np.corrcoef(age_resid[mask], mass_resid[mask])[0, 1])
    r_age_gamma = float(np.corrcoef(age_resid[mask], gamma_resid[mask])[0, 1])
    r_mass_gamma = float(np.corrcoef(mass_resid[mask], gamma_resid[mask])[0, 1])
    z_stat, p_value = steiger_z_dependent(r_age_mass, r_age_gamma, r_mass_gamma, n)
    return {
        "n": n,
        "rho_outcome_mass_given_z": r_age_mass,
        "rho_outcome_direct_given_z": r_age_gamma,
        "rho_mass_direct_given_z": r_mass_gamma,
        "delta_rho_direct_minus_mass_given_z": float(r_age_gamma - r_age_mass),
        "z_stat_direct_better_than_mass": z_stat,
        "p_value_direct_better_than_mass": p_value,
    }


def _bootstrap_steiger(age, log_mstar, gamma_dyn, z, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(age)
    z_stats = []
    deltas = []
    gamma_better = 0
    gamma_better_sig = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        out = _steiger_given_z(age[idx], log_mstar[idx], gamma_dyn[idx], z[idx])
        z_stat = out["z_stat_direct_better_than_mass"]
        delta = out["delta_rho_direct_minus_mass_given_z"]
        p_value = out["p_value_direct_better_than_mass"]
        if z_stat is not None and np.isfinite(z_stat):
            z_stats.append(float(z_stat))
            if z_stat > 0:
                gamma_better += 1
                if p_value is not None and np.isfinite(p_value) and p_value < 0.05:
                    gamma_better_sig += 1
        if delta is not None and np.isfinite(delta):
            deltas.append(float(delta))
    z_stats = np.asarray(z_stats, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    return {
        "n_boot": int(len(z_stats)),
        "z_stat_ci_95": [
            float(np.percentile(z_stats, 2.5)) if len(z_stats) >= 30 else None,
            float(np.percentile(z_stats, 97.5)) if len(z_stats) >= 30 else None,
        ],
        "delta_rho_ci_95": [
            float(np.percentile(deltas, 2.5)) if len(deltas) >= 30 else None,
            float(np.percentile(deltas, 97.5)) if len(deltas) >= 30 else None,
        ],
        "support_fraction_gamma_dyn_better": float(gamma_better / len(z_stats)) if len(z_stats) else None,
        "support_fraction_gamma_dyn_better_and_significant": float(gamma_better_sig / len(z_stats)) if len(z_stats) else None,
    }


def _draw_asymmetric(center, err_lo, err_hi, rng):
    center = np.asarray(center, dtype=float)
    err_lo = np.asarray(err_lo, dtype=float)
    err_hi = np.asarray(err_hi, dtype=float)
    draws = rng.standard_normal(size=center.shape)
    sigma_neg = np.where(np.isfinite(err_lo) & (err_lo > 0), err_lo, np.where(np.isfinite(err_hi) & (err_hi > 0), err_hi, 0.0))
    sigma_pos = np.where(np.isfinite(err_hi) & (err_hi > 0), err_hi, np.where(np.isfinite(err_lo) & (err_lo > 0), err_lo, 0.0))
    sigma = np.where(draws < 0, sigma_neg, sigma_pos)
    return center + draws * sigma


def _summarize_mc(values, p_values=None):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"median": None, "ci_95": [None, None], "support_fraction_positive": None, "support_fraction_p_lt_0_05": None}
    out = {
        "median": float(np.median(arr)),
        "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "support_fraction_positive": float(np.mean(arr > 0)),
        "support_fraction_p_lt_0_05": None,
    }
    if p_values is not None:
        p_arr = np.asarray([p for p in p_values if p is not None and np.isfinite(p)], dtype=float)
        if len(p_arr):
            out["support_fraction_p_lt_0_05"] = float(np.mean(p_arr < 0.05))
    return out


def _compute_direct_decoupling_metrics(outcome, mass, direct, z, with_bootstrap=True):
    outcome = np.asarray(outcome, dtype=float)
    mass = np.asarray(mass, dtype=float)
    direct = np.asarray(direct, dtype=float)
    z = np.asarray(z, dtype=float)

    rho_mass_outcome, p_mass_outcome, _ = _safe_spearman(mass, outcome)
    rho_direct_outcome, p_direct_outcome, _ = _safe_spearman(direct, outcome)
    rho_mass_direct, p_mass_direct, _ = _safe_spearman(mass, direct)

    partial_direct_z, p_partial_direct_z, n_partial_direct_z = partial_rank_correlation(direct, outcome, z)
    partial_mass_z, p_partial_mass_z, n_partial_mass_z = partial_rank_correlation(mass, outcome, z)
    partial_direct_mass_z, p_partial_direct_mass_z, n_partial_direct_mass_z = partial_rank_correlation(
        direct,
        outcome,
        np.column_stack([mass, z]),
    )
    partial_mass_direct_z, p_partial_mass_direct_z, n_partial_mass_direct_z = partial_rank_correlation(
        mass,
        outcome,
        np.column_stack([direct, z]),
    )

    steiger = _steiger_given_z(outcome, mass, direct, z)
    if with_bootstrap:
        ci_partial_direct_z = bootstrap_partial_rank_ci(direct, outcome, z, n_boot=N_BOOT, seed=RNG_SEED)
        ci_partial_mass_z = bootstrap_partial_rank_ci(mass, outcome, z, n_boot=N_BOOT, seed=RNG_SEED)
        ci_partial_direct_mass_z = bootstrap_partial_rank_ci(
            direct,
            outcome,
            np.column_stack([mass, z]),
            n_boot=N_BOOT,
            seed=RNG_SEED,
        )
        ci_partial_mass_direct_z = bootstrap_partial_rank_ci(
            mass,
            outcome,
            np.column_stack([direct, z]),
            n_boot=N_BOOT,
            seed=RNG_SEED,
        )
        steiger_boot = _bootstrap_steiger(outcome, mass, direct, z)
    else:
        ci_partial_direct_z = [np.nan, np.nan]
        ci_partial_mass_z = [np.nan, np.nan]
        ci_partial_direct_mass_z = [np.nan, np.nan]
        ci_partial_mass_direct_z = [np.nan, np.nan]
        steiger_boot = None

    return {
        "rho_mass_outcome": rho_mass_outcome,
        "p_mass_outcome": p_mass_outcome,
        "rho_direct_outcome": rho_direct_outcome,
        "p_direct_outcome": p_direct_outcome,
        "rho_mass_direct": rho_mass_direct,
        "p_mass_direct": p_mass_direct,
        "partial_rho_direct_outcome_given_z": float(partial_direct_z),
        "p_partial_direct_outcome_given_z": float(p_partial_direct_z),
        "n_partial_direct_outcome_given_z": int(n_partial_direct_z),
        "partial_rho_mass_outcome_given_z": float(partial_mass_z),
        "p_partial_mass_outcome_given_z": float(p_partial_mass_z),
        "n_partial_mass_outcome_given_z": int(n_partial_mass_z),
        "partial_rho_direct_outcome_given_mass_z": float(partial_direct_mass_z),
        "p_partial_direct_outcome_given_mass_z": float(p_partial_direct_mass_z),
        "n_partial_direct_outcome_given_mass_z": int(n_partial_direct_mass_z),
        "partial_rho_mass_outcome_given_direct_z": float(partial_mass_direct_z),
        "p_partial_mass_outcome_given_direct_z": float(p_partial_mass_direct_z),
        "n_partial_mass_outcome_given_direct_z": int(n_partial_mass_direct_z),
        "delta_partial_rho_direct_minus_mass_given_z": float(partial_direct_z - partial_mass_z),
        "delta_partial_rho_direct_minus_mass_given_competitor_z": float(partial_direct_mass_z - partial_mass_direct_z),
        "partial_direct_outcome_given_z_ci_95": [float(ci_partial_direct_z[0]), float(ci_partial_direct_z[1])],
        "partial_mass_outcome_given_z_ci_95": [float(ci_partial_mass_z[0]), float(ci_partial_mass_z[1])],
        "partial_direct_outcome_given_mass_z_ci_95": [float(ci_partial_direct_mass_z[0]), float(ci_partial_direct_mass_z[1])],
        "partial_mass_outcome_given_direct_z_ci_95": [float(ci_partial_mass_direct_z[0]), float(ci_partial_mass_direct_z[1])],
        "steiger_direct_vs_mass_given_z": steiger,
        "steiger_direct_vs_mass_given_z_bootstrap": steiger_boot,
    }


def _uncertainty_monte_carlo(df):
    needed = ["log_Mdyn_err_lo", "log_Mdyn_err_hi", "log_Mstar_err_lo", "log_Mstar_err_hi", "age_gyr_err_lo", "age_gyr_err_hi"]
    if any(col not in df.columns for col in needed):
        return None
    mask = np.ones(len(df), dtype=bool)
    for col in needed:
        mask &= np.isfinite(df[col].to_numpy(dtype=float))
    if int(mask.sum()) < 4:
        return None
    sub = df.loc[mask].copy().reset_index(drop=True)
    rng = np.random.default_rng(RNG_SEED)
    gamma_z = []
    gamma_z_p = []
    mass_z = []
    mass_z_p = []
    gamma_cond = []
    gamma_cond_p = []
    mass_cond = []
    mass_cond_p = []
    delta = []
    steiger_z = []
    steiger_p = []
    for _ in range(N_MONTE_CARLO):
        sim = sub.copy()
        sim["log_Mdyn"] = _draw_asymmetric(sim["log_Mdyn"], sim["log_Mdyn_err_lo"], sim["log_Mdyn_err_hi"], rng)
        sim["log_Mstar_obs"] = _draw_asymmetric(sim["log_Mstar_obs"], sim["log_Mstar_err_lo"], sim["log_Mstar_err_hi"], rng)
        sim["age_gyr"] = np.maximum(_draw_asymmetric(sim["age_gyr"], sim["age_gyr_err_lo"], sim["age_gyr_err_hi"], rng), 1e-6)
        sim["Gamma_t_dyn"] = np.power(10.0 ** sim["log_Mdyn"].to_numpy(dtype=float) / M_REF, ALPHA_0)
        sim["Gamma_t_phot"] = np.power(10.0 ** sim["log_Mstar_obs"].to_numpy(dtype=float) / M_REF, ALPHA_0)
        sim["t_tep"] = sim["t_univ"].to_numpy(dtype=float) * sim["Gamma_t_dyn"].to_numpy(dtype=float)
        metrics = _compute_metrics(sim, with_bootstrap=False)
        gamma_z.append(metrics["partial_rho_gamma_dyn_age_given_z"])
        gamma_z_p.append(metrics["p_partial_gamma_dyn_age_given_z"])
        mass_z.append(metrics["partial_rho_mstar_age_given_z"])
        mass_z_p.append(metrics["p_partial_mstar_age_given_z"])
        gamma_cond.append(metrics["partial_rho_gamma_dyn_age_given_mstar_z"])
        gamma_cond_p.append(metrics["p_partial_gamma_dyn_age_given_mstar_z"])
        mass_cond.append(metrics["partial_rho_mstar_age_given_gamma_dyn_z"])
        mass_cond_p.append(metrics["p_partial_mstar_age_given_gamma_dyn_z"])
        delta.append(metrics["delta_partial_rho_gamma_minus_mstar_given_z"])
        steiger_z.append(metrics["steiger_gamma_dyn_vs_mstar_given_z"]["z_stat_gamma_dyn_better_than_mstar"])
        steiger_p.append(metrics["steiger_gamma_dyn_vs_mstar_given_z"]["p_value_gamma_dyn_better_than_mstar"])
    return {
        "n_draws": int(N_MONTE_CARLO),
        "partial_gamma_dyn_age_given_z": _summarize_mc(gamma_z, gamma_z_p),
        "partial_mstar_age_given_z": _summarize_mc(mass_z, mass_z_p),
        "partial_gamma_dyn_age_given_mstar_z": _summarize_mc(gamma_cond, gamma_cond_p),
        "partial_mstar_age_given_gamma_dyn_z": _summarize_mc(mass_cond, mass_cond_p),
        "delta_partial_rho_gamma_minus_mstar_given_z": _summarize_mc(delta),
        "steiger_z_gamma_dyn_better_than_mstar": _summarize_mc(steiger_z, steiger_p),
    }


def _compute_metrics(df, with_bootstrap=True):
    age = df["age_gyr"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    log_mstar = df["log_Mstar_obs"].to_numpy(dtype=float)
    log_mdyn = df["log_Mdyn"].to_numpy(dtype=float)
    gamma_dyn = df["Gamma_t_dyn"].to_numpy(dtype=float)
    t_tep = df["t_tep"].to_numpy(dtype=float)

    generic = _compute_direct_decoupling_metrics(age, log_mstar, gamma_dyn, z, with_bootstrap=with_bootstrap)
    rho_mdyn_age, p_mdyn_age, _ = _safe_spearman(log_mdyn, age)
    rho_tep_age, p_tep_age, _ = _safe_spearman(t_tep, age)

    return {
        "rho_mstar_age": generic["rho_mass_outcome"],
        "p_mstar_age": generic["p_mass_outcome"],
        "rho_log_mdyn_age": rho_mdyn_age,
        "p_log_mdyn_age": p_mdyn_age,
        "rho_gamma_dyn_age": generic["rho_direct_outcome"],
        "p_gamma_dyn_age": generic["p_direct_outcome"],
        "rho_tep_age": rho_tep_age,
        "p_tep_age": p_tep_age,
        "rho_mstar_gamma_dyn": generic["rho_mass_direct"],
        "p_mstar_gamma_dyn": generic["p_mass_direct"],
        "partial_rho_gamma_dyn_age_given_z": generic["partial_rho_direct_outcome_given_z"],
        "p_partial_gamma_dyn_age_given_z": generic["p_partial_direct_outcome_given_z"],
        "n_partial_gamma_dyn_age_given_z": generic["n_partial_direct_outcome_given_z"],
        "partial_rho_mstar_age_given_z": generic["partial_rho_mass_outcome_given_z"],
        "p_partial_mstar_age_given_z": generic["p_partial_mass_outcome_given_z"],
        "n_partial_mstar_age_given_z": generic["n_partial_mass_outcome_given_z"],
        "partial_rho_gamma_dyn_age_given_mstar_z": generic["partial_rho_direct_outcome_given_mass_z"],
        "p_partial_gamma_dyn_age_given_mstar_z": generic["p_partial_direct_outcome_given_mass_z"],
        "n_partial_gamma_dyn_age_given_mstar_z": generic["n_partial_direct_outcome_given_mass_z"],
        "partial_rho_mstar_age_given_gamma_dyn_z": generic["partial_rho_mass_outcome_given_direct_z"],
        "p_partial_mstar_age_given_gamma_dyn_z": generic["p_partial_mass_outcome_given_direct_z"],
        "n_partial_mstar_age_given_gamma_dyn_z": generic["n_partial_mass_outcome_given_direct_z"],
        "delta_partial_rho_gamma_minus_mstar_given_z": generic["delta_partial_rho_direct_minus_mass_given_z"],
        "delta_partial_rho_gamma_minus_mstar_given_competitor_z": generic["delta_partial_rho_direct_minus_mass_given_competitor_z"],
        "partial_gamma_dyn_age_given_z_ci_95": generic["partial_direct_outcome_given_z_ci_95"],
        "partial_mstar_age_given_z_ci_95": generic["partial_mass_outcome_given_z_ci_95"],
        "partial_gamma_dyn_age_given_mstar_z_ci_95": generic["partial_direct_outcome_given_mass_z_ci_95"],
        "partial_mstar_age_given_gamma_dyn_z_ci_95": generic["partial_mass_outcome_given_direct_z_ci_95"],
        "steiger_gamma_dyn_vs_mstar_given_z": {
            "n": generic["steiger_direct_vs_mass_given_z"]["n"],
            "rho_age_mstar_given_z": generic["steiger_direct_vs_mass_given_z"]["rho_outcome_mass_given_z"],
            "rho_age_gamma_dyn_given_z": generic["steiger_direct_vs_mass_given_z"]["rho_outcome_direct_given_z"],
            "rho_mstar_gamma_dyn_given_z": generic["steiger_direct_vs_mass_given_z"]["rho_mass_direct_given_z"],
            "delta_rho_gamma_minus_mstar_given_z": generic["steiger_direct_vs_mass_given_z"]["delta_rho_direct_minus_mass_given_z"],
            "z_stat_gamma_dyn_better_than_mstar": generic["steiger_direct_vs_mass_given_z"]["z_stat_direct_better_than_mass"],
            "p_value_gamma_dyn_better_than_mstar": generic["steiger_direct_vs_mass_given_z"]["p_value_direct_better_than_mass"],
        },
        "steiger_gamma_dyn_vs_mstar_given_z_bootstrap": {
            "n_boot": generic["steiger_direct_vs_mass_given_z_bootstrap"]["n_boot"] if generic["steiger_direct_vs_mass_given_z_bootstrap"] is not None else None,
            "z_stat_ci_95": generic["steiger_direct_vs_mass_given_z_bootstrap"]["z_stat_ci_95"] if generic["steiger_direct_vs_mass_given_z_bootstrap"] is not None else [None, None],
            "delta_rho_ci_95": generic["steiger_direct_vs_mass_given_z_bootstrap"]["delta_rho_ci_95"] if generic["steiger_direct_vs_mass_given_z_bootstrap"] is not None else [None, None],
            "support_fraction_gamma_dyn_better": generic["steiger_direct_vs_mass_given_z_bootstrap"]["support_fraction_gamma_dyn_better"] if generic["steiger_direct_vs_mass_given_z_bootstrap"] is not None else None,
            "support_fraction_gamma_dyn_better_and_significant": generic["steiger_direct_vs_mass_given_z_bootstrap"]["support_fraction_gamma_dyn_better_and_significant"] if generic["steiger_direct_vs_mass_given_z_bootstrap"] is not None else None,
        },
    }


def _coerce_bool_series(series):
    if series.dtype == bool:
        return series.fillna(False)
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float) > 0
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def _classify_decoupling_support(metrics):
    direct_cond = metrics.get("partial_rho_direct_outcome_given_mass_z")
    direct_cond_p = metrics.get("p_partial_direct_outcome_given_mass_z")
    mass_cond = metrics.get("partial_rho_mass_outcome_given_direct_z")
    mass_cond_p = metrics.get("p_partial_mass_outcome_given_direct_z")
    direct_z = metrics.get("partial_rho_direct_outcome_given_z")
    direct_z_p = metrics.get("p_partial_direct_outcome_given_z")
    delta_z = metrics.get("delta_partial_rho_direct_minus_mass_given_z")

    if (
        direct_cond is not None
        and direct_cond_p is not None
        and direct_cond > 0
        and direct_cond_p < 0.05
        and (mass_cond_p is None or mass_cond_p >= 0.05)
    ):
        return "supportive_one_sided_conditional_asymmetry", True

    if (
        direct_z is not None
        and direct_z_p is not None
        and direct_z > 0
        and direct_z_p < 0.05
        and delta_z is not None
        and delta_z > 0
        and (mass_cond is None or direct_z >= mass_cond)
    ):
        return "directionally_supportive_with_small_sample_caveat", True

    return "inconclusive", False


def _dja_supportive_tally_robustness(n_quality_screened, n_with_balmer, resolution_source_counts):
    fallback_resolution_count = 0
    if isinstance(resolution_source_counts, dict):
        fallback_resolution_count = int(
            sum(float(value) for value in resolution_source_counts.values() if value is not None)
        )
    uses_any_fallback_resolution_model = bool(fallback_resolution_count > 0)
    all_quality_screened_widths_use_fallback_resolution = bool(
        n_quality_screened is not None
        and int(n_quality_screened) > 0
        and fallback_resolution_count >= int(n_quality_screened)
    )
    failed_criteria = []
    if n_with_balmer is None or int(n_with_balmer) < MIN_DJA_BALMER_N_FOR_SUPPORTIVE_TALLY:
        failed_criteria.append(f"n_with_balmer_below_{MIN_DJA_BALMER_N_FOR_SUPPORTIVE_TALLY}")
    if all_quality_screened_widths_use_fallback_resolution:
        failed_criteria.append("all_quality_screened_widths_use_fallback_resolution")
    return {
        "passes_supportive_tally_robustness": bool(len(failed_criteria) == 0),
        "n_quality_screened": int(n_quality_screened) if n_quality_screened is not None else None,
        "n_with_balmer": int(n_with_balmer) if n_with_balmer is not None else None,
        "fallback_resolution_count": int(fallback_resolution_count),
        "uses_any_fallback_resolution_model": uses_any_fallback_resolution_model,
        "all_quality_screened_widths_use_fallback_resolution": all_quality_screened_widths_use_fallback_resolution,
        "failed_criteria": failed_criteria,
    }


def _slim_beta_bootstrap(beta_bootstrap):
    if not isinstance(beta_bootstrap, dict):
        return None
    return {
        key: value
        for key, value in beta_bootstrap.items()
        if key != "samples"
    }


def _primary_suspense_branch(metrics, uncertainty_coverage, uncertainty_mc):
    generic_metrics = {
        "rho_mass_outcome": metrics.get("rho_mstar_age"),
        "p_mass_outcome": metrics.get("p_mstar_age"),
        "rho_direct_outcome": metrics.get("rho_gamma_dyn_age"),
        "p_direct_outcome": metrics.get("p_gamma_dyn_age"),
        "rho_mass_direct": metrics.get("rho_mstar_gamma_dyn"),
        "p_mass_direct": metrics.get("p_mstar_gamma_dyn"),
        "partial_rho_direct_outcome_given_z": metrics.get("partial_rho_gamma_dyn_age_given_z"),
        "p_partial_direct_outcome_given_z": metrics.get("p_partial_gamma_dyn_age_given_z"),
        "n_partial_direct_outcome_given_z": metrics.get("n_partial_gamma_dyn_age_given_z"),
        "partial_rho_mass_outcome_given_z": metrics.get("partial_rho_mstar_age_given_z"),
        "p_partial_mass_outcome_given_z": metrics.get("p_partial_mstar_age_given_z"),
        "n_partial_mass_outcome_given_z": metrics.get("n_partial_mstar_age_given_z"),
        "partial_rho_direct_outcome_given_mass_z": metrics.get("partial_rho_gamma_dyn_age_given_mstar_z"),
        "p_partial_direct_outcome_given_mass_z": metrics.get("p_partial_gamma_dyn_age_given_mstar_z"),
        "n_partial_direct_outcome_given_mass_z": metrics.get("n_partial_gamma_dyn_age_given_mstar_z"),
        "partial_rho_mass_outcome_given_direct_z": metrics.get("partial_rho_mstar_age_given_gamma_dyn_z"),
        "p_partial_mass_outcome_given_direct_z": metrics.get("p_partial_mstar_age_given_gamma_dyn_z"),
        "n_partial_mass_outcome_given_direct_z": metrics.get("n_partial_mstar_age_given_gamma_dyn_z"),
        "delta_partial_rho_direct_minus_mass_given_z": metrics.get("delta_partial_rho_gamma_minus_mstar_given_z"),
        "delta_partial_rho_direct_minus_mass_given_competitor_z": metrics.get("delta_partial_rho_gamma_minus_mstar_given_competitor_z"),
        "partial_direct_outcome_given_z_ci_95": metrics.get("partial_gamma_dyn_age_given_z_ci_95"),
        "partial_mass_outcome_given_z_ci_95": metrics.get("partial_mstar_age_given_z_ci_95"),
        "partial_direct_outcome_given_mass_z_ci_95": metrics.get("partial_gamma_dyn_age_given_mstar_z_ci_95"),
        "partial_mass_outcome_given_direct_z_ci_95": metrics.get("partial_mstar_age_given_gamma_dyn_z_ci_95"),
        "steiger_direct_vs_mass_given_z": {
            "n": metrics.get("steiger_gamma_dyn_vs_mstar_given_z", {}).get("n"),
            "rho_outcome_mass_given_z": metrics.get("steiger_gamma_dyn_vs_mstar_given_z", {}).get("rho_age_mstar_given_z"),
            "rho_outcome_direct_given_z": metrics.get("steiger_gamma_dyn_vs_mstar_given_z", {}).get("rho_age_gamma_dyn_given_z"),
            "rho_mass_direct_given_z": metrics.get("steiger_gamma_dyn_vs_mstar_given_z", {}).get("rho_mstar_gamma_dyn_given_z"),
            "delta_rho_direct_minus_mass_given_z": metrics.get("steiger_gamma_dyn_vs_mstar_given_z", {}).get("delta_rho_gamma_minus_mstar_given_z"),
            "z_stat_direct_better_than_mass": metrics.get("steiger_gamma_dyn_vs_mstar_given_z", {}).get("z_stat_gamma_dyn_better_than_mstar"),
            "p_value_direct_better_than_mass": metrics.get("steiger_gamma_dyn_vs_mstar_given_z", {}).get("p_value_gamma_dyn_better_than_mstar"),
        },
        "steiger_direct_vs_mass_given_z_bootstrap": {
            "n_boot": metrics.get("steiger_gamma_dyn_vs_mstar_given_z_bootstrap", {}).get("n_boot"),
            "z_stat_ci_95": metrics.get("steiger_gamma_dyn_vs_mstar_given_z_bootstrap", {}).get("z_stat_ci_95"),
            "delta_rho_ci_95": metrics.get("steiger_gamma_dyn_vs_mstar_given_z_bootstrap", {}).get("delta_rho_ci_95"),
            "support_fraction_gamma_dyn_better": metrics.get("steiger_gamma_dyn_vs_mstar_given_z_bootstrap", {}).get("support_fraction_gamma_dyn_better"),
            "support_fraction_gamma_dyn_better_and_significant": metrics.get("steiger_gamma_dyn_vs_mstar_given_z_bootstrap", {}).get("support_fraction_gamma_dyn_better_and_significant"),
        },
    }
    assessment, supportive = _classify_decoupling_support(generic_metrics)
    return {
        "label": "SUSPENSE spectral-age decoupling",
        "branch_type": "primary_direct_kinematic_decoupling",
        "source": "Slob et al. 2025 SUSPENSE (JWST NIRSpec)",
        "direct_predictor": "Gamma_t_dyn from M_dyn",
        "mass_predictor": "log_Mstar_obs",
        "outcome": "mass-weighted spectroscopic age",
        "n": metrics.get("n_partial_gamma_dyn_age_given_z"),
        "assessment": assessment,
        "supportive": supportive,
        "published_uncertainty_coverage": uncertainty_coverage,
        "uncertainty_monte_carlo": uncertainty_mc,
        "results": generic_metrics,
    }


def _load_step117_object_level_branch():
    if not STEP117_JSON.exists():
        return None
    payload = json.loads(STEP117_JSON.read_text())

    branch = None
    if payload.get("supplementary_direct_object_level_available"):
        branch = payload.get("supplementary_direct_object_level", {})
    elif payload.get("direct_kinematic_measurements_used"):
        branch = payload
    if not branch:
        return None

    table = branch.get("kinematic_table") or payload.get("kinematic_table", {})
    summary = branch.get("object_level_summary", {})
    upper_limits = branch.get("upper_limit_summary", {})
    observed = summary.get("mean_observed_excess_dex")
    corrected = summary.get("mean_corrected_excess_dex")
    exact_n = table.get("n_objects_exact_mdyn", summary.get("n_exact_mdyn_objects"))

    supportive = (
        observed is not None
        and corrected is not None
        and observed > 0
        and corrected <= 0
        and exact_n is not None
        and exact_n >= 3
    )
    directionally_supportive = observed is not None and corrected is not None and corrected < observed
    if supportive:
        assessment = "supportive_object_level_mass_resolution"
    elif directionally_supportive:
        assessment = "directionally_supportive_object_level_mass_resolution"
    else:
        assessment = "inconclusive_object_level_mass_resolution"

    return {
        "label": "Object-level M*/Mdyn anomaly resolution",
        "branch_type": "auxiliary_direct_object_level_kinematics",
        "source": table.get("sample_name") or branch.get("description"),
        "assessment": assessment,
        "supportive": bool(supportive or directionally_supportive),
        "analysis_class": branch.get("analysis_class"),
        "n_objects_total": summary.get("n_objects_total", table.get("n_objects")),
        "n_objects_exact_mdyn": table.get("n_objects_exact_mdyn", summary.get("n_exact_mdyn_objects")),
        "n_objects_upper_limit_only": table.get("n_objects_upper_limit_only", summary.get("n_upper_limit_objects")),
        "mean_observed_excess_dex": observed,
        "mean_corrected_excess_dex": corrected,
        "resolution_fraction_among_anomalous": summary.get("resolution_fraction_among_anomalous"),
        "mean_excess_metrics_basis": summary.get("mean_excess_metrics_basis"),
        "upper_limit_summary": {
            "mean_observed_excess_lower_bound_dex": upper_limits.get("mean_observed_excess_lower_bound_dex"),
            "mean_corrected_excess_lower_bound_dex": upper_limits.get("mean_corrected_excess_lower_bound_dex"),
            "n_objects": upper_limits.get("n_objects"),
        },
        "beta_bootstrap_summary": _slim_beta_bootstrap(branch.get("object_level_beta_bootstrap")),
        "source_breakdown": branch.get("source_breakdown"),
        "methodology_note": "Direct object-level kinematic closure branch; this is not the same predictor-comparison design as SUSPENSE, but it uses real dynamical masses rather than SED-only quantities.",
    }


def _load_same_regime_literature_branch():
    if not SAME_REGIME_JSON.exists():
        return None
    payload = json.loads(SAME_REGIME_JSON.read_text())
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    objects = payload.get("objects", []) if isinstance(payload, dict) else payload
    df = pd.DataFrame(objects)
    if len(df) == 0:
        return None

    for col in ["z", "log_Mstar_obs", "log_Mdyn", "log_Mdyn_upper", "sigma_kms", "sigma_kms_upper", "sfr10"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    exact_df = pd.DataFrame()
    if {"z", "log_Mstar_obs", "log_Mdyn"} <= set(df.columns):
        exact_df = df.dropna(subset=["z", "log_Mstar_obs", "log_Mdyn"]).copy()
    upper_df = pd.DataFrame()
    if {"z", "log_Mstar_obs", "log_Mdyn_upper"} <= set(df.columns):
        if "log_Mdyn" in df.columns:
            upper_df = df[df["log_Mdyn"].isna() & df["log_Mdyn_upper"].notna()].dropna(subset=["z", "log_Mstar_obs"]).copy()
        else:
            upper_df = df.dropna(subset=["z", "log_Mstar_obs", "log_Mdyn_upper"]).copy()

    combined_df = pd.concat([exact_df, upper_df], ignore_index=True, sort=False)
    if len(combined_df) == 0:
        return None

    source_breakdown = (
        {str(k): int(v) for k, v in combined_df["source"].fillna("unknown").value_counts().items()}
        if "source" in combined_df.columns
        else {}
    )
    sample_breakdown = (
        {str(k): int(v) for k, v in combined_df["sample_group"].fillna("unknown").value_counts().items()}
        if "sample_group" in combined_df.columns
        else {}
    )
    exact_log_mdyn_minus_mstar = (
        exact_df["log_Mdyn"].to_numpy(dtype=float) - exact_df["log_Mstar_obs"].to_numpy(dtype=float)
        if len(exact_df)
        else np.asarray([], dtype=float)
    )

    if len(exact_df) >= 10:
        assessment = "same_regime_direct_kinematic_sample_available"
    elif len(exact_df) >= 3:
        assessment = "same_regime_direct_kinematic_sample_small"
    else:
        assessment = "same_regime_direct_kinematic_sample_upper_limit_only"

    return {
        "label": "Same-regime literature kinematic sample",
        "branch_type": "contextual_same_regime_object_level_kinematics",
        "source": metadata.get("sample_name") or str(SAME_REGIME_JSON),
        "assessment": assessment,
        "supportive": None,
        "counts_toward_supportive_tally": False,
        "analysis_role": metadata.get("analysis_role"),
        "n_objects_total": int(len(combined_df)),
        "n_objects_exact_mdyn": int(len(exact_df)),
        "n_objects_upper_limit_only": int(len(upper_df)),
        "z_min": float(combined_df["z"].min()),
        "z_max": float(combined_df["z"].max()),
        "median_z": float(combined_df["z"].median()),
        "median_log_Mstar_obs": float(combined_df["log_Mstar_obs"].median()),
        "median_log_Mdyn_exact": float(exact_df["log_Mdyn"].median()) if len(exact_df) else None,
        "mean_exact_log_Mdyn_minus_Mstar_dex": float(np.mean(exact_log_mdyn_minus_mstar)) if len(exact_log_mdyn_minus_mstar) else None,
        "median_exact_log_Mdyn_minus_Mstar_dex": float(np.median(exact_log_mdyn_minus_mstar)) if len(exact_log_mdyn_minus_mstar) else None,
        "n_exact_mdyn_gt_mstar": int(np.sum(exact_log_mdyn_minus_mstar > 0)) if len(exact_log_mdyn_minus_mstar) else 0,
        "fraction_exact_mdyn_gt_mstar": float(np.mean(exact_log_mdyn_minus_mstar > 0)) if len(exact_log_mdyn_minus_mstar) else None,
        "source_breakdown": source_breakdown,
        "sample_breakdown": sample_breakdown,
        "objects": combined_df.replace({np.nan: None}).to_dict(orient="records"),
        "methodology_note": "Contextual same-regime mass-independent object-level kinematic sample at z>4; surfaced for direct-regime coverage, but not counted as a TEP sign-test branch because it is dominated by low-mass star-forming systems with Mdyn greater than or comparable to Mstar.",
    }


def _load_dja_sigma_balmer_branch():
    if not STEP169_CSV.exists():
        return None

    df = pd.read_csv(STEP169_CSV)
    if "quality_screen_pass" in df.columns:
        df = df[_coerce_bool_series(df["quality_screen_pass"])].copy()

    for col in ["log_balmer", "sigma_kms", "sigma_kms_err", "log_Mstar", "z", "reduced_chi2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["log_balmer", "sigma_kms", "log_Mstar", "z"]).copy()
    df = df[df["sigma_kms"] > 0].reset_index(drop=True)
    if len(df) < 4:
        return None

    meta = json.loads(STEP169_JSON.read_text()) if STEP169_JSON.exists() else {}
    quality_summary = meta.get("fit_summary", {}).get("quality_screened", {})
    quality_balmer = meta.get("pilot_balmer_sigma_test", {}).get("quality_screened", {})
    config = meta.get("config", {})
    selection = meta.get("selection", {})
    selection_mode = str(config.get("selection_mode", "pilot"))
    if selection_mode == "high_mass_same_regime":
        label = "DJA sigma-Balmer high-mass same-regime subset"
        source = "Public DJA/msaexp spectra, targeted z>=5 high-mass subset"
        methodology_note = "Targeted same-regime sigma-based decoupling test on the z>=5 high-mass DJA subset; results depend on public spec.fits availability and quality-screened fitted widths."
    else:
        label = "DJA sigma-Balmer pilot"
        source = "Public DJA/msaexp spectra, quality-screened pilot subset"
        methodology_note = "Pilot-quality sigma-based decoupling test using fitted emission-line widths and a Balmer proxy on the quality-screened subset; useful for directional replication, but not a replacement for the primary SUSPENSE age-based comparison."
    rho_mass_direct, p_mass_direct, n_mass_direct = _safe_spearman(
        df["log_Mstar"].to_numpy(dtype=float),
        df["sigma_kms"].to_numpy(dtype=float),
    )
    metrics = {
        "rho_mass_outcome": quality_balmer.get("raw_mass_vs_balmer", {}).get("rho"),
        "p_mass_outcome": quality_balmer.get("raw_mass_vs_balmer", {}).get("p"),
        "rho_direct_outcome": quality_balmer.get("raw_sigma_vs_balmer", {}).get("rho"),
        "p_direct_outcome": quality_balmer.get("raw_sigma_vs_balmer", {}).get("p"),
        "rho_mass_direct": rho_mass_direct,
        "p_mass_direct": p_mass_direct,
        "n_mass_direct": n_mass_direct,
        "partial_rho_direct_outcome_given_z": None,
        "p_partial_direct_outcome_given_z": None,
        "n_partial_direct_outcome_given_z": quality_balmer.get("n"),
        "partial_rho_mass_outcome_given_z": None,
        "p_partial_mass_outcome_given_z": None,
        "n_partial_mass_outcome_given_z": quality_balmer.get("n"),
        "partial_rho_direct_outcome_given_mass_z": quality_balmer.get("partial_sigma_given_mass_z", {}).get("rho"),
        "p_partial_direct_outcome_given_mass_z": quality_balmer.get("partial_sigma_given_mass_z", {}).get("p"),
        "n_partial_direct_outcome_given_mass_z": quality_balmer.get("partial_sigma_given_mass_z", {}).get("n"),
        "partial_rho_mass_outcome_given_direct_z": quality_balmer.get("partial_mass_given_sigma_z", {}).get("rho"),
        "p_partial_mass_outcome_given_direct_z": quality_balmer.get("partial_mass_given_sigma_z", {}).get("p"),
        "n_partial_mass_outcome_given_direct_z": quality_balmer.get("partial_mass_given_sigma_z", {}).get("n"),
        "delta_partial_rho_direct_minus_mass_given_z": None,
        "delta_partial_rho_direct_minus_mass_given_competitor_z": (
            quality_balmer.get("partial_sigma_given_mass_z", {}).get("rho")
            - quality_balmer.get("partial_mass_given_sigma_z", {}).get("rho")
            if quality_balmer.get("partial_sigma_given_mass_z", {}).get("rho") is not None
            and quality_balmer.get("partial_mass_given_sigma_z", {}).get("rho") is not None
            else None
        ),
        "partial_direct_outcome_given_z_ci_95": [None, None],
        "partial_mass_outcome_given_z_ci_95": [None, None],
        "partial_direct_outcome_given_mass_z_ci_95": [None, None],
        "partial_mass_outcome_given_direct_z_ci_95": [None, None],
        "steiger_direct_vs_mass_given_z": {
            "n": quality_balmer.get("n"),
            "rho_outcome_mass_given_z": None,
            "rho_outcome_direct_given_z": None,
            "rho_mass_direct_given_z": None,
            "delta_rho_direct_minus_mass_given_z": None,
            "z_stat_direct_better_than_mass": None,
            "p_value_direct_better_than_mass": None,
        },
        "steiger_direct_vs_mass_given_z_bootstrap": None,
    }
    assessment, supportive = _classify_decoupling_support(metrics)
    if not supportive:
        return None
    n_quality_screened = int(quality_summary.get("n_success")) if quality_summary.get("n_success") is not None else None
    n_with_balmer = int(len(df))
    resolution_source_counts = quality_summary.get("resolution_source_counts")
    supportive_tally_robustness = _dja_supportive_tally_robustness(
        n_quality_screened=n_quality_screened,
        n_with_balmer=n_with_balmer,
        resolution_source_counts=resolution_source_counts,
    )
    counts_toward_supportive_tally = supportive_tally_robustness["passes_supportive_tally_robustness"]
    return {
        "label": label,
        "branch_type": "auxiliary_sigma_balmer_pilot",
        "source": source,
        "assessment": assessment,
        "supportive": supportive,
        "counts_toward_supportive_tally": counts_toward_supportive_tally,
        "analysis_role": (
            "counted_auxiliary_direct_branch"
            if counts_toward_supportive_tally else
            "contextual_pilot_branch"
        ),
        "selection_mode": selection_mode,
        "n_targeted": selection.get("n_targeted"),
        "candidate_pool_n": selection.get("candidate_pool_n"),
        "candidate_pool_n_with_balmer": selection.get("candidate_pool_n_with_balmer"),
        "n_quality_screened": n_quality_screened,
        "n_with_balmer": n_with_balmer,
        "direct_predictor": "sigma_kms",
        "mass_predictor": "log_Mstar",
        "outcome": "log_balmer proxy",
        "resolution_source_counts": resolution_source_counts,
        "uses_fallback_resolution_model": supportive_tally_robustness["uses_any_fallback_resolution_model"],
        "all_quality_screened_widths_use_fallback_resolution": supportive_tally_robustness["all_quality_screened_widths_use_fallback_resolution"],
        "robustness_gate": supportive_tally_robustness,
        "results": metrics,
        "methodology_note": methodology_note,
    }


def _load_sigma_expansion_branch():
    """Load step_171 sigma-based kinematic expansion results as a counted branch."""
    if not STEP171_JSON.exists():
        return None
    try:
        payload = json.loads(STEP171_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("status") != "SUCCESS":
        return None
    assessment = payload.get("assessment", "unknown")
    supportive = bool(payload.get("supportive", False))
    sample = payload.get("sample", {})
    tests = payload.get("tests", {})
    t3 = tests.get("T3_gamma_sigma_only_prediction", {})
    t1 = tests.get("T1_mstar_sigma_evolution", {})
    t5 = tests.get("T5_highz_z_ge_4", {})
    t2 = tests.get("T2_fundamental_plane", {})
    methodology = payload.get("methodology", {})
    return {
        "label": "Sigma-based kinematic expansion (N={}, z={:.1f}-{:.1f})".format(
            sample.get("n_total", "?"),
            sample.get("z_min", 0),
            sample.get("z_max", 0),
        ),
        "branch_type": "auxiliary_sigma_expansion_mass_circularity_breaker",
        "source": "Combined SUSPENSE + Esdaile+21 + Tanaka+19 + de Graaff+24 + Saldana-Lopez+25 + Danhaive+25",
        "assessment": assessment,
        "supportive": supportive,
        "counts_toward_supportive_tally": supportive,
        "analysis_role": "counted_sigma_expansion_branch" if supportive else "contextual_sigma_expansion_branch",
        "n_objects_total": sample.get("n_total"),
        "n_sources": len(sample.get("source_counts", {})),
        "source_paper_breakdown": sample.get("source_paper_breakdown"),
        "z_min": sample.get("z_min"),
        "z_max": sample.get("z_max"),
        "z_median": sample.get("z_median"),
        "sigma_min_kms": sample.get("sigma_min_kms"),
        "sigma_max_kms": sample.get("sigma_max_kms"),
        "T3_partial_rho_gamma_mstar_given_sigma_z": t3.get("partial_rho_gamma_mstar_given_sigma_z"),
        "T3_p_partial_gamma_mstar_given_sigma_z": t3.get("p_partial_gamma_mstar_given_sigma_z"),
        "T3_ci_95": t3.get("ci_95_partial_gamma_mstar_given_sigma_z"),
        "T1_observed_rho_resid_z": t1.get("observed_residual_z_trend", {}).get("partial_rho_resid_z_given_sigma"),
        "T1_corrected_rho_resid_z": t1.get("corrected_residual_z_trend", {}).get("partial_rho_resid_z_given_sigma"),
        "T1_z_trend_improved": t1.get("z_trend_improved"),
        "T1_scatter_improved": t1.get("scatter_improved"),
        "T2_scatter_improved": t2.get("scatter_improved") if isinstance(t2, dict) else None,
        "T5_highz_n": t5.get("n"),
        "T5_highz_partial_rho_gamma": t5.get("partial_rho_gamma_mstar_given_sigma_z"),
        "T5_highz_p_gamma": t5.get("p_gamma_mstar"),
        "methodology_note": (
            "Sigma-only mass-circularity-breaking test on a combined N={} kinematic sample. "
            "Gamma_t is computed from velocity dispersion alone (sigma -> M_halo -> Gamma_t) "
            "with zero dependence on SED-fitted M* or Mdyn. The key test (T3) asks whether "
            "sigma-only Gamma_t adds predictive power for M*_obs beyond sigma and z individually."
        ).format(sample.get("n_total", "?")),
    }


def _build_federated_direct_package(metrics, uncertainty_coverage, uncertainty_mc):
    branches = {
        "suspense_spectral_age_decoupling": _primary_suspense_branch(metrics, uncertainty_coverage, uncertainty_mc),
    }

    step117_branch = _load_step117_object_level_branch()
    if step117_branch is not None:
        branches["object_level_mass_anomaly_resolution"] = step117_branch

    dja_branch = _load_dja_sigma_balmer_branch()
    if dja_branch is not None:
        branches["dja_sigma_balmer_pilot"] = dja_branch

    same_regime_branch = _load_same_regime_literature_branch()
    if same_regime_branch is not None:
        branches["same_regime_literature_kinematics"] = same_regime_branch

    sigma_expansion_branch = _load_sigma_expansion_branch()
    if sigma_expansion_branch is not None:
        branches["sigma_kinematic_expansion"] = sigma_expansion_branch

    available_labels = [branch["label"] for branch in branches.values()]
    counted_branches = [branch for branch in branches.values() if branch.get("counts_toward_supportive_tally", True)]
    counted_branch_labels = [branch["label"] for branch in counted_branches]
    contextual_branch_labels = [
        branch["label"]
        for branch in branches.values()
        if not branch.get("counts_toward_supportive_tally", True)
    ]
    supportive_labels = [branch["label"] for branch in counted_branches if branch.get("supportive")]
    primary_supportive = branches["suspense_spectral_age_decoupling"].get("supportive", False)
    auxiliary_supportive_labels = [
        branch["label"]
        for key, branch in branches.items()
        if key != "suspense_spectral_age_decoupling"
        and branch.get("counts_toward_supportive_tally", True)
        and branch.get("supportive")
    ]

    if primary_supportive and len(auxiliary_supportive_labels) >= 2:
        federated_assessment = "primary_direct_kinematic_support_with_two_auxiliary_direct_branch_alignments"
    elif primary_supportive and len(auxiliary_supportive_labels) >= 1:
        federated_assessment = "primary_direct_kinematic_support_with_auxiliary_direct_branch_alignment"
    elif primary_supportive:
        federated_assessment = "primary_direct_kinematic_support_only"
    else:
        federated_assessment = "insufficient_direct_kinematic_package"

    caveat_parts = [
        "Only the SUSPENSE branch is a direct age-based predictor-comparison test.",
    ]
    if step117_branch is not None:
        caveat_parts.append(
            "The object-level M*/Mdyn branch strengthens the broader direct kinematic package, "
            "but it remains auxiliary because it uses a different observable."
        )
    if dja_branch is not None and dja_branch.get("counts_toward_supportive_tally", True):
        caveat_parts.append(
            "The DJA sigma-Balmer branch strengthens the broader direct kinematic package, "
            "but it remains auxiliary because it uses a different observable and pilot-grade spectral resolution handling."
        )
    elif dja_branch is not None:
        caveat_parts.append(
            "The DJA sigma-Balmer pilot is surfaced as contextual pilot evidence rather than counted supportive evidence "
            "because the current supportive subset is very small and relies entirely on fallback spectral-resolution handling."
        )
    if same_regime_branch is not None:
        caveat_parts.append(
            "The same-regime literature sample is surfaced as contextual direct-regime coverage and is not counted as a TEP sign-test branch."
        )
    if sigma_expansion_branch is not None and sigma_expansion_branch.get("counts_toward_supportive_tally"):
        caveat_parts.append(
            "The sigma-based kinematic expansion branch uses a sigma-only Gamma_t (zero SED dependence) "
            "on a combined N={} sample to break the mass-proxy circularity.".format(
                sigma_expansion_branch.get("n_objects_total", "?")
            )
        )
    elif sigma_expansion_branch is not None:
        caveat_parts.append(
            "The sigma-based kinematic expansion branch is surfaced as contextual because its assessment "
            "did not meet the supportive threshold."
        )

    return {
        "summary": {
            "primary_branch_label": branches["suspense_spectral_age_decoupling"]["label"],
            "available_branch_labels": available_labels,
            "counted_branch_labels": counted_branch_labels,
            "contextual_branch_labels": contextual_branch_labels,
            "supportive_branch_labels": supportive_labels,
            "n_available_branches": int(len(branches)),
            "n_counted_branches": int(len(counted_branches)),
            "n_contextual_branches": int(len(contextual_branch_labels)),
            "n_supportive_branches": int(len(supportive_labels)),
            "all_available_branches_supportive": bool(len(supportive_labels) == len(counted_branches)),
            "all_counted_branches_supportive": bool(len(supportive_labels) == len(counted_branches)),
            "federated_assessment": federated_assessment,
            "caveat": " ".join(caveat_parts),
        },
        "branches": branches,
    }


def _load_suspense_sample():
    if not INPUT_JSON.exists():
        return None, None, {"step": STEP_NUM, "status": "FAILED", "reason": "No SUSPENSE kinematics dataset found."}

    df = pd.DataFrame(json.loads(INPUT_JSON.read_text()))
    needed = {"object_id", "z", "log_Mstar_obs", "log_Mdyn", "age_gyr"}
    missing = sorted(needed - set(df.columns))
    if missing:
        return None, None, {"step": STEP_NUM, "status": "FAILED", "reason": f"Missing required columns: {missing}"}

    df = df.dropna(subset=["z", "log_Mstar_obs", "log_Mdyn", "age_gyr"]).copy().reset_index(drop=True)
    df, uncertainty_coverage = _attach_published_uncertainties(df)
    df["t_univ"] = df["z"].apply(get_age_universe)
    df["Gamma_t_dyn"] = np.power(10.0 ** df["log_Mdyn"].to_numpy(dtype=float) / M_REF, ALPHA_0)
    df["Gamma_t_phot"] = np.power(10.0 ** df["log_Mstar_obs"].to_numpy(dtype=float) / M_REF, ALPHA_0)
    df["t_tep"] = df["t_univ"].to_numpy(dtype=float) * df["Gamma_t_dyn"].to_numpy(dtype=float)
    return df, uncertainty_coverage, None


def run():
    print_status(f"STEP {STEP_NUM}: TEP Kinematic Decoupling Test", "INFO")
    df, uncertainty_coverage, error = _load_suspense_sample()
    if error is not None:
        result = error
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    print_status(f"Running kinematic decoupling test on {len(df)} SUSPENSE galaxies", "INFO")
    metrics = _compute_metrics(df)
    uncertainty_mc = _uncertainty_monte_carlo(df)
    federated_direct_package = _build_federated_direct_package(metrics, uncertainty_coverage, uncertainty_mc)
    federated_assessment = federated_direct_package["summary"]["federated_assessment"]

    steiger = metrics["steiger_gamma_dyn_vs_mstar_given_z"]
    strong = (
        metrics["partial_rho_gamma_dyn_age_given_mstar_z"] > 0
        and metrics["p_partial_gamma_dyn_age_given_mstar_z"] < 0.05
        and metrics["p_partial_mstar_age_given_gamma_dyn_z"] >= 0.05
        and steiger["p_value_gamma_dyn_better_than_mstar"] is not None
        and steiger["p_value_gamma_dyn_better_than_mstar"] < 0.05
    )
    moderate = (
        metrics["partial_rho_gamma_dyn_age_given_z"] > 0
        and metrics["p_partial_gamma_dyn_age_given_z"] < 0.05
        and steiger["z_stat_gamma_dyn_better_than_mstar"] is not None
        and steiger["z_stat_gamma_dyn_better_than_mstar"] > 0
        and metrics["delta_partial_rho_gamma_minus_mstar_given_z"] > 0
    )
    if strong:
        assessment = "strong_direct_kinematic_support"
    elif moderate:
        assessment = "kinematic_support_with_small_sample_caveat"
    else:
        assessment = "insufficient_kinematic_support"

    print_status(f"  rho(Age, M_star) = {metrics['rho_mstar_age']:.3f} (p={metrics['p_mstar_age']:.3e})", "INFO")
    print_status(f"  rho(Age, Gamma_t_dyn) = {metrics['rho_gamma_dyn_age']:.3f} (p={metrics['p_gamma_dyn_age']:.3e})", "INFO")
    print_status(f"  rho(Age, Gamma_t_dyn | z) = {metrics['partial_rho_gamma_dyn_age_given_z']:.3f} (p={metrics['p_partial_gamma_dyn_age_given_z']:.3e})", "INFO")
    print_status(f"  rho(Age, M_star | z) = {metrics['partial_rho_mstar_age_given_z']:.3f} (p={metrics['p_partial_mstar_age_given_z']:.3e})", "INFO")
    print_status(f"  rho(Age, Gamma_t_dyn | M_star, z) = {metrics['partial_rho_gamma_dyn_age_given_mstar_z']:.3f} (p={metrics['p_partial_gamma_dyn_age_given_mstar_z']:.3e})", "INFO")
    print_status(f"  rho(Age, M_star | Gamma_t_dyn, z) = {metrics['partial_rho_mstar_age_given_gamma_dyn_z']:.3f} (p={metrics['p_partial_mstar_age_given_gamma_dyn_z']:.3e})", "INFO")
    if steiger["z_stat_gamma_dyn_better_than_mstar"] is not None:
        print_status(
            f"  Steiger(z-controlled ranks): Z={steiger['z_stat_gamma_dyn_better_than_mstar']:.3f} "
            f"(p={steiger['p_value_gamma_dyn_better_than_mstar']:.3e})",
            "INFO",
        )
    if uncertainty_mc is not None:
        gamma_mc = uncertainty_mc["partial_gamma_dyn_age_given_mstar_z"]
        mass_mc = uncertainty_mc["partial_mstar_age_given_gamma_dyn_z"]
        print_status(
            f"  MC robustness: med ρ(Age, Γ_dyn|M_star,z)={gamma_mc['median']:.3f} "
            f"vs med ρ(Age, M_star|Γ_dyn,z)={mass_mc['median']:.3f}",
            "INFO",
        )
    package_summary = federated_direct_package["summary"]
    print_status(
        f"  Direct-kinematic package: {package_summary['n_supportive_branches']}/{package_summary['n_available_branches']} "
        f"branches supportive",
        "INFO",
    )
    if package_summary["supportive_branch_labels"]:
        print_status(
            f"  Supportive branches: {', '.join(package_summary['supportive_branch_labels'])}",
            "INFO",
        )

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "assessment": assessment,
        "federated_assessment": federated_assessment,
        "n_galaxies": int(len(df)),
        "source": "Slob et al. 2025 SUSPENSE (JWST NIRSpec)",
        "methodology": {
            "comparison_goal": "Does a kinematically derived TEP predictor outperform photometric stellar mass in predicting spectral age?",
            "primary_controls": ["z"],
            "conditional_asymmetry_controls": ["competitor_predictor", "z"],
            "age_observable": "mass-weighted spectroscopic age",
            "direct_kinematic_package": "Primary SUSPENSE age-based comparison plus any available auxiliary direct object-level branches and contextual same-regime literature kinematic sample when available.",
        },
        "results": metrics,
        "robustness": {
            "published_uncertainty_coverage": uncertainty_coverage,
            "uncertainty_monte_carlo": uncertainty_mc,
        },
        "federated_direct_kinematic_package": federated_direct_package,
    }

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print_status(f"Assessment: {assessment}", "INFO")
    print_status(f"Federated assessment: {federated_assessment}", "INFO")
    return result


if __name__ == "__main__":
    run()
