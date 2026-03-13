#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
Step 117: Dynamical Mass Comparison — TEP Isochrony Correction

Uses real UNCOVER DR4 Gamma_t values to quantify the TEP isochrony
correction in the regime where published kinematic surveys have
identified M*/M_dyn anomalies.

IMPORTANT: Dynamical masses (M_dyn) cannot be derived from UNCOVER
photometry alone. This step does NOT manufacture individual kinematic
measurements. Instead:
  1. Loads real UNCOVER galaxies at z>4, log M*>9.5 (the kinematic regime)
  2. Computes TEP-corrected stellar masses using real Gamma_t values
  3. Characterises the distribution of corrections for the published regime
  4. Shows the correction magnitude is sufficient to resolve published
     M*/M_dyn tensions (de Graaff et al. 2024; Wang et al. 2024)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types (handles NaN, inf, float32)
from scripts.utils.tep_model import (
    compute_gamma_t as tep_compute_gamma_t,  # TEP model: Gamma_t formula
    stellar_to_halo_mass_behroozi_like,  # Stellar-to-halo mass from abundance matching
)

STEP_NUM  = "117"  # Pipeline step number (sequential 001-176)
STEP_NAME = "dynamical_mass_comparison"  # Dynamical mass comparison: TEP isochrony correction resolving M*/M_dyn tensions via M*_TEP-corrected = M*_obs × (Gamma_t)^0.5
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
INTERIM_DIR = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products

N_ML_HIGHZ = 0.5  # Mass-to-light power-law index for high-z isochrony correction
N_BOOT = 2000  # Bootstrap iterations for beta confidence interval

KINEMATIC_TABLE_CANDIDATES = [
    PROJECT_ROOT / "data" / "interim" / "literature_kinematic_sample.csv",
    PROJECT_ROOT / "data" / "interim" / "direct_kinematic_sample.csv",
    PROJECT_ROOT / "data" / "raw" / "kinematics" / "literature_kinematic_sample.csv",
    PROJECT_ROOT / "data" / "raw" / "kinematics" / "direct_kinematic_sample.csv",
    PROJECT_ROOT / "data" / "interim" / "literature_kinematic_sample.json",
    PROJECT_ROOT / "data" / "raw" / "kinematics" / "literature_kinematic_sample.json",
]


def _rename_first_match(df, target, aliases):
    for alias in aliases:
        if alias in df.columns and target not in df.columns:
            df = df.rename(columns={alias: target})
            break
    return df


def _load_direct_kinematic_table():
    path = next((candidate for candidate in KINEMATIC_TABLE_CANDIDATES if candidate.exists()), None)
    table_metadata = {}
    if path is None:
        return None, {
            "status": "missing_local_object_level_table",
            "searched_paths": [str(p) for p in KINEMATIC_TABLE_CANDIDATES],
            "required_columns_any_of": {
                "object_id": ["object_id", "id", "name"],
                "z": ["z", "z_spec", "redshift"],
                "log_Mstar_obs": ["log_Mstar_obs", "log_Mstar", "log_mstar", "log_mass_star"],
                "log_Mdyn": ["log_Mdyn", "log_mdyn", "log_M_dyn", "log_dyn_mass"],
                "or_linear_Mdyn": ["Mdyn", "M_dyn", "dyn_mass"],
                "optional_sigma_re": ["sigma_kms", "sigma", "re_kpc", "r_eff_kpc"],
                "optional_upper_limit_mdyn": ["log_Mdyn_upper", "Mdyn_upper"],
            },
        }

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "objects" in payload:
            df = pd.DataFrame(payload["objects"])
            if isinstance(payload.get("metadata"), dict):
                table_metadata = payload["metadata"].copy()
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            df = pd.DataFrame(payload)
    else:
        df = pd.read_csv(path)

    df = _rename_first_match(df, "object_id", ["object_id", "id", "name", "source_id"])
    df = _rename_first_match(df, "z", ["z", "z_spec", "redshift"])
    df = _rename_first_match(df, "log_Mstar_obs", ["log_Mstar_obs", "log_Mstar", "log_mstar", "log_mass_star"])
    df = _rename_first_match(df, "log_Mdyn", ["log_Mdyn", "log_mdyn", "log_M_dyn", "log_dyn_mass"])
    df = _rename_first_match(df, "Mdyn", ["Mdyn", "M_dyn", "dyn_mass"])
    df = _rename_first_match(df, "log_Mdyn_upper", ["log_Mdyn_upper", "log_mdyn_upper", "log_Mdyn_ul", "log_dyn_mass_upper"])
    df = _rename_first_match(df, "Mdyn_upper", ["Mdyn_upper", "M_dyn_upper", "dyn_mass_upper"])
    df = _rename_first_match(df, "sigma_kms", ["sigma_kms", "sigma", "velocity_dispersion"])
    df = _rename_first_match(df, "re_kpc", ["re_kpc", "r_eff_kpc", "rhalf_kpc", "size_kpc"])
    df = _rename_first_match(df, "source", ["source", "paper", "survey"])

    for col in ["z", "log_Mstar_obs", "log_Mdyn", "Mdyn", "log_Mdyn_upper", "Mdyn_upper", "sigma_kms", "re_kpc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "log_Mdyn" not in df.columns and "Mdyn" in df.columns:
        valid = df["Mdyn"] > 0
        df.loc[valid, "log_Mdyn"] = np.log10(df.loc[valid, "Mdyn"])

    if "log_Mdyn_upper" not in df.columns and "Mdyn_upper" in df.columns:
        valid = df["Mdyn_upper"] > 0
        df.loc[valid, "log_Mdyn_upper"] = np.log10(df.loc[valid, "Mdyn_upper"])

    if "log_Mdyn" not in df.columns and {"sigma_kms", "re_kpc"} <= set(df.columns):
        valid = (df["sigma_kms"] > 0) & (df["re_kpc"] > 0)
        mdyn = 5.0 * np.square(df.loc[valid, "sigma_kms"]) * df.loc[valid, "re_kpc"] / 4.302e-6
        df.loc[valid, "log_Mdyn"] = np.log10(mdyn)

    if "object_id" not in df.columns:
        df["object_id"] = np.arange(len(df))

    required = ["z", "log_Mstar_obs", "log_Mdyn"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return None, {
            "status": "table_present_but_missing_required_columns",
            "path": str(path),
            "missing_columns": missing,
            "columns_present": list(df.columns),
        }

    upper_limit_rows = pd.DataFrame()
    if "log_Mdyn_upper" in df.columns:
        upper_limit_rows = df[df["log_Mdyn"].isna() & df["log_Mdyn_upper"].notna()].copy()

    exact_rows = df.dropna(subset=required).copy()
    exact_rows = exact_rows[(exact_rows["z"] > 0) & (exact_rows["z"] < 15)]
    exact_rows = exact_rows[(exact_rows["log_Mstar_obs"] > 5) & (exact_rows["log_Mstar_obs"] < 13)]
    exact_rows = exact_rows[(exact_rows["log_Mdyn"] > 5) & (exact_rows["log_Mdyn"] < 13)]

    if not upper_limit_rows.empty:
        upper_limit_rows = upper_limit_rows.dropna(subset=["z", "log_Mstar_obs", "log_Mdyn_upper"]).copy()
        upper_limit_rows = upper_limit_rows[(upper_limit_rows["z"] > 0) & (upper_limit_rows["z"] < 15)]
        upper_limit_rows = upper_limit_rows[(upper_limit_rows["log_Mstar_obs"] > 5) & (upper_limit_rows["log_Mstar_obs"] < 13)]
        upper_limit_rows = upper_limit_rows[(upper_limit_rows["log_Mdyn_upper"] > 5) & (upper_limit_rows["log_Mdyn_upper"] < 13)]

    if len(exact_rows) < 3:
        return None, {
            "status": "table_present_but_underpowered",
            "path": str(path),
            "n_valid": int(len(exact_rows)),
        }

    combined_rows = pd.concat([exact_rows, upper_limit_rows], ignore_index=True, sort=False)

    meta = {
        "status": "loaded_local_object_level_table",
        "path": str(path),
        "n_objects": int(len(combined_rows)),
        "n_objects_exact_mdyn": int(len(exact_rows)),
        "n_objects_upper_limit_only": int(len(upper_limit_rows)),
        "sources": sorted(str(s) for s in combined_rows.get("source", pd.Series(dtype=str)).dropna().unique()),
        "sample_name": table_metadata.get("sample_name"),
        "analysis_role": table_metadata.get("analysis_role"),
        "authoritative_for_l4_regime": bool(table_metadata.get("authoritative_for_l4_regime", False)),
        "authoritative_for_beta_propagation": bool(table_metadata.get("authoritative_for_beta_propagation", False)),
        "n_upper_limit_rows_available": int(len(upper_limit_rows)),
        "upper_limit_objects": [
            {
                "object_id": str(row["object_id"]),
                "z": float(row["z"]),
                "log_Mstar_obs": float(row["log_Mstar_obs"]),
                "log_Mdyn_upper": float(row["log_Mdyn_upper"]),
                **({"source": str(row["source"])} if "source" in upper_limit_rows.columns and pd.notna(row.get("source")) else {}),
                **({"sigma_kms": float(row["sigma_kms"])} if "sigma_kms" in upper_limit_rows.columns and pd.notna(row.get("sigma_kms")) else {}),
                **({"re_kpc": float(row["re_kpc"])} if "re_kpc" in upper_limit_rows.columns and pd.notna(row.get("re_kpc")) else {}),
            }
            for _, row in upper_limit_rows.iterrows()
        ],
        "metadata": table_metadata,
    }
    return {
        "exact": exact_rows.reset_index(drop=True),
        "upper_limits": upper_limit_rows.reset_index(drop=True),
    }, meta


def _bootstrap_beta(log_gamma, observed_excess, calibration_mask=None):
    valid = np.isfinite(log_gamma) & np.isfinite(observed_excess) & (log_gamma > 0)
    if calibration_mask is not None:
        valid &= np.asarray(calibration_mask, dtype=bool)
    if valid.sum() < 3:
        return None

    rng = np.random.default_rng(42)
    lg = log_gamma[valid]
    ex = observed_excess[valid]
    samples = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(lg), len(lg))
        lg_i = lg[idx]
        ex_i = ex[idx]
        denom = float(np.sum(np.square(lg_i)))
        if denom <= 0:
            continue
        beta = float(np.sum(lg_i * ex_i) / denom)
        if np.isfinite(beta):
            samples.append(beta)

    if len(samples) < 50:
        return None

    samples = np.asarray(samples, dtype=float)
    return {
        "n_galaxies": int(len(lg)),
        "n_boot": int(len(samples)),
        "estimator": "bootstrap_slope_through_origin",
        "calibration_subset": (
            "anomalous_objects_only_if_available"
            if calibration_mask is not None
            else "all_objects_with_finite_positive_log_gamma"
        ),
        "median": float(np.median(samples)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "ci_95": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "samples": samples.tolist(),
    }


def _serialise_object_level_rows(df_direct, gamma_t_arr, delta_log_mstar, observed_excess, corrected_excess, df_upper_limits=None):
    rows = []
    for idx, row in df_direct.reset_index(drop=True).iterrows():
        payload = {
            "object_id": str(row["object_id"]),
            "z": float(row["z"]),
            "log_Mstar_obs": float(row["log_Mstar_obs"]),
            "log_Mdyn": float(row["log_Mdyn"]),
            "gamma_t": float(gamma_t_arr[idx]),
            "delta_logMstar_dex": float(delta_log_mstar[idx]),
            "observed_excess_dex": float(observed_excess[idx]),
            "corrected_excess_dex": float(corrected_excess[idx]),
            "anomalous_observed": bool(observed_excess[idx] > 0),
            "resolved_after_tep": bool(corrected_excess[idx] <= 0),
            "mdyn_limit_type": "exact",
        }
        if "source" in df_direct.columns and pd.notna(row.get("source")):
            payload["source"] = str(row["source"])
        if "sigma_kms" in df_direct.columns and pd.notna(row.get("sigma_kms")):
            payload["sigma_kms"] = float(row["sigma_kms"])
        if "re_kpc" in df_direct.columns and pd.notna(row.get("re_kpc")):
            payload["re_kpc"] = float(row["re_kpc"])
        rows.append(payload)

    if df_upper_limits is not None and len(df_upper_limits) > 0:
        z_upper = df_upper_limits["z"].to_numpy(dtype=float)
        log_mstar_obs_upper = df_upper_limits["log_Mstar_obs"].to_numpy(dtype=float)
        log_mh_upper = stellar_to_halo_mass_behroozi_like(log_mstar_obs_upper, z_upper)
        gamma_t_upper = tep_compute_gamma_t(log_mh_upper, z_upper)
        mass_correction_upper = np.power(np.maximum(gamma_t_upper, 0.01), N_ML_HIGHZ)
        delta_log_mstar_upper = np.log10(mass_correction_upper)
        log_mstar_tep_upper = log_mstar_obs_upper - delta_log_mstar_upper
        log_mdyn_upper = df_upper_limits["log_Mdyn_upper"].to_numpy(dtype=float)
        observed_excess_lower_bound = log_mstar_obs_upper - log_mdyn_upper
        corrected_excess_lower_bound = log_mstar_tep_upper - log_mdyn_upper

        for idx, row in df_upper_limits.reset_index(drop=True).iterrows():
            payload = {
                "object_id": str(row["object_id"]),
                "z": float(row["z"]),
                "log_Mstar_obs": float(row["log_Mstar_obs"]),
                "log_Mdyn_upper": float(row["log_Mdyn_upper"]),
                "gamma_t": float(gamma_t_upper[idx]),
                "delta_logMstar_dex": float(delta_log_mstar_upper[idx]),
                "observed_excess_lower_bound_dex": float(observed_excess_lower_bound[idx]),
                "corrected_excess_lower_bound_dex": float(corrected_excess_lower_bound[idx]),
                "anomalous_observed_lower_bound": bool(observed_excess_lower_bound[idx] > 0),
                "resolved_after_tep": None,
                "mdyn_limit_type": "upper_limit",
            }
            if "source" in df_upper_limits.columns and pd.notna(row.get("source")):
                payload["source"] = str(row["source"])
            if "sigma_kms" in df_upper_limits.columns and pd.notna(row.get("sigma_kms")):
                payload["sigma_kms"] = float(row["sigma_kms"])
            if "re_kpc" in df_upper_limits.columns and pd.notna(row.get("re_kpc")):
                payload["re_kpc"] = float(row["re_kpc"])
            rows.append(payload)

    return rows


def _run_direct_object_level(direct_sample):
    df_direct = direct_sample["exact"]
    df_upper_limits = direct_sample["upper_limits"]
    z_arr = df_direct["z"].to_numpy(dtype=float)
    log_mstar_obs = df_direct["log_Mstar_obs"].to_numpy(dtype=float)
    log_mh = stellar_to_halo_mass_behroozi_like(log_mstar_obs, z_arr)
    gamma_t_arr = tep_compute_gamma_t(log_mh, z_arr)
    mass_correction = np.power(np.maximum(gamma_t_arr, 0.01), N_ML_HIGHZ)
    delta_log_mstar = np.log10(mass_correction)
    log_mstar_tep = log_mstar_obs - delta_log_mstar
    log_mdyn = df_direct["log_Mdyn"].to_numpy(dtype=float)
    observed_excess = log_mstar_obs - log_mdyn
    corrected_excess = log_mstar_tep - log_mdyn

    anomalous = observed_excess > 0
    resolved = corrected_excess <= 0
    anomalous_and_resolved = anomalous & resolved

    reference_mask = anomalous if anomalous.any() else np.isfinite(observed_excess)
    published_excess_dex = float(np.mean(observed_excess[reference_mask]))
    tep_reduction_dex = float(np.mean(delta_log_mstar[reference_mask]))
    beta_boot = _bootstrap_beta(
        np.log10(np.maximum(gamma_t_arr, 1.0001)),
        observed_excess,
        calibration_mask=reference_mask,
    )
    if beta_boot is not None and len(df_upper_limits) > 0:
        beta_boot["n_upper_limit_rows_excluded"] = int(len(df_upper_limits))

    upper_limit_summary = None
    if len(df_upper_limits) > 0:
        z_upper = df_upper_limits["z"].to_numpy(dtype=float)
        log_mstar_obs_upper = df_upper_limits["log_Mstar_obs"].to_numpy(dtype=float)
        log_mh_upper = stellar_to_halo_mass_behroozi_like(log_mstar_obs_upper, z_upper)
        gamma_t_upper = tep_compute_gamma_t(log_mh_upper, z_upper)
        mass_correction_upper = np.power(np.maximum(gamma_t_upper, 0.01), N_ML_HIGHZ)
        delta_log_mstar_upper = np.log10(mass_correction_upper)
        log_mstar_tep_upper = log_mstar_obs_upper - delta_log_mstar_upper
        log_mdyn_upper = df_upper_limits["log_Mdyn_upper"].to_numpy(dtype=float)
        observed_excess_lower_bound = log_mstar_obs_upper - log_mdyn_upper
        corrected_excess_lower_bound = log_mstar_tep_upper - log_mdyn_upper
        upper_limit_summary = {
            "n_objects": int(len(df_upper_limits)),
            "mean_observed_excess_lower_bound_dex": float(np.mean(observed_excess_lower_bound)),
            "median_observed_excess_lower_bound_dex": float(np.median(observed_excess_lower_bound)),
            "mean_corrected_excess_lower_bound_dex": float(np.mean(corrected_excess_lower_bound)),
            "median_corrected_excess_lower_bound_dex": float(np.median(corrected_excess_lower_bound)),
            "n_lower_bound_anomalous": int(np.sum(observed_excess_lower_bound > 0)),
            "n_lower_bound_nonpositive_after_tep": int(np.sum(corrected_excess_lower_bound <= 0)),
        }

    source_breakdown = None
    combined_direct_rows = pd.concat([df_direct, df_upper_limits], ignore_index=True, sort=False)
    if "source" in combined_direct_rows.columns:
        source_breakdown = {
            str(source): int(count)
            for source, count in combined_direct_rows["source"].fillna("unknown").value_counts().items()
        }

    summary = {
        "step": STEP_NUM,
        "description": "TEP M*/M_dyn correction using direct object-level kinematic measurements",
        "analysis_class": "real_data_direct_object_level_kinematics",
        "direct_kinematic_measurements_used": True,
        "kinematic_table": {
            "n_objects": int(len(combined_direct_rows)),
            "n_objects_exact_mdyn": int(len(df_direct)),
            "n_objects_upper_limit_only": int(len(df_upper_limits)),
        },
        "n_kinematic_regime": int(len(combined_direct_rows)),
        "global_stats": {
            "mean_gamma_t": float(np.mean(gamma_t_arr)),
            "mean_mass_correction_factor": float(np.mean(mass_correction)),
            "mean_delta_logMstar_dex": float(np.mean(delta_log_mstar)),
        },
        "published_tension_resolution": {
            "regime": "Direct object-level kinematic sample",
            "published_excess_dex": published_excess_dex,
            "tep_reduction_dex": tep_reduction_dex,
            "resolved": bool(np.mean(corrected_excess[reference_mask]) <= 0),
        },
        "object_level_summary": {
            "n_objects_total": int(len(combined_direct_rows)),
            "n_exact_mdyn_objects": int(len(df_direct)),
            "n_upper_limit_objects": int(len(df_upper_limits)),
            "n_anomalous_observed": int(anomalous.sum()),
            "n_resolved_after_tep": int(anomalous_and_resolved.sum()),
            "resolution_fraction_among_anomalous": float(np.mean(anomalous_and_resolved[anomalous])) if anomalous.any() else None,
            "mean_observed_excess_dex": float(np.mean(observed_excess)),
            "median_observed_excess_dex": float(np.median(observed_excess)),
            "mean_corrected_excess_dex": float(np.mean(corrected_excess)),
            "median_corrected_excess_dex": float(np.median(corrected_excess)),
            "mean_excess_metrics_basis": "exact_mdyn_rows_only",
        },
        "object_level_beta_bootstrap": beta_boot,
        "objects": _serialise_object_level_rows(
            df_direct=df_direct,
            gamma_t_arr=gamma_t_arr,
            delta_log_mstar=delta_log_mstar,
            observed_excess=observed_excess,
            corrected_excess=corrected_excess,
            df_upper_limits=df_upper_limits,
        ),
        "methodology": {
            "source_data": "Local curated object-level kinematic table",
            "comparison_type": "Observed M* vs direct M_dyn per object",
            "mass_correction_formula": "M*_tep = M*_standard / Gamma_t^n_ML",
            "n_ML": N_ML_HIGHZ,
            "upper_limit_handling": (
                "Rows with log_Mdyn_upper but no exact log_Mdyn are included as censored upper-limit objects. "
                "They contribute conservative lower-bound excess metrics in upper_limit_summary and the object list, "
                "but are excluded from exact-mean excess summaries and beta bootstrap calculations."
            ),
        },
    }
    if upper_limit_summary is not None:
        summary["upper_limit_summary"] = upper_limit_summary
    if source_breakdown:
        summary["source_breakdown"] = source_breakdown
    return summary


def _run_regime_level_fallback():
    data_path = INTERIM_DIR / "step_002_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data not found: {data_path}. Run step_002 first.", "ERROR")
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    print_status(f"Loaded UNCOVER TEP catalog: N = {len(df)}", "PROCESS")

    mask = (df["z_phot"] > 4.0) & (df["log_Mstar"] > 9.5) & (df["gamma_t"].notna())
    df_kin = df[mask].copy()
    print_status(f"Kinematic regime (z>4, log M*>9.5): N = {len(df_kin)}", "PROCESS")

    if len(df_kin) < 5:
        print_status("Insufficient galaxies in kinematic regime.", "ERROR")
        raise ValueError("Too few galaxies after filter")

    gamma_t_arr  = df_kin["gamma_t"].values
    log_mstar    = df_kin["log_Mstar"].values
    z_arr        = df_kin["z_phot"].values

    mass_correction   = gamma_t_arr ** N_ML_HIGHZ
    log_mstar_tep     = log_mstar - np.log10(mass_correction)
    delta_log_mstar   = log_mstar - log_mstar_tep

    print_status(f"Mean Gamma_t: {gamma_t_arr.mean():.3f} ± {gamma_t_arr.std():.3f}", "INFO")
    print_status(f"Mean mass correction factor: {mass_correction.mean():.3f}×", "INFO")
    print_status(f"Mean Δlog M* (SED − TEP): {delta_log_mstar.mean():.3f} dex", "INFO")
    print_status(f"Median Δlog M*: {np.median(delta_log_mstar):.3f} dex", "INFO")

    bins = [
        ("z=4–5,  logM*=9.5–10.5",  (z_arr >= 4) & (z_arr < 5)  & (log_mstar >= 9.5) & (log_mstar < 10.5)),
        ("z=4–5,  logM*>10.5",       (z_arr >= 4) & (z_arr < 5)  & (log_mstar >= 10.5)),
        ("z=5–7,  logM*>9.5",        (z_arr >= 5) & (z_arr < 7)  & (log_mstar >= 9.5)),
        ("z>7,    logM*>9.5",        (z_arr >= 7)                 & (log_mstar >= 9.5)),
    ]
    bin_results = []
    for label, sel in bins:
        n_bin = sel.sum()
        if n_bin < 3:
            continue
        g  = gamma_t_arr[sel]
        mc = mass_correction[sel]
        bin_results.append({
            "regime":              label,
            "n":                   int(n_bin),
            "mean_gamma_t":        float(g.mean()),
            "median_gamma_t":      float(np.median(g)),
            "mean_correction":     float(mc.mean()),
            "median_correction":   float(np.median(mc)),
            "mean_delta_logMstar": float(delta_log_mstar[sel].mean()),
        })
        print_status(
            f"  {label}: N={n_bin}, mean Γ_t={g.mean():.2f}, "
            f"mean corr={mc.mean():.2f}× (Δlog M*={delta_log_mstar[sel].mean():.2f} dex)",
            "INFO"
        )

    rubies_regime = (z_arr >= 4.0) & (z_arr < 5.5) & (log_mstar >= 10.5)
    n_rubies = rubies_regime.sum()
    mean_delta_rubies = float(delta_log_mstar[rubies_regime].mean()) if n_rubies > 0 else np.nan
    published_tension_dex = 0.15
    resolved = (n_rubies > 0) and (mean_delta_rubies >= published_tension_dex)

    print_status("\n--- Model Prediction vs Published Tension ---", "INFO")
    print_status(f"Published M*/M_dyn excess (de Graaff+24): ~{published_tension_dex:.2f} dex", "INFO")
    print_status(f"TEP mean reduction in this regime: {mean_delta_rubies:.2f} dex", "INFO")
    print_status(f"Resolution achieved: {'YES' if resolved else 'NO'}", "SUCCESS" if resolved else "ERROR")

    return {
        "step": STEP_NUM,
        "description": "TEP M*/M_dyn correction magnitude (real UNCOVER data)",
        "analysis_class": "real_data_derived_regime_comparison",
        "direct_kinematic_measurements_used": False,
        "n_kinematic_regime": int(len(df_kin)),
        "global_stats": {
            "mean_gamma_t": float(gamma_t_arr.mean()),
            "mean_mass_correction_factor": float(mass_correction.mean()),
            "mean_delta_logMstar_dex": float(delta_log_mstar.mean())
        },
        "sub_regimes": bin_results,
        "published_tension_resolution": {
            "regime": "z~4.5, log M*>10.5 (RUBIES-like)",
            "published_excess_dex": published_tension_dex,
            "tep_reduction_dex": mean_delta_rubies,
            "resolved": bool(resolved)
        },
        "object_level_beta_bootstrap": None,
        "methodology": {
            "source_data": "Real UNCOVER DR4 photometry (step_002)",
            "comparison_type": "Real-data-derived mass correction compared against published external kinematic regime tension",
            "mass_correction_formula": "M*_tep = M*_standard / Gamma_t^n_ML",
            "n_ML": N_ML_HIGHZ
        }
    }


def _run_with_direct_sample(direct_sample, direct_meta):
    direct_summary = _run_direct_object_level(direct_sample)
    direct_summary["kinematic_table"].update(direct_meta)

    if bool(direct_meta.get("authoritative_for_l4_regime", False)):
        direct_summary["kinematic_table"]["used_as_primary_analysis"] = True
        direct_summary["supplementary_regime_level"] = _run_regime_level_fallback()
        direct_summary["methodology"] = dict(direct_summary.get("methodology", {}))
        direct_summary["methodology"]["downstream_beta_source"] = "direct_object_level_bootstrap"
        return direct_summary

    regime_summary = _run_regime_level_fallback()
    regime_summary["analysis_class"] = "real_data_derived_regime_comparison_with_supplementary_object_level_kinematics"
    regime_summary["kinematic_table"] = {
        **direct_meta,
        "used_as_primary_analysis": False,
        "supplementary_object_level_summary_available": True,
    }
    regime_summary["supplementary_direct_object_level"] = direct_summary
    regime_summary["supplementary_direct_object_level_available"] = True
    regime_summary["supplementary_direct_object_level_beta_bootstrap"] = direct_summary.get("object_level_beta_bootstrap")
    regime_summary["methodology"] = dict(regime_summary.get("methodology", {}))
    regime_summary["methodology"]["supplementary_object_level_role"] = direct_meta.get("analysis_role")
    regime_summary["methodology"]["supplementary_object_level_source"] = direct_meta.get("sample_name")
    if bool(direct_meta.get("authoritative_for_beta_propagation", False)) and direct_summary.get("object_level_beta_bootstrap") is not None:
        regime_summary["object_level_beta_bootstrap"] = direct_summary.get("object_level_beta_bootstrap")
        regime_summary["methodology"]["downstream_beta_source"] = "supplementary_direct_object_level_bootstrap"
    else:
        regime_summary["methodology"]["downstream_beta_source"] = "regime_level_fallback"
    return regime_summary


def run():
    print_status("=" * 65, "TITLE")
    print_status(f"STEP {STEP_NUM}: TEP M*/M_dyn Correction — Real UNCOVER Data", "TITLE")
    print_status("=" * 65, "TITLE")

    direct_sample, direct_meta = _load_direct_kinematic_table()
    if direct_sample is not None:
        print_status(
            f"Loaded local object-level kinematic table: {direct_meta['path']} (N={direct_meta['n_objects']})",
            "INFO",
        )
        if direct_meta.get("authoritative_for_l4_regime", False):
            print_status("Direct object-level sample is marked primary for L4.", "INFO")
        else:
            print_status("Direct object-level sample loaded as supplementary; matched regime-level fallback remains primary.", "INFO")
        if direct_meta.get("authoritative_for_beta_propagation", False):
            print_status("Direct object-level sample is marked authoritative for downstream beta propagation.", "INFO")
        summary = _run_with_direct_sample(direct_sample, direct_meta)
    else:
        print_status(
            "No local object-level kinematic table detected; using regime-level fallback.",
            "INFO",
        )
        summary = _run_regime_level_fallback()
        summary["kinematic_table"] = direct_meta

    output_path = RESULTS_DIR / f"step_{STEP_NUM}_dynamical_mass_comparison.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=safe_json_default)
    print_status(f"Results saved to {output_path.name}", "SUCCESS")

    return summary

if __name__ == "__main__":
    run()

