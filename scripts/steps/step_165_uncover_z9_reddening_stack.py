import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types

STEP_NUM = "165"  # Pipeline step number
STEP_NAME = "uncover_z9_reddening_stack"  # Used in log / output filenames
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
for path in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

INPUT_CSV = INTERIM_PATH / "step_152_uncover_dr4_full_sps.csv"
INPUT_NULL_AUDIT = OUTPUT_PATH / "step_164_uncover_z9_null_audit.json"
BOOTSTRAP_N = 4000
RNG_SEED = 165


def _clip_p(value: float) -> float:
    return max(float(value), 1e-300)


def _weighted_mean(values: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-3, None)
    weights = 1.0 / np.square(sigma)
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def _bootstrap_weighted_delta(low_values: np.ndarray, low_sigma: np.ndarray, high_values: np.ndarray, high_sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draws = np.empty(BOOTSTRAP_N, dtype=float)
    n_low = len(low_values)
    n_high = len(high_values)
    for i in range(BOOTSTRAP_N):
        idx_low = rng.integers(0, n_low, size=n_low)
        idx_high = rng.integers(0, n_high, size=n_high)
        low_mean = _weighted_mean(low_values[idx_low], low_sigma[idx_low])
        high_mean = _weighted_mean(high_values[idx_high], high_sigma[idx_high])
        draws[i] = high_mean - low_mean
    return draws


def _summarize_metric(low: pd.DataFrame, high: pd.DataFrame, metric: str, sigma_col: str) -> dict | None:
    if metric not in low.columns or metric not in high.columns:
        return None
    low_sub = low[[metric, sigma_col]].replace([np.inf, -np.inf], np.nan).dropna()
    high_sub = high[[metric, sigma_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(low_sub) < 5 or len(high_sub) < 5:
        return None
    rng = np.random.default_rng(RNG_SEED + abs(hash(metric)) % 1000)
    boot = _bootstrap_weighted_delta(
        low_sub[metric].to_numpy(dtype=float),
        low_sub[sigma_col].to_numpy(dtype=float),
        high_sub[metric].to_numpy(dtype=float),
        high_sub[sigma_col].to_numpy(dtype=float),
        rng,
    )
    delta_weighted = _weighted_mean(high_sub[metric].to_numpy(dtype=float), high_sub[sigma_col].to_numpy(dtype=float)) - _weighted_mean(low_sub[metric].to_numpy(dtype=float), low_sub[sigma_col].to_numpy(dtype=float))
    mw = mannwhitneyu(high_sub[metric], low_sub[metric], alternative="two-sided")
    supportive_p = (float(np.sum(boot <= 0.0)) + 1.0) / (len(boot) + 1.0)
    return {
        "metric": metric,
        "n_low": int(len(low_sub)),
        "n_high": int(len(high_sub)),
        "low_median": float(np.nanmedian(low_sub[metric])),
        "high_median": float(np.nanmedian(high_sub[metric])),
        "low_weighted_mean": float(_weighted_mean(low_sub[metric].to_numpy(dtype=float), low_sub[sigma_col].to_numpy(dtype=float))),
        "high_weighted_mean": float(_weighted_mean(high_sub[metric].to_numpy(dtype=float), high_sub[sigma_col].to_numpy(dtype=float))),
        "delta_weighted_high_minus_low": float(delta_weighted),
        "bootstrap_ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_supportive_p_one_sided": _clip_p(supportive_p),
        "mannwhitney_p_two_sided": _clip_p(mw.pvalue),
    }


def _build_contrast(df: pd.DataFrame, subset_label: str, low_q: float, high_q: float) -> dict | None:
    gamma_lo = float(df["gamma_t"].quantile(low_q))
    gamma_hi = float(df["gamma_t"].quantile(high_q))
    low = df[df["gamma_t"] <= gamma_lo].copy()
    high = df[df["gamma_t"] >= gamma_hi].copy()
    if len(low) < 10 or len(high) < 10:
        return None
    metrics = {}
    for metric in ["dust2", "UV", "VJ"]:
        summary = _summarize_metric(low, high, metric, "sigma_dust2")
        if summary is not None:
            metrics[metric] = summary
    return {
        "subset_label": subset_label,
        "gamma_t_quantiles": [float(low_q), float(high_q)],
        "gamma_t_threshold_low": gamma_lo,
        "gamma_t_threshold_high": gamma_hi,
        "n_total": int(len(df)),
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "median_gamma_t_low": float(np.nanmedian(low["gamma_t"])),
        "median_gamma_t_high": float(np.nanmedian(high["gamma_t"])),
        "raw_spearman_gamma_t_vs_dust2": {
            "rho": float(spearmanr(df["gamma_t"], df["dust2"]).statistic),
            "p": _clip_p(spearmanr(df["gamma_t"], df["dust2"]).pvalue),
        },
        "metrics": metrics,
    }


def _assessment(primary: dict | None) -> str:
    if not primary:
        return "No stable contrast could be computed from the current z=9-12 UNCOVER subset."
    dust = primary.get("metrics", {}).get("dust2")
    uv = primary.get("metrics", {}).get("UV")
    vj = primary.get("metrics", {}).get("VJ")
    if not dust:
        return "The stack ran, but dust2 was unavailable in one of the contrast groups."
    dust_pos = dust["delta_weighted_high_minus_low"] > 0
    dust_ci_pos = dust["bootstrap_ci_95"][0] > 0
    colour_pos = bool(uv and uv["delta_weighted_high_minus_low"] > 0) and bool(vj and vj["delta_weighted_high_minus_low"] > 0)
    if dust_ci_pos and colour_pos:
        return "The posterior-broad z=9-12 tail shows a supportive stacked reddening contrast: high-Gamma_t objects are redder on average in dust2 and rest-frame colours even where individual posteriors are broad."
    if dust_pos and colour_pos:
        return "The posterior-broad z=9-12 tail shows a directional stacked reddening contrast, but the present catalog-level surrogate remains suggestive rather than decisive."
    if dust_pos:
        return "The stack gives a positive dust2 contrast for high-Gamma_t objects, but the colour proxies are not jointly aligned strongly enough to claim resolution."
    return "The stacked reddening surrogate does not recover a clear positive contrast in the current z=9-12 catalog-level data."


def run():
    print_status(f"STEP {STEP_NUM}: UNCOVER z=9-12 stacked reddening surrogate", "INFO")
    if not INPUT_CSV.exists():
        result = {
            "step": STEP_NUM,
            "name": STEP_NAME,
            "status": "FAILED_NO_DATA",
            "note": f"Missing {INPUT_CSV}. Run step_152_uncover_dr4_full_sps.py first.",
        }
        (OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json").write_text(json.dumps(result, indent=2))
        return result

    df = pd.read_csv(INPUT_CSV)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[(df["z_phot"] >= 9.0) & (df["z_phot"] < 12.0)].copy()
    df = df.dropna(subset=["gamma_t", "dust2", "dust2_16", "dust2_84", "UV", "VJ"]).copy()
    df["sigma_dust2"] = 0.5 * np.abs(df["dust2_84"] - df["dust2_16"])
    df["sigma_dust2"] = np.clip(df["sigma_dust2"], 1e-3, None)
    df["posterior_broad"] = df["sigma_dust2"] >= float(df["sigma_dust2"].median())

    full_q33 = _build_contrast(df, "all_objects_q33_q67", 1.0 / 3.0, 2.0 / 3.0)
    full_q25 = _build_contrast(df, "all_objects_q25_q75", 0.25, 0.75)
    broad = df[df["posterior_broad"]].copy()
    broad_q33 = _build_contrast(broad, "posterior_broad_q33_q67", 1.0 / 3.0, 2.0 / 3.0)
    broad_q25 = _build_contrast(broad, "posterior_broad_q25_q75", 0.25, 0.75)

    null_audit = None
    if INPUT_NULL_AUDIT.exists():
        try:
            null_audit = json.loads(INPUT_NULL_AUDIT.read_text())
        except Exception:
            null_audit = None

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "description": "Catalog-level stacked reddening surrogate for the UNCOVER z=9-12 null branch",
        "input_catalog": str(INPUT_CSV),
        "rationale": "No local extracted spectra are available in the canonical workspace, so the analysis uses broad-posterior z=9-12 catalog objects as a surrogate for faint or weakly constrained targets.",
        "selection": {
            "z_window": [9.0, 12.0],
            "required_columns": ["gamma_t", "dust2", "dust2_16", "dust2_84", "UV", "VJ"],
            "n_total_after_selection": int(len(df)),
            "n_posterior_broad": int(df["posterior_broad"].sum()),
            "median_sigma_dust2": float(df["sigma_dust2"].median()),
        },
        "contrasts": {
            "all_objects_q33_q67": full_q33,
            "all_objects_q25_q75": full_q25,
            "posterior_broad_q33_q67": broad_q33,
            "posterior_broad_q25_q75": broad_q25,
        },
        "primary_contrast": broad_q25,
        "null_audit_context": null_audit.get("diagnosis") if isinstance(null_audit, dict) else None,
    }
    result["assessment"] = _assessment(result["primary_contrast"])

    out_csv = INTERIM_PATH / f"step_{STEP_NUM}_{STEP_NAME}.csv"
    df.to_csv(out_csv, index=False)
    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    out_json.write_text(json.dumps(result, indent=2, default=safe_json_default))

    dust = (result.get("primary_contrast") or {}).get("metrics", {}).get("dust2")
    if dust:
        print_status(
            f"Primary broad-tail dust2 contrast Δ={dust['delta_weighted_high_minus_low']:.3f} with 95% CI [{dust['bootstrap_ci_95'][0]:.3f}, {dust['bootstrap_ci_95'][1]:.3f}]",
            "INFO",
        )
    print_status(result["assessment"], "INFO")
    return result


if __name__ == "__main__":
    run()
