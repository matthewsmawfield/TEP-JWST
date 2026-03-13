#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): <0.5s.
"""
TEP-JWST Step 164: UNCOVER z=9-12 null audit

Diagnose why the UNCOVER DR4 full SPS dust-Gamma_t correlation becomes null in
z=9-12 despite remaining positive in z=7-8 and z=8-9. This step uses the live
step_152 interim catalog and audits three concrete failure modes:

1. Sample-size collapse in the ultra-high-z tail.
2. Dust-posterior compression and larger parameter uncertainties.
3. Whether the signal returns in the highest-quality subset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging

STEP_NUM = "164"  # Pipeline step number
STEP_NAME = "uncover_z9_null_audit"  # Used in log / output filenames
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
for path in [OUTPUT_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

PHOTO_CSV = INTERIM_PATH / "step_152_uncover_dr4_full_sps.csv"
ZSPEC_CSV = INTERIM_PATH / "step_152_uncover_dr4_zspec.csv"

BINS = [
    (7.0, 8.0, "z=7-8"),
    (8.0, 9.0, "z=8-9"),
    (9.0, 12.0, "z=9-12"),
]


def p_clip(value):
    return max(float(value), 1e-300)



def summarize_bin(df_sub: pd.DataFrame, z_lo: float, z_hi: float, label: str) -> dict:
    sub = df_sub[(df_sub["z_phot"] >= z_lo) & (df_sub["z_phot"] < z_hi)].copy()
    n = len(sub)
    if n < 10:
        return {"label": label, "z_lo": z_lo, "z_hi": z_hi, "n": int(n), "status": "underpowered"}

    rho, p = spearmanr(sub["gamma_t"], sub["dust2"])

    quality_metric = np.square(sub["sigma_dust2"].fillna(np.inf).values) + np.square(sub["sigma_z_rel"].fillna(np.inf).values)
    qcut = np.nanpercentile(quality_metric[np.isfinite(quality_metric)], 50) if np.isfinite(quality_metric).any() else np.inf
    highq = sub[quality_metric <= qcut].copy()
    highq_stats = None
    if len(highq) >= 10:
        rho_hq, p_hq = spearmanr(highq["gamma_t"], highq["dust2"])
        highq_stats = {
            "n": int(len(highq)),
            "rho": float(rho_hq),
            "p": p_clip(p_hq),
        }

    summary = {
        "label": label,
        "z_lo": z_lo,
        "z_hi": z_hi,
        "n": int(n),
        "rho": float(rho),
        "p": p_clip(p),
        "dust2_median": float(np.nanmedian(sub["dust2"])),
        "dust2_iqr": float(np.nanpercentile(sub["dust2"], 75) - np.nanpercentile(sub["dust2"], 25)),
        "dust2_p90_minus_p10": float(np.nanpercentile(sub["dust2"], 90) - np.nanpercentile(sub["dust2"], 10)),
        "median_sigma_dust2": float(np.nanmedian(sub["sigma_dust2"])),
        "median_sigma_log_mstar": float(np.nanmedian(sub["sigma_log_Mstar"])),
        "median_sigma_z_rel": float(np.nanmedian(sub["sigma_z_rel"])),
        "fraction_t_eff_gt_0p3": float(np.mean(sub["t_eff"] >= 0.3)),
        "median_gamma_t": float(np.nanmedian(sub["gamma_t"])),
        "median_log_mstar": float(np.nanmedian(sub["log_Mstar"])),
        "high_quality_subset": highq_stats,
    }
    return summary



def build_diagnosis(bin_map: dict[str, dict]) -> dict:
    z89 = bin_map.get("z=8-9")
    z912 = bin_map.get("z=9-12")
    if not z89 or not z912 or z89.get("status") == "underpowered" or z912.get("status") == "underpowered":
        return {
            "status": "insufficient_bins",
            "assessment": "step_152 does not currently provide enough z>8 sample depth for the planned audit.",
        }

    n_ratio = z912["n"] / max(z89["n"], 1)
    range_ratio = z912["dust2_p90_minus_p10"] / max(z89["dust2_p90_minus_p10"], 1e-6)
    dust_sigma_ratio = z912["median_sigma_dust2"] / max(z89["median_sigma_dust2"], 1e-6)
    z_sigma_ratio = z912["median_sigma_z_rel"] / max(z89["median_sigma_z_rel"], 1e-6)
    highq = z912.get("high_quality_subset") or {}

    flags = {
        "sample_size_collapse": bool(n_ratio < 0.7),
        "dust_posterior_compression": bool(range_ratio < 0.75),
        "uncertainty_inflation": bool(dust_sigma_ratio > 1.15 or z_sigma_ratio > 1.15),
        "high_quality_subset_recovers_signal": bool(highq and highq.get("rho", 0.0) > 0 and highq.get("p", 1.0) < 0.1),
    }

    if flags["high_quality_subset_recovers_signal"]:
        assessment = "The z=9-12 null is consistent with uncertainty dilution rather than a clean signal disappearance."
    elif flags["sample_size_collapse"] or flags["dust_posterior_compression"] or flags["uncertainty_inflation"]:
        assessment = "The z=9-12 null is likely sensitivity-limited: the tail sample is smaller and/or the dust posteriors are more compressed or uncertain than in z=8-9."
    else:
        assessment = "The z=9-12 null persists without a clear posterior-compression signature; treat it as a genuine live limitation pending more data."

    return {
        "status": "success",
        "n_ratio_z9_12_vs_z8_9": float(n_ratio),
        "dust_dynamic_range_ratio_z9_12_vs_z8_9": float(range_ratio),
        "dust_uncertainty_ratio_z9_12_vs_z8_9": float(dust_sigma_ratio),
        "z_uncertainty_ratio_z9_12_vs_z8_9": float(z_sigma_ratio),
        "flags": flags,
        "assessment": assessment,
    }



def run():
    print_status(f"STEP {STEP_NUM}: UNCOVER z=9-12 null audit", "INFO")
    if not PHOTO_CSV.exists():
        result = {
            "step": STEP_NUM,
            "name": STEP_NAME,
            "status": "FAILED_NO_DATA",
            "note": f"Missing {PHOTO_CSV}. Run step_152_uncover_dr4_full_sps.py first.",
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        out_path.write_text(json.dumps(result, indent=2))
        return result

    df = pd.read_csv(PHOTO_CSV)
    df = df.dropna(subset=["z_phot", "gamma_t", "dust2", "log_Mstar", "t_eff"]).copy()
    df = df[(df["z_phot"] >= 7.0) & (df["z_phot"] < 12.0)]

    if "dust2_16" in df.columns and "dust2_84" in df.columns:
        df["sigma_dust2"] = 0.5 * np.abs(df["dust2_84"] - df["dust2_16"])
    else:
        df["sigma_dust2"] = np.nan
    if "log_Mstar_16" in df.columns and "log_Mstar_84" in df.columns:
        df["sigma_log_Mstar"] = 0.5 * np.abs(df["log_Mstar_84"] - df["log_Mstar_16"])
    else:
        df["sigma_log_Mstar"] = np.nan
    if "z_16" in df.columns and "z_84" in df.columns:
        df["sigma_z_rel"] = 0.5 * np.abs(df["z_84"] - df["z_16"]) / np.clip(1.0 + df["z_phot"], 1e-6, None)
    else:
        df["sigma_z_rel"] = np.nan

    bin_summaries = [summarize_bin(df, z_lo, z_hi, label) for z_lo, z_hi, label in BINS]
    bin_map = {row["label"]: row for row in bin_summaries}
    diagnosis = build_diagnosis(bin_map)

    zspec_context = None
    if ZSPEC_CSV.exists():
        zspec = pd.read_csv(ZSPEC_CSV)
        zspec_context = {
            "n_total": int(len(zspec)),
            "n_z_gt_4": int((zspec["z_spec"] > 4).sum()) if "z_spec" in zspec.columns else None,
            "n_z_gt_5": int((zspec["z_spec"] > 5).sum()) if "z_spec" in zspec.columns else None,
            "n_z_gt_7": int((zspec["z_spec"] > 7).sum()) if "z_spec" in zspec.columns else None,
        }

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "description": "UNCOVER DR4 audit of the z=9-12 dust-Gamma_t null branch",
        "photoz_source": str(PHOTO_CSV),
        "zspec_context": zspec_context,
        "bin_summaries": bin_summaries,
        "diagnosis": diagnosis,
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Saved: {out_path}")
    print_status(diagnosis.get("assessment", "Audit complete."), "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
