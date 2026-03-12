# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.9s.
"""
step_155_jades_dr5_morphology.py

JADES DR5 photometry (94k sources, GOODS-S) cross-matched with JADES DR4
spectroscopic catalog via NIRCam_DR5_ID. Tests whether deeper gravitational
potentials (higher Gamma_t) host more compact galaxies (smaller half-light
radius), as predicted by TEP-enhanced early mass assembly.

TEP prediction: deeper potentials → earlier, more concentrated mass assembly
→ smaller effective radii at fixed mass/redshift.

Data:
  - JADES DR5: data/raw/jades_hainline/jades_dr5_goods_s_photometry.fits
    (Robertson et al. 2026; 94,000 sources, GOODS-S-Deep)
  - JADES DR4 spec-z: data/raw/jades_hainline/JADES_DR4_spectroscopic_catalog.fits
    (Scholtz, Carniani et al. 2025; 5,190 sources, 2,858 good spec-z)
"""

import json
import logging

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.downloader import smart_download
from scripts.utils.rank_stats import partial_rank_correlation
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like


STEP_NUM = "155"
STEP_NAME = "jades_dr5_morphology"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DR4_PATH = ROOT / "data/raw/jades_hainline/JADES_DR4_spectroscopic_catalog.fits"
DR5_PATH = ROOT / "data/raw/jades_hainline/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits"
DR5_FALLBACK_PATHS = [
    ROOT / "data/raw/JADES_z_gt_8_Candidates_Hainline_et_al.fits",
]
DR5_DOWNLOAD_URLS = [
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/catalogs/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits",
]
PHYSICAL_PATH = ROOT / "data/interim/jades_highz_physical.csv"
OUTPUT = ROOT / "results/outputs/step_155_jades_dr5_morphology.json"

H0 = 70.0
OM = 0.3


def rest_frame_rhalf(rhalf_arcsec, z, filter_wave_um, target_wave_um=0.5):
    """
    Select the filter closest to rest-frame target_wave_um at given z,
    and convert arcsec to physical kpc.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=OM)
    da = cosmo.angular_diameter_distance(z).value  # Mpc
    rhalf_kpc = rhalf_arcsec * (np.pi / 180.0 / 3600.0) * da * 1e3
    return rhalf_kpc


def resolve_dr5_catalog():
    candidates = [DR5_PATH] + DR5_FALLBACK_PATHS
    for path in candidates:
        if path.exists() and path.stat().st_size > 200 * 1e6:
            return path
    for path in candidates:
        if path.exists():
            return path
    try:
        ok = smart_download(
            url=DR5_DOWNLOAD_URLS[0],
            dest=DR5_PATH,
            min_size_mb=200,
            logger=logger,
        )
        if ok:
            return DR5_PATH
    except Exception as e:
        logger.warning(f"Could not obtain JADES DR5 catalog: {e}")
    return None


def run_assoc(label, mask, gamma, predictor, controls, expected_sign):
    m = mask & np.isfinite(gamma) & np.isfinite(predictor)
    for ctrl in controls:
        m &= np.isfinite(ctrl)
    n = int(m.sum())
    if n < 20:
        log.info(f"  {label}: N={n} (skip)")
        return None
    rho_raw, p_raw = spearmanr(gamma[m], predictor[m])
    try:
        rho_partial, p_partial, _ = partial_rank_correlation(
            gamma[m],
            predictor[m],
            [ctrl[m] for ctrl in controls],
        )
    except Exception:
        rho_partial, p_partial = np.nan, np.nan
    log.info(
        f"  {label}: raw ρ={rho_raw:.3f}, p={p_raw:.4g}; "
        f"partial ρ={rho_partial:.3f}, p={p_partial:.4g}; N={n}"
    )
    return {
        "expected_sign": expected_sign,
        "rho_raw": float(rho_raw),
        "p_raw": float(p_raw),
        "rho_partial_mass_z": float(rho_partial),
        "p_partial_mass_z": float(p_partial),
        "sign_correct_raw": bool(np.sign(rho_raw) == np.sign(expected_sign)),
        "sign_correct_partial": bool(np.sign(rho_partial) == np.sign(expected_sign)) if np.isfinite(rho_partial) else False,
        "N": n,
    }


def evaluate_sample(sample_name, matched_z, log_mstar, gt, matched_rhalf277, matched_rhalf444, matched_q, matched_gini, control_variables, mass_source):
    valid_mass = np.isfinite(log_mstar)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=OM)
    da_mpc = np.array([cosmo.angular_diameter_distance(z).value for z in matched_z], dtype=float)
    rhalf_kpc = matched_rhalf277 * (np.pi / 180.0 / 3600.0) * da_mpc * 1e3
    log_rhalf = np.log10(np.clip(rhalf_kpc, 1e-3, None))
    log_rhalf_f444 = np.log10(
        np.clip(
            matched_rhalf444 * (np.pi / 180.0 / 3600.0) * da_mpc * 1e3,
            1e-3,
            None,
        )
    )
    log_sigma = log_mstar - 2 * log_rhalf
    results = {}
    bins = [
        (7, 99, "z_gt_7"),
        (5, 99, "z_gt_5"),
        (4, 99, "z_gt_4"),
    ]
    seen_counts = set()
    for zmin, zmax, label in bins:
        base_mask = (matched_z >= zmin) & (matched_z < zmax) & valid_mass
        n_mask = int(base_mask.sum())
        if n_mask < 20 or n_mask in seen_counts:
            continue
        seen_counts.add(n_mask)
        assoc_specs = [
            (f"rhalf_f277_{label}", log_rhalf, -1.0),
            (f"rhalf_f444_{label}", log_rhalf_f444, -1.0),
            (f"axis_ratio_{label}", matched_q, -1.0),
            (f"gini_{label}", matched_gini, +1.0),
            (f"sigma_star_{label}", log_sigma, +1.0),
        ]
        for key, predictor, expected_sign in assoc_specs:
            res = run_assoc(
                key,
                base_mask,
                gt,
                predictor,
                controls=[log_mstar, matched_z],
                expected_sign=expected_sign,
            )
            if res is not None:
                results[key] = res
    supportive_partial = [
        key for key, value in results.items()
        if value["sign_correct_partial"]
        and np.isfinite(value["p_partial_mass_z"])
        and value["p_partial_mass_z"] < 0.05
        and key.split("_z_")[0] in {"gini", "sigma_star", "rhalf_f277", "rhalf_f444"}
    ]
    strongest_key = None
    strongest_score = -np.inf
    for key, value in results.items():
        if np.isfinite(value["rho_partial_mass_z"]):
            score = abs(value["rho_partial_mass_z"])
            if score > strongest_score:
                strongest_key = key
                strongest_score = score
    return {
        "sample_name": sample_name,
        "mass_source": mass_source,
        "n_matched": int(len(matched_z)),
        "n_with_mass": int(valid_mass.sum()),
        "control_variables": control_variables,
        "headline": {
            "n_structural_proxies_supportive_after_mass_z_control": int(len(supportive_partial)),
            "supportive_partial_keys": supportive_partial,
            "strongest_partial_key": strongest_key,
            "conclusion": (
                "controlled_structural_support_present"
                if len(supportive_partial) >= 1 else
                "raw_only_or_mixed_structural_signal"
            ),
        },
        "results": results,
    }


def main():
    log.info("=" * 60)
    log.info("step_155: JADES DR5 morphology × DR4 spec-z TEP test")
    log.info("=" * 60)

    # --- Load DR4 spec-z ---
    log.info("Loading JADES DR4 spectroscopic catalog...")
    if not DR4_PATH.exists():
        log.error(f"Missing: {DR4_PATH} — run step_149 (JADES DR4 ingestion) first. Aborting.")
        return {"status": "aborted", "reason": "missing JADES_DR4_spectroscopic_catalog.fits"}
    with fits.open(DR4_PATH) as f:
        obs = f["Obs_info"].data

    dr4_id = obs["NIRCam_DR5_ID"].astype(np.int64)
    z_spec = obs["z_Spec"].astype(float)
    z_flag = obs["z_Spec_flag"]
    muv = obs["MUV"].astype(float)

    # Quality cuts: good spec-z (A/B flags), z > 0
    good = (
        np.isin(z_flag, ["A", "B"])
        & (z_spec > 0)
        & (dr4_id > 0)
    )
    log.info(f"DR4 good spec-z with DR5 ID: N = {good.sum()}")

    dr4_proxy_df = pd.DataFrame(
        {
            "ID": dr4_id[good].astype(np.int64),
            "z_spec": z_spec[good].astype(float),
            "MUV": muv[good].astype(float),
        }
    )

    # --- Load DR5 morphology ---
    log.info("Loading JADES DR5 photometry (SIZE HDU)...")
    dr5_path = resolve_dr5_catalog()
    if dr5_path is None:
        log.warning(f"JADES DR5 file not found: {DR5_PATH}")
        log.warning("Skipping step_155 — download jades_dr5_goods_s_photometry.fits to enable.")
        result = {"status": "skipped", "reason": "JADES DR5 photometry file not available",
                  "file_expected": str(DR5_PATH)}
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as fh:
            json.dump(result, fh, indent=2)
        return result
    with fits.open(dr5_path) as f:
        size = f["SIZE"].data

    dr5_df = pd.DataFrame(
        {
            "ID": size["ID"].astype(np.int64),
            "rhalf_f277": size["F277W_RHALF"].astype(float),
            "rhalf_f444": size["F444W_RHALF"].astype(float),
            "q": size["Q"].astype(float),
            "gini": size["GINI"].astype(float),
        }
    ).drop_duplicates(subset="ID")

    # --- Cross-match ---
    log.info("Cross-matching DR4 spec-z with DR5 morphology...")
    dr4_matched = dr4_proxy_df.merge(dr5_df, on="ID", how="inner")
    dr4_matched = dr4_matched[
        np.isfinite(dr4_matched["rhalf_f277"]) & (dr4_matched["rhalf_f277"] > 0)
    ].copy()
    log.info(f"DR4 proxy sample with valid F277W_RHALF: N = {len(dr4_matched)}")

    matched_muv = dr4_matched["MUV"].to_numpy(dtype=float)
    log_mstar_proxy = np.where(np.isfinite(matched_muv), -0.35 * (matched_muv + 20.0) + 9.0, np.nan)
    matched_z_proxy = dr4_matched["z_spec"].to_numpy(dtype=float)
    log_mh_proxy = stellar_to_halo_mass_behroozi_like(log_mstar_proxy, matched_z_proxy)
    gt_proxy = compute_gamma_t(log_mh_proxy, matched_z_proxy)
    legacy_sample = evaluate_sample(
        "dr4_spec_muv_proxy",
        matched_z_proxy,
        log_mstar_proxy,
        gt_proxy,
        dr4_matched["rhalf_f277"].to_numpy(dtype=float),
        dr4_matched["rhalf_f444"].to_numpy(dtype=float),
        dr4_matched["q"].to_numpy(dtype=float),
        dr4_matched["gini"].to_numpy(dtype=float),
        ["MUV-derived log_Mstar", "z_spec"],
        "MUV-derived log_Mstar",
    )

    samples = {"dr4_spec_muv_proxy": legacy_sample}
    preferred_sample_name = "dr4_spec_muv_proxy"

    if PHYSICAL_PATH.exists():
        log.info("Loading JADES physical catalog for direct-mass morphology control...")
        physical_df = pd.read_csv(PHYSICAL_PATH)
        if {"ID", "z_best", "log_Mstar"}.issubset(physical_df.columns):
            physical_df = physical_df.copy()
            physical_df["ID"] = pd.to_numeric(physical_df["ID"], errors="coerce")
            physical_df["z_best"] = pd.to_numeric(physical_df["z_best"], errors="coerce")
            physical_df["z_spec"] = pd.to_numeric(physical_df.get("z_spec", np.nan), errors="coerce")
            physical_df["log_Mstar"] = pd.to_numeric(physical_df["log_Mstar"], errors="coerce")
            if "log_Mhalo" in physical_df.columns:
                physical_df["log_Mhalo"] = pd.to_numeric(physical_df["log_Mhalo"], errors="coerce")
            else:
                physical_df["log_Mhalo"] = np.nan
            physical_df["z_use"] = np.where(
                np.isfinite(physical_df["z_spec"]) & (physical_df["z_spec"] > 0),
                physical_df["z_spec"],
                physical_df["z_best"],
            )
            physical_matched = physical_df.merge(dr5_df, on="ID", how="inner")
            physical_matched = physical_matched[
                np.isfinite(physical_matched["rhalf_f277"]) & (physical_matched["rhalf_f277"] > 0)
            ].copy()
            log.info(f"Physical-mass sample with valid F277W_RHALF: N = {len(physical_matched)}")
            if len(physical_matched) > 0:
                matched_z_direct = physical_matched["z_use"].to_numpy(dtype=float)
                log_mstar_direct = physical_matched["log_Mstar"].to_numpy(dtype=float)
                log_mh_direct = physical_matched["log_Mhalo"].to_numpy(dtype=float)
                missing_halo = ~np.isfinite(log_mh_direct)
                if missing_halo.any():
                    log_mh_direct[missing_halo] = stellar_to_halo_mass_behroozi_like(
                        log_mstar_direct[missing_halo],
                        matched_z_direct[missing_halo],
                    )
                gt_direct = compute_gamma_t(log_mh_direct, matched_z_direct)
                direct_sample = evaluate_sample(
                    "physical_catalog_direct_mass",
                    matched_z_direct,
                    log_mstar_direct,
                    gt_direct,
                    physical_matched["rhalf_f277"].to_numpy(dtype=float),
                    physical_matched["rhalf_f444"].to_numpy(dtype=float),
                    physical_matched["q"].to_numpy(dtype=float),
                    physical_matched["gini"].to_numpy(dtype=float),
                    ["log_Mstar", "z_best_or_z_spec"],
                    "direct_log_Mstar",
                )
                samples["physical_catalog_direct_mass"] = direct_sample
                legacy_support = legacy_sample["headline"]["n_structural_proxies_supportive_after_mass_z_control"]
                direct_support = direct_sample["headline"]["n_structural_proxies_supportive_after_mass_z_control"]
                if (
                    direct_support > legacy_support
                    or (
                        direct_support == legacy_support
                        and direct_sample["mass_source"] == "direct_log_Mstar"
                    )
                ):
                    preferred_sample_name = "physical_catalog_direct_mass"

    preferred_sample = samples[preferred_sample_name]

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"  Preferred sample: {preferred_sample_name}")
    for sample_name, sample in samples.items():
        log.info(
            f"  {sample_name}: N_matched={sample['n_matched']}, N_with_mass={sample['n_with_mass']}, "
            f"supportive={sample['headline']['n_structural_proxies_supportive_after_mass_z_control']}"
        )
        for k, v in sample["results"].items():
            log.info(
                f"  {sample_name}.{k}: raw ρ={v['rho_raw']:.3f}, partial ρ={v['rho_partial_mass_z']:.3f}, "
                f"N={v['N']}"
            )
    log.info("=" * 60)

    result = {
        "step": STEP_NUM,
        "description": "JADES DR5 morphology: controlled structural support for L2",
        "catalog_used": str(dr5_path),
        "preferred_sample": preferred_sample_name,
        "n_matched": preferred_sample["n_matched"],
        "n_with_mass": preferred_sample["n_with_mass"],
        "n_with_muv": legacy_sample["n_with_mass"],
        "control_variables": preferred_sample["control_variables"],
        "headline": preferred_sample["headline"],
        "results": preferred_sample["results"],
        "samples": samples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
