#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.6s.
"""
TEP-JWST Step 158: DJA NIRSpec Ha/Hb Balmer decrement vs Gamma_t (Table B5)

Spectroscopic dust measurement using the Balmer decrement Ha/Hb.
Uses the JADES DR4 spectroscopic catalog (R100 5-pixel extraction) which
provides Hb (4861A) and the Ha+[NII] blend (6563A + [NII] 6584A) for
all available sources with SNR>3 detections.

Note: The blend Blnd_HBaA_N2_flux includes [NII] contamination.
At z>6 the [NII] fraction is small (low metallicity), so the blend
serves as a Halpha proxy. At z<6 the blend should be interpreted
cautiously (noted as caveat in manuscript Table B5).

Data: JADES DR4 spectroscopic catalog (de Graaff et al.; D'Eugenio et al. 2025)
Requires: data/raw/jades_hainline/JADES_DR4_spectroscopic_catalog.fits

Author: Matthew L. Smawfield
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.rank_stats import partial_rank_correlation
from scripts.utils.tep_model import (
    compute_gamma_t,
    stellar_to_halo_mass_behroozi_like,
)
from scripts.utils.downloader import smart_download

STEP_NUM  = "158"
STEP_NAME = "dja_balmer_decrement"

DATA_PATH   = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH   = PROJECT_ROOT / "logs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

JADES_SPEC_FILE = DATA_PATH / "jades_hainline" / "JADES_DR4_spectroscopic_catalog.fits"
DJA_CATALOG_FILE = DATA_PATH / "dja_msaexp_emission_lines_v4.4.csv.gz"
DJA_DOWNLOAD_URLS = [
    "https://zenodo.org/records/15472354/files/dja_msaexp_emission_lines_v4.4.csv.gz?download=1",
    "https://zenodo.org/api/records/15472354/files/dja_msaexp_emission_lines_v4.4.csv.gz/content",
]

# Redshift bins for Table B5 (z ranges where Ha/Hb both fall in NIRSpec)
Z_BINS = [
    (2.0, 4.0, "z=2-4"),
    (4.0, 5.0, "z=4-5"),
    (5.0, 6.0, "z=5-6"),
    (6.0, 7.0, "z=6-7"),
]
SNR_MIN = 3.0  # minimum SNR for both lines


def partial_spearman(x, y, controls):
    """Spearman partial correlation after regressing out control variables."""
    rho, p, _ = partial_rank_correlation(x, y, controls)
    return float(rho), float(p)


def bootstrap_ci(x, y, n_boot=200, seed=42):
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos.append(float(r))
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def try_download_dja():
    try:
        return smart_download(
            url=DJA_DOWNLOAD_URLS[0],
            dest=DJA_CATALOG_FILE,
            min_size_mb=10,
            fallback_urls=DJA_DOWNLOAD_URLS[1:],
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"Download attempt failed: {e}")
        return False


def _coerce_log_mass(series):
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values) & (values > 0)]
    if len(finite) == 0:
        return values
    if np.nanmedian(finite) > 100:
        values = values.where(values > 0)
        return np.log10(values)
    return values


def load_jades_balmer():
    """Load JADES DR4 Halpha, Hbeta, and spec-z from catalog."""
    from astropy.io import fits
    from astropy.cosmology import Planck18 as cosmo

    if not JADES_SPEC_FILE.exists():
        logger.error(f"JADES spec catalog not found: {JADES_SPEC_FILE}")
        return None

    logger.info(f"Loading JADES DR4 spectroscopic catalog ...")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(JADES_SPEC_FILE) as hdul:
            obs  = hdul[1].data   # Obs_info
            r100 = hdul[2].data   # R100_5pix (line fluxes)

            n_rows = min(len(obs), len(r100))
            logger.info(f"  Catalog rows: {n_rows:,}")

            for i in range(n_rows):
                try:
                    z_val = float(r100["z_PRISM"][i])
                except Exception:
                    z_val = np.nan
                try:
                    z_flag = str(r100["Z_FLAG"][i])
                except Exception:
                    z_flag = None
                try:
                    ha_blend = float(r100["Blnd_HBaA_N2_flux"][i])
                    ha_err   = float(r100["Blnd_HBaA_N2_flux_err"][i])
                    hb_flux  = float(r100["HBaB_4861_flux"][i])
                    hb_err   = float(r100["HBaB_4861_flux_err"][i])
                    rows.append({
                        "z_spec":   z_val,
                        "z_flag":   z_flag,
                        "ha_blend": ha_blend,
                        "ha_err":   ha_err,
                        "hb_flux":  hb_flux,
                        "hb_err":   hb_err,
                    })
                except Exception:
                    pass

    if not rows:
        return None

    df = pd.DataFrame(rows)
    logger.info(f"  Rows loaded: {len(df):,}")

    # Quality cuts: valid redshift, positive fluxes
    # Note: Z_FLAG in JADES DR4 is a string (e.g. '{2, 4}'), not an int grade
    df = df[
        (df["z_spec"] > 0) & (df["z_spec"] < 10) &
        (df["ha_blend"] > 0) &
        (df["hb_flux"] > 0)
    ].copy()

    # SNR cuts
    df["ha_snr"] = df["ha_blend"] / df["ha_err"].replace(0, np.nan)
    df["hb_snr"] = df["hb_flux"] / df["hb_err"].replace(0, np.nan)
    df = df[
        df["ha_snr"].notna() & (df["ha_snr"] > SNR_MIN) &
        df["hb_snr"].notna() & (df["hb_snr"] > SNR_MIN)
    ].copy()

    # Balmer decrement: Ha/Hb ratio (proxy for dust, Av)
    df["balmer_ratio"] = df["ha_blend"] / df["hb_flux"]
    df["log_balmer"] = np.log10(np.clip(df["balmer_ratio"], 1e-4, None))
    logger.info(f"  Valid Balmer sources (SNR>{SNR_MIN}): {len(df):,}")

    # Median Balmer ratio (Intrinsic = 2.86 for case B recombination)
    logger.info(f"  Median Ha/Hb ratio: {df['balmer_ratio'].median():.3f}")

    # Assign approximate stellar masses using median-mass by redshift bin
    # (JADES spec catalog doesn't carry SED masses directly)
    # Use a proxy: assign log_Mstar ~ 9 + (z - 5) * 0.1 as a nominal value,
    # then compute Gamma_t from that. The correlation will reflect whether
    # higher-z (deeper potential) objects have higher Balmer ratios.
    # NOTE: For a clean analysis, the DJA catalog with SED masses is preferred.
    df = df.copy()
    df["t_cosmic"] = cosmo.age(df["z_spec"].values).value

    return df


def load_dja_balmer():
    """Load DJA merged catalog with Balmer line fluxes (if available)."""
    if not DJA_CATALOG_FILE.exists() and not try_download_dja():
        return None, None

    logger.info(f"Loading DJA NIRSpec merged catalog for Balmer analysis ...")
    try:
        table = pd.read_csv(DJA_CATALOG_FILE, compression="infer", low_memory=False)
        if len(table) == 0:
            return None, None

        z_series = None
        for col in ["z_best", "zgrade", "z_prism", "z_grating", "zline"]:
            if col in table.columns:
                z_series = pd.to_numeric(table[col], errors="coerce")
                break

        if z_series is None or "line_hb" not in table.columns:
            return None, None

        if "line_ha_nii" in table.columns:
            ha_blend = pd.to_numeric(table["line_ha_nii"], errors="coerce")
            ha_err = pd.to_numeric(table.get("line_ha_nii_err"), errors="coerce") if "line_ha_nii_err" in table.columns else np.nan
        else:
            ha = pd.to_numeric(table.get("line_ha"), errors="coerce") if "line_ha" in table.columns else np.nan
            nii_6549 = pd.to_numeric(table.get("line_nii_6549"), errors="coerce") if "line_nii_6549" in table.columns else 0.0
            nii_6584 = pd.to_numeric(table.get("line_nii_6584"), errors="coerce") if "line_nii_6584" in table.columns else 0.0
            ha_blend = ha + nii_6549 + nii_6584
            ha_err = np.nan

        df = pd.DataFrame({
            "z_spec": z_series,
            "ha_blend": ha_blend,
            "hb_flux": pd.to_numeric(table["line_hb"], errors="coerce"),
        })
        if not np.isscalar(ha_err):
            df["ha_err"] = ha_err
        if "line_hb_err" in table.columns:
            df["hb_err"] = pd.to_numeric(table["line_hb_err"], errors="coerce")
        if "phot_mass" in table.columns:
            df["log_Mstar"] = _coerce_log_mass(table["phot_mass"])
        if "grade" in table.columns:
            df["grade"] = pd.to_numeric(table["grade"], errors="coerce")
        if "valid" in table.columns:
            df["valid"] = pd.to_numeric(table["valid"], errors="coerce")
        if "sn_line" in table.columns:
            df["sn_line"] = pd.to_numeric(table["sn_line"], errors="coerce")
        if "objid" in table.columns:
            df["objid"] = pd.to_numeric(table["objid"], errors="coerce")
        elif "srcid" in table.columns:
            df["objid"] = pd.to_numeric(table["srcid"], errors="coerce")

        if "objid" in df.columns:
            if "grade" not in df.columns:
                df["grade"] = np.nan
            if "sn_line" not in df.columns:
                df["sn_line"] = np.nan
            df = df.sort_values(["objid", "grade", "sn_line"], ascending=[True, False, False], na_position="last")
            df = df.drop_duplicates(subset="objid", keep="first")

        if "valid" in df.columns and df["valid"].notna().any():
            df = df[df["valid"] > 0]
        elif "grade" in df.columns and df["grade"].notna().any():
            df = df[df["grade"] >= 2]

        df = df[
            (df["z_spec"] > 0) & (df["z_spec"] < 10) &
            (df["ha_blend"] > 0) & (df["hb_flux"] > 0)
        ].copy()
        if "ha_err" in df.columns and "hb_err" in df.columns:
            df["ha_snr"] = df["ha_blend"] / df["ha_err"].replace(0, np.nan)
            df["hb_snr"] = df["hb_flux"] / df["hb_err"].replace(0, np.nan)
            df = df[
                df["ha_snr"].notna() & (df["ha_snr"] > SNR_MIN) &
                df["hb_snr"].notna() & (df["hb_snr"] > SNR_MIN)
            ].copy()
        df["balmer_ratio"] = df["ha_blend"] / df["hb_flux"]
        df["log_balmer"] = np.log10(df["balmer_ratio"].clip(1e-4))
        return df, f"DJA NIRSpec v4.4 ({DJA_CATALOG_FILE.name})"
    except Exception as e:
        logger.warning(f"Could not read DJA Balmer data: {e}")
    return None, None


def analyze_bin(sub, z_lo, z_hi, label, has_mass):
    """Compute Spearman rho and partial rho for one redshift bin."""
    n = len(sub)
    if n < 10:
        return {"label": label, "n": n, "note": "insufficient_sample"}

    # Use t_cosmic as proxy predictor if no Gamma_t
    if has_mass and "gamma_t" in sub.columns:
        x = sub["gamma_t"].values
    else:
        x = sub["t_cosmic"].values

    y = sub["log_balmer"].values

    rho, p = spearmanr(x, y)
    p = max(float(p), 1e-300)

    median_ratio = float(sub["balmer_ratio"].median())

    res = {
        "label":         label,
        "z_lo":          z_lo,
        "z_hi":          z_hi,
        "n":             n,
        "rho":           float(rho),
        "p":             p,
        "median_ratio":  median_ratio,
    }

    # Partial rho if masses available
    if has_mass and "log_Mstar" in sub.columns and sub["log_Mstar"].notna().sum() > 10:
        sub_valid = sub.dropna(subset=["log_Mstar", "gamma_t", "log_balmer"])
        if len(sub_valid) >= 10:
            try:
                p_rho, p_p = partial_spearman(
                    sub_valid["gamma_t"].values,
                    sub_valid["log_balmer"].values,
                    controls=[sub_valid["log_Mstar"].values, sub_valid["z_spec"].values]
                )
                p_p = max(float(p_p), 1e-300)
                ci_lo, ci_hi = bootstrap_ci(
                    sub_valid["gamma_t"].values, sub_valid["log_balmer"].values
                )
                res.update({
                    "partial_rho": p_rho,
                    "partial_p":   p_p,
                    "ci_lo":       ci_lo,
                    "ci_hi":       ci_hi,
                })
            except Exception:
                pass

    return res


def run():
    print_status(f"STEP {STEP_NUM}: DJA NIRSpec Balmer decrement (Table B5)", "INFO")

    # Try DJA catalog first
    dja_result = load_dja_balmer()
    df, catalog_used = dja_result if dja_result else (None, None)

    # Fallback: JADES spec catalog
    if df is None or len(df) == 0:
        df = load_jades_balmer()
        catalog_used = "JADES DR4 spectroscopic catalog (fallback; merged external product unavailable)"

    using_dja_catalog = catalog_used is not None and str(catalog_used).startswith("DJA")

    if df is None or len(df) == 0:
        result = {
            "step": STEP_NUM, "name": STEP_NAME,
            "status": "SKIPPED_NO_DATA",
            "note": f"Neither DJA catalog nor JADES spec catalog available.",
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    has_mass = "log_Mstar" in df.columns and df["log_Mstar"].notna().any()
    predictor_used = "gamma_t" if has_mass else "t_cosmic_fallback"

    # If masses available: compute Gamma_t
    if has_mass:
        from astropy.cosmology import Planck18 as cosmo
        valid = df[df["log_Mstar"].notna() & (df["log_Mstar"] > 5)].copy()
        valid["log_Mh"] = stellar_to_halo_mass_behroozi_like(
            valid["log_Mstar"].values, valid["z_spec"].values
        )
        valid["gamma_t"] = compute_gamma_t(valid["log_Mh"].values, valid["z_spec"].values)
        df = valid

    # Also compute t_cosmic for all (as fallback predictor)
    if "t_cosmic" not in df.columns:
        from astropy.cosmology import Planck18 as cosmo
        df["t_cosmic"] = cosmo.age(df["z_spec"].values).value

    logger.info(f"Total sources with Balmer detection: N={len(df):,}")
    logger.info(f"  z range: {df['z_spec'].min():.2f} - {df['z_spec'].max():.2f}")
    logger.info(f"  Median Ha/Hb: {df['balmer_ratio'].median():.3f}")

    # Full sample correlation
    df_gt2 = df[df["z_spec"] > 2]
    if len(df_gt2) >= 10:
        if has_mass and "gamma_t" in df_gt2.columns:
            rho_full, p_full = spearmanr(df_gt2["gamma_t"].values, df_gt2["log_balmer"].values)
        else:
            rho_full, p_full = spearmanr(df_gt2["t_cosmic"].values, df_gt2["log_balmer"].values)
        p_full = max(float(p_full), 1e-300)
        logger.info(f"  Full z>2: rho={rho_full:.3f}, p={p_full:.2e}, N={len(df_gt2):,}")
    else:
        rho_full, p_full = None, None

    # Partial rho (full sample)
    partial_rho_full, partial_p_full = None, None
    if has_mass and "gamma_t" in df.columns:
        valid_all = df.dropna(subset=["gamma_t", "log_balmer", "log_Mstar", "z_spec"])
        valid_all = valid_all[valid_all["z_spec"] > 2]
        if len(valid_all) >= 20:
            try:
                partial_rho_full, partial_p_full = partial_spearman(
                    valid_all["gamma_t"].values,
                    valid_all["log_balmer"].values,
                    controls=[valid_all["log_Mstar"].values, valid_all["z_spec"].values]
                )
                partial_p_full = max(partial_p_full, 1e-300)
                ci_lo_full, ci_hi_full = bootstrap_ci(
                    valid_all["gamma_t"].values, valid_all["log_balmer"].values
                )
                logger.info(
                    f"  Full z>2 partial rho: {partial_rho_full:.3f}, p={partial_p_full:.2e}, "
                    f"N={len(valid_all):,}"
                )
            except Exception as e:
                logger.warning(f"Partial rho failed: {e}")

    # Redshift-binned analysis
    bin_results = []
    for z_lo, z_hi, label in Z_BINS:
        sub = df[(df["z_spec"] >= z_lo) & (df["z_spec"] < z_hi)].copy()
        res = analyze_bin(sub, z_lo, z_hi, label, has_mass)
        bin_results.append(res)
        logger.info(
            f"  {label}: N={res.get('n',0):,}, "
            f"rho={res.get('rho', float('nan')):.3f}, "
            f"partial_rho={res.get('partial_rho', None)}"
        )

    reproducible_dja_available = bool(
        using_dja_catalog
        and partial_rho_full is not None
        and partial_p_full is not None
    )
    if reproducible_dja_available:
        result_status = "SUCCESS"
        result_description = "DJA NIRSpec Ha/Hb Balmer decrement — Table B5"
        result_note = None
    elif using_dja_catalog:
        result_status = "SUCCESS_REFERENCE_ONLY"
        result_description = "DJA Balmer decrement loaded, but live Table B5 was not reproduced"
        result_note = (
            "DJA merged catalog loaded, but the mass-complete partial Gamma_t result "
            "required for the live Table B5 reproduction is unavailable."
        )
    else:
        result_status = "SUCCESS_REFERENCE_ONLY"
        result_description = "JADES fallback Balmer reference; live Table B5 not reproduced"
        result_note = (
            "JADES fallback loaded; the DJA merged product is unavailable, so this output "
            "is a reference-only fallback and is not used as the reproducible Balmer confirmation."
        )

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": result_status,
        "description": result_description,
        "catalog_used": catalog_used,
        "predictor_used": predictor_used,
        "reproducible_dja_available": reproducible_dja_available,
        "manuscript_table_reproduced": reproducible_dja_available,
        "n_total_z2": len(df_gt2),
        "rho_full":         float(rho_full) if rho_full is not None else None,
        "p_full":           float(p_full)   if p_full   is not None else None,
        "partial_rho_full": float(partial_rho_full) if partial_rho_full is not None else None,
        "partial_p_full":   float(partial_p_full)   if partial_p_full   is not None else None,
        "redshift_bins":    bin_results,
        "note": result_note,
        "caveat": (
            "Ha flux uses Blnd_HBaA_N2 (Halpha+[NII] blend). "
            "[NII] contamination significant at z<6; interpret cautiously. "
            "JADES DR4 catalog (D'Eugenio et al. 2025, msaexp pipeline)."
        ),
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete. N(z>2)={len(df_gt2):,}", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
