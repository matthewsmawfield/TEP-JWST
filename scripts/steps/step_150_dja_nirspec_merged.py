#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.1s.
"""
TEP-JWST Step 150: DJA NIRSpec Merged v4.4 catalog — rho(Gamma_t, log M*)

Ingests the DAWN JWST Archive (DJA) merged NIRSpec spectroscopic catalog v4.4
(Brammer et al.; de Graaff et al. 2024; 19,445 sources across 50+ JWST programs)
and computes rho(Gamma_t, log M*) by redshift bin, producing Table B2 statistics.

The DJA catalog is publicly available from the DJA GitHub/S3 archive.
If the catalog is not present, this step attempts to download it automatically.

Data: DJA NIRSpec Merged v4.4 (Brammer et al. 2024)
Fallback: JADES DR4 spectroscopic catalog

Author: Matthew L. Smawfield
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import (
    compute_gamma_t,
    stellar_to_halo_mass_behroozi_like,
)
from scripts.utils.downloader import smart_download

STEP_NUM  = "150"
STEP_NAME = "dja_nirspec_merged"

DATA_PATH   = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH   = PROJECT_ROOT / "logs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# DJA NIRSpec merged catalog candidates (try in order)
DJA_CATALOG_FILE = DATA_PATH / "dja_msaexp_emission_lines_v4.4.csv.gz"
DJA_DOWNLOAD_URLS = [
    "https://zenodo.org/records/15472354/files/dja_msaexp_emission_lines_v4.4.csv.gz?download=1",
    "https://zenodo.org/api/records/15472354/files/dja_msaexp_emission_lines_v4.4.csv.gz/content",
]

# Fallback: JADES DR4 spectroscopic catalog (already present)
JADES_SPEC_FILE = DATA_PATH / "jades_hainline" / "JADES_DR4_spectroscopic_catalog.fits"

# Redshift bins for Table B2
Z_BINS = [(5, 7), (7, 9), (9, 99)]


def try_download_dja():
    """Attempt to download the DJA NIRSpec merged catalog."""
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


def load_dja_merged_catalog():
    if not DJA_CATALOG_FILE.exists() and not try_download_dja():
        return None, None

    logger.info(f"Loading DJA merged table: {DJA_CATALOG_FILE.name}")
    try:
        table = pd.read_csv(DJA_CATALOG_FILE, compression="infer", low_memory=False)
    except Exception as e:
        logger.warning(f"Could not read DJA merged table: {e}")
        return None, None

    if len(table) == 0:
        return None, None

    z_series = None
    for col in ["z_best", "zgrade", "z_prism", "z_grating", "zline"]:
        if col in table.columns:
            z_series = pd.to_numeric(table[col], errors="coerce")
            break

    mass_series = None
    for col in ["phot_mass", "mass", "log_mass", "log_Mstar"]:
        if col in table.columns:
            mass_series = _coerce_log_mass(table[col])
            break

    if z_series is None or mass_series is None:
        logger.warning("DJA merged table lacks the required redshift or stellar-mass columns.")
        return None, None

    df = pd.DataFrame({
        "z": z_series,
        "log_Mstar": mass_series,
    })
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

    df = df[(df["z"] > 0) & (df["z"] < 15) & (df["log_Mstar"] > 5) & (df["log_Mstar"] < 13)]
    if len(df) == 0:
        return None, None

    logger.info(f"DJA merged table loaded: N={len(df):,}")
    return df, f"DJA NIRSpec v4.4 merged table ({DJA_CATALOG_FILE.name})"


def load_jades_spec_catalog():
    """Load JADES DR4 spectroscopic catalog as fallback."""
    from astropy.io import fits
    from astropy.cosmology import Planck18 as cosmo

    if not JADES_SPEC_FILE.exists():
        return None

    logger.info("Loading JADES DR4 spectroscopic catalog (fallback) ...")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(JADES_SPEC_FILE) as hdul:
            # Use Obs_info extension for z and basic info
            obs = hdul[1].data
            # Try to get spec-z and NIRCam ID
            for i in range(min(len(obs), 5190)):
                try:
                    z_flag = None
                    z_val  = None
                    # Try R100_5pix for z (Z_FLAG is a string in DR4, not int)
                    if len(hdul) > 2:
                        row2 = hdul[2].data[i]
                        try:
                            z_val = float(row2["z_PRISM"])
                        except Exception:
                            pass
                        try:
                            z_flag = str(row2["Z_FLAG"])
                        except Exception:
                            z_flag = None
                    rows.append({"z_spec": z_val, "z_flag": z_flag})
                except Exception:
                    break

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Z_FLAG in JADES DR4 is a string (e.g. '{2, 4}'), not int; filter by valid z_PRISM
    df = df[df["z_spec"].notna() & (df["z_spec"] > 0) & (df["z_spec"] < 15)]
    logger.info(f"JADES spec-z valid: N={len(df):,}")
    return df


def analyze_bin(df, z_lo, z_hi):
    """Spearman rho(Gamma_t, log M*) in a redshift bin."""
    sub = df[(df["z"] >= z_lo) & (df["z"] < z_hi)].dropna(
        subset=["gamma_t", "log_Mstar"]
    )
    n = len(sub)
    if n < 5:
        return {"z_lo": z_lo, "z_hi": z_hi, "n": n, "note": "insufficient_sample"}
    rho, p = spearmanr(sub["gamma_t"].values, sub["log_Mstar"].values)
    return {
        "z_lo":  z_lo,
        "z_hi":  z_hi if z_hi < 90 else 99,
        "n":     n,
        "rho":   float(rho),
        "p":     max(float(p), 1e-300),
    }


def run():
    print_status(f"STEP {STEP_NUM}: DJA NIRSpec Merged v4.4 — rho(Gamma_t, log M*)", "INFO")

    if not DJA_CATALOG_FILE.exists():
        logger.info("DJA merged table not found locally; attempting public v4.4 download.")

    df, catalog_used = load_dja_merged_catalog()

    # ---- Fallback: JADES DR4 spec catalog ----
    if df is None or len(df) == 0:
        logger.info("Falling back to JADES DR4 spectroscopic catalog ...")
        jades_df = load_jades_spec_catalog()
        if jades_df is not None and len(jades_df) > 0:
            df = jades_df.rename(columns={"z_spec": "z"})
            
            # Try to cross-match with JADES physical catalog for stellar masses
            phys_path = PROJECT_ROOT / "data" / "interim" / "jades_highz_physical.csv"
            has_mass = False
            if phys_path.exists():
                phys = pd.read_csv(phys_path)
                if "z_spec" in phys.columns and "log_Mstar" in phys.columns:
                    # Match on z_spec to within 0.01
                    phys_spec = phys[phys["z_spec"].notna()].copy()
                    from scipy.spatial import cKDTree
                    if len(phys_spec) > 0:
                        tree = cKDTree(phys_spec[["z_spec"]].values)
                        dists, idxs = tree.query(df[["z"]].values, distance_upper_bound=0.01)
                        valid_match = dists != np.inf
                        
                        df["log_Mstar"] = np.nan
                        df.loc[valid_match, "log_Mstar"] = phys_spec.iloc[idxs[valid_match]]["log_Mstar"].values
                        has_mass = df["log_Mstar"].notna().sum() > 10
            
            if not has_mass:
                # Estimate mass using empirical proxy if no match
                df["log_Mstar"] = 8.5 + (df["z"] - 5) * 0.1
                catalog_used = "JADES DR4 spectroscopic catalog (fallback; M* empirical proxy)"
            else:
                catalog_used = "JADES DR4 spectroscopic catalog (matched to physical catalog)"

    using_dja_catalog = catalog_used is not None and "DJA" in str(catalog_used)

    if df is None or len(df) == 0:
        result = {
            "step": STEP_NUM, "name": STEP_NAME,
            "status": "SKIPPED_NO_CATALOG",
            "note": (
                "DJA NIRSpec merged catalog not available. "
                "Download the public v4.4 merged table from Zenodo "
                "(https://zenodo.org/records/15472354) and place at: "
                f"{DJA_CATALOG_FILE}"
            ),
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.warning(f"Step {STEP_NUM} skipped: catalog unavailable.")
        return result

    from astropy.cosmology import Planck18 as cosmo

    # Compute TEP quantities (if mass available)
    has_mass = df["log_Mstar"].notna().any()
    if has_mass:
        valid = df[df["log_Mstar"].notna()].copy()
        valid["log_Mh"] = stellar_to_halo_mass_behroozi_like(
            valid["log_Mstar"].values, valid["z"].values
        )
        valid["gamma_t"] = compute_gamma_t(valid["log_Mh"].values, valid["z"].values)
        df = valid

        # Full sample
        df_z5 = df[df["z"] >= 5]
        rho_full, p_full = spearmanr(df_z5["gamma_t"].values, df_z5["log_Mstar"].values)

        # Redshift bins
        bin_results = [analyze_bin(df, z_lo, z_hi) for z_lo, z_hi in Z_BINS]

        logger.info(f"Full z>5: N={len(df_z5):,}, rho={rho_full:.3f}, p={p_full:.2e}")
        for b in bin_results:
            logger.info(f"  z={b['z_lo']}-{b['z_hi']}: N={b.get('n',0):,}, rho={b.get('rho',float('nan')):.3f}")

        if using_dja_catalog:
            result_status = "SUCCESS"
            result_description = "DJA NIRSpec rho(Gamma_t, log M*) — Table B2"
            result_note = None
        else:
            result_status = "SUCCESS_REFERENCE_ONLY"
            result_description = "JADES fallback spectroscopic mass-proxy reference; live Table B2 not reproduced"
            result_note = (
                "The DJA merged cross-survey catalog is unavailable in this workspace. "
                "This fallback uses JADES DR4 plus matched or empirical masses and is not "
                "counted as the reproducible Table B2 confirmation."
            )

        result = {
            "step":   STEP_NUM,
            "name":   STEP_NAME,
            "status": result_status,
            "description": result_description,
            "catalog_used": catalog_used,
            "reproducible_dja_available": bool(using_dja_catalog),
            "manuscript_table_reproduced": bool(using_dja_catalog),
            "n_total": len(df_z5),
            "rho_full": float(rho_full),
            "p_full":   max(float(p_full), 1e-300),
            "redshift_bins": bin_results,
            "note": result_note,
        }
    else:
        result_status = "PARTIAL_NO_MASS" if using_dja_catalog else "PARTIAL_REFERENCE_ONLY"
        result_note = (
            "Catalog loaded but stellar masses unavailable; cannot compute rho(Gamma_t, M*)."
            if using_dja_catalog else
            "JADES fallback loaded without usable stellar masses; this does not reproduce the live DJA Table B2 result."
        )
        result = {
            "step":   STEP_NUM,
            "name":   STEP_NAME,
            "status": result_status,
            "catalog_used": catalog_used,
            "reproducible_dja_available": False,
            "manuscript_table_reproduced": False,
            "n_total": len(df),
            "note": result_note,
        }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete ({result['status']})", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
