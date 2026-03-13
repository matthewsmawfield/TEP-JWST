#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 3.2s.
"""
TEP-JWST Step 156: DJA GOODS-S morphology cross-matched with spec-z (Yang+2025)

DJA GOODS-S morphology cross-matched with spec-z (Yang+2025)


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging

STEP_NUM  = "156"  # Pipeline step number
STEP_NAME = "dja_gds_morphology"  # Used in log / output filenames

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH   = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

import warnings
import numpy as np
from scipy.stats import spearmanr  # Rank correlation for morphology analysis

DATA_RAW  = PROJECT_ROOT / "data" / "raw"  # Raw external catalogues
INTERIM   = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products

JADES_PHOT_FILE = DATA_RAW / "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"
JADES_SPEC_FILE = DATA_RAW / "jades_hainline" / "JADES_DR4_spectroscopic_catalog.fits"

Z_MIN  = 4.0
Z_MAX  = 12.0
Z_BINS = [(4, 6), (6, 8), (8, 12)]

# Morphology columns in JADES photometry SIZE extension
MORPH_COLS = ["GINI", "ASYMMETRY", "CONCENTRATION", "M20", "FWHM", "R_KRON", "Q"]


def load_jades_phot_size():
    """Load the SIZE extension of JADES photometry for morphological parameters."""
    if not JADES_PHOT_FILE.exists():
        return None, None
    from astropy.io import fits
    warnings.filterwarnings("ignore")
    try:
        with fits.open(JADES_PHOT_FILE, memmap=False) as hdul:
            # Find the SIZE extension
            size_ext = None
            flag_ext = None
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'name') and 'SIZE' in str(hdu.name).upper():
                    size_ext = i
                if hasattr(hdu, 'name') and 'FLAG' in str(hdu.name).upper():
                    flag_ext = i

            if size_ext is None:
                # Try ext 3 (known structure)
                size_ext = min(3, len(hdul) - 1)
            if flag_ext is None:
                flag_ext = min(2, len(hdul) - 1)

            size_data = hdul[size_ext].data
            flag_data = hdul[flag_ext].data

            if size_data is None:
                return None, None

            size_cols = [c.name.upper() for c in size_data.columns]
            flag_cols = [c.name.upper() for c in flag_data.columns] if flag_data is not None else []

            logger.info(f"  SIZE ext cols: {size_cols[:10]}")
            logger.info(f"  FLAG ext cols: {flag_cols[:10]}")

            # Find ID column for crossmatch
            id_col_size = next((c for c in size_cols if 'ID' in c), None)
            id_col_flag = next((c for c in flag_cols if 'ID' in c), None)

            return size_data, flag_data, size_cols, flag_cols, id_col_size, id_col_flag
    except Exception as e:
        logger.warning(f"  Failed to load JADES photometry SIZE ext: {e}")
        return None, None, None, None, None, None


def load_jades_spec_redshifts():
    """Load JADES DR4 spec redshifts from the spectroscopic catalog."""
    if not JADES_SPEC_FILE.exists():
        return None
    from astropy.io import fits
    warnings.filterwarnings("ignore")
    try:
        rows = []
        with fits.open(JADES_SPEC_FILE, memmap=False) as hdul:
            for ext_i in range(1, len(hdul)):
                try:
                    data = hdul[ext_i].data
                    if data is None or len(data) == 0:
                        continue
                    cols = [c.name.upper() for c in data.columns]
                    z_col = next((c for c in ["Z_PRISM", "REDSHIFT", "Z"] if c in cols), None)
                    id_col = next((c for c in ["NIRCAM_DR5_ID", "NIRCAM_DR3_ID", "ID", "UNIQUE_ID"] if c in cols), None)
                    if z_col and id_col:
                        for row in data:
                            try:
                                z = float(row[z_col])
                                source_id = int(row[id_col]) if row[id_col] else None
                                if z > 0 and source_id:
                                    rows.append({"id": source_id, "z": z})
                            except Exception:
                                pass
                        if rows:
                            logger.info(f"  Spec-z from ext {ext_i}: N={len(rows):,}")
                            break
                except Exception:
                    continue
        return rows if rows else None
    except Exception as e:
        logger.warning(f"  Failed to load JADES spec-z: {e}")
        return None


def run():
    print_status(f"STEP {STEP_NUM}: GOODS-S morphology vs. Gamma_t analysis", "INFO")
    import pandas as pd
    from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like

    result_loaded = load_jades_phot_size()
    if result_loaded[0] is None:
        # Fallback: use JADES interim + morphology proxy
        interim_file = INTERIM / "jades_highz_physical.csv"
        if interim_file.exists():
            df = pd.read_csv(interim_file)
            df = df[(df["z_best"] > Z_MIN)].dropna(subset=["z_best", "log_Mstar"])
            log_mh = stellar_to_halo_mass_behroozi_like(
                df["log_Mstar"].values, df["z_best"].values
            )
            df["gamma_t"] = compute_gamma_t(log_mh, df["z_best"].values)
            # Use age_ratio as morphology proxy (younger/older = late-type/early-type)
            if "age_ratio" in df.columns:
                rho, p = spearmanr(df["gamma_t"].values, df["age_ratio"].values)
                morph_key = "age_ratio"
            else:
                rho, p = float("nan"), float("nan")
                morph_key = "none"
            result = {
                "step":   STEP_NUM,
                "name":   STEP_NAME,
                "status": "SUCCESS_PARTIAL",
                "description": "GOODS-S morphology — JADES photometry FITS not found; using interim age_ratio proxy",
                "note":   "JADES photometry FITS not found. Used age_ratio from jades_highz_physical.csv as morphology proxy.",
                "source": "jades_highz_physical.csv",
                "n":      int(len(df)),
                "morph_proxy":        morph_key,
                "rho_gamma_t_morph":  float(rho) if np.isfinite(rho) else None,
                "p_gamma_t_morph":    max(float(p), 1e-300) if np.isfinite(p) else None,
            }
        else:
            result = {
                "step":   STEP_NUM,
                "name":   STEP_NAME,
                "status": "SKIPPED_NO_DATA",
                "note":   "JADES photometry FITS not found; run step_149 first.",
            }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    size_data, flag_data, size_cols, flag_cols, id_col_size, id_col_flag = result_loaded

    # Load spec-z for ID crossmatch
    spec_rows = load_jades_spec_redshifts()

    # Build morphology dataframe
    morph_rows = []
    avail_morcols = [c for c in MORPH_COLS if c in size_cols]
    for i, row in enumerate(size_data):
        try:
            entry = {}
            if id_col_size:
                entry["id"] = int(row[id_col_size])
            for mc in avail_morcols:
                try:
                    entry[mc.lower()] = float(row[mc])
                except Exception:
                    entry[mc.lower()] = np.nan
            morph_rows.append(entry)
        except Exception:
            pass

    if not morph_rows:
        result = {"step": STEP_NUM, "name": STEP_NAME, "status": "SKIPPED_NO_MORPHDATA"}
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    df_morph = pd.DataFrame(morph_rows)
    logger.info(f"  Morphology rows: {len(df_morph):,}")

    # Crossmatch with spec-z
    if spec_rows and id_col_size:
        df_spec = pd.DataFrame(spec_rows)
        df_merge = df_morph.merge(df_spec, on="id", how="inner")
        df_merge = df_merge[(df_merge["z"] > Z_MIN) & (df_merge["z"] < Z_MAX)]
        logger.info(f"  After spec-z crossmatch z>{Z_MIN}: N={len(df_merge):,}")
    else:
        df_merge = pd.DataFrame()
        logger.warning("  No spec-z crossmatch available")

    if len(df_merge) < 5:
        result = {
            "step":   STEP_NUM,
            "name":   STEP_NAME,
            "status": "SUCCESS_PARTIAL",
            "description": "GOODS-S morphology — insufficient spec-z crossmatch",
            "n_morph":   len(df_morph),
            "n_matched": len(df_merge),
            "note":   "Morphology loaded but spec-z crossmatch yields <5 sources at z>4.",
        }
        out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # Compute Gamma_t from z (use fiducial M*=9.5)
    log_mh = stellar_to_halo_mass_behroozi_like(
        np.full(len(df_merge), 9.5), df_merge["z"].values
    )
    df_merge["gamma_t"] = compute_gamma_t(log_mh, df_merge["z"].values)

    # Correlations with morphological parameters
    morph_results = []
    for mc in [c.lower() for c in avail_morcols]:
        if mc not in df_merge.columns:
            continue
        sub = df_merge.dropna(subset=[mc])
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub["gamma_t"].values, sub[mc].values)
        morph_results.append({
            "parameter": mc,
            "n":         len(sub),
            "rho":       float(rho),
            "p":         max(float(p), 1e-300),
        })
        logger.info(f"  {mc}: N={len(sub)}, rho(Gamma_t, {mc})={rho:.3f}")

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "GOODS-S morphological parameters vs. Gamma_t from JADES photometry",
        "n_morph":          len(df_morph),
        "n_specz_matched":  len(df_merge),
        "morphology_cols":  avail_morcols,
        "morphology_correlations": morph_results,
        "conclusion": (
            f"GOODS-S morphology crossmatch (N={len(df_merge):,} at z>4): "
            + (f"rho(Gamma_t, {morph_results[0]['parameter']})={morph_results[0]['rho']:.3f}. " if morph_results else "No morphological correlations computed. ")
            + "TEP predicts compact (high-Gini) galaxies in deeper potentials due to faster inside-out growth."
        ),
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
