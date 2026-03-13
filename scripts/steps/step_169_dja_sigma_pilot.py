#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats
from scipy.optimize import least_squares
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.downloader import smart_download  # Robust HTTP download utility
from scripts.utils.logger import TEPLogger, print_status, set_step_logger  # Centralised logging
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting & JSON serialiser
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like  # Shared TEP model

STEP_NUM = "169"  # Pipeline step number
STEP_NAME = "dja_sigma_pilot"  # Used in log / output filenames
DATA_PATH = PROJECT_ROOT / "data" / "raw"  # Raw external catalogues
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
for path in [OUTPUT_PATH, INTERIM_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

DJA_CATALOG_FILE = DATA_PATH / "dja_msaexp_emission_lines_v4.4.csv.gz"
LOCAL_SPECTRA_DIR = DATA_PATH / "dja_specfits"
DJA_S3_BASE = "https://s3.amazonaws.com/msaexp-nirspec/extractions"
C_KMS = 299792.458
AUTO_DOWNLOAD = os.environ.get("TEP_DJA_SIGMA_AUTODOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
SELECTION_MODE = os.environ.get("TEP_DJA_SIGMA_SELECTION_MODE", "pilot").strip().lower()
DEFAULT_MAX_TARGETS = "44" if SELECTION_MODE == "high_mass_same_regime" else "24"
MAX_TARGETS = max(1, int(os.environ.get("TEP_DJA_SIGMA_MAX_TARGETS", DEFAULT_MAX_TARGETS)))
MIN_TARGET_Z = float(os.environ.get("TEP_DJA_SIGMA_MIN_TARGET_Z", "5.0"))
MAX_TARGET_Z = float(os.environ.get("TEP_DJA_SIGMA_MAX_TARGET_Z", "12.5"))
DEFAULT_MIN_TARGET_LOG_MSTAR = "10.0" if SELECTION_MODE == "high_mass_same_regime" else "5.0"
MIN_TARGET_LOG_MSTAR = float(os.environ.get("TEP_DJA_SIGMA_MIN_TARGET_LOG_MSTAR", DEFAULT_MIN_TARGET_LOG_MSTAR))
REQUIRE_BALMER = os.environ.get("TEP_DJA_SIGMA_REQUIRE_BALMER", "0").strip().lower() in {"1", "true", "yes", "on"}
MIN_LINE_SNR = float(os.environ.get("TEP_DJA_SIGMA_MIN_LINE_SNR", "8.0"))
MIN_GRADE = float(os.environ.get("TEP_DJA_SIGMA_MIN_GRADE", "3.0"))
MIN_SUCCESS_FOR_CORRELATION = 6
MIN_WINDOW_POINTS = 12
MIN_SIGMA_KMS_QUALITY = float(os.environ.get("TEP_DJA_SIGMA_MIN_SIGMA_KMS_QUALITY", "10.0"))
MAX_SIGMA_FRAC_ERR_QUALITY = float(os.environ.get("TEP_DJA_SIGMA_MAX_FRAC_ERR_QUALITY", "1.0"))
MAX_REDUCED_CHI2_QUALITY = float(os.environ.get("TEP_DJA_SIGMA_MAX_REDUCED_CHI2_QUALITY", "50.0"))
MIN_AMP_SNR_QUALITY = float(os.environ.get("TEP_DJA_SIGMA_MIN_AMP_SNR_QUALITY", "10.0"))
FALLBACK_R_BY_GRATING = {
    "G140M": 1000.0,
    "G235M": 1000.0,
    "G395M": 1000.0,
    "G140H": 2700.0,
    "G235H": 2700.0,
    "G395H": 2700.0,
}
LINE_DEFS = [
    {"name": "oiii_5007", "rest_wave_a": 5008.24, "flux_col": "line_oiii_5007", "err_col": "line_oiii_5007_err", "window_um": 0.012},
    {"name": "hb", "rest_wave_a": 4862.68, "flux_col": "line_hb", "err_col": "line_hb_err", "window_um": 0.012},
    {"name": "ha", "rest_wave_a": 6564.61, "flux_col": "line_ha", "err_col": "line_ha_err", "window_um": 0.015},
]


def _clip_p(value):
    return format_p_value(value)


def _to_native(array_like):
    arr = np.array(array_like)
    if hasattr(arr.dtype, "byteorder") and arr.dtype.byteorder == ">":
        return arr.astype(arr.dtype.newbyteorder("="))
    return arr


def _coerce_log_mass(series):
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values) & (values > 0)]
    if len(finite) == 0:
        return values
    if np.nanmedian(finite) > 100:
        values = values.where(values > 0)
        return np.log10(values)
    return values


def _resolution_rank(grating):
    text = str(grating).upper()
    if text.endswith("H"):
        return 2
    if text.endswith("M"):
        return 1
    return 0


def _public_url(row):
    return f"{DJA_S3_BASE}/{row['root']}/{row['file']}"


def _local_path(row):
    return LOCAL_SPECTRA_DIR / str(row["root"]) / str(row["file"])


def _balmer_available_mask(df):
    ha = pd.to_numeric(df.get("line_ha"), errors="coerce")
    ha_nii = pd.to_numeric(df.get("line_ha_nii"), errors="coerce")
    hb = pd.to_numeric(df.get("line_hb"), errors="coerce")
    mask = ((ha > 0) | (ha_nii > 0)) & (hb > 0)
    return mask.fillna(False).to_numpy(dtype=bool)


def _load_catalog():
    if not DJA_CATALOG_FILE.exists():
        return None, f"Missing merged DJA catalog: {DJA_CATALOG_FILE}"

    usecols = [
        "file",
        "srcid",
        "grating",
        "filter",
        "dataset",
        "root",
        "objid",
        "grade",
        "valid",
        "sn_line",
        "z_best",
        "z_grating",
        "z_prism",
        "zline",
        "phot_mass",
        "mass",
        "log_mass",
        "log_Mstar",
        "line_oiii_5007",
        "line_oiii_5007_err",
        "line_hb",
        "line_hb_err",
        "line_ha",
        "line_ha_err",
        "line_ha_nii",
        "line_ha_nii_err",
    ]
    table = pd.read_csv(DJA_CATALOG_FILE, compression="infer", usecols=lambda c: c in usecols, low_memory=False)
    if len(table) == 0:
        return None, "Merged DJA catalog is empty."

    z_series = None
    for col in ["z_best", "z_grating", "z_prism", "zline"]:
        if col in table.columns:
            series = pd.to_numeric(table[col], errors="coerce")
            if series.notna().any():
                z_series = series
                break
    mass_series = None
    for col in ["phot_mass", "mass", "log_mass", "log_Mstar"]:
        if col in table.columns:
            series = _coerce_log_mass(table[col])
            if series.notna().any():
                mass_series = series
                break
    if z_series is None or mass_series is None:
        return None, "Merged DJA catalog lacks usable redshift or mass columns for the sigma pilot."

    df = pd.DataFrame(
        {
            "file": table["file"].astype(str),
            "srcid": pd.to_numeric(table.get("srcid"), errors="coerce"),
            "grating": table["grating"].astype(str),
            "filter": table["filter"].astype(str),
            "dataset": table["dataset"].astype(str),
            "root": table["root"].astype(str),
            "objid": pd.to_numeric(table.get("objid"), errors="coerce"),
            "grade": pd.to_numeric(table.get("grade"), errors="coerce"),
            "valid": pd.to_numeric(table.get("valid"), errors="coerce"),
            "sn_line": pd.to_numeric(table.get("sn_line"), errors="coerce"),
            "z": z_series,
            "log_Mstar": mass_series,
            "line_ha": pd.to_numeric(table.get("line_ha"), errors="coerce"),
            "line_hb": pd.to_numeric(table.get("line_hb"), errors="coerce"),
            "line_ha_nii": pd.to_numeric(table.get("line_ha_nii"), errors="coerce"),
        }
    )

    for line in LINE_DEFS:
        flux = pd.to_numeric(table.get(line["flux_col"]), errors="coerce")
        err = pd.to_numeric(table.get(line["err_col"]), errors="coerce")
        df[f"snr_{line['name']}"] = flux / err.replace(0, np.nan)

    if df["valid"].notna().any():
        df = df[df["valid"] > 0].copy()
    if df["grade"].notna().any():
        df = df[df["grade"] >= MIN_GRADE].copy()
    df = df[df["grating"].str.upper() != "PRISM"].copy()
    df = df[df["z"].between(MIN_TARGET_Z, MAX_TARGET_Z) & df["log_Mstar"].between(MIN_TARGET_LOG_MSTAR, 13.0)].copy()
    if len(df) == 0:
        return None, "No medium/high-resolution DJA spectra remained after the selection cuts."

    df["has_balmer"] = _balmer_available_mask(df)
    if REQUIRE_BALMER:
        df = df[df["has_balmer"]].copy()
    if len(df) == 0:
        return None, "No DJA spectra remained after the Balmer-availability requirement."

    snr_cols = [f"snr_{line['name']}" for line in LINE_DEFS]
    snr_frame = df[snr_cols].apply(pd.to_numeric, errors="coerce")
    df["best_line_snr"] = snr_frame.max(axis=1, skipna=True)
    safe_idx = snr_frame.fillna(-np.inf).idxmax(axis=1)
    df["best_line"] = safe_idx.where(df["best_line_snr"].notna()).str.replace("snr_", "", regex=False)
    df = df[df["best_line_snr"] >= MIN_LINE_SNR].copy()
    if len(df) == 0:
        return None, "No DJA spectra passed the line-S/N threshold for the sigma pilot."

    if df["objid"].notna().any():
        df["resolution_rank"] = df["grating"].map(_resolution_rank)
        df = df.sort_values(
            ["objid", "resolution_rank", "best_line_snr", "sn_line"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        df = df.drop_duplicates(subset="objid", keep="first")

    candidate_pool_n = int(len(df))
    candidate_pool_balmer_n = int(df["has_balmer"].sum()) if "has_balmer" in df.columns else None
    df = df.sort_values(["best_line_snr", "sn_line"], ascending=[False, False], na_position="last").head(MAX_TARGETS).copy()
    df["gamma_t_mass_proxy"] = compute_gamma_t(
        stellar_to_halo_mass_behroozi_like(df["log_Mstar"].to_numpy(dtype=float), df["z"].to_numpy(dtype=float)),
        df["z"].to_numpy(dtype=float),
    )
    df["public_url"] = df.apply(_public_url, axis=1)
    df["local_path"] = df.apply(lambda row: str(_local_path(row)), axis=1)
    df = df.reset_index(drop=True)
    df.attrs["candidate_pool_n"] = candidate_pool_n
    df.attrs["candidate_pool_balmer_n"] = candidate_pool_balmer_n
    return df, None


def _ensure_local_spectrum(row):
    path = _local_path(row)
    if path.exists() and path.stat().st_size > 10000:
        return path, "local_existing"
    if not AUTO_DOWNLOAD:
        return None, "missing_public_spectrum_download_disabled"
    ok = smart_download(
        url=_public_url(row),
        dest=path,
        min_size_mb=0.01,
        logger=logger,
        n_workers=4,
        timeout=180,
    )
    if ok:
        return path, "downloaded_public_specfits"
    return None, "download_failed"


def _load_spectrum(path):
    with fits.open(path, memmap=False) as hdul:
        data = None
        names = None
        grating = None
        for hdu in hdul[1:]:
            cols = getattr(getattr(hdu, "columns", None), "names", None)
            if not cols:
                continue
            lower = {col.lower(): col for col in cols}
            if hdu.name == "SLITS" and "grating" in lower and len(hdu.data):
                grating = str(hdu.data[0][lower["grating"]]).strip().upper()
            if data is None and "wave" in lower and "flux" in lower and ("full_err" in lower or "err" in lower):
                data = hdu.data
                names = lower
        if data is None:
            raise ValueError(f"No SPEC1D-like binary table found in {path}")

    wave = _to_native(data[names["wave"]]).astype(float)
    if np.nanmedian(wave[np.isfinite(wave)]) > 100.0:
        wave = wave / 1e4
    flux = _to_native(data[names["flux"]]).astype(float)
    err_key = names.get("full_err", names.get("err"))
    err = _to_native(data[err_key]).astype(float)
    valid = np.ones_like(wave, dtype=bool)
    if "valid" in names:
        valid &= _to_native(data[names["valid"]]).astype(bool)
    R = np.full_like(wave, np.nan, dtype=float)
    if "r" in names:
        R = _to_native(data[names["r"]]).astype(float)
    resolution_source = "per_pixel_r"
    if not np.isfinite(R).any():
        if grating in FALLBACK_R_BY_GRATING:
            R = np.full_like(wave, FALLBACK_R_BY_GRATING[grating], dtype=float)
            resolution_source = f"grating_fallback_{grating}"
        else:
            resolution_source = "missing"

    mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0) & valid
    return {
        "wave_um": wave[mask],
        "flux": flux[mask],
        "err": err[mask],
        "R": R[mask],
        "resolution_source": resolution_source,
        "grating": grating,
    }


def _fit_line(spectrum, z, line_def):
    center0 = line_def["rest_wave_a"] * (1.0 + z) / 1e4
    wave = spectrum["wave_um"]
    flux = spectrum["flux"]
    err = spectrum["err"]
    resolution = spectrum["R"]
    mask = np.abs(wave - center0) <= line_def["window_um"]
    if mask.sum() < MIN_WINDOW_POINTS:
        return {"fit_status": "insufficient_window_points", "line": line_def["name"]}

    wave = wave[mask]
    flux = flux[mask]
    err = err[mask]
    resolution = resolution[mask]
    valid_R = resolution[np.isfinite(resolution) & (resolution > 0)]
    if len(valid_R) == 0:
        return {"fit_status": "missing_resolution", "line": line_def["name"]}
    median_R = float(np.nanmedian(valid_R))
    sigma_inst_kms = float(C_KMS / (2.354820045 * median_R))
    continuum0 = float(np.nanmedian(flux))
    amplitude0 = float(np.nanmax(flux) - continuum0)
    if not np.isfinite(amplitude0) or amplitude0 <= 0:
        amplitude0 = max(float(np.nanpercentile(flux, 95) - continuum0), 0.0)
    center_bounds = center0 * np.array([1.0 - 800.0 / C_KMS, 1.0 + 800.0 / C_KMS])

    def _model(params):
        c0, c1, amp, center, sigma_kms = params
        sigma_lambda_inst = center / (median_R * 2.354820045)
        sigma_lambda_int = center * max(sigma_kms, 1e-6) / C_KMS
        sigma_lambda_tot = np.sqrt(sigma_lambda_inst**2 + sigma_lambda_int**2)
        profile = np.exp(-0.5 * ((wave - center) / sigma_lambda_tot) ** 2)
        return c0 + c1 * (wave - center0) + amp * profile

    def _resid(params):
        return (_model(params) - flux) / err

    x0 = np.array([continuum0, 0.0, max(amplitude0, 0.0), center0, 80.0], dtype=float)
    bounds = (
        np.array([-np.inf, -np.inf, 0.0, center_bounds[0], 5.0], dtype=float),
        np.array([np.inf, np.inf, np.inf, center_bounds[1], 800.0], dtype=float),
    )

    fit = least_squares(_resid, x0=x0, bounds=bounds, method="trf")
    if not fit.success:
        return {"fit_status": "fit_failed", "line": line_def["name"]}

    params = fit.x.astype(float)
    dof = max(len(wave) - len(params), 1)
    chi2 = float(np.sum(np.square(_resid(params))))
    reduced_chi2 = chi2 / dof
    sigma_err = np.nan
    amp_err = np.nan
    try:
        jt_j = fit.jac.T @ fit.jac
        cov = np.linalg.pinv(jt_j) * reduced_chi2
        sigma_err = float(np.sqrt(max(cov[4, 4], 0.0)))
        amp_err = float(np.sqrt(max(cov[2, 2], 0.0)))
    except Exception:
        pass

    amp_snr = None
    if np.isfinite(amp_err) and amp_err > 0:
        amp_snr = float(params[2] / amp_err)

    fit_status = "success"
    if not np.isfinite(sigma_err):
        fit_status = "success_no_covariance"
    return {
        "fit_status": fit_status,
        "line": line_def["name"],
        "sigma_kms": float(params[4]),
        "sigma_kms_err": None if not np.isfinite(sigma_err) else sigma_err,
        "amplitude_snr": amp_snr,
        "median_R": median_R,
        "sigma_inst_kms": sigma_inst_kms,
        "resolution_source": spectrum.get("resolution_source"),
        "line_center_um": float(params[3]),
        "reduced_chi2": reduced_chi2,
        "n_window_points": int(len(wave)),
    }


def _residualize(values, controls):
    x = np.column_stack([np.ones(len(values))] + [np.asarray(c, dtype=float) for c in controls])
    beta = np.linalg.lstsq(x, np.asarray(values, dtype=float), rcond=None)[0]
    return np.asarray(values, dtype=float) - x @ beta


def _partial_spearman(x, y, controls):
    arrays = [np.asarray(x, dtype=float), np.asarray(y, dtype=float)] + [np.asarray(c, dtype=float) for c in controls]
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    if int(mask.sum()) < MIN_SUCCESS_FOR_CORRELATION:
        return {"rho": None, "p": None, "n": int(mask.sum())}
    xr = stats.rankdata(arrays[0][mask])
    yr = stats.rankdata(arrays[1][mask])
    control_ranks = [stats.rankdata(arr[mask]) for arr in arrays[2:]]
    x_resid = _residualize(xr, control_ranks)
    y_resid = _residualize(yr, control_ranks)
    rho = float(np.corrcoef(x_resid, y_resid)[0, 1])
    dof = int(mask.sum()) - len(control_ranks) - 2
    if not np.isfinite(rho) or dof <= 0 or abs(rho) >= 1:
        return {"rho": None if not np.isfinite(rho) else rho, "p": None, "n": int(mask.sum())}
    t_stat = rho * np.sqrt(dof / max(1e-12, 1.0 - rho**2))
    p = 2.0 * stats.t.sf(abs(t_stat), dof)
    return {"rho": rho, "p": _clip_p(p), "n": int(mask.sum())}


def _assessment(fetch_stats, fit_df, balmer_test):
    if fit_df.empty and fetch_stats["n_missing_with_download_disabled"] > 0:
        return "The DJA sigma pilot code path is now wired through individual public .spec.fits URLs, but no local spectra were available and public downloading was disabled for this run."
    if fit_df.empty:
        return "The DJA sigma pilot found candidate spectra, but no stable instrumental-resolution-aware line-width fits were recovered in this run."
    if balmer_test["n"] < MIN_SUCCESS_FOR_CORRELATION:
        return "The DJA sigma pilot recovered line widths for a non-zero subset, but the Balmer-comparison branch remains underpowered after requiring both a stable sigma fit and a Balmer dust estimate."
    rho_sigma = balmer_test["partial_sigma_given_mass_z"].get("rho")
    rho_mass = balmer_test["partial_mass_given_sigma_z"].get("rho")
    if rho_sigma is not None and rho_mass is not None and abs(rho_sigma) > abs(rho_mass):
        return "Within the fitted DJA pilot subset, the Balmer dust proxy tracks the fitted emission-line width at least as strongly as photometric stellar mass after mass-plus-redshift control, which is the intended direction for the sigma-based decoupling audit."
    return "The DJA sigma pilot recovered a fitted subset and a direct dust-versus-sigma comparison, but the current pilot sample remains mixed rather than decisively sigma-dominated."


def _coerce_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _quality_screen_mask(fit_df):
    if len(fit_df) == 0:
        return np.zeros(0, dtype=bool)
    sigma = pd.to_numeric(fit_df.get("sigma_kms"), errors="coerce")
    sigma_err = pd.to_numeric(fit_df.get("sigma_kms_err"), errors="coerce")
    amp_snr = pd.to_numeric(fit_df.get("amplitude_snr"), errors="coerce")
    reduced_chi2 = pd.to_numeric(fit_df.get("reduced_chi2"), errors="coerce")
    frac_err = sigma_err / sigma.replace(0, np.nan)
    mask = (
        np.isfinite(sigma)
        & (sigma > MIN_SIGMA_KMS_QUALITY)
        & np.isfinite(sigma_err)
        & np.isfinite(frac_err)
        & (frac_err <= MAX_SIGMA_FRAC_ERR_QUALITY)
        & np.isfinite(amp_snr)
        & (amp_snr >= MIN_AMP_SNR_QUALITY)
        & np.isfinite(reduced_chi2)
        & (reduced_chi2 <= MAX_REDUCED_CHI2_QUALITY)
    )
    return np.asarray(mask, dtype=bool)


def _run_balmer_sigma_test(fit_df):
    balmer_df = fit_df[np.isfinite(pd.to_numeric(fit_df.get("log_balmer"), errors="coerce"))].copy() if len(fit_df) else pd.DataFrame()
    raw_sigma = {"rho": None, "p": None, "n": int(len(balmer_df))}
    raw_mass = {"rho": None, "p": None, "n": int(len(balmer_df))}
    partial_sigma = {"rho": None, "p": None, "n": int(len(balmer_df))}
    partial_mass = {"rho": None, "p": None, "n": int(len(balmer_df))}
    if len(balmer_df) >= MIN_SUCCESS_FOR_CORRELATION:
        rho_sigma, p_sigma = spearmanr(balmer_df["log_balmer"], balmer_df["sigma_kms"])
        rho_mass, p_mass = spearmanr(balmer_df["log_balmer"], balmer_df["log_Mstar"])
        raw_sigma = {"rho": float(rho_sigma), "p": _clip_p(p_sigma), "n": int(len(balmer_df))}
        raw_mass = {"rho": float(rho_mass), "p": _clip_p(p_mass), "n": int(len(balmer_df))}
        partial_sigma = _partial_spearman(
            balmer_df["log_balmer"].to_numpy(dtype=float),
            balmer_df["sigma_kms"].to_numpy(dtype=float),
            [balmer_df["log_Mstar"].to_numpy(dtype=float), balmer_df["z"].to_numpy(dtype=float)],
        )
        partial_mass = _partial_spearman(
            balmer_df["log_balmer"].to_numpy(dtype=float),
            balmer_df["log_Mstar"].to_numpy(dtype=float),
            [balmer_df["sigma_kms"].to_numpy(dtype=float), balmer_df["z"].to_numpy(dtype=float)],
        )
    return {
        "n": int(len(balmer_df)),
        "raw_sigma_vs_balmer": raw_sigma,
        "raw_mass_vs_balmer": raw_mass,
        "partial_sigma_given_mass_z": partial_sigma,
        "partial_mass_given_sigma_z": partial_mass,
    }, balmer_df


def _fit_summary_payload(fit_df):
    return {
        "n_success": int(len(fit_df)),
        "median_sigma_kms": float(np.nanmedian(fit_df["sigma_kms"])) if len(fit_df) else None,
        "median_sigma_kms_err": float(np.nanmedian(pd.to_numeric(fit_df.get("sigma_kms_err"), errors="coerce"))) if len(fit_df) and "sigma_kms_err" in fit_df.columns else None,
        "median_reduced_chi2": float(np.nanmedian(pd.to_numeric(fit_df.get("reduced_chi2"), errors="coerce"))) if len(fit_df) and "reduced_chi2" in fit_df.columns else None,
        "median_amplitude_snr": float(np.nanmedian(pd.to_numeric(fit_df.get("amplitude_snr"), errors="coerce"))) if len(fit_df) and "amplitude_snr" in fit_df.columns else None,
        "line_counts": {str(k): int(v) for k, v in fit_df["line"].value_counts().items()} if len(fit_df) and "line" in fit_df.columns else {},
        "grating_counts": {str(k): int(v) for k, v in fit_df["grating"].value_counts().items()} if len(fit_df) and "grating" in fit_df.columns else {},
        "resolution_source_counts": {str(k): int(v) for k, v in fit_df["resolution_source"].value_counts().items()} if len(fit_df) and "resolution_source" in fit_df.columns else {},
    }


def _pilot_target_records(result_df):
    desired = [
        "objid",
        "srcid",
        "z",
        "log_Mstar",
        "grating",
        "file",
        "root",
        "best_line",
        "best_line_snr",
        "spectrum_fetch_status",
        "fit_status",
        "line",
        "sigma_kms",
        "sigma_kms_err",
        "amplitude_snr",
        "median_R",
        "resolution_source",
        "reduced_chi2",
        "n_window_points",
        "log_balmer",
        "public_url",
        "local_path",
    ]
    available = [col for col in desired if col in result_df.columns]
    if not available:
        return []
    return result_df[available].replace({np.nan: None}).to_dict(orient="records")


def run():
    print_status(f"STEP {STEP_NUM}: DJA sigma pilot", "INFO")
    df, error = _load_catalog()
    if df is None:
        result = {
            "step": STEP_NUM,
            "name": STEP_NAME,
            "status": "FAILED_NO_CANDIDATES",
            "note": error,
        }
        out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        out_json.write_text(json.dumps(result, indent=2))
        return result

    rows = []
    fetch_stats = {
        "download_enabled": AUTO_DOWNLOAD,
        "n_candidates": int(len(df)),
        "n_local_existing": 0,
        "n_downloaded": 0,
        "n_missing_with_download_disabled": 0,
        "n_download_failed": 0,
    }

    for _, row in df.iterrows():
        payload = row.to_dict()
        path, fetch_status = _ensure_local_spectrum(row)
        payload["spectrum_fetch_status"] = fetch_status
        if fetch_status == "local_existing":
            fetch_stats["n_local_existing"] += 1
        elif fetch_status == "downloaded_public_specfits":
            fetch_stats["n_downloaded"] += 1
        elif fetch_status == "missing_public_spectrum_download_disabled":
            fetch_stats["n_missing_with_download_disabled"] += 1
        elif fetch_status == "download_failed":
            fetch_stats["n_download_failed"] += 1

        balmer_flux = payload.get("line_ha")
        if not np.isfinite(balmer_flux) or balmer_flux <= 0:
            balmer_flux = payload.get("line_ha_nii")
        hb_flux = payload.get("line_hb")
        if np.isfinite(balmer_flux) and balmer_flux > 0 and np.isfinite(hb_flux) and hb_flux > 0:
            payload["log_balmer"] = float(np.log10(balmer_flux / hb_flux))
        else:
            payload["log_balmer"] = None

        if path is None:
            payload["fit_status"] = "spectrum_unavailable"
            rows.append(payload)
            continue

        try:
            spectrum = _load_spectrum(path)
        except Exception as exc:
            payload["fit_status"] = f"spectrum_read_failed: {exc}"
            rows.append(payload)
            continue

        line_order = sorted(
            LINE_DEFS,
            key=lambda item: payload.get(f"snr_{item['name']}") if np.isfinite(payload.get(f"snr_{item['name']}", np.nan)) else -np.inf,
            reverse=True,
        )
        best_result = None
        for line in line_order:
            snr_value = payload.get(f"snr_{line['name']}")
            if not np.isfinite(snr_value) or snr_value < MIN_LINE_SNR:
                continue
            fit_result = _fit_line(spectrum, payload["z"], line)
            if str(fit_result.get("fit_status", "")).startswith("success"):
                best_result = fit_result
                break
            if best_result is None:
                best_result = fit_result

        if best_result is not None:
            payload.update(best_result)
        else:
            payload["fit_status"] = "no_line_with_sufficient_catalog_snr"
        rows.append(payload)

    result_df = pd.DataFrame(rows)
    if len(result_df) == 0:
        result_df = df.copy()

    fit_mask = result_df["fit_status"].astype(str).str.startswith("success")
    if "sigma_kms" in result_df.columns:
        fit_mask &= pd.to_numeric(result_df["sigma_kms"], errors="coerce").notna()
    fit_df = result_df[fit_mask].copy()
    if len(fit_df) > 0:
        fit_df = _coerce_numeric(
            fit_df,
            ["sigma_kms", "sigma_kms_err", "amplitude_snr", "reduced_chi2", "log_Mstar", "z", "log_balmer", "median_R", "n_window_points"],
        )
        fit_df["sigma_frac_err"] = fit_df["sigma_kms_err"] / fit_df["sigma_kms"].replace(0, np.nan)
        fit_df["quality_screen_pass"] = _quality_screen_mask(fit_df)
        fit_df["quality_screen_reason"] = np.where(fit_df["quality_screen_pass"], "pass", "screened_out")
        result_df = result_df.copy()
        result_df["quality_screen_pass"] = False
        result_df["quality_screen_reason"] = None
        result_df.loc[fit_df.index, "quality_screen_pass"] = fit_df["quality_screen_pass"].to_numpy(dtype=bool)
        result_df.loc[fit_df.index, "quality_screen_reason"] = fit_df["quality_screen_reason"].astype(object).to_numpy()
        result_df.loc[fit_df.index, "sigma_frac_err"] = fit_df["sigma_frac_err"].to_numpy(dtype=float)
    else:
        result_df = result_df.copy()
        result_df["quality_screen_pass"] = False
        result_df["quality_screen_reason"] = None

    quality_df = fit_df[fit_df["quality_screen_pass"]].copy() if len(fit_df) else pd.DataFrame()
    all_fit_test, balmer_df = _run_balmer_sigma_test(fit_df)
    quality_fit_test, quality_balmer_df = _run_balmer_sigma_test(quality_df)

    out_csv = INTERIM_PATH / f"step_{STEP_NUM}_{STEP_NAME}.csv"
    result_df.to_csv(out_csv, index=False)

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "description": "DJA pilot emission-line width extraction from individual public spec.fits files with a Balmer-vs-sigma decoupling audit on the fitted subset",
        "input_catalog": str(DJA_CATALOG_FILE),
        "config": {
            "download_enabled": AUTO_DOWNLOAD,
            "selection_mode": SELECTION_MODE,
            "max_targets": MAX_TARGETS,
            "min_target_z": MIN_TARGET_Z,
            "max_target_z": MAX_TARGET_Z,
            "min_target_log_Mstar": MIN_TARGET_LOG_MSTAR,
            "require_balmer": REQUIRE_BALMER,
            "min_line_snr": MIN_LINE_SNR,
            "min_grade": MIN_GRADE,
            "quality_min_sigma_kms": MIN_SIGMA_KMS_QUALITY,
            "quality_max_sigma_frac_err": MAX_SIGMA_FRAC_ERR_QUALITY,
            "quality_max_reduced_chi2": MAX_REDUCED_CHI2_QUALITY,
            "quality_min_amplitude_snr": MIN_AMP_SNR_QUALITY,
            "download_opt_in_env": "TEP_DJA_SIGMA_AUTODOWNLOAD=1",
            "pilot_target_cap_env": "TEP_DJA_SIGMA_MAX_TARGETS",
        },
        "selection": {
            "candidate_pool_n": int(df.attrs.get("candidate_pool_n", len(df))),
            "candidate_pool_n_with_balmer": int(df.attrs.get("candidate_pool_balmer_n", int(df["has_balmer"].sum()) if "has_balmer" in df.columns else 0)),
            "n_targeted": int(len(df)),
            "n_targeted_with_balmer": int(df["has_balmer"].sum()) if "has_balmer" in df.columns else 0,
            "grating_counts": {str(k): int(v) for k, v in df["grating"].value_counts().items()},
            "best_line_counts": {str(k): int(v) for k, v in df["best_line"].value_counts().items()},
            "median_best_line_snr": float(np.nanmedian(df["best_line_snr"])) if len(df) else None,
            "median_z": float(np.nanmedian(df["z"])) if len(df) else None,
            "median_log_Mstar": float(np.nanmedian(df["log_Mstar"])) if len(df) else None,
        },
        "spectrum_fetch": fetch_stats,
        "fit_summary": {
            "all_success": _fit_summary_payload(fit_df),
            "quality_screened": _fit_summary_payload(quality_df),
            "quality_screen": {
                "n_pass": int(len(quality_df)),
                "n_rejected": int(len(fit_df) - len(quality_df)),
                "pass_fraction": float(len(quality_df) / len(fit_df)) if len(fit_df) else None,
            },
        },
        "pilot_balmer_sigma_test": {
            "all_success": all_fit_test,
            "quality_screened": quality_fit_test,
        },
        "interim_table": str(out_csv),
        "pilot_targets": _pilot_target_records(result_df),
    }
    result["assessment"] = _assessment(fetch_stats, quality_df, quality_fit_test)

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, "w") as handle:
        json.dump(result, handle, indent=2, default=safe_json_default)
    print_status(f"Step {STEP_NUM} complete ({result['status']})", "INFO")
    return result


if __name__ == "__main__":
    run()
