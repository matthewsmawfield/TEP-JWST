import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import safe_json_default
from scripts.utils.tep_model import ALPHA_0, compute_gamma_t as tep_gamma

STEP_NUM = "166"
STEP_NAME = "jades_z9_beta_contrast"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"
for path in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

BOOTSTRAP_N = 4000
RNG_SEED = 166


def _clip_p(value: float) -> float:
    return max(float(value), 1e-300)


def _to_native(arr):
    arr = np.array(arr)
    if hasattr(arr.dtype, "byteorder") and arr.dtype.byteorder == ">":
        return arr.astype(arr.dtype.newbyteorder("="))
    return arr


def load_jades_highz_photometry() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run step_014_jwst_uv_slope.py or place the local JADES catalog first.")

    with fits.open(DATA_PATH) as hdu:
        kron = hdu["KRON"].data
        photoz = hdu["PHOTOZ"].data

    df_kron = pd.DataFrame({
        "id": _to_native(kron["ID"]),
        "ra": _to_native(kron["RA"]),
        "dec": _to_native(kron["DEC"]),
        "F115W": _to_native(kron["F115W_KRON"]),
        "F115W_err": _to_native(kron["F115W_KRON_S"]),
        "F150W": _to_native(kron["F150W_KRON"]),
        "F150W_err": _to_native(kron["F150W_KRON_S"]),
        "F200W": _to_native(kron["F200W_KRON"]),
        "F200W_err": _to_native(kron["F200W_KRON_S"]),
        "F277W": _to_native(kron["F277W_KRON"]),
        "F277W_err": _to_native(kron["F277W_KRON_S"]),
        "F356W": _to_native(kron["F356W_KRON"]),
        "F356W_err": _to_native(kron["F356W_KRON_S"]),
        "F444W": _to_native(kron["F444W_KRON"]),
        "F444W_err": _to_native(kron["F444W_KRON_S"]),
    })
    df_pz = pd.DataFrame({
        "id": _to_native(photoz["ID"]),
        "z_phot": _to_native(photoz["EAZY_z_a"]),
        "z_l95": _to_native(photoz["EAZY_l95"]),
        "z_u95": _to_native(photoz["EAZY_u95"]),
    })
    df = df_kron.merge(df_pz, on="id", how="inner")
    df["z_best"] = df["z_phot"]
    df["P_z_gt_7"] = (df["z_l95"] > 7).astype(float)
    df = df[(df["z_best"] > 8) & (df["P_z_gt_7"] > 0.5)].copy()

    d_l_pc = np.array([Planck18.luminosity_distance(z).to("pc").value for z in df["z_best"]])
    f_njy = df["F150W"].astype(float).to_numpy()
    f_njy = np.where(f_njy > 0, f_njy, np.nan)
    m_ab = -2.5 * np.log10(f_njy * 1e-9 / 3631)
    dm = 5.0 * np.log10(d_l_pc / 10.0)
    k_corr = 2.5 * np.log10(1.0 + df["z_best"].to_numpy())
    df["MUV"] = m_ab - dm + k_corr
    return df


def calculate_uv_slope(df: pd.DataFrame) -> pd.DataFrame:
    wavelengths = {
        "F115W": 1.154,
        "F150W": 1.501,
        "F200W": 1.989,
        "F277W": 2.762,
        "F356W": 3.568,
        "F444W": 4.421,
    }
    beta_values = []
    beta_errors = []

    for _, row in df.iterrows():
        z = row["z_best"]
        uv_bands = []
        for band, obs_wave in wavelengths.items():
            rest_wave = obs_wave / (1.0 + z)
            if 0.15 < rest_wave < 0.30:
                flux = row[band]
                flux_err = row[f"{band}_err"]
                if flux > 0 and flux_err > 0 and flux / flux_err > 2:
                    uv_bands.append((band, obs_wave, flux, flux_err))
        if len(uv_bands) >= 2:
            _, wave1, flux1, err1 = uv_bands[0]
            _, wave2, flux2, err2 = uv_bands[-1]
            log_flux_ratio = np.log10(flux2 / flux1)
            log_wave_ratio = np.log10(wave2 / wave1)
            beta = log_flux_ratio / log_wave_ratio - 2.0
            rel_err1 = err1 / flux1
            rel_err2 = err2 / flux2
            log_flux_err = np.sqrt(rel_err1**2 + rel_err2**2) / np.log(10)
            beta_err = log_flux_err / abs(log_wave_ratio)
            beta_values.append(beta)
            beta_errors.append(beta_err)
        else:
            beta_values.append(np.nan)
            beta_errors.append(np.nan)

    out = df.copy()
    out["beta"] = beta_values
    out["beta_err"] = beta_errors
    return out


def calculate_gamma_t(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_Mhalo"] = out["MUV"] / (-2.5) + 6.0
    out["log_Mhalo"] = np.clip(out["log_Mhalo"], 10.0, 14.0)
    out["gamma_t"] = tep_gamma(out["log_Mhalo"].to_numpy(), out["z_best"].to_numpy(), alpha_0=ALPHA_0)
    return out


def _weighted_mean(values: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-3, None)
    weights = 1.0 / np.square(sigma)
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def _bootstrap_delta(low_values: np.ndarray, low_sigma: np.ndarray, high_values: np.ndarray, high_sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draws = np.empty(BOOTSTRAP_N, dtype=float)
    for i in range(BOOTSTRAP_N):
        idx_low = rng.integers(0, len(low_values), size=len(low_values))
        idx_high = rng.integers(0, len(high_values), size=len(high_values))
        draws[i] = _weighted_mean(high_values[idx_high], high_sigma[idx_high]) - _weighted_mean(low_values[idx_low], low_sigma[idx_low])
    return draws


def _contrast(df: pd.DataFrame, low_q: float, high_q: float) -> dict | None:
    low_thr = float(df["gamma_t"].quantile(low_q))
    high_thr = float(df["gamma_t"].quantile(high_q))
    low = df[df["gamma_t"] <= low_thr].copy()
    high = df[df["gamma_t"] >= high_thr].copy()
    if len(low) < 6 or len(high) < 6:
        return None
    rng = np.random.default_rng(RNG_SEED + int(low_q * 100) + int(high_q * 100))
    boot = _bootstrap_delta(
        low["beta"].to_numpy(dtype=float),
        low["beta_err"].to_numpy(dtype=float),
        high["beta"].to_numpy(dtype=float),
        high["beta_err"].to_numpy(dtype=float),
        rng,
    )
    return {
        "gamma_t_quantiles": [float(low_q), float(high_q)],
        "gamma_t_threshold_low": low_thr,
        "gamma_t_threshold_high": high_thr,
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "median_gamma_t_low": float(np.nanmedian(low["gamma_t"])),
        "median_gamma_t_high": float(np.nanmedian(high["gamma_t"])),
        "median_beta_low": float(np.nanmedian(low["beta"])),
        "median_beta_high": float(np.nanmedian(high["beta"])),
        "weighted_beta_low": float(_weighted_mean(low["beta"].to_numpy(dtype=float), low["beta_err"].to_numpy(dtype=float))),
        "weighted_beta_high": float(_weighted_mean(high["beta"].to_numpy(dtype=float), high["beta_err"].to_numpy(dtype=float))),
        "delta_weighted_beta_high_minus_low": float(_weighted_mean(high["beta"].to_numpy(dtype=float), high["beta_err"].to_numpy(dtype=float)) - _weighted_mean(low["beta"].to_numpy(dtype=float), low["beta_err"].to_numpy(dtype=float))),
        "bootstrap_ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_supportive_p_one_sided": _clip_p((float(np.sum(boot <= 0.0)) + 1.0) / (len(boot) + 1.0)),
        "mean_MUV_high_minus_low": float(high["MUV"].mean() - low["MUV"].mean()),
    }


def _assessment(primary: dict | None, rho: float, p: float) -> str:
    if not primary:
        return "The JADES z=9-12 beta-contrast step was underpowered after conservative quality cuts."
    delta = primary["delta_weighted_beta_high_minus_low"]
    ci_lo = primary["bootstrap_ci_95"][0]
    if delta > 0 and ci_lo > 0:
        return "The JADES z=9-12 beta sample supports the same direction as the UNCOVER stack: high-Gamma_t objects have redder mean UV slopes."
    if delta > 0 and rho > 0:
        return "The JADES z=9-12 beta sample is directionally supportive but remains a lower-power photometric companion rather than a decisive standalone test."
    return "The JADES z=9-12 beta sample does not provide a clean positive contrast after conservative cuts."


def run():
    print_status(f"STEP {STEP_NUM}: JADES z=9-12 beta contrast", "INFO")
    try:
        df = load_jades_highz_photometry()
    except FileNotFoundError as exc:
        result = {
            "step": STEP_NUM,
            "name": STEP_NAME,
            "status": "FAILED_NO_DATA",
            "note": str(exc),
        }
        (OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json").write_text(json.dumps(result, indent=2))
        return result

    df = calculate_uv_slope(df)
    df = calculate_gamma_t(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    quality = df.dropna(subset=["beta", "beta_err", "gamma_t", "z_best", "MUV"]).copy()
    quality = quality[(quality["z_best"] >= 9.0) & (quality["z_best"] < 12.0)].copy()
    quality = quality[(quality["beta_err"] <= 2.5) & quality["MUV"].between(-22.0, -16.0)].copy()
    quality["redder_beta"] = quality["beta"]

    rho, p = spearmanr(quality["gamma_t"], quality["beta"]) if len(quality) >= 6 else (np.nan, np.nan)
    contrast_q33 = _contrast(quality, 1.0 / 3.0, 2.0 / 3.0) if len(quality) >= 12 else None
    contrast_q25 = _contrast(quality, 0.25, 0.75) if len(quality) >= 12 else None
    primary = contrast_q25 if contrast_q25 is not None else contrast_q33

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "description": "JADES photometric z=9-12 UV-slope companion analysis split by predicted Gamma_t",
        "input_catalog": str(DATA_PATH),
        "selection": {
            "base_highz_count": int(len(df)),
            "n_with_beta": int(df["beta"].notna().sum()),
            "z_window": [9.0, 12.0],
            "beta_err_max": 2.5,
            "MUV_window": [-22.0, -16.0],
            "n_final": int(len(quality)),
            "median_z": float(np.nanmedian(quality["z_best"])) if len(quality) else None,
        },
        "raw_correlation": {
            "rho_gamma_t_vs_beta": None if np.isnan(rho) else float(rho),
            "p": None if np.isnan(p) else _clip_p(p),
        },
        "contrasts": {
            "q33_q67": contrast_q33,
            "q25_q75": contrast_q25,
        },
        "primary_contrast": primary,
    }
    result["assessment"] = _assessment(primary, float(rho) if not np.isnan(rho) else np.nan, float(p) if not np.isnan(p) else np.nan)

    out_csv = INTERIM_PATH / f"step_{STEP_NUM}_{STEP_NAME}.csv"
    quality.to_csv(out_csv, index=False)
    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    out_json.write_text(json.dumps(result, indent=2, default=safe_json_default))

    if primary is not None:
        print_status(
            f"Primary beta contrast Δ={primary['delta_weighted_beta_high_minus_low']:.3f} with 95% CI [{primary['bootstrap_ci_95'][0]:.3f}, {primary['bootstrap_ci_95'][1]:.3f}]",
            "INFO",
        )
    print_status(result["assessment"], "INFO")
    return result


if __name__ == "__main__":
    run()
