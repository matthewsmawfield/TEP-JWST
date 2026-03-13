#!/usr/bin/env python3
"""
TEP-JWST Step 142: LRD M_BH/M_* Combined Growth Model

Formally models the combined TEP + physically motivated accretion framework
for Little Red Dot (LRD) overmassive black holes.

The galactic centre (BH location) has a deeper potential well than the bulk
stellar disk.  TEP predicts Gamma_t(centre) > Gamma_t(disk), so black holes
accumulate effective time faster than the stellar halo.  This step quantifies
whether the combined model — TEP differential shear plus physically motivated
seed masses and modest super-Eddington duty cycles — fully accounts for the
observed M_BH/M_* ratios at z > 4 without requiring exotic physics.

This step uses the step_132 population-level result as a conservative baseline
and evaluates representative-host growth scenarios:
  S0: Standard (no TEP) — 100 M_sun seed, Eddington-limited
  S1: TEP-only — 100 M_sun seed, Eddington-limited, TEP differential shear
  S2: TEP + intermediate seed (10^3 M_sun) — Eddington-limited
  S3: TEP + mild super-Eddington (f_Edd=1.3, 100 M_sun seed)
  S4: TEP combined (500 M_sun seed, f_Edd=1.1)

Observational anchor: median observed LRD M_BH/M_* ~ 0.01–0.05 (median ~0.03)
from the published compilation regime (Matthee et al. 2024; Greene et al. 2024;
Kokorev et al. 2024; Maiolino et al. 2024; Pacucci et al. 2024).

Outputs:
  - results/outputs/step_142_lrd_mbh_mstar_prediction.json
  - results/interim/step_142_lrd_mbh_mstar_prediction.csv
Author: Matthew L. Smawfield
"""

import json
import sys
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import ALPHA_0, ALPHA_UNCERTAINTY, Z_REF  # Shared TEP model constants

STEP_NUM  = "142"  # Pipeline step number
STEP_NAME = "lrd_mbh_mstar_prediction"  # Used in log / output filenames

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
OUTPUT_PATH  = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH    = PROJECT_ROOT / "logs"  # Log directory
DATA_PATH    = PROJECT_ROOT / "data" / "raw"  # Raw data directory
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"
for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

G = 4.301e-6            # kpc (km/s)^2 / M_sun
T_SALPETER = 0.045      # Gyr — Salpeter e-folding time at Eddington limit
PHI_REF_VIR = 360.0**2  # Reference potential depth (V_vir^2 for M=10^12 at z=5.5)
LOCAL_MBH_MSTAR = 1e-3  # Local M_BH/M_* baseline (Kormendy & Ho 2013)
CONCENTRATION_FACTOR = 10.0  # Phi_cen / Phi_vir for compact LRDs (step_132)
LRD_CATALOG_PATH = DATA_PATH / "kokorev_lrd_catalog_v1.1.fits"
CEERS_HIGHZ_SAMPLE_PATH = DATA_INTERIM_PATH / "ceers_highz_sample.csv"
DIRECT_MASS_MATCH_ARCSEC = 0.1

MUV_REF = -19.5
LOG_MSTAR_REF = 8.7
ML_SLOPE = 0.4

# Observed LRD regime from published compilations
# Matthee et al. 2024; Greene et al. 2024; Kokorev et al. 2024;
# Maiolino et al. 2024; Pacucci et al. 2024
OBSERVED_LOG_MBH_MSTAR_MEDIAN = -1.5   # log10(0.03)
OBSERVED_LOG_MBH_MSTAR_LO     = -2.0   # log10(0.01)
OBSERVED_LOG_MBH_MSTAR_HI     = -1.3   # log10(0.05)

# Growth scenarios
SCENARIOS = {
    "S0_standard": {
        "label": "Standard (no TEP)",
        "M_seed": 1e2,
        "f_Edd": 1.0,
        "use_tep": False,
    },
    "S1_tep_only": {
        "label": "TEP-only (100 M_sun seed, Eddington)",
        "M_seed": 1e2,
        "f_Edd": 1.0,
        "use_tep": True,
    },
    "S2_tep_intermediate_seed": {
        "label": "TEP + intermediate seed (10^3 M_sun)",
        "M_seed": 1e3,
        "f_Edd": 1.0,
        "use_tep": True,
    },
    "S3_tep_mild_super_eddington": {
        "label": "TEP + mild super-Eddington (f=1.3, 100 M_sun)",
        "M_seed": 1e2,
        "f_Edd": 1.3,
        "use_tep": True,
    },
    "S4_tep_combined": {
        "label": "TEP combined (500 M_sun, f=1.1)",
        "M_seed": 5e2,
        "f_Edd": 1.1,
        "use_tep": True,
    },
}


def _maybe_scalar(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr


def muv_to_log_mstar(muv):
    return -ML_SLOPE * (np.asarray(muv) - MUV_REF) + LOG_MSTAR_REF


def load_real_lrd_sample():
    df = Table.read(LRD_CATALOG_PATH).to_pandas()
    col_map = {
        "z_phot": "z",
        "av": "Av",
        "lbol": "log_Lbol",
        "muv": "Muv",
        "r_eff_50_phys": "Re_pc",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df[pd.notna(df["z"]) & (df["z"] > 4) & (df["z"] < 10)].copy()
    df["Re_pc"] = pd.to_numeric(df["Re_pc"], errors="coerce")
    df["Muv"] = pd.to_numeric(df["Muv"], errors="coerce")
    if "log_Mstar" not in df.columns:
        df["log_Mstar"] = np.nan
    df["log_Mstar"] = pd.to_numeric(df["log_Mstar"], errors="coerce")
    inferred_mass = (
        df["log_Mstar"].isna() &
        pd.notna(df["Muv"]) &
        (df["Muv"] < 0) &
        (df["Muv"] > -30)
    )
    df.loc[inferred_mass, "log_Mstar"] = muv_to_log_mstar(df.loc[inferred_mass, "Muv"])
    df["log_Mstar_source"] = np.where(
        inferred_mass,
        "MUV_proxy",
        np.where(pd.notna(df["log_Mstar"]), "catalog", "missing"),
    )
    df["Re_kpc"] = df["Re_pc"] / 1000.0
    valid = (
        pd.notna(df["log_Mstar"]) &
        pd.notna(df["Re_pc"]) &
        (df["Re_pc"] > 0)
    )
    return df.loc[valid].copy()


def build_ceers_direct_mass_subset(df_lrd, match_radius_arcsec=DIRECT_MASS_MATCH_ARCSEC):
    if not CEERS_HIGHZ_SAMPLE_PATH.exists():
        return pd.DataFrame()

    df_ceers = pd.read_csv(CEERS_HIGHZ_SAMPLE_PATH)
    required = {"ra", "dec", "log_Mstar"}
    if not required.issubset(df_ceers.columns):
        return pd.DataFrame()

    mask_lrd = (
        df_lrd["field"].astype(str).eq("ceers-full") &
        pd.notna(df_lrd["ra"]) &
        pd.notna(df_lrd["dec"])
    )
    df_lrd_ceers = df_lrd.loc[mask_lrd].copy()
    if len(df_lrd_ceers) == 0:
        return pd.DataFrame()

    lrd_coords = SkyCoord(df_lrd_ceers["ra"].to_numpy(dtype=float) * u.deg,
                          df_lrd_ceers["dec"].to_numpy(dtype=float) * u.deg)
    ceers_coords = SkyCoord(df_ceers["ra"].to_numpy(dtype=float) * u.deg,
                            df_ceers["dec"].to_numpy(dtype=float) * u.deg)
    idx, sep2d, _ = lrd_coords.match_to_catalog_sky(ceers_coords)
    matched = sep2d.arcsec < match_radius_arcsec
    if not np.any(matched):
        return pd.DataFrame()

    df_matched = df_lrd_ceers.loc[matched].copy()
    df_matched["match_sep_arcsec"] = sep2d.arcsec[matched]
    df_matched["log_Mstar_muv_proxy"] = df_matched["log_Mstar"]
    df_matched["log_Mstar"] = df_ceers.iloc[idx[matched]]["log_Mstar"].to_numpy(dtype=float)
    df_matched["log_Mstar_source"] = "CEERS_direct"
    return df_matched


def fit_empirical_mass_calibration(df_direct):
    features = ["Muv", "Av", "log_Lbol", "z"]
    df_fit = df_direct.dropna(subset=features + ["log_Mstar"]).copy()
    if len(df_fit) < 10:
        return None

    X = df_fit[features].to_numpy(dtype=float)
    y = df_fit["log_Mstar"].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)

    loo = LeaveOneOut()
    preds = np.empty_like(y)
    for train_idx, test_idx in loo.split(X):
        loo_model = LinearRegression().fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = loo_model.predict(X[test_idx])[0]

    loo_mae = float(mean_absolute_error(y, preds))
    loo_corr = float(np.corrcoef(y, preds)[0, 1]) if len(y) > 1 else np.nan

    return {
        "model": model,
        "features": features,
        "training_sample_size": int(len(df_fit)),
        "loo_mae_dex": loo_mae,
        "loo_pred_true_corr": loo_corr,
        "coefficients": {
            feature: float(coef) for feature, coef in zip(features, model.coef_)
        },
        "intercept": float(model.intercept_),
        "training_feature_ranges": {
            feature: {
                "min": float(df_fit[feature].min()),
                "max": float(df_fit[feature].max()),
            }
            for feature in features
        },
    }


# =============================================================================
# TEP POTENTIAL-RATIO APPROACH (consistent with step_132)
# =============================================================================

def get_tep_gamma_potential(phi_local, phi_ref, z, alpha_0=ALPHA_0):
    """
    Compute Gamma_t from potential depth ratio (power-law form).
    Consistent with step_132_lrd_validation.py.
    """
    alpha_z = alpha_0 * np.sqrt(1 + z)
    z_fac = (1 + z) / (1 + Z_REF)
    exponent = alpha_z * z_fac * 0.5
    ratio = np.maximum(phi_local / phi_ref, 1e-6)
    return ratio**exponent


def calculate_differential_shear(z, log_Mh, concentration, alpha_0=ALPHA_0):
    """
    Compute differential temporal shear between galactic centre and halo.
    Uses the physical potential-ratio approach from step_132.

    Returns (gamma_halo, gamma_cen, delta_gamma, t_cosmic_Gyr).
    """
    M_h = 10**log_Mh
    R_vir = 30.0 * (M_h / 1e11)**(1/3) * (10.0 / (1 + z))  # kpc
    Phi_vir = G * M_h / R_vir
    Phi_cen = Phi_vir * concentration

    gamma_halo = get_tep_gamma_potential(Phi_vir, PHI_REF_VIR, z, alpha_0)
    gamma_cen  = get_tep_gamma_potential(Phi_cen, PHI_REF_VIR, z, alpha_0)

    t_cosmic = cosmo.age(z).value  # Gyr
    delta_gamma = gamma_cen - gamma_halo
    return float(gamma_halo), float(gamma_cen), float(delta_gamma), float(t_cosmic)


def estimate_halo_mass(log_Mstar, z):
    """LRD-specific Behroozi+19-like relation (consistent with step_132)."""
    log_Mh = log_Mstar + 1.5 + 0.1 * (z - 5)
    return float(np.clip(log_Mh, 9.0, 14.0))


def compute_mbh_mstar(delta_gamma, t_cosmic_Gyr, M_seed, M_star, f_Edd,
                       use_tep):
    """
    Compute predicted M_BH/M_* for a given scenario.

    BH grows as: M_BH = M_seed * exp(f_Edd * Delta_Gamma * t_cosmic / t_Salpeter)
    where Delta_Gamma is the differential temporal shear (centre minus halo).
    Without TEP, Delta_Gamma = 0 and M_BH = M_seed (no differential boost).

    The *stellar* mass M_* is the observed value (already TEP-inflated if TEP is
    active, but we compare to observed ratios so this is the right quantity).
    """
    delta_gamma = np.asarray(delta_gamma, dtype=float)
    t_cosmic_Gyr = np.asarray(t_cosmic_Gyr, dtype=float)
    M_star = np.asarray(M_star, dtype=float)
    if not use_tep:
        ratio = np.full(np.broadcast(delta_gamma, t_cosmic_Gyr, M_star).shape,
                        LOCAL_MBH_MSTAR, dtype=float)
        extra_efolds = np.zeros_like(ratio)
        return _maybe_scalar(ratio), _maybe_scalar(extra_efolds)
    extra_efolds = f_Edd * delta_gamma * t_cosmic_Gyr / T_SALPETER
    extra_efolds_clipped = np.clip(extra_efolds, None, 50.0)
    M_BH = M_seed * np.exp(extra_efolds_clipped)
    ratio = np.divide(
        M_BH,
        M_star,
        out=np.full_like(M_BH, np.nan, dtype=float),
        where=M_star > 0,
    )
    ratio = np.minimum(ratio, 1.0)
    return _maybe_scalar(ratio), _maybe_scalar(extra_efolds)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: LRD M_BH/M_* Combined Growth Model", "INFO")
    print_status("=" * 70, "INFO")

    # -------------------------------------------------------------------------
    # 1. Load step_132 population results if available
    # -------------------------------------------------------------------------
    step132_path = OUTPUT_PATH / "step_132_lrd_validation.json"
    pop_delta_gamma_median = None
    pop_log_boost_median = None
    if step132_path.exists():
        with open(step132_path) as f:
            s132 = json.load(f)
        pop_delta_gamma_median = s132.get("temporal_enhancement", {}).get(
            "median_delta_gamma")
        pop_log_boost_median = s132.get("bh_growth_boost", {}).get(
            "median_log_boost")
        n_pop = s132.get("sample_size", 0)
        print_status(f"Loaded step_132 population: N={n_pop}, "
                     f"median ΔΓ={pop_delta_gamma_median:.3f}, "
                     f"median log_boost={pop_log_boost_median:.2f}", "INFO")
    else:
        print_status("step_132 output not found; using real-sample estimates only",
                     "WARN")

    # -------------------------------------------------------------------------
    # 2. Load real LRD sample
    # -------------------------------------------------------------------------
    df_lrd = load_real_lrd_sample()
    df_lrd_ceers_direct = build_ceers_direct_mass_subset(df_lrd)
    print_status(
        f"\nLoaded real LRD sample: N={len(df_lrd)}, "
        f"median z={df_lrd['z'].median():.2f}, "
        f"median log_M*={df_lrd['log_Mstar'].median():.2f}, "
        f"median Re={df_lrd['Re_pc'].median():.0f} pc",
        "INFO",
    )
    if len(df_lrd_ceers_direct) > 0:
        print_status(
            f"Direct-mass CEERS overlap: N={len(df_lrd_ceers_direct)}, "
            f"median log_M*={df_lrd_ceers_direct['log_Mstar'].median():.2f}, "
            f"median Δlog_M*(direct - MUV)="
            f"{(df_lrd_ceers_direct['log_Mstar'] - df_lrd_ceers_direct['log_Mstar_muv_proxy']).median():+.2f}",
            "INFO",
        )
    print_status(f"Observed LRD M_BH/M*: {10**OBSERVED_LOG_MBH_MSTAR_LO:.3f}"
                 f"–{10**OBSERVED_LOG_MBH_MSTAR_HI:.3f} "
                 f"(median {10**OBSERVED_LOG_MBH_MSTAR_MEDIAN:.3f})", "INFO")

    # -------------------------------------------------------------------------
    # 3. Compute differential shear for the real object-level sample
    # -------------------------------------------------------------------------
    shear_rows = []
    for _, row in df_lrd.iterrows():
        z = float(row["z"])
        log_Mstar = float(row["log_Mstar"])
        log_Mh = estimate_halo_mass(log_Mstar, z)
        concentration = float(np.clip(500.0 / row["Re_pc"], 5.0, 50.0))
        g_halo, g_cen, dg, t_cos = calculate_differential_shear(z, log_Mh, concentration)
        shear_rows.append({
            "id": row["id"],
            "field": row.get("field", "unknown"),
            "z": z,
            "Muv": float(row["Muv"]) if pd.notna(row["Muv"]) else np.nan,
            "log_Mstar": log_Mstar,
            "log_Mstar_source": row["log_Mstar_source"],
            "Re_pc": float(row["Re_pc"]),
            "concentration": concentration,
            "log_Mh": log_Mh,
            "gamma_halo": g_halo, "gamma_cen": g_cen,
            "delta_gamma": dg, "t_cosmic_Gyr": t_cos,
        })

    df_shear = pd.DataFrame(shear_rows)
    print_status(
        f"\nSample shear summary: median ΔΓ={df_shear['delta_gamma'].median():.3f}, "
        f"median concentration={df_shear['concentration'].median():.1f}, "
        f"median t_cos={df_shear['t_cosmic_Gyr'].median():.3f} Gyr",
        "INFO",
    )

    median_dg = float(df_shear["delta_gamma"].median())
    median_t  = float(df_shear["t_cosmic_Gyr"].median())

    # -------------------------------------------------------------------------
    # 4. Evaluate each growth scenario
    # -------------------------------------------------------------------------
    print_status("\n" + "=" * 70, "INFO")
    print_status("GROWTH SCENARIO COMPARISON", "INFO")
    print_status("=" * 70, "INFO")

    scenario_results = {}
    M_star_arr = np.power(10.0, df_shear["log_Mstar"].to_numpy(dtype=float))
    for skey, spar in SCENARIOS.items():
        ratios, extra_efolds = compute_mbh_mstar(
            df_shear["delta_gamma"].to_numpy(dtype=float),
            df_shear["t_cosmic_Gyr"].to_numpy(dtype=float),
            spar["M_seed"],
            M_star_arr,
            spar["f_Edd"],
            spar["use_tep"],
        )
        ratios = np.asarray(ratios, dtype=float)
        extra_efolds = np.asarray(extra_efolds, dtype=float)
        log_ratios = np.log10(np.maximum(ratios, 1e-20))
        median_log_ratio = float(np.median(log_ratios))
        offset_from_obs = median_log_ratio - OBSERVED_LOG_MBH_MSTAR_MEDIAN

        in_observed_regime = (
            (log_ratios >= OBSERVED_LOG_MBH_MSTAR_LO) &
            (log_ratios <= OBSERVED_LOG_MBH_MSTAR_HI)
        )
        n_redshifts_in_observed_regime = int(np.sum(in_observed_regime))
        covers_observed = bool(n_redshifts_in_observed_regime > 0)

        scenario_results[skey] = {
            "label": spar["label"],
            "M_seed": spar["M_seed"],
            "f_Edd": spar["f_Edd"],
            "use_tep": spar["use_tep"],
            "median_log_mbh_mstar": median_log_ratio,
            "offset_from_observed_median_dex": offset_from_obs,
            "covers_observed_range": covers_observed,
            "n_objects_in_observed_regime": n_redshifts_in_observed_regime,
            "fraction_objects_in_observed_regime": float(np.mean(in_observed_regime)),
            "median_extra_efolds": float(np.median(extra_efolds)),
            "p16_log_mbh_mstar": float(np.percentile(log_ratios, 16)),
            "p84_log_mbh_mstar": float(np.percentile(log_ratios, 84)),
        }
        df_shear[f"log_mbh_mstar_{skey}"] = log_ratios

        status = "✓ CLOSES GAP" if abs(offset_from_obs) < 0.5 else "✗ gap remains"
        print_status(f"  {spar['label']:50s}  "
                     f"median log(M_BH/M*)={median_log_ratio:+.2f}  "
                     f"offset={offset_from_obs:+.2f} dex  {status}", "INFO")

    # -------------------------------------------------------------------------
    # 5. Monte Carlo error propagation on the combined model (S4)
    # -------------------------------------------------------------------------
    print_status("\nMonte Carlo error propagation (N=5000)...", "INFO")
    n_mc = 5000
    rng = np.random.default_rng(42)

    alpha_samples = rng.normal(ALPHA_0, ALPHA_UNCERTAINTY, n_mc)
    alpha_samples = np.clip(alpha_samples, 0.1, 1.5)
    z_arr = df_shear["z"].to_numpy(dtype=float)
    log_mh_arr = df_shear["log_Mh"].to_numpy(dtype=float)
    concentration_arr = df_shear["concentration"].to_numpy(dtype=float)
    t_cosmic_arr = df_shear["t_cosmic_Gyr"].to_numpy(dtype=float)
    M_h_arr = np.power(10.0, log_mh_arr)
    R_vir_arr = 30.0 * np.power(M_h_arr / 1e11, 1 / 3) * (10.0 / (1.0 + z_arr))
    phi_vir_arr = G * M_h_arr / R_vir_arr
    phi_cen_arr = phi_vir_arr * concentration_arr

    mc_results = {}
    for skey in [
        "S2_tep_intermediate_seed",
        "S3_tep_mild_super_eddington",
        "S4_tep_combined",
    ]:
        spar = SCENARIOS[skey]
        mc_log_ratios = []
        for a0 in alpha_samples:
            gamma_halo_mc = get_tep_gamma_potential(phi_vir_arr, PHI_REF_VIR, z_arr, alpha_0=a0)
            gamma_cen_mc = get_tep_gamma_potential(phi_cen_arr, PHI_REF_VIR, z_arr, alpha_0=a0)
            dg_mc = gamma_cen_mc - gamma_halo_mc
            ratio_mc, _ = compute_mbh_mstar(
                dg_mc, t_cosmic_arr, spar["M_seed"], M_star_arr, spar["f_Edd"], True)
            mc_log_ratios.append(np.median(np.log10(np.maximum(ratio_mc, 1e-20))))

        mc_log_ratios = np.asarray(mc_log_ratios, dtype=float)
        mc_median = float(np.median(mc_log_ratios))
        mc_ci_lo = float(np.percentile(mc_log_ratios, 2.5))
        mc_ci_hi = float(np.percentile(mc_log_ratios, 97.5))
        mc_frac_in_range = float(np.mean(
            (mc_log_ratios >= OBSERVED_LOG_MBH_MSTAR_LO) &
            (mc_log_ratios <= OBSERVED_LOG_MBH_MSTAR_HI)
        ))
        mc_results[skey] = {
            "n_draws": n_mc,
            "sample_size": int(len(df_shear)),
            "statistic": "population_median_log_mbh_mstar",
            "median_log_mbh_mstar_population": mc_median,
            "ci_95_lo": mc_ci_lo,
            "ci_95_hi": mc_ci_hi,
            "fraction_in_observed_range": mc_frac_in_range,
            "median_in_observed_range": bool(
                OBSERVED_LOG_MBH_MSTAR_LO <= mc_median <= OBSERVED_LOG_MBH_MSTAR_HI
            ),
        }
        print_status(
            f"  {spar['label']:40s} sample median={mc_median:.2f}, "
            f"95% CI [{mc_ci_lo:.2f}, {mc_ci_hi:.2f}], "
            f"in-range={mc_frac_in_range:.1%}",
            "INFO",
        )

    empirical_mass_calibration = None
    empirical_full_sample = None
    if len(df_lrd_ceers_direct) > 0:
        empirical_mass_calibration = fit_empirical_mass_calibration(df_lrd_ceers_direct)

    if empirical_mass_calibration is not None:
        empirical_model = empirical_mass_calibration["model"]
        empirical_features = empirical_mass_calibration["features"]
        empirical_calibration_output = {
            key: value for key, value in empirical_mass_calibration.items()
            if key != "model"
        }
        df_lrd_empirical = df_lrd.dropna(subset=empirical_features + ["Re_pc"]).copy()
        df_lrd_empirical["log_Mstar_empirical"] = np.clip(
            empirical_model.predict(df_lrd_empirical[empirical_features].to_numpy(dtype=float)),
            7.0,
            12.0,
        )

        empirical_rows = []
        for _, row in df_lrd_empirical.iterrows():
            z = float(row["z"])
            log_Mstar = float(row["log_Mstar_empirical"])
            log_Mh = estimate_halo_mass(log_Mstar, z)
            concentration = float(np.clip(500.0 / row["Re_pc"], 5.0, 50.0))
            g_halo, g_cen, dg, t_cos = calculate_differential_shear(z, log_Mh, concentration)
            empirical_rows.append({
                "id": row["id"],
                "field": row.get("field", "unknown"),
                "z": z,
                "Muv": float(row["Muv"]) if pd.notna(row["Muv"]) else np.nan,
                "Av": float(row["Av"]) if pd.notna(row["Av"]) else np.nan,
                "log_Lbol": float(row["log_Lbol"]) if pd.notna(row["log_Lbol"]) else np.nan,
                "log_Mstar_empirical": log_Mstar,
                "log_Mstar_muv_proxy": float(row["log_Mstar"]),
                "Re_pc": float(row["Re_pc"]),
                "concentration": concentration,
                "log_Mh": log_Mh,
                "delta_gamma": dg,
                "t_cosmic_Gyr": t_cos,
            })

        df_empirical_shear = pd.DataFrame(empirical_rows)
        M_star_empirical = np.power(10.0, df_empirical_shear["log_Mstar_empirical"].to_numpy(dtype=float))
        empirical_scenarios = {}
        for skey, spar in SCENARIOS.items():
            ratios_emp, extra_efolds_emp = compute_mbh_mstar(
                df_empirical_shear["delta_gamma"].to_numpy(dtype=float),
                df_empirical_shear["t_cosmic_Gyr"].to_numpy(dtype=float),
                spar["M_seed"],
                M_star_empirical,
                spar["f_Edd"],
                spar["use_tep"],
            )
            ratios_emp = np.asarray(ratios_emp, dtype=float)
            extra_efolds_emp = np.asarray(extra_efolds_emp, dtype=float)
            log_ratios_emp = np.log10(np.maximum(ratios_emp, 1e-20))
            in_observed_emp = (
                (log_ratios_emp >= OBSERVED_LOG_MBH_MSTAR_LO) &
                (log_ratios_emp <= OBSERVED_LOG_MBH_MSTAR_HI)
            )
            empirical_scenarios[skey] = {
                "median_log_mbh_mstar": float(np.median(log_ratios_emp)),
                "offset_from_observed_median_dex": float(np.median(log_ratios_emp) - OBSERVED_LOG_MBH_MSTAR_MEDIAN),
                "fraction_objects_in_observed_regime": float(np.mean(in_observed_emp)),
                "median_extra_efolds": float(np.median(extra_efolds_emp)),
                "p16_log_mbh_mstar": float(np.percentile(log_ratios_emp, 16)),
                "p84_log_mbh_mstar": float(np.percentile(log_ratios_emp, 84)),
            }

        z_emp = df_empirical_shear["z"].to_numpy(dtype=float)
        log_mh_emp = df_empirical_shear["log_Mh"].to_numpy(dtype=float)
        conc_emp = df_empirical_shear["concentration"].to_numpy(dtype=float)
        t_emp = df_empirical_shear["t_cosmic_Gyr"].to_numpy(dtype=float)
        M_h_emp = np.power(10.0, log_mh_emp)
        R_vir_emp = 30.0 * np.power(M_h_emp / 1e11, 1 / 3) * (10.0 / (1.0 + z_emp))
        phi_vir_emp = G * M_h_emp / R_vir_emp
        phi_cen_emp = phi_vir_emp * conc_emp

        empirical_mc_results = {}
        for skey in [
            "S2_tep_intermediate_seed",
            "S3_tep_mild_super_eddington",
            "S4_tep_combined",
        ]:
            spar = SCENARIOS[skey]
            mc_emp = []
            for a0 in alpha_samples:
                gamma_halo_mc = get_tep_gamma_potential(phi_vir_emp, PHI_REF_VIR, z_emp, alpha_0=a0)
                gamma_cen_mc = get_tep_gamma_potential(phi_cen_emp, PHI_REF_VIR, z_emp, alpha_0=a0)
                dg_mc = gamma_cen_mc - gamma_halo_mc
                ratio_mc, _ = compute_mbh_mstar(
                    dg_mc, t_emp, spar["M_seed"], M_star_empirical, spar["f_Edd"], True)
                mc_emp.append(np.median(np.log10(np.maximum(ratio_mc, 1e-20))))

            mc_emp = np.asarray(mc_emp, dtype=float)
            empirical_mc_results[skey] = {
                "n_draws": n_mc,
                "sample_size": int(len(df_empirical_shear)),
                "statistic": "population_median_log_mbh_mstar",
                "median_log_mbh_mstar_population": float(np.median(mc_emp)),
                "ci_95_lo": float(np.percentile(mc_emp, 2.5)),
                "ci_95_hi": float(np.percentile(mc_emp, 97.5)),
                "fraction_in_observed_range": float(np.mean(
                    (mc_emp >= OBSERVED_LOG_MBH_MSTAR_LO) &
                    (mc_emp <= OBSERVED_LOG_MBH_MSTAR_HI)
                )),
                "median_in_observed_range": bool(
                    OBSERVED_LOG_MBH_MSTAR_LO <= np.median(mc_emp) <= OBSERVED_LOG_MBH_MSTAR_HI
                ),
            }

        empirical_plausible_keys = [
            key for key, value in empirical_scenarios.items()
            if SCENARIOS[key]["use_tep"] and abs(value["offset_from_observed_median_dex"]) < 0.5
        ]
        empirical_robust_keys = [
            key for key, value in empirical_mc_results.items()
            if value["median_in_observed_range"]
        ]

        empirical_full_sample = {
            "sample_size": int(len(df_empirical_shear)),
            "median_z": float(df_empirical_shear["z"].median()),
            "median_log_mstar_empirical": float(df_empirical_shear["log_Mstar_empirical"].median()),
            "median_log_mstar_muv_proxy": float(df_empirical_shear["log_Mstar_muv_proxy"].median()),
            "median_mass_offset_empirical_minus_muv": float(
                (df_empirical_shear["log_Mstar_empirical"] - df_empirical_shear["log_Mstar_muv_proxy"]).median()
            ),
            "median_delta_gamma": float(df_empirical_shear["delta_gamma"].median()),
            "median_concentration": float(df_empirical_shear["concentration"].median()),
            "mass_calibration": empirical_calibration_output,
            "scenarios": empirical_scenarios,
            "monte_carlo_population": empirical_mc_results,
            "plausible_closure_scenarios": empirical_plausible_keys,
            "robust_closure_scenarios": empirical_robust_keys,
        }

        print_status(
            f"\nEmpirical CEERS-calibrated branch: N={len(df_empirical_shear)}, "
            f"LOO MAE={empirical_calibration_output['loo_mae_dex']:.2f} dex, "
            f"median log_M*={empirical_full_sample['median_log_mstar_empirical']:.2f}, "
            f"median ΔΓ={empirical_full_sample['median_delta_gamma']:.3f}",
            "INFO",
        )
        for skey in ["S1_tep_only", "S2_tep_intermediate_seed", "S3_tep_mild_super_eddington", "S4_tep_combined"]:
            item = empirical_scenarios[skey]
            print_status(
                f"  {SCENARIOS[skey]['label']:50s}  "
                f"median log(M_BH/M*)={item['median_log_mbh_mstar']:+.2f}  "
                f"offset={item['offset_from_observed_median_dex']:+.2f} dex",
                "INFO",
            )
        for skey in ["S2_tep_intermediate_seed", "S3_tep_mild_super_eddington", "S4_tep_combined"]:
            item = empirical_mc_results[skey]
            print_status(
                f"  MC {SCENARIOS[skey]['label']:43s} median={item['median_log_mbh_mstar_population']:.2f}, "
                f"95% CI [{item['ci_95_lo']:.2f}, {item['ci_95_hi']:.2f}], "
                f"in-range={item['fraction_in_observed_range']:.1%}",
                "INFO",
            )

    ceers_direct_subset = None
    if len(df_lrd_ceers_direct) > 0:
        ceers_shear_rows = []
        for _, row in df_lrd_ceers_direct.iterrows():
            z = float(row["z"])
            log_Mstar = float(row["log_Mstar"])
            log_Mh = estimate_halo_mass(log_Mstar, z)
            concentration = float(np.clip(500.0 / row["Re_pc"], 5.0, 50.0))
            g_halo, g_cen, dg, t_cos = calculate_differential_shear(z, log_Mh, concentration)
            ceers_shear_rows.append({
                "id": row["id"],
                "z": z,
                "log_Mstar": log_Mstar,
                "log_Mstar_muv_proxy": float(row["log_Mstar_muv_proxy"]),
                "Re_pc": float(row["Re_pc"]),
                "concentration": concentration,
                "log_Mh": log_Mh,
                "delta_gamma": dg,
                "t_cosmic_Gyr": t_cos,
            })

        df_ceers_shear = pd.DataFrame(ceers_shear_rows)
        M_star_ceers = np.power(10.0, df_ceers_shear["log_Mstar"].to_numpy(dtype=float))
        ceers_scenario_results = {}
        for skey, spar in SCENARIOS.items():
            ratios_ceers, _ = compute_mbh_mstar(
                df_ceers_shear["delta_gamma"].to_numpy(dtype=float),
                df_ceers_shear["t_cosmic_Gyr"].to_numpy(dtype=float),
                spar["M_seed"],
                M_star_ceers,
                spar["f_Edd"],
                spar["use_tep"],
            )
            log_ratios_ceers = np.log10(np.maximum(np.asarray(ratios_ceers, dtype=float), 1e-20))
            ceers_scenario_results[skey] = {
                "median_log_mbh_mstar": float(np.median(log_ratios_ceers)),
                "offset_from_observed_median_dex": float(np.median(log_ratios_ceers) - OBSERVED_LOG_MBH_MSTAR_MEDIAN),
                "fraction_objects_in_observed_regime": float(np.mean(
                    (log_ratios_ceers >= OBSERVED_LOG_MBH_MSTAR_LO) &
                    (log_ratios_ceers <= OBSERVED_LOG_MBH_MSTAR_HI)
                )),
            }

        ceers_direct_subset = {
            "sample_size": int(len(df_ceers_shear)),
            "match_radius_arcsec": float(DIRECT_MASS_MATCH_ARCSEC),
            "median_z": float(df_ceers_shear["z"].median()),
            "median_log_mstar_direct": float(df_ceers_shear["log_Mstar"].median()),
            "median_log_mstar_muv_proxy": float(df_ceers_shear["log_Mstar_muv_proxy"].median()),
            "median_mass_offset_direct_minus_muv": float(
                (df_ceers_shear["log_Mstar"] - df_ceers_shear["log_Mstar_muv_proxy"]).median()
            ),
            "median_delta_gamma": float(df_ceers_shear["delta_gamma"].median()),
            "median_concentration": float(df_ceers_shear["concentration"].median()),
            "scenarios": ceers_scenario_results,
        }
        print_status(
            f"\nCEERS direct-mass subset: median ΔΓ={ceers_direct_subset['median_delta_gamma']:.3f}, "
            f"median log_M*={ceers_direct_subset['median_log_mstar_direct']:.2f}",
            "INFO",
        )
        for skey in ["S1_tep_only", "S2_tep_intermediate_seed", "S3_tep_mild_super_eddington", "S4_tep_combined"]:
            item = ceers_scenario_results[skey]
            print_status(
                f"  {SCENARIOS[skey]['label']:50s}  "
                f"median log(M_BH/M*)={item['median_log_mbh_mstar']:+.2f}  "
                f"offset={item['offset_from_observed_median_dex']:+.2f} dex",
                "INFO",
            )

    s1_offset = scenario_results["S1_tep_only"]["offset_from_observed_median_dex"]
    plausible_closure_keys = [
        key for key, value in scenario_results.items()
        if value["use_tep"] and abs(value["offset_from_observed_median_dex"]) < 0.5
    ]
    best_plausible_key = min(
        plausible_closure_keys,
        key=lambda key: abs(scenario_results[key]["offset_from_observed_median_dex"]),
    ) if plausible_closure_keys else None
    robust_closure_keys = [
        key for key, value in mc_results.items()
        if value["median_in_observed_range"]
    ]
    best_robust_key = max(
        robust_closure_keys,
        key=lambda key: mc_results[key]["fraction_in_observed_range"],
    ) if robust_closure_keys else None
    best_plausible_offset = (
        scenario_results[best_plausible_key]["offset_from_observed_median_dex"]
        if best_plausible_key is not None else None
    )
    additional_closure_dex = (
        s1_offset - best_plausible_offset
        if best_plausible_offset is not None else 0.0
    )
    empirical_best_plausible_key = None
    empirical_best_robust_key = None
    if empirical_full_sample is not None:
        empirical_plausible_keys = empirical_full_sample["plausible_closure_scenarios"]
        empirical_robust_keys = empirical_full_sample["robust_closure_scenarios"]
        if empirical_plausible_keys:
            empirical_best_plausible_key = min(
                empirical_plausible_keys,
                key=lambda key: abs(
                    empirical_full_sample["scenarios"][key]["offset_from_observed_median_dex"]
                ),
            )
        if empirical_robust_keys:
            empirical_best_robust_key = max(
                empirical_robust_keys,
                key=lambda key: empirical_full_sample["monte_carlo_population"][key]["fraction_in_observed_range"],
            )

    print_status("\n" + "=" * 70, "INFO")
    print_status("GAP CLOSURE SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    print_status(f"  Conservative MUV-proxy TEP-only offset: {s1_offset:+.2f} dex",
                 "INFO")
    if best_plausible_key is not None:
        print_status(
            f"  Best conservative deterministic path:  "
            f"{scenario_results[best_plausible_key]['label']} "
            f"({best_plausible_offset:+.2f} dex)",
            "INFO",
        )
        print_status(
            f"  Additional closure vs conservative S1: "
            f"{additional_closure_dex:+.2f} dex",
            "INFO",
        )
    if empirical_full_sample is not None:
        print_status(
            f"  Empirical CEERS-calibrated median S1:  "
            f"{empirical_full_sample['scenarios']['S1_tep_only']['offset_from_observed_median_dex']:+.2f} dex",
            "INFO",
        )
        if empirical_best_plausible_key is not None:
            print_status(
                f"  Best empirical deterministic path:     "
                f"{SCENARIOS[empirical_best_plausible_key]['label']} "
                f"({empirical_full_sample['scenarios'][empirical_best_plausible_key]['offset_from_observed_median_dex']:+.2f} dex)",
                "INFO",
            )
        if empirical_best_robust_key is not None:
            print_status(
                f"  Best empirical MC-supported path:      "
                f"{SCENARIOS[empirical_best_robust_key]['label']} "
                f"({empirical_full_sample['monte_carlo_population'][empirical_best_robust_key]['fraction_in_observed_range']:.1%} in-range)",
                "INFO",
            )

    if empirical_best_robust_key is not None:
        assessment = "mass_model_sensitive_real_sample_empirical_calibration_supports_closure"
        conclusion_text = (
            "The upgraded real-sample analysis shows that LRD gap closure is "
            "dominated by stellar-mass estimation. Under the conservative MUV-only "
            f"mass proxy, the full Kokorev population undercloses badly (TEP-only {s1_offset:+.1f} dex). "
            "But a CEERS direct-mass calibration trained on matched real LRDs "
            f"(N={empirical_mass_calibration['training_sample_size']}, leave-one-out MAE "
            f"{empirical_mass_calibration['loo_mae_dex']:.2f} dex) shifts the full real "
            "sample into population-median near-closure, with Monte Carlo support "
            f"strongest for {SCENARIOS[empirical_best_robust_key]['label']}."
        )
    elif empirical_best_plausible_key is not None:
        assessment = "mass_model_sensitive_real_sample_empirical_calibration_identifies_plausible_closure"
        conclusion_text = (
            "The upgraded real-sample analysis shows a sharp mass-model sensitivity. "
            f"The conservative MUV-only branch undercloses (TEP-only {s1_offset:+.1f} dex), "
            "while the CEERS-calibrated empirical-mass branch yields deterministic "
            f"near-closure for {SCENARIOS[empirical_best_plausible_key]['label']}. "
            "The honest interpretation is that real-sample closure becomes plausible "
            "once the LRD stellar-mass proxy is anchored to direct masses, but the "
            "result remains calibration-dependent."
        )
    else:
        assessment = "combined_model_narrows_gap_under_conservative_mass_proxy"
        conclusion_text = (
            "The combined TEP + accretion model narrows the M_BH/M_* gap "
            f"from {abs(s1_offset):.1f} dex under TEP-only "
            "but does not fully close it under the tested parameters."
        )

    print_status(f"\nAssessment: {assessment}", "INFO")

    # -------------------------------------------------------------------------
    # 7. Save outputs
    # -------------------------------------------------------------------------
    # CSV: per-redshift shear table
    csv_path = INTERIM_PATH / f"step_{STEP_NUM}_{STEP_NAME}.csv"
    df_shear.to_csv(csv_path, index=False)
    print_status(f"Saved: {csv_path}", "INFO")

    result = {
        "step":       STEP_NUM,
        "name":       STEP_NAME,
        "status":     "complete",
        "alpha_0":    float(ALPHA_0),
        "alpha_uncertainty": float(ALPHA_UNCERTAINTY),
        "real_sample": {
            "catalog": str(LRD_CATALOG_PATH.name),
            "sample_size": int(len(df_shear)),
            "median_z": float(df_shear["z"].median()),
            "median_log_mstar": float(df_shear["log_Mstar"].median()),
            "median_re_pc": float(df_shear["Re_pc"].median()),
            "median_concentration": float(df_shear["concentration"].median()),
            "mass_source_counts": df_lrd["log_Mstar_source"].value_counts().to_dict(),
        },
        "observed_regime": {
            "log_mbh_mstar_median": OBSERVED_LOG_MBH_MSTAR_MEDIAN,
            "log_mbh_mstar_lo":     OBSERVED_LOG_MBH_MSTAR_LO,
            "log_mbh_mstar_hi":     OBSERVED_LOG_MBH_MSTAR_HI,
            "sources": "Matthee+2024; Greene+2024; Kokorev+2024; Maiolino+2024; Pacucci+2024",
        },
        "differential_shear": {
            "median_delta_gamma": median_dg,
            "median_t_cosmic_Gyr": median_t,
        },
        "scenarios": scenario_results,
        "monte_carlo_population_muv_proxy": mc_results,
        "gap_closure": {
            "conservative_muv_proxy": {
                "tep_only_offset_dex": s1_offset,
                "best_plausible_scenario": best_plausible_key,
                "best_plausible_offset_dex": best_plausible_offset,
                "additional_closure_dex": additional_closure_dex,
                "plausible_closure_scenarios": plausible_closure_keys,
                "robust_closure_scenarios": robust_closure_keys,
                "gap_closed_population_median": bool(best_robust_key is not None),
            },
            "empirical_ceers_calibrated": None if empirical_full_sample is None else {
                "best_plausible_scenario": empirical_best_plausible_key,
                "best_plausible_offset_dex": (
                    empirical_full_sample["scenarios"][empirical_best_plausible_key]["offset_from_observed_median_dex"]
                    if empirical_best_plausible_key is not None else None
                ),
                "best_robust_scenario": empirical_best_robust_key,
                "best_robust_fraction_in_range": (
                    empirical_full_sample["monte_carlo_population"][empirical_best_robust_key]["fraction_in_observed_range"]
                    if empirical_best_robust_key is not None else None
                ),
                "plausible_closure_scenarios": empirical_full_sample["plausible_closure_scenarios"],
                "robust_closure_scenarios": empirical_full_sample["robust_closure_scenarios"],
                "gap_closed_population_median": bool(empirical_best_robust_key is not None),
            },
        },
        "assessment": assessment,
        "conclusion": conclusion_text,
        "step_132_population_baseline": {
            "available": step132_path.exists(),
            "median_delta_gamma": pop_delta_gamma_median,
            "median_log_boost":   pop_log_boost_median,
        },
        "empirical_ceers_mass_calibrated_full_sample": empirical_full_sample,
        "direct_mass_ceers_subset": ceers_direct_subset,
        "physical_motivation": {
            "intermediate_seeds": (
                "Seeds of 10^3 M_sun are consistent with Population III "
                "remnants in dense nuclear star clusters (Madau & Rees 2001; "
                "Devecchi & Volonteri 2009) and fall well below the heavy-seed "
                "abundance problem threshold (10^5 M_sun)."
            ),
            "mild_super_eddington": (
                "Accretion at 1.5-2.5x Eddington is routinely observed in "
                "local ultraluminous X-ray sources (King et al. 2023; "
                "Middleton et al. 2015) and confirmed stable over 10^7 yr "
                "timescales in radiation-magnetohydrodynamic simulations "
                "(Jiang et al. 2019). This is far below the 10x required "
                "without TEP."
            ),
        },
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    return result


main = run

if __name__ == "__main__":
    run()
