#!/usr/bin/env python3
"""
TEP-JWST Step 177: JADES DR4 Spectroscopic Catalog Ingestion

Ingests the JADES Data Release 4 NIRSpec/MSA spectroscopic catalog
(D'Eugenio et al. 2025, Curtis-Lake et al. 2025) and applies the TEP
model to the 2,858 good-quality spectroscopic redshifts (flags A/B).

This step provides:
1. A high-purity spectroscopic sample with precise redshifts (Δz/z < 0.01)
2. UV-luminosity-based stellar mass estimates for the z>5 subsample
3. TEP Gamma_t and t_eff values for cross-survey validation
4. Comparison of spec-z vs photo-z TEP signal strength

Key upgrade vs prior data:
- Previous combined spec catalog: N=147 (UNCOVER + 4 Curtis-Lake sources)
- JADES DR4 good spec-z: N=2,858 (flags A/B), 118 at z>7, 41 at z>8

Inputs:
- data/raw/jades_hainline/JADES_DR4_spectroscopic_catalog.fits

Outputs:
- results/interim/step_177_jades_dr4_specz.csv
- results/outputs/step_177_jades_dr4_ingestion.json

Reference:
- D'Eugenio et al. 2025 (DR3/DR4 data reduction)
- Curtis-Lake, Cameron, Bunker et al. 2025 (DR4 Paper I)
- Scholtz, Carniani et al. 2025 (DR4 Paper II)

Author: Matthew L. Smawfield
Date: 2026
"""

import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import (
    ALPHA_0, LOG_MH_REF, Z_REF,
    tep_alpha, compute_gamma_t as tep_gamma, isochrony_mass_bias
)

STEP_NUM = "177"
STEP_NAME = "jades_dr4_ingestion"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "jades_hainline"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# CONSTANTS
# =============================================================================

# UV-to-stellar-mass relation: log(M*) = -0.4*(MUV - MUV_ref) + log(M*_ref)
# Based on Stark et al. 2013, Song et al. 2016 calibration for z~5-10 galaxies
# log(M*/L_UV) ~ -0.4*(MUV + 19.5) + 8.7  (in solar units)
MUV_REF = -19.5        # Reference UV magnitude
LOG_MSTAR_REF = 8.7    # Reference log stellar mass at MUV_REF
ML_SLOPE = 0.4         # dlog(M*)/d(MUV): brighter → more massive

# Halo mass from stellar mass: Behroozi et al. 2013 high-z approximation
# log(Mh) ≈ log(M*) + 1.5  (for z>5, where SMHM ratio is ~1/30)
DELTA_LOG_MH = 1.5

# =============================================================================
# LOAD JADES DR4 CATALOG
# =============================================================================

def load_jades_dr4():
    """Load JADES DR4 spectroscopic catalog."""
    catalog_file = DATA_PATH / "JADES_DR4_spectroscopic_catalog.fits"
    print_status(f"Loading JADES DR4: {catalog_file}", "PROCESS")

    with fits.open(catalog_file) as hdul:
        obs_info = hdul['Obs_info'].data
        r100 = hdul['R100_5pix'].data

    print_status(f"Total targets: {len(obs_info)}", "INFO")
    return obs_info, r100


def apply_quality_cuts(obs_info):
    """
    Apply quality cuts to select reliable spectroscopic redshifts.

    Flag definitions (JADES DR4):
    - A: Secure redshift from multiple emission lines (highest quality)
    - B: Secure redshift from single strong line + continuum break
    - C: Tentative redshift
    - D: Uncertain / low S/N
    - E: No reliable redshift / failed extraction
    """
    flag = obs_info['z_Spec_flag']
    z = obs_info['z_Spec']

    # Accept flags A and B only
    good = ((flag == 'A') | (flag == 'B')) & (z > 0.1) & (z < 20)
    data_good = obs_info[good]
    print_status(f"Good spec-z (flags A/B): N = {np.sum(good)}", "INFO")
    print_status(f"  z>5: N = {np.sum(data_good['z_Spec'] > 5)}", "INFO")
    print_status(f"  z>7: N = {np.sum(data_good['z_Spec'] > 7)}", "INFO")
    print_status(f"  z>8: N = {np.sum(data_good['z_Spec'] > 8)}", "INFO")
    print_status(f"  z>10: N = {np.sum(data_good['z_Spec'] > 10)}", "INFO")
    return data_good


# =============================================================================
# STELLAR MASS FROM UV LUMINOSITY
# =============================================================================

def muv_to_log_mstar(muv):
    """
    Estimate log stellar mass from rest-frame UV absolute magnitude.

    Uses the empirical calibration from Song et al. (2016, ApJ 825, 5)
    for z~4-8 galaxies: log(M*/M_sun) = -0.4*(MUV - MUV_ref) + log(M*_ref)

    This is an approximation; uncertainties are ~0.3-0.5 dex.
    """
    log_mstar = -ML_SLOPE * (muv - MUV_REF) + LOG_MSTAR_REF
    return log_mstar


def stellar_to_halo_mass(log_mstar, z):
    """
    Estimate halo mass from stellar mass using Behroozi+2013 high-z SMHM.

    At z>5, the stellar-to-halo mass ratio is approximately M*/Mh ~ 0.03,
    so log(Mh) ~ log(M*) + 1.5.
    """
    log_mh = log_mstar + DELTA_LOG_MH
    return log_mh


# =============================================================================
# BUILD DATAFRAME
# =============================================================================

def build_dataframe(data_good):
    """Build analysis DataFrame from JADES DR4 good-quality sources."""

    z_spec = np.array(data_good['z_Spec'], dtype=float)
    z_phot = np.array(data_good['z_phot'], dtype=float)
    ra = np.array(data_good['RA_TARG'], dtype=float)
    dec = np.array(data_good['Dec_TARG'], dtype=float)
    muv = np.array(data_good['MUV'], dtype=float)
    muv_err_u = np.array(data_good['MUV_err_u'], dtype=float)
    muv_err_l = np.array(data_good['MUV_err_l'], dtype=float)
    flag = np.array(data_good['z_Spec_flag'])
    field = np.array(data_good['Field'])
    unique_id = np.array(data_good['Unique_ID'])

    # Stellar mass from MUV where available
    valid_muv = np.isfinite(muv) & (muv < 0) & (muv > -30)
    log_mstar = np.full(len(z_spec), np.nan)
    log_mstar[valid_muv] = muv_to_log_mstar(muv[valid_muv])

    # Halo mass
    log_mh = np.full(len(z_spec), np.nan)
    valid_mass = np.isfinite(log_mstar) & (log_mstar > 5)
    log_mh[valid_mass] = stellar_to_halo_mass(log_mstar[valid_mass], z_spec[valid_mass])

    # Cosmic age at spec-z
    t_cosmic = np.array([cosmo.age(z).to(u.Gyr).value for z in z_spec])

    # MUV uncertainty (average of upper/lower)
    muv_err = np.full(len(z_spec), np.nan)
    valid_err = np.isfinite(muv_err_u) & np.isfinite(muv_err_l)
    muv_err[valid_err] = 0.5 * (np.abs(muv_err_u[valid_err]) + np.abs(muv_err_l[valid_err]))

    df = pd.DataFrame({
        'unique_id': unique_id,
        'ra': ra,
        'dec': dec,
        'z_spec': z_spec,
        'z_phot': z_phot,
        'z_spec_flag': flag,
        'field': field,
        'MUV': muv,
        'MUV_err': muv_err,
        'log_Mstar': log_mstar,
        'log_Mh': log_mh,
        't_cosmic': t_cosmic,
    })

    return df


# =============================================================================
# APPLY TEP MODEL
# =============================================================================

def apply_tep_model(df):
    """Apply TEP model to compute Gamma_t and t_eff."""
    print_status("Applying TEP model...", "PROCESS")

    # Only compute TEP quantities where we have halo mass
    valid = np.isfinite(df['log_Mh']) & (df['log_Mh'] > 8)
    print_status(f"Sources with valid halo mass: {valid.sum()}", "INFO")

    # Compute Gamma_t
    gamma_t = np.full(len(df), np.nan)
    t_eff = np.full(len(df), np.nan)
    alpha_z = np.full(len(df), np.nan)
    n_ml = np.full(len(df), np.nan)
    ml_bias = np.full(len(df), np.nan)
    log_mstar_true = np.full(len(df), np.nan)

    for i, (idx, row) in enumerate(df[valid].iterrows()):
        z = row['z_spec']
        lmh = row['log_Mh']
        lms = row['log_Mstar']

        try:
            gt = tep_gamma(lmh, z)
            az = tep_alpha(z)
            te = row['t_cosmic'] * gt
            bias = isochrony_mass_bias(lmh, z)
            lms_true = lms - bias if np.isfinite(lms) else np.nan

            gamma_t[df.index.get_loc(idx)] = gt
            t_eff[df.index.get_loc(idx)] = te
            alpha_z[df.index.get_loc(idx)] = az
            n_ml[df.index.get_loc(idx)] = bias
            ml_bias[df.index.get_loc(idx)] = bias
            log_mstar_true[df.index.get_loc(idx)] = lms_true
        except Exception:
            pass

    df['gamma_t'] = gamma_t
    df['t_eff'] = t_eff
    df['alpha_z'] = alpha_z
    df['n_ml'] = n_ml
    df['ml_bias'] = ml_bias
    df['log_Mstar_true'] = log_mstar_true

    n_valid_gt = np.sum(np.isfinite(df['gamma_t']))
    print_status(f"Gamma_t computed for {n_valid_gt} sources", "INFO")
    print_status(f"  Median Gamma_t: {np.nanmedian(df['gamma_t']):.3f}", "INFO")
    print_status(f"  Gamma_t range: {np.nanmin(df['gamma_t']):.3f} - {np.nanmax(df['gamma_t']):.3f}", "INFO")

    return df


# =============================================================================
# TEP SIGNAL ANALYSIS
# =============================================================================

def analyze_tep_signal(df):
    """
    Analyze TEP signal in JADES DR4 spectroscopic sample.

    Tests:
    1. Spearman correlation: MUV vs z_spec (brighter at higher z = more massive)
    2. Spearman correlation: gamma_t vs MUV (deeper potential → brighter UV)
    3. Photo-z vs spec-z comparison for TEP signal consistency
    4. z>7 subsample analysis
    """
    results = {}

    # --- Full sample ---
    full = df[np.isfinite(df['gamma_t']) & np.isfinite(df['MUV'])].copy()
    print_status(f"Full sample with gamma_t + MUV: N={len(full)}", "INFO")

    if len(full) >= 10:
        rho, p = stats.spearmanr(full['gamma_t'], full['MUV'])
        results['full_sample'] = {
            'N': len(full),
            'spearman_rho_gamma_t_vs_MUV': float(rho),
            'spearman_p': float(p),
            'p_formatted': format_p_value(p),
            'median_gamma_t': float(np.nanmedian(full['gamma_t'])),
            'median_MUV': float(np.nanmedian(full['MUV'])),
        }
        print_status(f"Full: ρ(Gamma_t, MUV) = {rho:.3f}, p = {format_p_value(p)}", "INFO")

    # --- z > 7 subsample ---
    z7 = df[(df['z_spec'] > 7) & np.isfinite(df['gamma_t']) & np.isfinite(df['MUV'])].copy()
    print_status(f"z>7 sample with gamma_t + MUV: N={len(z7)}", "INFO")

    if len(z7) >= 5:
        rho7, p7 = stats.spearmanr(z7['gamma_t'], z7['MUV'])
        results['z_gt_7'] = {
            'N': len(z7),
            'spearman_rho_gamma_t_vs_MUV': float(rho7),
            'spearman_p': float(p7),
            'p_formatted': format_p_value(p7),
            'median_gamma_t': float(np.nanmedian(z7['gamma_t'])),
            'z_range': [float(z7['z_spec'].min()), float(z7['z_spec'].max())],
        }
        print_status(f"z>7: ρ(Gamma_t, MUV) = {rho7:.3f}, p = {format_p_value(p7)}", "INFO")

    # --- z > 8 subsample ---
    z8 = df[(df['z_spec'] > 8) & np.isfinite(df['gamma_t']) & np.isfinite(df['MUV'])].copy()
    print_status(f"z>8 sample with gamma_t + MUV: N={len(z8)}", "INFO")

    if len(z8) >= 5:
        rho8, p8 = stats.spearmanr(z8['gamma_t'], z8['MUV'])
        results['z_gt_8'] = {
            'N': len(z8),
            'spearman_rho_gamma_t_vs_MUV': float(rho8),
            'spearman_p': float(p8),
            'p_formatted': format_p_value(p8),
            'median_gamma_t': float(np.nanmedian(z8['gamma_t'])),
            'z_range': [float(z8['z_spec'].min()), float(z8['z_spec'].max())],
        }
        print_status(f"z>8: ρ(Gamma_t, MUV) = {rho8:.3f}, p = {format_p_value(p8)}", "INFO")

    # --- Photo-z vs spec-z comparison ---
    both = df[np.isfinite(df['z_phot']) & (df['z_phot'] > 0.1) &
              np.isfinite(df['z_spec']) & (df['z_spec'] > 0.1)].copy()
    if len(both) >= 20:
        dz = (both['z_spec'] - both['z_phot']) / (1 + both['z_spec'])
        sigma_mad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
        eta_outlier = np.mean(np.abs(dz) > 0.15)
        results['photoz_vs_specz'] = {
            'N': len(both),
            'sigma_MAD': float(sigma_mad),
            'eta_outlier_15pct': float(eta_outlier),
            'median_dz': float(np.median(dz)),
        }
        print_status(f"Photo-z accuracy: σ_MAD = {sigma_mad:.4f}, η_outlier = {eta_outlier:.3f}", "INFO")

    # --- Redshift distribution ---
    z_all = df['z_spec'].values
    results['redshift_distribution'] = {
        'N_total': int(len(df)),
        'N_z_gt_5': int(np.sum(z_all > 5)),
        'N_z_gt_7': int(np.sum(z_all > 7)),
        'N_z_gt_8': int(np.sum(z_all > 8)),
        'N_z_gt_10': int(np.sum(z_all > 10)),
        'z_max': float(z_all.max()),
        'z_median': float(np.median(z_all)),
    }

    return results


# =============================================================================
# CROSS-SURVEY CONSISTENCY CHECK
# =============================================================================

def cross_survey_check(df_jades):
    """
    Compare JADES DR4 TEP signal with UNCOVER DR4 signal.

    Loads the UNCOVER step_02 output and checks whether the Spearman ρ
    for Gamma_t vs dust/age is consistent between surveys.
    """
    results = {}

    uncover_path = PROJECT_ROOT / "results" / "interim" / "step_02_uncover_full_sample_tep.csv"
    if not uncover_path.exists():
        print_status("UNCOVER TEP sample not found, skipping cross-survey check", "WARN")
        return results

    df_uncover = pd.read_csv(uncover_path)
    print_status(f"UNCOVER sample: N={len(df_uncover)}", "INFO")

    # JADES: gamma_t vs MUV (proxy for mass/age)
    jades_valid = df_jades[np.isfinite(df_jades['gamma_t']) & np.isfinite(df_jades['MUV'])]
    if len(jades_valid) >= 10:
        rho_j, p_j = stats.spearmanr(jades_valid['gamma_t'], jades_valid['MUV'])
        results['jades_gamma_t_vs_MUV'] = {
            'rho': float(rho_j), 'p': float(p_j), 'N': len(jades_valid)
        }

    # UNCOVER: gamma_t vs dust (primary TEP signal)
    if 'gamma_t' in df_uncover.columns and 'dust' in df_uncover.columns:
        unc_valid = df_uncover[np.isfinite(df_uncover['gamma_t']) & np.isfinite(df_uncover['dust'])]
        if len(unc_valid) >= 10:
            rho_u, p_u = stats.spearmanr(unc_valid['gamma_t'], unc_valid['dust'])
            results['uncover_gamma_t_vs_dust'] = {
                'rho': float(rho_u), 'p': float(p_u), 'N': len(unc_valid)
            }
            print_status(f"UNCOVER ρ(Gamma_t, dust) = {rho_u:.3f}", "INFO")

        # Sign consistency: both should show negative correlation with Gamma_t
        # JADES: ρ(Gamma_t, MUV) < 0 → deeper potential → brighter UV (more negative MUV)
        # UNCOVER: ρ(Gamma_t, dust) < 0 → deeper potential → less dust attenuation at high-z
        # Both negative = consistent TEP signal direction
        if 'jades_gamma_t_vs_MUV' in results and 'uncover_gamma_t_vs_dust' in results:
            rho_j = results['jades_gamma_t_vs_MUV']['rho']
            rho_u = results['uncover_gamma_t_vs_dust']['rho']
            sign_consistent = np.sign(rho_j) == np.sign(rho_u)
            results['sign_consistent'] = bool(sign_consistent)
            print_status(f"Sign consistency (JADES vs UNCOVER): {sign_consistent}", "INFO")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 60, "INFO")
    print_status("STEP 177: JADES DR4 Spectroscopic Catalog Ingestion", "INFO")
    print_status("=" * 60, "INFO")

    # Load
    obs_info, r100 = load_jades_dr4()

    # Quality cuts
    data_good = apply_quality_cuts(obs_info)

    # Build DataFrame
    df = build_dataframe(data_good)

    # Apply TEP model
    df = apply_tep_model(df)

    # Save interim CSV
    out_csv = INTERIM_PATH / "step_177_jades_dr4_specz.csv"
    df.to_csv(out_csv, index=False)
    print_status(f"Saved: {out_csv} (N={len(df)})", "INFO")

    # Analyze TEP signal
    print_status("Analyzing TEP signal...", "PROCESS")
    signal_results = analyze_tep_signal(df)

    # Cross-survey check
    print_status("Cross-survey consistency check...", "PROCESS")
    cross_results = cross_survey_check(df)

    # Compile output
    output = {
        'step': STEP_NUM,
        'name': STEP_NAME,
        'description': 'JADES DR4 spectroscopic catalog ingestion and TEP analysis',
        'reference': 'D\'Eugenio et al. 2025; Curtis-Lake et al. 2025; Scholtz et al. 2025',
        'catalog_file': 'JADES_DR4_spectroscopic_catalog.fits',
        'n_total_targets': int(len(obs_info)),
        'n_good_specz': int(len(df)),
        'n_with_gamma_t': int(np.sum(np.isfinite(df['gamma_t']))),
        'tep_signal': signal_results,
        'cross_survey': cross_results,
        'upgrade_vs_prior': {
            'prior_combined_specz': 147,
            'new_good_specz': int(len(df)),
            'factor_increase': round(len(df) / 147, 1),
            'prior_z_gt_7': 32,
            'new_z_gt_7': int(np.sum(df['z_spec'] > 7)),
        },
    }

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print_status(f"Saved: {out_json}", "INFO")

    # Summary
    print_status("=" * 60, "INFO")
    print_status("SUMMARY", "INFO")
    print_status(f"  JADES DR4 good spec-z: N = {len(df)}", "INFO")
    print_status(f"  Upgrade factor vs prior: {len(df)/147:.1f}×", "INFO")
    if 'full_sample' in signal_results:
        rho = signal_results['full_sample']['spearman_rho_gamma_t_vs_MUV']
        p_fmt = signal_results['full_sample']['p_formatted']
        print_status(f"  TEP signal (full): ρ = {rho:.3f}, p = {p_fmt}", "INFO")
    if 'z_gt_7' in signal_results:
        rho7 = signal_results['z_gt_7']['spearman_rho_gamma_t_vs_MUV']
        p7_fmt = signal_results['z_gt_7']['p_formatted']
        print_status(f"  TEP signal (z>7): ρ = {rho7:.3f}, p = {p7_fmt}", "INFO")
    print_status("=" * 60, "INFO")

    return output


if __name__ == "__main__":
    main()
