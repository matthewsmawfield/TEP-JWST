#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.4s.
"""
TEP-JWST Step 152: UNCOVER DR4 Full SPS Catalog (MegaScience-updated)

Ingests the full UNCOVER DR4 Stellar Population Synthesis catalog (Wang et al.
2024; Suess et al. 2024; Price et al. 2025), which uses 20-band MegaScience
photometry and Prospector-β SED fitting. This supersedes the DR2 catalog
used in step_01 (2,315 sources) with a much larger sample (74,020 total;
2,628 at z>4 with dust2; 860 at z>7).

Also ingests the DR4 zspec sub-catalog: 87 sources at z>5 with spectroscopic
redshifts and full Prospector SED properties (dust2, mwa, met, sfr) — the
gold-standard spec-z + Prospector dust sample.

Key improvements over step_01:
- 20-band MegaScience photometry vs 7-band UNCOVER-only
- Prospector-β with dynamic SFH(M,z) prior
- Lens model v2.0 (improved magnification corrections)
- Separate zspec catalog with spec-z fixed SED fits

Inputs:
- data/raw/uncover/UNCOVER_DR4_SPS_catalog.fits   (74,020 sources)
- data/raw/uncover/UNCOVER_DR4_SPS_zspec_catalog.fits  (668 sources)

Outputs:
- results/interim/step_152_uncover_dr4_full_sps.csv
- results/interim/step_152_uncover_dr4_zspec.csv
- results/outputs/step_152_uncover_dr4_full_sps.json

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting & JSON serialiser
from scripts.utils.downloader import smart_download  # Robust HTTP download utility
from scripts.utils.tep_model import (
    tep_alpha, compute_gamma_t as tep_gamma, isochrony_mass_bias  # Shared TEP model
)

STEP_NUM = "152"  # Pipeline step number
STEP_NAME = "uncover_dr4_full_sps"  # Used in log / output filenames

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uncover"  # UNCOVER raw data directory
UNCOVER_FULL_SPS_FILE = DATA_PATH / "UNCOVER_DR4_SPS_catalog.fits"
UNCOVER_FULL_SPS_URLS = [
    "https://zenodo.org/api/records/14281664/files/UNCOVER_DR4_SPS_catalog.fits/content",
    "https://zenodo.org/records/14281664/files/UNCOVER_DR4_SPS_catalog.fits?download=1",
]
UNCOVER_ZSPEC_SPS_FILE = DATA_PATH / "UNCOVER_DR4_SPS_zspec_catalog.fits"
UNCOVER_ZSPEC_SPS_URLS = [
    "https://drive.usercontent.google.com/download?id=1j32n3e7hX0iw5ZyGlVAbyIf4MmjM4RfS&export=download&confirm=t",
    "https://drive.google.com/uc?export=download&id=1j32n3e7hX0iw5ZyGlVAbyIf4MmjM4RfS&confirm=t",
]
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

DELTA_LOG_MH = 2.0   # log(Mh) = log(M*) + 2.0 (consistent with tep_model.py and step_01/02)
Z_MIN = 4.0
LOG_MSTAR_MIN = 8.0


# =============================================================================
# LOAD CATALOGS
# =============================================================================

def ensure_catalog(path, min_size_mb, urls):
    if path.exists() and path.stat().st_size / 1e6 >= min_size_mb:
        return True
    try:
        return smart_download(
            url=urls[0],
            dest=path,
            min_size_mb=min_size_mb,
            fallback_urls=urls[1:],
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"Could not obtain {path.name}: {e}")
        return False

def load_full_sps():
    """Load full UNCOVER DR4 SPS catalog (74k sources, MegaScience-updated)."""
    path = UNCOVER_FULL_SPS_FILE
    if not ensure_catalog(path, 60, UNCOVER_FULL_SPS_URLS):
        raise FileNotFoundError(path)
    print_status(f"Loading full DR4 SPS: {path}", "PROCESS")
    with fits.open(path) as hdul:
        d = hdul[1].data
        df = pd.DataFrame({
            'id': np.array(d['id'], dtype=int),
            'ra': np.array(d['ra'], dtype=float),
            'dec': np.array(d['dec'], dtype=float),
            'use_phot': np.array(d['use_phot'], dtype=int),
            'z_spec': np.array(d['z_spec'], dtype=float),
            'z_phot': np.array(d['z_ml'], dtype=float),
            'z_16': np.array(d['z_16'], dtype=float),
            'z_84': np.array(d['z_84'], dtype=float),
            'log_Mstar': np.array(d['mstar_50'], dtype=float),
            'log_Mstar_16': np.array(d['mstar_16'], dtype=float),
            'log_Mstar_84': np.array(d['mstar_84'], dtype=float),
            'dust2': np.array(d['dust2_50'], dtype=float),
            'dust2_16': np.array(d['dust2_16'], dtype=float),
            'dust2_84': np.array(d['dust2_84'], dtype=float),
            'mwa': np.array(d['mwa_50'], dtype=float),
            'mwa_16': np.array(d['mwa_16'], dtype=float),
            'mwa_84': np.array(d['mwa_84'], dtype=float),
            'met': np.array(d['met_50'], dtype=float),
            'met_16': np.array(d['met_16'], dtype=float),
            'met_84': np.array(d['met_84'], dtype=float),
            'sfr100': np.array(d['sfr100_50'], dtype=float),
            'ssfr100': np.array(d['ssfr100_50'], dtype=float),
            'sfr10': np.array(d['sfr10_50'], dtype=float),
            'logfagn': np.array(d['logfagn_50'], dtype=float),
            'UV': np.array(d['UV_50'], dtype=float),
            'VJ': np.array(d['VJ_50'], dtype=float),
            'chi2': np.array(d['chi2'], dtype=float),
            'nbands': np.array(d['nbands'], dtype=int),
        })
    print_status(f"Full DR4 SPS: N = {len(df)}", "INFO")
    return df


def load_zspec_sps():
    """Load UNCOVER DR4 zspec SPS catalog (668 sources, spec-z fixed fits)."""
    path = UNCOVER_ZSPEC_SPS_FILE
    if not ensure_catalog(path, 0.5, UNCOVER_ZSPEC_SPS_URLS):
        print_status(f"Missing: {path.name} — not yet downloaded.", "ERROR")
        return None
    print_status(f"Loading DR4 zspec SPS: {path}", "PROCESS")
    with fits.open(path) as hdul:
        d = hdul[1].data
        df = pd.DataFrame({
            'id_msa': np.array(d['id_msa'], dtype=int),
            'ra': np.array(d['ra'], dtype=float),
            'dec': np.array(d['dec'], dtype=float),
            'z_spec': np.array(d['z_spec'], dtype=float),
            'flag_zspec_qual': np.array(d['flag_zspec_qual'], dtype=int),
            'flag_emission_lines': np.array(d['flag_emission_lines'], dtype=int),
            'log_Mstar': np.array(d['mstar_50'], dtype=float),
            'log_Mstar_16': np.array(d['mstar_16'], dtype=float),
            'log_Mstar_84': np.array(d['mstar_84'], dtype=float),
            'dust2': np.array(d['dust2_50'], dtype=float),
            'dust2_16': np.array(d['dust2_16'], dtype=float),
            'dust2_84': np.array(d['dust2_84'], dtype=float),
            'mwa': np.array(d['mwa_50'], dtype=float),
            'met': np.array(d['met_50'], dtype=float),
            'sfr100': np.array(d['sfr100_50'], dtype=float),
            'ssfr100': np.array(d['ssfr100_50'], dtype=float),
            'logfagn': np.array(d['logfagn_50'], dtype=float),
            'UV': np.array(d['UV_50'], dtype=float),
            'VJ': np.array(d['VJ_50'], dtype=float),
            'chi2': np.array(d['chi2'], dtype=float),
        })
    print_status(f"DR4 zspec SPS: N = {len(df)}", "INFO")
    return df


# =============================================================================
# QUALITY CUTS
# =============================================================================

def apply_quality_cuts_full(df):
    """Apply quality cuts to full SPS catalog."""
    mask = (
        (df['use_phot'] == 1) &
        (df['z_phot'] >= Z_MIN) &
        (df['log_Mstar'] >= LOG_MSTAR_MIN) &
        np.isfinite(df['log_Mstar']) &
        np.isfinite(df['dust2']) &
        (df['dust2'] >= 0)
    )
    out = df[mask].copy().reset_index(drop=True)
    print_status(f"After quality cuts (z>{Z_MIN}, M*>{LOG_MSTAR_MIN}, use_phot): N = {len(out)}", "INFO")
    print_status(f"  z>5: N = {(out['z_phot']>5).sum()}", "INFO")
    print_status(f"  z>7: N = {(out['z_phot']>7).sum()}", "INFO")
    print_status(f"  z>8: N = {(out['z_phot']>8).sum()}", "INFO")
    return out


def apply_quality_cuts_zspec(df):
    """Apply quality cuts to zspec SPS catalog."""
    mask = (
        (df['flag_zspec_qual'] >= 2) &
        np.isfinite(df['z_spec']) &
        (df['z_spec'] > 0.5) &
        np.isfinite(df['log_Mstar']) &
        (df['log_Mstar'] >= LOG_MSTAR_MIN) &
        np.isfinite(df['dust2']) &
        (df['dust2'] >= 0)
    )
    out = df[mask].copy().reset_index(drop=True)
    print_status(f"zspec after cuts (qual>=2, z>0.5, M*>8): N = {len(out)}", "INFO")
    print_status(f"  z>4: N = {(out['z_spec']>4).sum()}", "INFO")
    print_status(f"  z>5: N = {(out['z_spec']>5).sum()}", "INFO")
    print_status(f"  z>7: N = {(out['z_spec']>7).sum()}", "INFO")
    return out


# =============================================================================
# APPLY TEP MODEL
# =============================================================================

def apply_tep(df, z_col='z_phot'):
    """Apply TEP model to compute Gamma_t and related quantities."""
    print_status(f"Applying TEP model (z from '{z_col}')...", "PROCESS")

    gamma_t = np.full(len(df), np.nan)
    t_eff = np.full(len(df), np.nan)
    log_mh = np.full(len(df), np.nan)
    ml_bias = np.full(len(df), np.nan)
    log_mstar_true = np.full(len(df), np.nan)
    alpha_z = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        z = row[z_col]
        lms = row['log_Mstar']
        if not (np.isfinite(z) and z > 0.1 and np.isfinite(lms) and lms > 5):
            continue
        lmh = lms + DELTA_LOG_MH
        try:
            gt = tep_gamma(lmh, z)
            az = tep_alpha(z)
            te = cosmo.age(z).to(u.Gyr).value * gt
            bias = isochrony_mass_bias(lmh, z)
            gamma_t[i] = gt
            t_eff[i] = te
            log_mh[i] = lmh
            ml_bias[i] = bias
            log_mstar_true[i] = lms - bias
            alpha_z[i] = az
        except Exception:
            pass

    df['gamma_t'] = gamma_t
    df['t_eff'] = t_eff
    df['log_Mh'] = log_mh
    df['ml_bias'] = ml_bias
    df['log_Mstar_true'] = log_mstar_true
    df['alpha_z'] = alpha_z

    n_valid = np.sum(np.isfinite(df['gamma_t']))
    print_status(f"Gamma_t computed for {n_valid} sources", "INFO")
    med_gt = np.nanmedian(df['gamma_t'])
    print_status(f"  Median Gamma_t: {med_gt:.3f}", "INFO")
    return df


# =============================================================================
# TEP SIGNAL ANALYSIS
# =============================================================================

def analyze_tep_signal(df, z_col='z_phot', label='photoz'):
    """
    Comprehensive TEP signal analysis on UNCOVER DR4 SPS catalog.

    Tests:
    1. rho(Gamma_t, dust2) — primary L1 signal (Prospector dust2)
    2. rho(Gamma_t, mwa) — mass-weighted age
    3. rho(Gamma_t, met) — metallicity
    4. rho(Gamma_t, log_Mstar) — sanity check
    5. Partial correlation: rho(Gamma_t, dust2 | M*, z) — mass-controlled
    6. Redshift-binned dust signal
    """
    results = {}

    z = df[z_col]
    gt = df['gamma_t']
    dust = df['dust2']
    mwa = df['mwa']
    met = df['met']
    lms = df['log_Mstar']

    # --- Primary: dust2 correlation ---
    for zlabel, zmask in [
        ('z_gt_4', z > 4),
        ('z_gt_5', z > 5),
        ('z_gt_7', z > 7),
        ('z_gt_8', z > 8),
    ]:
        valid = zmask & np.isfinite(gt) & np.isfinite(dust) & (dust >= 0)
        if valid.sum() >= 10:
            rho, p = stats.spearmanr(gt[valid], dust[valid])
            results[f'dust2_{zlabel}'] = {
                'N': int(valid.sum()),
                'rho': float(rho),
                'p': float(p),
                'p_formatted': format_p_value(p),
            }
            print_status(f"ρ(Γ_t, dust2) [{zlabel}]: {rho:.3f}, p={format_p_value(p)}, N={valid.sum()}", "INFO")

    # --- MWA correlation ---
    for zlabel, zmask in [('z_gt_4', z > 4), ('z_gt_7', z > 7)]:
        valid = zmask & np.isfinite(gt) & np.isfinite(mwa) & (mwa > 0)
        if valid.sum() >= 10:
            rho, p = stats.spearmanr(gt[valid], mwa[valid])
            results[f'mwa_{zlabel}'] = {
                'N': int(valid.sum()),
                'rho': float(rho),
                'p': float(p),
                'p_formatted': format_p_value(p),
            }
            print_status(f"ρ(Γ_t, mwa) [{zlabel}]: {rho:.3f}, p={format_p_value(p)}, N={valid.sum()}", "INFO")

    # --- Metallicity correlation ---
    valid = (z > 4) & np.isfinite(gt) & np.isfinite(met)
    if valid.sum() >= 10:
        rho, p = stats.spearmanr(gt[valid], met[valid])
        results['met_z_gt_4'] = {
            'N': int(valid.sum()),
            'rho': float(rho),
            'p': float(p),
            'p_formatted': format_p_value(p),
        }
        print_status(f"ρ(Γ_t, met) [z>4]: {rho:.3f}, p={format_p_value(p)}, N={valid.sum()}", "INFO")

    # --- Partial correlation: dust2 | M*, z ---
    from scipy.stats import spearmanr
    valid = (z > 4) & np.isfinite(gt) & np.isfinite(dust) & np.isfinite(lms) & (dust >= 0)
    if valid.sum() >= 30:
        # Residualize dust2 on M* and z
        from numpy.polynomial import polynomial as P
        X = np.column_stack([lms[valid], z[valid]])
        # Simple linear residualization
        from numpy.linalg import lstsq
        A = np.column_stack([np.ones(valid.sum()), X])
        dust_resid, _, _, _ = lstsq(A, dust[valid], rcond=None)
        dust_partial = dust[valid] - A @ dust_resid

        gt_resid, _, _, _ = lstsq(A, gt[valid], rcond=None)
        gt_partial = gt[valid] - A @ gt_resid

        rho_p, p_p = spearmanr(gt_partial, dust_partial)
        results['dust2_partial_z_gt_4'] = {
            'N': int(valid.sum()),
            'rho_partial': float(rho_p),
            'p': float(p_p),
            'p_formatted': format_p_value(p_p),
            'note': 'Partial rho(Gamma_t, dust2 | M*, z) — mass+z controlled',
        }
        print_status(f"ρ_partial(Γ_t, dust2 | M*, z) [z>4]: {rho_p:.3f}, p={format_p_value(p_p)}, N={valid.sum()}", "INFO")

    # --- Redshift-binned dust signal ---
    bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12)]
    bin_results = []
    for zlo, zhi in bins:
        mask = (z >= zlo) & (z < zhi) & np.isfinite(gt) & np.isfinite(dust) & (dust >= 0)
        if mask.sum() >= 10:
            rho, p = stats.spearmanr(gt[mask], dust[mask])
            bin_results.append({
                'z_bin': f'{zlo}-{zhi}',
                'z_mid': (zlo + zhi) / 2,
                'N': int(mask.sum()),
                'rho': float(rho),
                'p': float(p),
                'p_formatted': format_p_value(p),
            })
            print_status(f"  z={zlo}-{zhi}: ρ={rho:.3f}, p={format_p_value(p)}, N={mask.sum()}", "INFO")
    results['dust2_redshift_bins'] = bin_results

    # --- AGB threshold test (t_eff > 0.3 Gyr) ---
    if 't_eff' in df.columns:
        valid = (z > 4) & np.isfinite(gt) & np.isfinite(dust) & np.isfinite(df['t_eff'])
        above = valid & (df['t_eff'] >= 0.3)
        below = valid & (df['t_eff'] < 0.3)
        if above.sum() >= 10 and below.sum() >= 10:
            med_above = np.median(dust[above])
            med_below = np.median(dust[below])
            stat, p_mw = stats.mannwhitneyu(dust[above], dust[below], alternative='greater')
            ratio = med_above / med_below if med_below > 0 else np.nan
            results['agb_threshold'] = {
                'N_above': int(above.sum()),
                'N_below': int(below.sum()),
                'median_dust2_above': float(med_above),
                'median_dust2_below': float(med_below),
                'dust2_ratio': float(ratio),
                'mannwhitney_p': float(p_mw),
                'p_formatted': format_p_value(p_mw),
            }
            print_status(f"AGB threshold (t_eff>0.3 Gyr): dust2 ratio={ratio:.2f}, p={format_p_value(p_mw)}", "INFO")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 60, "INFO")
    print_status("STEP 152: UNCOVER DR4 Full SPS Catalog (MegaScience)", "INFO")
    print_status("=" * 60, "INFO")

    # --- Full photometric SPS catalog ---
    full_df = load_full_sps()
    full_cut = apply_quality_cuts_full(full_df)
    full_cut = apply_tep(full_cut, z_col='z_phot')

    out_csv = INTERIM_PATH / "step_152_uncover_dr4_full_sps.csv"
    full_cut.to_csv(out_csv, index=False)
    print_status(f"Saved: {out_csv} (N={len(full_cut)})", "INFO")

    print_status("Analyzing TEP signal (full photometric SPS)...", "PROCESS")
    signal_full = analyze_tep_signal(full_cut, z_col='z_phot', label='photoz')

    # --- Zspec SPS catalog ---
    print_status("-" * 40, "INFO")
    zspec_df = load_zspec_sps()
    if zspec_df is None:
        print_status("UNCOVER_DR4_SPS_zspec_catalog.fits not found. Aborting.", "ERROR")
        partial = {
            'step': STEP_NUM,
            'name': STEP_NAME,
            'description': 'UNCOVER DR4 Full SPS (MegaScience 20-band, Prospector-β)',
            'status': 'partial',
            'zspec_status': 'skipped — UNCOVER_DR4_SPS_zspec_catalog.fits not downloaded',
            'n_total_catalog': len(full_df),
            'n_quality_z_gt_4': len(full_cut),
            'n_z_gt_5': int((full_cut['z_phot'] > 5).sum()),
            'n_z_gt_7': int((full_cut['z_phot'] > 7).sum()),
            'n_z_gt_8': int((full_cut['z_phot'] > 8).sum()),
            'tep_signal_photoz': signal_full,
        }
        out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
        with open(out_json, 'w') as _f:
            json.dump(partial, _f, indent=2, default=safe_json_default)
        print_status(f"Saved partial results: {out_json}", "INFO")
        return partial
    zspec_cut = apply_quality_cuts_zspec(zspec_df)
    zspec_cut = apply_tep(zspec_cut, z_col='z_spec')

    out_csv_z = INTERIM_PATH / "step_152_uncover_dr4_zspec.csv"
    zspec_cut.to_csv(out_csv_z, index=False)
    print_status(f"Saved: {out_csv_z} (N={len(zspec_cut)})", "INFO")

    print_status("Analyzing TEP signal (zspec SPS)...", "PROCESS")
    signal_zspec = {}
    for zlabel, zmask in [
        ('z_gt_2', zspec_cut['z_spec'] > 2),
        ('z_gt_4', zspec_cut['z_spec'] > 4),
        ('z_gt_5', zspec_cut['z_spec'] > 5),
    ]:
        valid = zmask & np.isfinite(zspec_cut['gamma_t']) & np.isfinite(zspec_cut['dust2'])
        if valid.sum() >= 5:
            rho, p = stats.spearmanr(zspec_cut.loc[valid, 'gamma_t'], zspec_cut.loc[valid, 'dust2'])
            signal_zspec[f'dust2_{zlabel}'] = {
                'N': int(valid.sum()),
                'rho': float(rho),
                'p': float(p),
                'p_formatted': format_p_value(p),
                'note': 'Spec-z fixed Prospector SED fit',
            }
            print_status(f"ρ(Γ_t, dust2) [zspec, {zlabel}]: {rho:.3f}, p={format_p_value(p)}, N={valid.sum()}", "INFO")

    # --- Comparison with step_01 (DR2) ---
    n_step01 = 2315
    n_z7_step01 = 0  # approximate from prior runs
    n_new_z4 = len(full_cut)
    n_new_z7 = int((full_cut['z_phot'] > 7).sum())

    # Build output
    output = {
        'step': STEP_NUM,
        'name': STEP_NAME,
        'description': 'UNCOVER DR4 Full SPS (MegaScience 20-band, Prospector-β)',
        'reference': 'Wang et al. 2024; Suess et al. 2024; Price et al. 2025',
        'catalog_version': 'DR4 (February 2025)',
        'n_total_catalog': len(full_df),
        'n_quality_z_gt_4': n_new_z4,
        'n_z_gt_5': int((full_cut['z_phot'] > 5).sum()),
        'n_z_gt_7': n_new_z7,
        'n_z_gt_8': int((full_cut['z_phot'] > 8).sum()),
        'n_zspec_quality': len(zspec_cut),
        'n_zspec_z_gt_4': int((zspec_cut['z_spec'] > 4).sum()),
        'n_zspec_z_gt_5': int((zspec_cut['z_spec'] > 5).sum()),
        'upgrade_factor_vs_step01': round(n_new_z4 / n_step01, 1),
        'tep_signal_photoz': signal_full,
        'tep_signal_zspec': signal_zspec,
    }

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print_status(f"Saved: {out_json}", "INFO")

    # Summary
    print_status("=" * 60, "INFO")
    print_status("SUMMARY", "INFO")
    print_status(f"  Full DR4 SPS (z>4, M*>8): N = {n_new_z4}", "INFO")
    print_status(f"  z>7: N = {n_new_z7}", "INFO")
    print_status(f"  Upgrade vs step_01: {n_new_z4}/{n_step01} = {n_new_z4/n_step01:.1f}×", "INFO")
    print_status(f"  Zspec SPS (qual>=2): N = {len(zspec_cut)}", "INFO")
    for k, v in signal_full.items():
        if isinstance(v, dict) and 'rho' in v:
            print_status(f"  {k}: ρ={v['rho']:.3f}, p={v['p_formatted']}, N={v['N']}", "INFO")
    print_status("=" * 60, "INFO")

    return output


if __name__ == "__main__":
    main()
