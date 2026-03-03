#!/usr/bin/env python3
"""
Step 142: Blind Validation Protocol

Implements time-split and field-split blind validation using REAL survey
data from UNCOVER, CEERS, and COSMOS-Web to test whether TEP signatures
generalize to held-out data splits.

Data sources:
- UNCOVER: results/interim/step_02_uncover_full_sample_tep.csv (N=2315)
- CEERS: data/interim/ceers_z8_sample.csv (N=82, z>8 only)
- COSMOS-Web: data/interim/cosmosweb_z8_sample.csv (N=2606, z>8 only)
"""

import json
import numpy as np
np.random.seed(42)
import sys
from pathlib import Path
from scipy import stats
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import (
    compute_gamma_t as tep_compute_gamma_t,
    stellar_to_halo_mass as tep_stellar_to_halo_mass,
    ALPHA_0, ALPHA_UNCERTAINTY,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
INTERIM_DIR = PROJECT_ROOT / "results" / "interim"
DATA_DIR = PROJECT_ROOT / "data" / "interim"


def load_real_surveys():
    """
    Load REAL survey data from the pipeline's data files.
    Returns dict of {survey_name: DataFrame} with columns:
    z, log_Mstar, dust, gamma_t, ra, dec (where available).
    """
    surveys = {}

    # --- UNCOVER (full sample with TEP columns from step_02) ---
    uncover_path = INTERIM_DIR / "step_02_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        df = pd.read_csv(uncover_path)
        df = df.rename(columns={'z_phot': 'z'})
        df = df.dropna(subset=['z', 'log_Mstar', 'dust', 'gamma_t'])
        surveys['UNCOVER'] = df
        print(f"  UNCOVER: {len(df)} galaxies loaded (real data)")
    else:
        print(f"  WARNING: UNCOVER data not found at {uncover_path}")

    # --- CEERS z>8 sample ---
    ceers_path = DATA_DIR / "ceers_z8_sample.csv"
    if ceers_path.exists():
        df = pd.read_csv(ceers_path)
        df = df.rename(columns={'z_phot': 'z'})
        # Compute gamma_t for CEERS (not pre-computed)
        df['log_Mh'] = df.apply(
            lambda r: float(tep_stellar_to_halo_mass(r['log_Mstar'], r['z'])), axis=1
        )
        df['gamma_t'] = df.apply(
            lambda r: float(tep_compute_gamma_t(r['log_Mh'], r['z'], alpha_0=ALPHA_0)), axis=1
        )
        df = df.dropna(subset=['z', 'log_Mstar', 'dust', 'gamma_t'])
        surveys['CEERS'] = df
        print(f"  CEERS: {len(df)} galaxies loaded (real data)")
    else:
        print(f"  WARNING: CEERS data not found at {ceers_path}")

    # --- COSMOS-Web z>8 sample ---
    cw_path = DATA_DIR / "cosmosweb_z8_sample.csv"
    if cw_path.exists():
        df = pd.read_csv(cw_path)
        df = df.rename(columns={'z_phot': 'z'})
        # Compute gamma_t for COSMOS-Web (not pre-computed)
        df['log_Mh'] = df.apply(
            lambda r: float(tep_stellar_to_halo_mass(r['log_Mstar'], r['z'])), axis=1
        )
        df['gamma_t'] = df.apply(
            lambda r: float(tep_compute_gamma_t(r['log_Mh'], r['z'], alpha_0=ALPHA_0)), axis=1
        )
        df = df.dropna(subset=['z', 'log_Mstar', 'dust', 'gamma_t'])
        surveys['COSMOS-Web'] = df
        print(f"  COSMOS-Web: {len(df)} galaxies loaded (real data)")
    else:
        print(f"  WARNING: COSMOS-Web data not found at {cw_path}")

    return surveys


def compute_dust_gamma_correlation(df, z_min=8.0):
    """Compute dust-gamma_t Spearman correlation for z > z_min sample."""
    mask = df['z'] > z_min
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())

    rho, p = stats.spearmanr(df.loc[mask, 'gamma_t'], df.loc[mask, 'dust'])
    return float(rho), float(p), int(mask.sum())


def time_split_validation(df, train_frac=0.6):
    """Split by redshift: low-z train, high-z test."""
    df_sorted = df.sort_values('z').reset_index(drop=True)
    n_train = int(len(df_sorted) * train_frac)
    return df_sorted.iloc[:n_train], df_sorted.iloc[n_train:]


def field_split_validation(df):
    """Split by RA: East (RA<median) vs West (RA>=median)."""
    if 'ra' not in df.columns:
        # If no RA, split by index (first half / second half)
        mid = len(df) // 2
        return df.iloc[:mid], df.iloc[mid:]
    median_ra = df['ra'].median()
    return df[df['ra'] < median_ra], df[df['ra'] >= median_ra]


def run_analysis():
    """Run blind validation analysis on REAL survey data."""

    print("=" * 60)
    print("Step 142: Blind Validation Protocol (Real Data)")
    print("=" * 60)

    surveys = load_real_surveys()

    if not surveys:
        print("ERROR: No survey data loaded. Cannot run validation.")
        return None

    results = {'time_split': [], 'field_split': [], 'cross_survey': []}

    # 1. Time-split validation
    print("\n1. TIME-SPLIT VALIDATION")
    print("-" * 40)

    for name, df in surveys.items():
        train, test = time_split_validation(df)
        rho_train, p_train, n_train = compute_dust_gamma_correlation(train)
        rho_test, p_test, n_test = compute_dust_gamma_correlation(test)

        gen = 'success' if (not np.isnan(p_test) and p_test < 0.05) else 'fail'
        result = {
            'survey': name,
            'train_n': int(len(train)),
            'train_n_z8': n_train,
            'train_rho': rho_train if not np.isnan(rho_train) else None,
            'train_p': format_p_value(p_train),
            'test_n': int(len(test)),
            'test_n_z8': n_test,
            'test_rho': rho_test if not np.isnan(rho_test) else None,
            'test_p': format_p_value(p_test),
            'generalization': gen,
        }
        results['time_split'].append(result)
        rho_s = f"{rho_test:.3f}" if not np.isnan(rho_test) else "N/A"
        p_s = f"{p_test:.2e}" if not np.isnan(p_test) else "N/A"
        print(f"  {name}: test N_z8={n_test}, ρ={rho_s}, p={p_s} → {gen}")

    # 2. Field-split validation
    print("\n2. FIELD-SPLIT VALIDATION")
    print("-" * 40)

    for name, df in surveys.items():
        east, west = field_split_validation(df)
        rho_e, p_e, n_e = compute_dust_gamma_correlation(east)
        rho_w, p_w, n_w = compute_dust_gamma_correlation(west)

        consistent = 'consistent' if (
            not np.isnan(rho_e) and not np.isnan(rho_w)
            and np.sign(rho_e) == np.sign(rho_w)
        ) else 'inconsistent'

        result = {
            'survey': name,
            'east_n': int(len(east)), 'east_n_z8': n_e,
            'east_rho': rho_e if not np.isnan(rho_e) else None,
            'east_p': format_p_value(p_e),
            'west_n': int(len(west)), 'west_n_z8': n_w,
            'west_rho': rho_w if not np.isnan(rho_w) else None,
            'west_p': format_p_value(p_w),
            'consistency': consistent,
        }
        results['field_split'].append(result)
        re_s = f"{rho_e:.3f}" if not np.isnan(rho_e) else "N/A"
        rw_s = f"{rho_w:.3f}" if not np.isnan(rho_w) else "N/A"
        print(f"  {name}: East ρ={re_s}, West ρ={rw_s} → {consistent}")

    # 3. Cross-survey validation
    print("\n3. CROSS-SURVEY VALIDATION")
    print("-" * 40)

    survey_names = list(surveys.keys())
    for train_name in survey_names:
        test_names = [s for s in survey_names if s != train_name]
        test_df = pd.concat([surveys[s] for s in test_names], ignore_index=True)

        rho_train, p_train, n_train = compute_dust_gamma_correlation(surveys[train_name])
        rho_test, p_test, n_test = compute_dust_gamma_correlation(test_df)

        gen = 'success' if (not np.isnan(p_test) and p_test < 0.05) else 'fail'
        result = {
            'train_survey': train_name, 'test_surveys': test_names,
            'train_n_z8': n_train,
            'train_rho': rho_train if not np.isnan(rho_train) else None,
            'train_p': format_p_value(p_train),
            'test_n_z8': n_test,
            'test_rho': rho_test if not np.isnan(rho_test) else None,
            'test_p': format_p_value(p_test),
            'generalization': gen,
        }
        results['cross_survey'].append(result)
        rt_s = f"{rho_test:.3f}" if not np.isnan(rho_test) else "N/A"
        pt_s = f"{p_test:.2e}" if not np.isnan(p_test) else "N/A"
        print(f"  Train={train_name}, Test={test_names}: test ρ={rt_s}, p={pt_s} → {gen}")

    # Summary
    valid_time = [r for r in results['time_split'] if (r['test_n_z8'] or 0) >= 5]
    valid_field = [r for r in results['field_split']
                   if (r['east_n_z8'] or 0) >= 5 and (r['west_n_z8'] or 0) >= 5]
    valid_cross = [r for r in results['cross_survey'] if (r['test_n_z8'] or 0) >= 5]

    n_ts = sum(1 for r in valid_time if r['generalization'] == 'success')
    n_fs = sum(1 for r in valid_field if r['consistency'] == 'consistent')
    n_cs = sum(1 for r in valid_cross if r['generalization'] == 'success')

    total = len(valid_time) + len(valid_field) + len(valid_cross)
    total_ok = n_ts + n_fs + n_cs

    if total == 0:
        grade, interp = 'INSUFFICIENT', 'Not enough data for validation.'
    elif total_ok / total >= 0.8:
        grade = 'STRONG'
        interp = (f"TEP signatures generalize: {total_ok}/{total} validation tests passed. "
                  f"Time-split ({n_ts}/{len(valid_time)}), field-split ({n_fs}/{len(valid_field)}), "
                  f"cross-survey ({n_cs}/{len(valid_cross)}).")
    elif total_ok / total >= 0.5:
        grade = 'MODERATE'
        interp = f"Moderate generalization: {total_ok}/{total} tests passed."
    else:
        grade = 'WEAK'
        interp = f"Weak generalization: {total_ok}/{total} tests passed."

    summary = {
        'time_split': {'n_tests': len(valid_time), 'n_success': n_ts,
                        'success_rate': float(n_ts / len(valid_time)) if valid_time else 0},
        'field_split': {'n_tests': len(valid_field), 'n_consistent': n_fs,
                         'consistency_rate': float(n_fs / len(valid_field)) if valid_field else 0},
        'cross_survey': {'n_tests': len(valid_cross), 'n_success': n_cs,
                          'success_rate': float(n_cs / len(valid_cross)) if valid_cross else 0},
        'overall_validation': grade,
        'interpretation': interp,
    }

    print(f"\nOverall: {grade}")
    print(f"Interpretation: {interp}")

    output = {
        'step': 142,
        'description': 'Blind Validation Protocol (Real Survey Data)',
        'results': results,
        'summary': summary,
        'methodology': {
            'data': 'Real JWST survey data (UNCOVER DR4, CEERS, COSMOS-Web)',
            'time_split': 'Train on low-z (60%), test on high-z (40%)',
            'field_split': 'Split by RA median into two sky halves',
            'cross_survey': 'Train on one survey, test on remaining surveys',
            'success_criterion': 'p < 0.05 on held-out data',
        }
    }

    output_path = RESULTS_DIR / "step_142_blind_validation.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_analysis()
