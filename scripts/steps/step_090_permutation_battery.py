#!/usr/bin/env python3
""" 
Step 111: Permutation Battery (Empirical p-values)

Computes empirical (permutation) p-values for key z>8 dust tests.

Nulls implemented:
- Simple shuffle of dust values.
- Stratified shuffles to preserve potential confounds:
  - within mass bins
  - within redshift bins
  - within (mass, redshift) bins

Outputs:
- results/outputs/step_090_permutation_battery.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like  # TEP model: Gamma_t formula, stellar-to-halo mass from abundance matching

STEP_NUM = "090"  # Pipeline step number (sequential 001-176)
STEP_NAME = "permutation_battery"  # Permutation battery: empirical p-values via stratified shuffles (simple, mass bins, redshift bins, mass×redshift bins) for z>8 dust tests

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products (CSV format from prior steps)

for p in [LOGS_PATH, OUTPUT_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


def _safe_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _fisher_z_from_rho(rho):
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(np.arctanh(rho))


def load_survey_data():
    surveys = {}

    uncover_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        df = pd.read_csv(uncover_path)
        df['survey'] = 'UNCOVER'
        surveys['UNCOVER'] = df

    ceers_path = DATA_INTERIM_PATH / "ceers_z8_sample.csv"
    if ceers_path.exists():
        df = pd.read_csv(ceers_path)
        df['survey'] = 'CEERS'
        surveys['CEERS'] = df

    cosmosweb_path = DATA_INTERIM_PATH / "cosmosweb_z8_sample.csv"
    if cosmosweb_path.exists():
        df = pd.read_csv(cosmosweb_path)
        df['survey'] = 'COSMOS-Web'
        surveys['COSMOS-Web'] = df

    return surveys


def ensure_gamma_t(df):
    df = df.copy()

    if 'z_phot' not in df.columns:
        if 'z' in df.columns:
            df['z_phot'] = df['z']
        elif 'redshift' in df.columns:
            df['z_phot'] = df['redshift']
        else:
            return df

    if 'log_Mstar' not in df.columns:
        if 'mass' in df.columns:
            df['log_Mstar'] = df['mass']
        else:
            return df

    if 'gamma_t' in df.columns and bool(df['gamma_t'].notna().any()):
        return df

    z_phot = df['z_phot'].astype(float).to_numpy()

    if 'log_Mh' in df.columns:
        df['log_Mh'] = pd.to_numeric(df['log_Mh'], errors='coerce')
    else:
        df['log_Mh'] = np.nan

    mh = df['log_Mh'].astype(float).to_numpy()
    missing = np.isnan(mh)
    if np.any(missing):
        mstar = df['log_Mstar'].astype(float).to_numpy()
        mh[missing] = stellar_to_halo_mass_behroozi_like(mstar[missing], z_phot[missing])
        df['log_Mh'] = mh

    df['gamma_t'] = tep_gamma(df['log_Mh'].astype(float).to_numpy(), z_phot)
    return df


def _prepare_arrays(df, z_min, dust_positive_only):
    df = df.copy()

    if 'z_phot' not in df.columns:
        return None

    df = df[df['z_phot'] > float(z_min)]

    required = ['gamma_t', 'dust', 'log_Mstar', 'z_phot']
    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=required)

    if dust_positive_only:
        df = df[df['dust'].astype(float) > 0]

    if len(df) < 10:
        return None

    x = df['gamma_t'].astype(float).to_numpy()
    y = df['dust'].astype(float).to_numpy()
    m = df['log_Mstar'].astype(float).to_numpy()
    z = df['z_phot'].astype(float).to_numpy()

    return {
        'x': x,
        'y': y,
        'mass': m,
        'z': z,
        'n': int(len(df)),
    }


def _centered_ranks(arr):
    r = stats.rankdata(arr, method='average').astype(float)
    r -= np.mean(r)
    return r


def _spearman_rho_from_centered_ranks(rx_c, ry_c):
    denom = float(np.sqrt(np.sum(rx_c**2) * np.sum(ry_c**2)))
    if denom <= 0:
        return None
    return _safe_float(float(np.sum(rx_c * ry_c) / denom))


def _stratify_groups(mass, z, mode, mass_bin_width, z_bin_width):
    n = len(mass)

    if mode == 'none':
        return [np.arange(n, dtype=int)]

    if mode == 'mass':
        m0 = float(np.floor(np.nanmin(mass) / mass_bin_width) * mass_bin_width)
        bins = np.floor((mass - m0) / float(mass_bin_width)).astype(int)
        groups = {}
        for i, b in enumerate(bins):
            groups.setdefault(int(b), []).append(i)
        return [np.array(v, dtype=int) for v in groups.values()]

    if mode == 'z':
        z0 = float(np.floor(np.nanmin(z) / z_bin_width) * z_bin_width)
        bins = np.floor((z - z0) / float(z_bin_width)).astype(int)
        groups = {}
        for i, b in enumerate(bins):
            groups.setdefault(int(b), []).append(i)
        return [np.array(v, dtype=int) for v in groups.values()]

    if mode == 'mass_z':
        m0 = float(np.floor(np.nanmin(mass) / mass_bin_width) * mass_bin_width)
        z0 = float(np.floor(np.nanmin(z) / z_bin_width) * z_bin_width)
        mb = np.floor((mass - m0) / float(mass_bin_width)).astype(int)
        zb = np.floor((z - z0) / float(z_bin_width)).astype(int)
        groups = {}
        for i, (b1, b2) in enumerate(zip(mb, zb)):
            key = (int(b1), int(b2))
            groups.setdefault(key, []).append(i)
        return [np.array(v, dtype=int) for v in groups.values()]

    return [np.arange(n, dtype=int)]


def _permute_within_groups(rng, y_c, groups, out):
    for idx in groups:
        if len(idx) <= 1:
            out[idx] = y_c[idx]
            continue
        perm = rng.permutation(idx)
        out[idx] = y_c[perm]


def permutation_test_one_dataset(x, y, mass, z, n_perm, seed, stratify_mode, mass_bin_width, z_bin_width):
    rx_c = _centered_ranks(x)
    ry_c = _centered_ranks(y)

    rho_obs = _spearman_rho_from_centered_ranks(rx_c, ry_c)
    if rho_obs is None:
        return None

    z_obs = _fisher_z_from_rho(rho_obs)

    denom = float(np.sqrt(np.sum(rx_c**2) * np.sum(ry_c**2)))
    if denom <= 0:
        return None

    groups = _stratify_groups(mass, z, stratify_mode, mass_bin_width, z_bin_width)

    rng = np.random.default_rng(int(seed))
    y_work = np.empty_like(ry_c)

    z_perm = np.zeros(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        if stratify_mode == 'none':
            perm = rng.permutation(len(ry_c))
            rho_p = float(np.sum(rx_c * ry_c[perm]) / denom)
        else:
            _permute_within_groups(rng, ry_c, groups, y_work)
            rho_p = float(np.sum(rx_c * y_work) / denom)
        rho_p = float(np.clip(rho_p, -0.999999, 0.999999))
        z_perm[i] = float(np.arctanh(rho_p))

    if z_obs >= 0:
        p_one = float((np.sum(z_perm >= z_obs) + 1.0) / (len(z_perm) + 1.0))
    else:
        p_one = float((np.sum(z_perm <= z_obs) + 1.0) / (len(z_perm) + 1.0))
    p_two = float((np.sum(np.abs(z_perm) >= abs(z_obs)) + 1.0) / (len(z_perm) + 1.0))

    qs = np.quantile(z_perm, [0.01, 0.05, 0.5, 0.95, 0.99])

    return {
        'rho_observed': _safe_float(rho_obs),
        'z_observed': _safe_float(z_obs),
        'n_perm': int(n_perm),
        'seed': int(seed),
        'stratify': stratify_mode,
        'p_empirical_one_sided': _safe_float(p_one),
        'p_empirical_two_sided': _safe_float(p_two),
        'null_z_mean': _safe_float(float(np.mean(z_perm))),
        'null_z_std': _safe_float(float(np.std(z_perm))),
        'null_z_quantiles': {
            'q01': _safe_float(qs[0]),
            'q05': _safe_float(qs[1]),
            'q50': _safe_float(qs[2]),
            'q95': _safe_float(qs[3]),
            'q99': _safe_float(qs[4]),
        },
    }


def combine_meta_fixed_from_z(per_survey):
    items = []
    for name, payload in per_survey.items():
        z_val = payload.get('z_observed')
        n = payload.get('n')
        if z_val is None or n is None:
            continue
        w = float(max(1.0, float(n) - 3.0))
        items.append((name, float(z_val), w, int(n)))

    if len(items) == 0:
        return None

    z_vals = np.array([t[1] for t in items], dtype=float)
    weights = np.array([t[2] for t in items], dtype=float)

    z_fixed = float(np.sum(weights * z_vals) / np.sum(weights))
    se_fixed = float(1.0 / np.sqrt(np.sum(weights)))

    rho_fixed = float(np.tanh(z_fixed))
    ci = (float(np.tanh(z_fixed - 1.96 * se_fixed)), float(np.tanh(z_fixed + 1.96 * se_fixed)))

    return {
        'z_fixed': _safe_float(z_fixed),
        'rho_fixed': _safe_float(rho_fixed),
        'se_z_fixed': _safe_float(se_fixed),
        'ci_rho_lower': _safe_float(ci[0]),
        'ci_rho_upper': _safe_float(ci[1]),
        'n_total': int(sum(t[3] for t in items)),
        'k': int(len(items)),
    }


def permutation_meta_fixed(surveys_data, n_perm, seed, stratify_mode, mass_bin_width, z_bin_width):
    rng = np.random.default_rng(int(seed))

    pre = {}
    for name, dat in surveys_data.items():
        x = dat['x']
        y = dat['y']
        rx_c = _centered_ranks(x)
        ry_c = _centered_ranks(y)
        denom = float(np.sqrt(np.sum(rx_c**2) * np.sum(ry_c**2)))
        if denom <= 0:
            return None

        groups = _stratify_groups(dat['mass'], dat['z'], stratify_mode, mass_bin_width, z_bin_width)

        pre[name] = {
            'rx_c': rx_c,
            'ry_c': ry_c,
            'denom': denom,
            'groups': groups,
            'n': int(dat['n']),
        }

    weights = {name: float(max(1.0, pre[name]['n'] - 3.0)) for name in pre.keys()}
    w_sum = float(np.sum(list(weights.values())))
    if w_sum <= 0:
        return None

    z_obs_parts = {}
    for name, dat in surveys_data.items():
        obs = _spearman_rho_from_centered_ranks(pre[name]['rx_c'], pre[name]['ry_c'])
        if obs is None:
            return None
        z_obs_parts[name] = _fisher_z_from_rho(obs)

    z_obs_fixed = float(np.sum([weights[name] * z_obs_parts[name] for name in pre.keys()]) / w_sum)

    z_perm_fixed = np.zeros(int(n_perm), dtype=float)
    y_work = {name: np.empty_like(pre[name]['ry_c']) for name in pre.keys()}

    for i in range(int(n_perm)):
        z_parts = []
        for name in pre.keys():
            rx_c = pre[name]['rx_c']
            ry_c = pre[name]['ry_c']
            denom = pre[name]['denom']

            if stratify_mode == 'none':
                perm = rng.permutation(len(ry_c))
                rho_p = float(np.sum(rx_c * ry_c[perm]) / denom)
            else:
                _permute_within_groups(rng, ry_c, pre[name]['groups'], y_work[name])
                rho_p = float(np.sum(rx_c * y_work[name]) / denom)

            rho_p = float(np.clip(rho_p, -0.999999, 0.999999))
            z_parts.append(float(np.arctanh(rho_p)))

        z_parts = np.array(z_parts, dtype=float)
        z_perm_fixed[i] = float(np.sum([weights[name] * z_parts[j] for j, name in enumerate(pre.keys())]) / w_sum)

    if z_obs_fixed >= 0:
        p_one = float((np.sum(z_perm_fixed >= z_obs_fixed) + 1.0) / (len(z_perm_fixed) + 1.0))
    else:
        p_one = float((np.sum(z_perm_fixed <= z_obs_fixed) + 1.0) / (len(z_perm_fixed) + 1.0))
    p_two = float((np.sum(np.abs(z_perm_fixed) >= abs(z_obs_fixed)) + 1.0) / (len(z_perm_fixed) + 1.0))

    qs = np.quantile(z_perm_fixed, [0.01, 0.05, 0.5, 0.95, 0.99])

    return {
        'z_fixed_observed': _safe_float(z_obs_fixed),
        'rho_fixed_observed': _safe_float(float(np.tanh(z_obs_fixed))),
        'n_perm': int(n_perm),
        'seed': int(seed),
        'stratify': stratify_mode,
        'p_empirical_one_sided': _safe_float(p_one),
        'p_empirical_two_sided': _safe_float(p_two),
        'null_z_fixed_mean': _safe_float(float(np.mean(z_perm_fixed))),
        'null_z_fixed_std': _safe_float(float(np.std(z_perm_fixed))),
        'null_z_fixed_quantiles': {
            'q01': _safe_float(qs[0]),
            'q05': _safe_float(qs[1]),
            'q50': _safe_float(qs[2]),
            'q95': _safe_float(qs[3]),
            'q99': _safe_float(qs[4]),
        },
    }


def main():
    print_status(f"STEP {STEP_NUM}: Permutation Battery", "TITLE")

    config = {
        'z_min': 8.0,
        'n_perm': 2000,
        'seed': 42,
        'mass_bin_width_dex': 0.25,
        'z_bin_width': 0.5,
        'nulls': ['none', 'mass', 'z', 'mass_z'],
        'analyses': [
            {'name': 'dust_positive_only', 'dust_positive_only': True},
            {'name': 'all_dust', 'dust_positive_only': False},
        ],
    }

    surveys = load_survey_data()
    if not surveys:
        print_status("No survey data found.", "ERROR")
        return

    print_status(f"Surveys loaded: {list(surveys.keys())}", "INFO")

    results = {
        'config': config,
        'sources': {
            'UNCOVER': str(INTERIM_PATH / 'step_002_uncover_full_sample_tep.csv'),
            'CEERS': str(DATA_INTERIM_PATH / 'ceers_z8_sample.csv'),
            'COSMOS-Web': str(DATA_INTERIM_PATH / 'cosmosweb_z8_sample.csv'),
        },
        'analyses': {},
    }

    for analysis in config['analyses']:
        analysis_name = analysis['name']
        dust_positive_only = bool(analysis['dust_positive_only'])

        per_survey_data = {}
        per_survey_n = {}
        for name, df in surveys.items():
            df_std = ensure_gamma_t(df)
            dat = _prepare_arrays(df_std, config['z_min'], dust_positive_only)
            if dat is None:
                continue
            per_survey_data[name] = dat
            per_survey_n[name] = int(dat['n'])

        if len(per_survey_data) == 0:
            continue

        payload = {
            'per_survey_n': per_survey_n,
            'per_survey': {},
            'meta_fixed_observed': None,
            'meta_fixed_permutations': {},
        }

        per_survey_obs = {}
        for name, dat in per_survey_data.items():
            rx_c = _centered_ranks(dat['x'])
            ry_c = _centered_ranks(dat['y'])
            rho_obs = _spearman_rho_from_centered_ranks(rx_c, ry_c)
            if rho_obs is None:
                continue
            per_survey_obs[name] = {
                'rho_observed': _safe_float(rho_obs),
                'z_observed': _safe_float(_fisher_z_from_rho(rho_obs)),
                'n': int(dat['n']),
            }

        payload['meta_fixed_observed'] = combine_meta_fixed_from_z(per_survey_obs)

        for null_mode in config['nulls']:
            payload['per_survey'][null_mode] = {}
            for name, dat in per_survey_data.items():
                out = permutation_test_one_dataset(
                    dat['x'],
                    dat['y'],
                    dat['mass'],
                    dat['z'],
                    n_perm=int(config['n_perm']),
                    seed=int(config['seed']),
                    stratify_mode=str(null_mode),
                    mass_bin_width=float(config['mass_bin_width_dex']),
                    z_bin_width=float(config['z_bin_width']),
                )
                if out is not None:
                    out['n'] = int(dat['n'])
                payload['per_survey'][null_mode][name] = out

            meta_perm = permutation_meta_fixed(
                per_survey_data,
                n_perm=int(config['n_perm']),
                seed=int(config['seed']),
                stratify_mode=str(null_mode),
                mass_bin_width=float(config['mass_bin_width_dex']),
                z_bin_width=float(config['z_bin_width']),
            )
            payload['meta_fixed_permutations'][null_mode] = meta_perm

        results['analyses'][analysis_name] = payload

    out_json = OUTPUT_PATH / 'step_090_permutation_battery.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, allow_nan=False)

    print_status(f"Saved: {out_json}", "SUCCESS")


if __name__ == '__main__':
    main()
