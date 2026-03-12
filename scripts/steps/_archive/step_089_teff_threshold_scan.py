#!/usr/bin/env python3
"""
Step 089: t_eff Threshold Scan (AGB-timescale discovery)

Scans candidate t_eff thresholds and evaluates how well each threshold separates
high-z galaxies into dusty vs dust-poor subsets.

Primary metric: dust-detection odds ratio above vs below threshold.

Outputs:
- results/outputs/step_089_teff_threshold_scan.json
- results/figures/figure_089_teff_threshold_scan.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like

STEP_NUM = "089"
STEP_NAME = "teff_threshold_scan"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)


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


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _fisher_exact_greater(table_2x2):
    try:
        odds_ratio, p_value = stats.fisher_exact(table_2x2, alternative='greater')
    except TypeError:
        odds_ratio, p_value = stats.fisher_exact(table_2x2)
    return _safe_float(odds_ratio), _safe_float(p_value)


def _log_odds_ratio(det_above, non_above, det_below, non_below, correction=0.5):
    a1 = float(det_above) + float(correction)
    a0 = float(non_above) + float(correction)
    b1 = float(det_below) + float(correction)
    b0 = float(non_below) + float(correction)
    return _safe_float(np.log((a1 / a0) / (b1 / b0)))


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

    if 'log_Mh' in df.columns and bool(pd.to_numeric(df['log_Mh'], errors='coerce').notna().any()):
        log_mh = pd.to_numeric(df['log_Mh'], errors='coerce').astype(float).to_numpy()
        missing = np.isnan(log_mh)
        if np.any(missing):
            log_mstar = df['log_Mstar'].astype(float).to_numpy()
            log_mh[missing] = stellar_to_halo_mass_behroozi_like(log_mstar[missing], z_phot[missing])
        df['log_Mh'] = log_mh
    else:
        log_mstar = df['log_Mstar'].astype(float).to_numpy()
        df['log_Mh'] = stellar_to_halo_mass_behroozi_like(log_mstar, z_phot)

    df['gamma_t'] = tep_gamma(df['log_Mh'].astype(float).to_numpy(), z_phot)
    return df


def add_times(df):
    df = df.copy()
    df = df.dropna(subset=['z_phot', 'log_Mstar', 'dust', 'gamma_t'])
    z_phot = df['z_phot'].astype(float).to_numpy()
    t_cosmic = cosmo.age(z_phot).value
    gamma_t = df['gamma_t'].astype(float).to_numpy()

    df['t_cosmic_gyr'] = t_cosmic
    df['t_eff_gyr'] = t_cosmic * gamma_t
    return df


def threshold_scan(df, threshold_grid, z_min=8.0, min_group=10):
    df = df[df['z_phot'] > z_min].copy()
    df = df.dropna(subset=['dust', 't_eff_gyr'])

    n_total = int(len(df))
    if n_total < 2 * int(min_group):
        return None

    dust = df['dust'].astype(float).to_numpy()
    teff = df['t_eff_gyr'].astype(float).to_numpy()

    rows = []
    for thr in threshold_grid:
        thr = float(thr)
        above = teff > thr
        n_above = int(np.sum(above))
        n_below = int(len(teff) - n_above)

        det_above = int(np.sum(dust[above] > 0))
        non_above = int(n_above - det_above)
        det_below = int(np.sum(dust[~above] > 0))
        non_below = int(n_below - det_below)

        row = {
            'teff_threshold_gyr': thr,
            'n_above': n_above,
            'n_below': n_below,
            'det_above': det_above,
            'non_det_above': non_above,
            'det_below': det_below,
            'non_det_below': non_below,
        }

        if min(n_above, n_below) >= int(min_group):
            frac_above = det_above / max(n_above, 1)
            frac_below = det_below / max(n_below, 1)
            log_or = _log_odds_ratio(det_above, non_above, det_below, non_below)
            odds_ratio, p_fisher = _fisher_exact_greater([[det_above, non_above], [det_below, non_below]])

            row.update({
                'det_frac_above': _safe_float(frac_above),
                'det_frac_below': _safe_float(frac_below),
                'det_frac_delta': _safe_float(frac_above - frac_below),
                'odds_ratio': odds_ratio,
                'log_odds_ratio': log_or,
                'p_fisher_greater': p_fisher,
            })

            dust_above = dust[above]
            dust_below = dust[~above]
            if len(dust_above) >= int(min_group) and len(dust_below) >= int(min_group):
                try:
                    u_stat, p_u = stats.mannwhitneyu(dust_above, dust_below, alternative='greater')
                except TypeError:
                    u_stat, p_u = stats.mannwhitneyu(dust_above, dust_below)
                mean_above = float(np.mean(dust_above))
                mean_below = float(np.mean(dust_below))

                row.update({
                    'mean_dust_above': _safe_float(mean_above),
                    'mean_dust_below': _safe_float(mean_below),
                    'mean_ratio': _safe_float(mean_above / max(mean_below, 1e-12)),
                    'mannwhitney_u': _safe_float(u_stat),
                    'p_mannwhitney_greater': _safe_float(p_u),
                })

        rows.append(row)

    best = None
    valid = [r for r in rows if r.get('log_odds_ratio') is not None]
    if valid:
        best = max(valid, key=lambda r: r['log_odds_ratio'])

    return {
        'n': n_total,
        'z_min': float(z_min),
        'min_group': int(min_group),
        'scan': rows,
        'best_by_log_odds_ratio': best,
    }


def bootstrap_best_threshold(df, threshold_grid, z_min=8.0, min_group=10, n_boot=400, seed=42):
    df = df[df['z_phot'] > z_min].copy()
    df = df.dropna(subset=['dust', 't_eff_gyr'])

    n = int(len(df))
    if n < 2 * int(min_group):
        return None

    dust = df['dust'].astype(float).to_numpy()
    teff = df['t_eff_gyr'].astype(float).to_numpy()

    rng = np.random.default_rng(int(seed))
    best_thresholds = []

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        dust_b = dust[idx]
        teff_b = teff[idx]

        best_thr = None
        best_score = None

        for thr in threshold_grid:
            thr = float(thr)
            above = teff_b > thr
            n_above = int(np.sum(above))
            n_below = int(len(teff_b) - n_above)
            if min(n_above, n_below) < int(min_group):
                continue

            det_above = int(np.sum(dust_b[above] > 0))
            non_above = int(n_above - det_above)
            det_below = int(np.sum(dust_b[~above] > 0))
            non_below = int(n_below - det_below)

            score = _log_odds_ratio(det_above, non_above, det_below, non_below)
            if score is None:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_thr = thr

        if best_thr is not None:
            best_thresholds.append(float(best_thr))

    if len(best_thresholds) == 0:
        return None

    qs = np.quantile(best_thresholds, [0.16, 0.5, 0.84])
    return {
        'n_boot': int(n_boot),
        'seed': int(seed),
        'threshold_q16': _safe_float(qs[0]),
        'threshold_q50': _safe_float(qs[1]),
        'threshold_q84': _safe_float(qs[2]),
    }


def make_figure(results, out_path: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

        combined = results.get('combined')
        if not combined or not combined.get('scan'):
            return False

        scan = combined['scan']
        xs = [r['teff_threshold_gyr'] for r in scan if r.get('log_odds_ratio') is not None]
        ys = [r['log_odds_ratio'] for r in scan if r.get('log_odds_ratio') is not None]

        if len(xs) == 0:
            return False

        ax.plot(xs, ys, color='steelblue', lw=1.5)
        ax.set_xlabel('t_eff threshold (Gyr)')
        ax.set_ylabel('log odds ratio (dust detected above vs below)')
        ax.set_title('t_eff threshold scan (combined)')

        best = combined.get('best_by_log_odds_ratio')
        if best and best.get('teff_threshold_gyr') is not None:
            ax.axvline(best['teff_threshold_gyr'], color='darkorange', ls='--', lw=1.5)

        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return True
    except Exception:
        return False


def main():
    print_status(f"STEP {STEP_NUM}: t_eff Threshold Scan", "TITLE")

    config = {
        'z_min': 8.0,
        'threshold_min_gyr': 0.05,
        'threshold_max_gyr': 2.0,
        'threshold_step_gyr': 0.01,
        'min_group': 10,
        'bootstrap': {
            'n_boot': 400,
            'seed': 42,
        },
    }

    threshold_grid = np.arange(
        float(config['threshold_min_gyr']),
        float(config['threshold_max_gyr']) + 1e-12,
        float(config['threshold_step_gyr']),
    )

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
        'per_survey': {},
    }

    combined_rows = []

    for name, df in surveys.items():
        df_std = ensure_gamma_t(df)
        df_std = add_times(df_std)

        payload = {
            'threshold_scan': threshold_scan(
                df_std,
                threshold_grid,
                z_min=float(config['z_min']),
                min_group=int(config['min_group']),
            ),
            'bootstrap_best_threshold': bootstrap_best_threshold(
                df_std,
                threshold_grid,
                z_min=float(config['z_min']),
                min_group=int(config['min_group']),
                n_boot=int(config['bootstrap']['n_boot']),
                seed=int(config['bootstrap']['seed']),
            ),
        }
        results['per_survey'][name] = payload

        df_z = df_std[df_std['z_phot'] > float(config['z_min'])].dropna(subset=['z_phot', 'log_Mstar', 'dust', 'gamma_t', 't_eff_gyr']).copy()
        if len(df_z) > 0:
            combined_rows.append(df_z[['survey', 'z_phot', 'log_Mstar', 'dust', 'gamma_t', 't_cosmic_gyr', 't_eff_gyr']])

        scan_summary = payload.get('threshold_scan')
        best = scan_summary.get('best_by_log_odds_ratio') if scan_summary else None
        if best:
            print_status(
                f"{name}: best t_eff≈{best['teff_threshold_gyr']:.2f} Gyr, Δf_det={best.get('det_frac_delta', 0):+.2f}",
                "INFO",
            )

    if combined_rows:
        df_all = pd.concat(combined_rows, ignore_index=True)
        results['combined'] = threshold_scan(
            df_all,
            threshold_grid,
            z_min=float(config['z_min']),
            min_group=int(config['min_group']),
        )
        results['combined_bootstrap_best_threshold'] = bootstrap_best_threshold(
            df_all,
            threshold_grid,
            z_min=float(config['z_min']),
            min_group=int(config['min_group']),
            n_boot=int(config['bootstrap']['n_boot']),
            seed=int(config['bootstrap']['seed']),
        )

        fig_path = FIGURES_PATH / 'figure_089_teff_threshold_scan.png'
        wrote = make_figure(results, fig_path)
        results['figure'] = {
            'path': str(fig_path),
            'written': bool(wrote),
        }

    out_json = OUTPUT_PATH / 'step_089_teff_threshold_scan.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, allow_nan=False)

    print_status(f"Saved: {out_json}", "SUCCESS")


if __name__ == '__main__':
    main()
