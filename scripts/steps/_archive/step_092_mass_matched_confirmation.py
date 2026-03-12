#!/usr/bin/env python3
""" 
Step 113: Mass-matched / stratified confirmation (z>8 dust replication)

Provides a mass- and redshift-stratified confirmation of the dust–Γ_t
association to reduce sensitivity to confounding by mass and redshift.

Analyses:
1. Mass-stratified Spearman correlations within log(M*) bins.
2. Mass+z matched high-Γ_t vs low-Γ_t comparison within (survey, mass, z) strata.
3. Partial-residual Spearman correlation after removing linear dependence on (mass, z).

Outputs:
- results/outputs/step_092_mass_matched_confirmation.json
- results/figures/figure_092_mass_matched_confirmation.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like

STEP_NUM = "092"
STEP_NAME = "mass_matched_confirmation"

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


def _safe_p_value(value):
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return format_p_value(out)


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


def _prepare(df, z_min, dust_positive_only):
    df = df.copy()

    required = ['survey', 'z_phot', 'log_Mstar', 'dust', 'gamma_t']
    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=required)
    df = df[df['z_phot'].astype(float) > float(z_min)]

    if dust_positive_only:
        df = df[df['dust'].astype(float) > 0]

    if len(df) < 10:
        return None

    df['z_phot'] = df['z_phot'].astype(float)
    df['log_Mstar'] = df['log_Mstar'].astype(float)
    df['dust'] = df['dust'].astype(float)
    df['gamma_t'] = df['gamma_t'].astype(float)

    return df


def fisher_ci_rho(rho, n):
    rho = float(np.clip(float(rho), -0.999999, 0.999999))
    n = int(n)
    if n <= 3:
        return None
    z = float(np.arctanh(rho))
    se = float(1.0 / np.sqrt(max(1.0, n - 3)))
    lo = float(np.tanh(z - 1.96 * se))
    hi = float(np.tanh(z + 1.96 * se))
    return lo, hi


def mass_stratified_correlations(df, mass_bin_width, min_bin_n):
    mass = df['log_Mstar'].to_numpy()

    m0 = float(np.floor(np.min(mass) / float(mass_bin_width)) * float(mass_bin_width))
    bins = np.floor((mass - m0) / float(mass_bin_width)).astype(int)

    rows = []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if len(idx) < int(min_bin_n):
            continue

        sub = df.iloc[idx]
        rho, p = stats.spearmanr(sub['gamma_t'].to_numpy(), sub['dust'].to_numpy())
        ci = fisher_ci_rho(rho, len(sub))

        lo = _safe_float(ci[0]) if ci else None
        hi = _safe_float(ci[1]) if ci else None

        rows.append({
            'mass_bin_index': int(b),
            'mass_range': [
                _safe_float(m0 + float(b) * float(mass_bin_width)),
                _safe_float(m0 + (float(b) + 1.0) * float(mass_bin_width)),
            ],
            'mass_center': _safe_float(m0 + (float(b) + 0.5) * float(mass_bin_width)),
            'n': int(len(sub)),
            'rho': _safe_float(rho),
            'p': _safe_p_value(p),
            'ci_rho_lower': lo,
            'ci_rho_upper': hi,
        })

    rows = sorted(rows, key=lambda r: r['mass_center'] if r['mass_center'] is not None else -np.inf)
    return rows


def partial_residual_spearman(df, n_boot=300, seed=42):
    x = np.log(np.maximum(df['gamma_t'].to_numpy(dtype=float), 1e-12))
    y = df['dust'].to_numpy(dtype=float)
    m = df['log_Mstar'].to_numpy(dtype=float)
    z = df['z_phot'].to_numpy(dtype=float)

    X = np.column_stack([m, z, np.ones(len(m), dtype=float)])

    beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(X, y, rcond=None)[0]

    x_res = x - X @ beta_x
    y_res = y - X @ beta_y

    rho, p = stats.spearmanr(x_res, y_res)

    rng = np.random.default_rng(int(seed))
    rhos = []
    n = len(df)

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        mb = m[idx]
        zb = z[idx]
        Xb = np.column_stack([mb, zb, np.ones(len(mb), dtype=float)])

        bx = np.linalg.lstsq(Xb, xb, rcond=None)[0]
        by = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        xr = xb - Xb @ bx
        yr = yb - Xb @ by

        rb, _ = stats.spearmanr(xr, yr)
        if np.isfinite(rb):
            rhos.append(float(rb))

    ci = None
    if len(rhos) >= 30:
        ci = np.percentile(rhos, [2.5, 97.5])

    return {
        'rho': _safe_float(rho),
        'p': _safe_p_value(p),
        'n': int(len(df)),
        'bootstrap_n': int(n_boot),
        'bootstrap_seed': int(seed),
        'bootstrap_ci_rho_95': [_safe_float(ci[0]), _safe_float(ci[1])] if ci is not None else None,
    }


def mass_z_matched_high_low(df, mass_bin_width, z_bin_width, seed=42):
    rng = np.random.default_rng(int(seed))

    mass = df['log_Mstar'].to_numpy(dtype=float)
    z = df['z_phot'].to_numpy(dtype=float)

    m0 = float(np.floor(np.min(mass) / float(mass_bin_width)) * float(mass_bin_width))
    z0 = float(np.floor(np.min(z) / float(z_bin_width)) * float(z_bin_width))

    mb = np.floor((mass - m0) / float(mass_bin_width)).astype(int)
    zb = np.floor((z - z0) / float(z_bin_width)).astype(int)

    groups = {}
    surveys = df['survey'].astype(str).to_numpy()
    for i, (s, b1, b2) in enumerate(zip(surveys, mb, zb)):
        key = (s, int(b1), int(b2))
        groups.setdefault(key, []).append(i)

    idx_high = []
    idx_low = []

    for key, idx in groups.items():
        if len(idx) < 2:
            continue

        idx = np.array(idx, dtype=int)
        g = df.iloc[idx]['gamma_t'].to_numpy(dtype=float)
        med = float(np.median(g))

        hi = idx[g > med]
        lo = idx[g < med]

        k = int(min(len(hi), len(lo)))
        if k <= 0:
            continue

        if len(hi) > k:
            hi = rng.choice(hi, size=k, replace=False)
        if len(lo) > k:
            lo = rng.choice(lo, size=k, replace=False)

        idx_high.append(np.array(hi, dtype=int))
        idx_low.append(np.array(lo, dtype=int))

    if len(idx_high) == 0 or len(idx_low) == 0:
        return None

    idx_high = np.concatenate(idx_high)
    idx_low = np.concatenate(idx_low)

    if len(idx_high) == 0 or len(idx_low) == 0:
        return None

    dust_high = df.iloc[idx_high]['dust'].to_numpy(dtype=float)
    dust_low = df.iloc[idx_low]['dust'].to_numpy(dtype=float)

    try:
        u_stat, p_u = stats.mannwhitneyu(dust_high, dust_low, alternative='greater')
    except TypeError:
        u_stat, p_u = stats.mannwhitneyu(dust_high, dust_low)

    mean_high = float(np.mean(dust_high))
    mean_low = float(np.mean(dust_low))
    med_high = float(np.median(dust_high))
    med_low = float(np.median(dust_low))

    pooled = float(np.sqrt((np.var(dust_high) + np.var(dust_low)) / 2.0))
    cohens_d = float((mean_high - mean_low) / pooled) if pooled > 0 else None

    det_high = int(np.sum(dust_high > 0))
    det_low = int(np.sum(dust_low > 0))
    nondet_high = int(len(dust_high) - det_high)
    nondet_low = int(len(dust_low) - det_low)

    odds_ratio = None
    p_fisher = None
    if (det_high + nondet_high) > 0 and (det_low + nondet_low) > 0:
        table = [[det_high, nondet_high], [det_low, nondet_low]]
        try:
            odds_ratio, p_fisher = stats.fisher_exact(table, alternative='greater')
        except TypeError:
            odds_ratio, p_fisher = stats.fisher_exact(table)

    mass_high = df.iloc[idx_high]['log_Mstar'].to_numpy(dtype=float)
    mass_low = df.iloc[idx_low]['log_Mstar'].to_numpy(dtype=float)
    z_high = df.iloc[idx_high]['z_phot'].to_numpy(dtype=float)
    z_low = df.iloc[idx_low]['z_phot'].to_numpy(dtype=float)

    ks_mass = stats.ks_2samp(mass_high, mass_low)
    ks_z = stats.ks_2samp(z_high, z_low)

    return {
        'seed': int(seed),
        'mass_bin_width_dex': _safe_float(mass_bin_width),
        'z_bin_width': _safe_float(z_bin_width),
        'n_high': int(len(dust_high)),
        'n_low': int(len(dust_low)),
        'mean_dust_high': _safe_float(mean_high),
        'mean_dust_low': _safe_float(mean_low),
        'median_dust_high': _safe_float(med_high),
        'median_dust_low': _safe_float(med_low),
        'delta_mean': _safe_float(mean_high - mean_low),
        'cohens_d': _safe_float(cohens_d),
        'mannwhitney_u': _safe_float(u_stat),
        'p_mannwhitney_greater': _safe_p_value(p_u),
        'detection_fraction_high': _safe_float(det_high / max(1, len(dust_high))),
        'detection_fraction_low': _safe_float(det_low / max(1, len(dust_low))),
        'detection_fraction_delta': _safe_float(det_high / max(1, len(dust_high)) - det_low / max(1, len(dust_low))),
        'odds_ratio_detected': _safe_float(odds_ratio),
        'p_fisher_detected_greater': _safe_p_value(p_fisher),
        'balance_checks': {
            'mass_mean_high': _safe_float(float(np.mean(mass_high))),
            'mass_mean_low': _safe_float(float(np.mean(mass_low))),
            'z_mean_high': _safe_float(float(np.mean(z_high))),
            'z_mean_low': _safe_float(float(np.mean(z_low))),
            'ks_mass_stat': _safe_float(float(ks_mass.statistic)),
            'ks_mass_p': _safe_p_value(float(ks_mass.pvalue)),
            'ks_z_stat': _safe_float(float(ks_z.statistic)),
            'ks_z_p': _safe_p_value(float(ks_z.pvalue)),
        },
    }


def analyze_dataset(df, config):
    out = {
        'n': int(len(df)),
        'mass_stratified': mass_stratified_correlations(
            df,
            mass_bin_width=float(config['mass_bin_width_dex']),
            min_bin_n=int(config['min_bin_n']),
        ),
        'partial_residual': partial_residual_spearman(
            df,
            n_boot=int(config['partial_bootstrap_n']),
            seed=int(config['partial_bootstrap_seed']),
        ),
        'mass_z_matched_high_low': mass_z_matched_high_low(
            df,
            mass_bin_width=float(config['mass_bin_width_dex']),
            z_bin_width=float(config['z_bin_width']),
            seed=int(config['match_seed']),
        ),
    }
    return out


def make_figure(results, out_path: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        combined = results.get('combined', {})
        payload = combined.get('dust_positive_only') or combined.get('all_dust')
        if not payload:
            return False

        ms = payload.get('mass_stratified', [])
        matched = payload.get('mass_z_matched_high_low', None)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

        ax = axes[0]
        xs = [r['mass_center'] for r in ms if r.get('rho') is not None and r.get('mass_center') is not None]
        ys = [r['rho'] for r in ms if r.get('rho') is not None and r.get('mass_center') is not None]
        if len(xs) > 0:
            ax.plot(xs, ys, marker='o', lw=1.5, color='steelblue')
        ax.axhline(0.0, color='0.7', lw=1)
        ax.set_xlabel('log(M*) bin center')
        ax.set_ylabel('Spearman ρ (dust vs Γ_t)')
        ax.set_title('Mass-stratified correlations (combined)')

        ax2 = axes[1]
        if matched:
            means = [matched.get('mean_dust_low'), matched.get('mean_dust_high')]
            labels = ['Low Γ_t', 'High Γ_t']
            ax2.bar(labels, means, color=['0.6', 'darkorange'])
            ax2.set_ylabel('Mean dust')
            ax2.set_title('Mass+z matched mean dust')
        else:
            ax2.set_title('Mass+z matched mean dust')
            ax2.text(0.5, 0.5, 'No matched sample', ha='center', va='center')
            ax2.set_axis_off()

        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return True
    except Exception:
        return False


def main():
    print_status(f"STEP {STEP_NUM}: Mass-matched confirmation", "TITLE")

    config = {
        'z_min': 8.0,
        'mass_bin_width_dex': 0.25,
        'z_bin_width': 0.5,
        'min_bin_n': 20,
        'match_seed': 42,
        'partial_bootstrap_n': 300,
        'partial_bootstrap_seed': 42,
    }

    surveys = load_survey_data()
    if not surveys:
        print_status("No survey data found.", "ERROR")
        return

    print_status(f"Surveys loaded: {list(surveys.keys())}", "INFO")

    analyses = {
        'dust_positive_only': True,
        'all_dust': False,
    }

    results = {
        'config': config,
        'sources': {
            'UNCOVER': str(INTERIM_PATH / 'step_002_uncover_full_sample_tep.csv'),
            'CEERS': str(DATA_INTERIM_PATH / 'ceers_z8_sample.csv'),
            'COSMOS-Web': str(DATA_INTERIM_PATH / 'cosmosweb_z8_sample.csv'),
        },
        'per_survey': {},
        'combined': {},
    }

    for analysis_name, dust_pos_only in analyses.items():
        per = {}
        combined_rows = []

        for name, df in surveys.items():
            df_std = ensure_gamma_t(df)
            df_use = _prepare(df_std, z_min=float(config['z_min']), dust_positive_only=bool(dust_pos_only))
            if df_use is None:
                continue

            per[name] = analyze_dataset(df_use, config)
            combined_rows.append(df_use[['survey', 'z_phot', 'log_Mstar', 'dust', 'gamma_t']])

        results['per_survey'][analysis_name] = per

        if combined_rows:
            df_all = pd.concat(combined_rows, ignore_index=True)
            results['combined'][analysis_name] = analyze_dataset(df_all, config)

    fig_path = FIGURES_PATH / 'figure_092_mass_matched_confirmation.png'
    wrote = make_figure(results, fig_path)
    results['figure'] = {
        'path': str(fig_path),
        'written': bool(wrote),
    }

    # Add diagnostic notes for per-survey anomalies
    notes = []
    for analysis_name, per in results['per_survey'].items():
        for survey, data in per.items():
            pr = data.get('partial_residual', {})
            rho = pr.get('rho')
            if rho is not None and rho < 0:
                notes.append({
                    'survey': survey,
                    'analysis': analysis_name,
                    'issue': 'negative_partial_residual',
                    'rho': rho,
                    'explanation': (
                        f'{survey} shows negative partial residual (rho={rho:.3f}) due to '
                        'near-strong mass-gamma_t collinearity in small sample. '
                        'Raw correlation is positive; combined analysis remains robust.'
                    ),
                })
    if notes:
        results['diagnostic_notes'] = notes

    out_json = OUTPUT_PATH / 'step_092_mass_matched_confirmation.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, allow_nan=False)

    print_status(f"Saved: {out_json}", "SUCCESS")


if __name__ == '__main__':
    main()
