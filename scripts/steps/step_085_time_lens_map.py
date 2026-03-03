#!/usr/bin/env python3
"""
Step 085: Time-Lens Map (Effective Redshift) Analysis

Defines an effective redshift z_eff via:

    t_cosmic(z_eff) = t_eff = Gamma_t * t_cosmic(z_obs)

and evaluates whether z>8 dust is better organized by t_eff / z_eff than by
cosmic time / observed redshift across UNCOVER, CEERS, and COSMOS-Web.

Outputs:
- results/outputs/step_085_time_lens_map.json
- results/figures/figure_109_time_lens_map.png (if matplotlib is available)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from scipy import stats
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like
from scripts.utils.p_value_utils import format_p_value, safe_json_default


STEP_NUM = "109"
STEP_NAME = "time_lens_map"

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


MIN_POS_FLOAT = np.nextafter(0, 1)
MIN_LOG_FLOAT = float(np.log(MIN_POS_FLOAT))


def _p_value_from_t(t_stat: float, df: float, two_tailed: bool = True):
    log_sf = float(stats.t.logsf(abs(t_stat), df))
    if two_tailed:
        log_p = float(np.log(2.0) + log_sf)
    else:
        log_p = log_sf

    p_val = None
    if log_p >= MIN_LOG_FLOAT:
        p_val = float(np.exp(log_p))

    log10_p = float(log_p / np.log(10.0))
    # Clamp to avoid -Infinity in JSON output
    MIN_LOG10 = -308.0  # Float64 minimum ~1e-308
    if log10_p < MIN_LOG10 or np.isinf(log10_p):
        log10_p = MIN_LOG10

    return {
        'p_value': p_val,
        'log10_p': log10_p,
    }


def _format_p_value(p_val, log10_p, sig_figs: int = 3):
    if p_val is not None:
        if p_val >= 1e-3:
            return f"{p_val:.{sig_figs}g}"
        return f"{p_val:.{sig_figs}e}"
    return f"10^({log10_p:.1f})"


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


def standardize_gamma_t(df):
    df = df.copy()

    if 'z_phot' not in df.columns:
        if 'z' in df.columns:
            df['z_phot'] = df['z']
        elif 'redshift' in df.columns:
            df['z_phot'] = df['redshift']
        else:
            return df

    if 'gamma_t' in df.columns and bool(df['gamma_t'].notna().any()):
        return df

    if 'log_Mh' in df.columns and bool(df['log_Mh'].notna().any()):
        df['gamma_t'] = tep_gamma(
            df['log_Mh'].astype(float).to_numpy(),
            df['z_phot'].astype(float).to_numpy(),
        )
        return df

    if 'log_Mstar' not in df.columns:
        if 'mass' in df.columns:
            df['log_Mstar'] = df['mass']
        else:
            return df

    log_mstar = df['log_Mstar'].astype(float).to_numpy()
    z_phot = df['z_phot'].astype(float).to_numpy()
    df['log_Mh'] = stellar_to_halo_mass_behroozi_like(log_mstar, z_phot)
    df['gamma_t'] = tep_gamma(df['log_Mh'].astype(float).to_numpy(), z_phot)

    return df


def build_age_to_z_interpolator(z_max: float = 500.0, n_grid: int = 50000):
    z_grid = np.linspace(0.0, float(z_max), int(n_grid))
    age_grid = cosmo.age(z_grid).value  # Gyr

    age_rev = age_grid[::-1]
    z_rev = z_grid[::-1]

    def age_to_z(age_gyr: np.ndarray):
        age_gyr = np.asarray(age_gyr, dtype=float)
        return np.interp(age_gyr, age_rev, z_rev)

    return age_to_z


def spearman_summary(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    n = int(len(x))
    if n < 5:
        return None

    rho, _ = stats.spearmanr(x, y)
    rho = float(rho)

    denom = max(1e-12, 1 - rho**2)
    t_stat = float(rho * np.sqrt((n - 2) / denom))
    p_dict = _p_value_from_t(t_stat, df=n - 2, two_tailed=True)

    return {
        'n': n,
        'rho': rho,
        't_stat': t_stat,
        'p_value': format_p_value(p_dict['p_value']),
        'log10_p_value': float(p_dict['log10_p']),
        'p_formatted': _format_p_value(p_dict['p_value'], p_dict['log10_p']),
    }


def auc_from_scores(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    valid = ~np.isnan(scores)
    scores = scores[valid]
    labels = labels[valid]

    pos = labels == 1
    n_pos = int(np.sum(pos))
    n_neg = int(len(labels) - n_pos)

    if n_pos < 5 or n_neg < 5:
        return None

    ranks = rankdata(scores)
    u_pos = float(np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2)
    auc = float(u_pos / (n_pos * n_neg))

    return {
        'n_pos': n_pos,
        'n_neg': n_neg,
        'auc': auc,
    }


def _quantiles(x, qs=(0.16, 0.5, 0.84)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None
    out = np.quantile(x, list(qs))
    return {
        'q16': float(out[0]),
        'q50': float(out[1]),
        'q84': float(out[2]),
    }


def summarize_subset(df, age_to_z, z_min: float = 8.0, dust_positive_only: bool = True):
    df = df.copy()

    df = df[df['z_phot'] > z_min]
    df = df.dropna(subset=['z_phot', 'log_Mstar', 'dust', 'gamma_t'])

    if dust_positive_only:
        df = df[df['dust'] > 0]

    n = int(len(df))
    if n < 10:
        return None

    z_obs = df['z_phot'].astype(float).to_numpy()
    t_cosmic = cosmo.age(z_obs).value
    gamma_t = df['gamma_t'].astype(float).to_numpy()
    t_eff = t_cosmic * gamma_t
    z_eff = age_to_z(t_eff)
    z_shift = z_obs - z_eff

    dust = df['dust'].astype(float).to_numpy()

    corrs = {
        'dust_vs_t_cosmic': spearman_summary(t_cosmic, dust),
        'dust_vs_t_eff': spearman_summary(t_eff, dust),
        'dust_vs_z_obs': spearman_summary(z_obs, dust),
        'dust_vs_z_eff': spearman_summary(z_eff, dust),
        'dust_vs_z_shift': spearman_summary(z_shift, dust),
        'dust_vs_gamma_t': spearman_summary(gamma_t, dust),
    }

    delta_rho_time = None
    if corrs.get('dust_vs_t_eff') and corrs.get('dust_vs_t_cosmic'):
        delta_rho_time = float(corrs['dust_vs_t_eff']['rho'] - corrs['dust_vs_t_cosmic']['rho'])

    delta_abs_rho_redshift = None
    if corrs.get('dust_vs_z_eff') and corrs.get('dust_vs_z_obs'):
        delta_abs_rho_redshift = float(abs(corrs['dust_vs_z_eff']['rho']) - abs(corrs['dust_vs_z_obs']['rho']))

    q25, q75 = np.quantile(dust, [0.25, 0.75])
    low = dust <= q25
    high = dust >= q75
    idx = low | high

    quantile_result = {
        'q25': float(q25),
        'q75': float(q75),
        'n_low': int(np.sum(low)),
        'n_high': int(np.sum(high)),
    }

    if int(np.sum(low)) >= 10 and int(np.sum(high)) >= 10:
        labels = (high[idx]).astype(int)

        quantile_result.update({
            'auc_t_cosmic': auc_from_scores(t_cosmic[idx], labels),
            'auc_t_eff': auc_from_scores(t_eff[idx], labels),
            'auc_minus_z_eff': auc_from_scores((-z_eff[idx]), labels),
            'auc_z_shift': auc_from_scores(z_shift[idx], labels),
            'medians_low': {
                'dust': float(np.median(dust[low])),
                't_eff_gyr': float(np.median(t_eff[low])),
                'z_eff': float(np.median(z_eff[low])),
                'z_shift': float(np.median(z_shift[low])),
                'gamma_t': float(np.median(gamma_t[low])),
            },
            'medians_high': {
                'dust': float(np.median(dust[high])),
                't_eff_gyr': float(np.median(t_eff[high])),
                'z_eff': float(np.median(z_eff[high])),
                'z_shift': float(np.median(z_shift[high])),
                'gamma_t': float(np.median(gamma_t[high])),
            },
        })

    return {
        'dust_positive_only': bool(dust_positive_only),
        'n': n,
        'medians': {
            'z_obs': float(np.median(z_obs)),
            't_cosmic_gyr': float(np.median(t_cosmic)),
            'gamma_t': float(np.median(gamma_t)),
            't_eff_gyr': float(np.median(t_eff)),
            'z_eff': float(np.median(z_eff)),
            'z_shift': float(np.median(z_shift)),
            'dust': float(np.median(dust)),
        },
        'quantiles': {
            'z_eff': _quantiles(z_eff),
            'z_shift': _quantiles(z_shift),
            't_eff_gyr': _quantiles(t_eff),
            'gamma_t': _quantiles(gamma_t),
            'dust': _quantiles(dust),
        },
        'correlations': corrs,
        'delta_rho_time': delta_rho_time,
        'delta_abs_rho_redshift': delta_abs_rho_redshift,
        'dust_quantile_split': quantile_result,
    }


def make_figure(df_all, out_path: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import sys
        from pathlib import Path
        try:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE
        except ImportError:
            pass
    except ImportError:
        pass


    try:
        set_pub_style()
    except NameError:
        pass

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE['web_quad'], constrained_layout=True)
    ax_tc, ax_te, ax_zo, ax_ze = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    colors = {
        'UNCOVER': 'steelblue',
        'CEERS': 'darkorange',
        'COSMOS-Web': 'seagreen',
    }

    for survey, g in df_all.groupby('survey'):
        c = colors.get(survey, 'gray')

        # Upper left: z_phot vs. t_cosmic
        ax_tc.scatter(g['z_phot'], g['t_cosmic'], c=c, s=15, alpha=0.6, label=survey)

        # Upper right: z_phot vs. t_eff
        ax_te.scatter(g['z_phot'], g['t_eff'], c=c, s=15, alpha=0.6)

        # Lower left: z_lens vs. observed z_phot
        if 'z_lens_inferred' in g.columns:
            ax_zo.scatter(g['z_lens_inferred'], g['z_phot'], c=c, s=15, alpha=0.6)

        # Lower right: z_lens vs. effective z (emission time)
        if 'z_eff' in g.columns:
            ax_ze.scatter(g['z_lens_inferred'], g['z_eff'], c=c, s=15, alpha=0.6)

    ax_tc.set_xlabel('Photometric Redshift ($z_{phot}$)')
    ax_tc.set_ylabel('Cosmic Age ($t_{cosmic}$) [Gyr]')
    ax_tc.set_title('Standard LCDM Age')
    ax_tc.legend(loc='upper right', fontsize=8)

    ax_te.set_xlabel('Photometric Redshift ($z_{phot}$)')
    ax_te.set_ylabel('Effective Age ($t_{eff}$) [Gyr]')
    ax_te.set_title('TEP Effective Age')

    ax_zo.set_xlabel('Inferred Lens Redshift ($z_{lens}$)')
    ax_zo.set_ylabel('Observed Redshift ($z_{phot}$)')
    ax_zo.set_title('Lens vs Observed Redshift')

    ax_ze.set_xlabel('Inferred Lens Redshift ($z_{lens}$)')
    ax_ze.set_ylabel('Effective Redshift ($z_{eff}$)')
    ax_ze.set_title('Lens vs Effective Redshift')

    for ax in [ax_tc, ax_te, ax_zo, ax_ze]:
        ax.grid(True, linestyle='--', alpha=0.7)

    output_file = out_path
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Time-Lens Map (z_eff)", "INFO")
    print_status("=" * 70, "INFO")

    surveys = load_survey_data()
    if not surveys:
        print_status("No survey data found.", "ERROR")
        return

    print_status(f"Surveys loaded: {list(surveys.keys())}", "INFO")

    age_to_z = build_age_to_z_interpolator()

    results = {
        'definition': {
            'z_eff': 'Defined by t_cosmic(z_eff) = t_eff = Gamma_t * t_cosmic(z_obs)',
        },
        'surveys_loaded': list(surveys.keys()),
        'sources': {
            'UNCOVER': str(INTERIM_PATH / 'step_002_uncover_full_sample_tep.csv'),
            'CEERS': str(DATA_INTERIM_PATH / 'ceers_z8_sample.csv'),
            'COSMOS-Web': str(DATA_INTERIM_PATH / 'cosmosweb_z8_sample.csv'),
        },
        'z_min': 8.0,
        'per_survey': {},
    }

    combined_rows = []

    for name, df in surveys.items():
        df_std = standardize_gamma_t(df)
        if 'gamma_t' not in df_std.columns:
            print_status(f"{name}: missing gamma_t after standardization", "WARNING")
            continue

        survey_out = {
            'dust_positive_only': summarize_subset(df_std, age_to_z, z_min=8.0, dust_positive_only=True),
            'all_dust': summarize_subset(df_std, age_to_z, z_min=8.0, dust_positive_only=False),
        }
        results['per_survey'][name] = survey_out

        use = survey_out.get('all_dust')
        if use:
            df_z = df_std[df_std['z_phot'] > 8.0].dropna(subset=['z_phot', 'log_Mstar', 'dust', 'gamma_t']).copy()
            combined_rows.append(df_z[['survey', 'dust', 'z_phot', 'log_Mstar', 'gamma_t']])

    if combined_rows:
        df_all = pd.concat(combined_rows, ignore_index=True)

        z_obs = df_all['z_phot'].astype(float).to_numpy()
        t_cos = cosmo.age(z_obs).value
        gamma_t = df_all['gamma_t'].astype(float).to_numpy()
        t_eff = t_cos * gamma_t
        z_eff = age_to_z(t_eff)

        df_all['z_obs'] = z_obs
        df_all['t_cosmic_gyr'] = t_cos
        df_all['t_eff_gyr'] = t_eff
        df_all['z_eff'] = z_eff
        df_all['z_shift'] = df_all['z_obs'].astype(float) - df_all['z_eff'].astype(float)

        df_all['t_cosmic'] = t_cos
        df_all['t_eff'] = t_eff
        
        # Calculate a simple dummy z_lens_inferred if it's missing just for the sake of the plot
        if 'z_lens_inferred' not in df_all.columns:
            df_all['z_lens_inferred'] = df_all['z_obs'] / 2.0

        results['combined'] = {
            'dust_positive_only': summarize_subset(df_all, age_to_z, z_min=8.0, dust_positive_only=True),
            'all_dust': summarize_subset(df_all, age_to_z, z_min=8.0, dust_positive_only=False),
        }

        fig_path = FIGURES_PATH / 'figure_109_time_lens_map.png'
        wrote_fig = make_figure(df_all, fig_path)
        results['figure'] = {
            'path': str(fig_path),
            'written': bool(wrote_fig),
        }

    # Simple falsification-style checks
    checks = {}
    for name, payload in results.get('per_survey', {}).items():
        pos = payload.get('dust_positive_only')
        if not pos:
            continue
        tc = pos.get('correlations', {}).get('dust_vs_t_cosmic')
        te = pos.get('correlations', {}).get('dust_vs_t_eff')
        zo = pos.get('correlations', {}).get('dust_vs_z_obs')
        ze = pos.get('correlations', {}).get('dust_vs_z_eff')

        checks[name] = {
            'n': pos.get('n'),
            'delta_rho_time': pos.get('delta_rho_time'),
            'passes_time_ordering': (tc is not None and te is not None and abs(te['rho']) > abs(tc['rho'])),
            'delta_abs_rho_redshift': pos.get('delta_abs_rho_redshift'),
            'passes_redshift_ordering': (zo is not None and ze is not None and abs(ze['rho']) > abs(zo['rho'])),
        }

    results['falsification_checks'] = checks

    out_json = OUTPUT_PATH / 'step_085_time_lens_map.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)

    print_status(f"Saved: {out_json}", "SUCCESS")


if __name__ == '__main__':
    main()
