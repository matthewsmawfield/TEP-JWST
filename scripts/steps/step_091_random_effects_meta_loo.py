#!/usr/bin/env python3
""" 
Step 112: Random-effects meta-analysis + leave-one-out influence (z>8 dust replication)

Combines the cross-survey dust–Γ_t correlations using both fixed-effects and
random-effects (DerSimonian–Laird) models, and evaluates leave-one-out
sensitivity.

Outputs:
- results/outputs/step_091_random_effects_meta_loo.json
- results/figures/figure_091_random_effects_meta_loo.png
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
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300) & JSON serialiser for numpy types
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like  # TEP model: Gamma_t formula, stellar-to-halo mass from abundance matching

STEP_NUM = "091"  # Pipeline step number (sequential 001-176)
STEP_NAME = "random_effects_meta_loo"  # Random-effects meta-analysis: DerSimonian-Laird model combining dust-Gamma_t correlations across surveys with leave-one-out sensitivity analysis

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory (PNG/PDF for manuscript)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products (CSV format from prior steps)

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

MIN_POS_FLOAT = np.nextafter(0, 1)
MIN_LOG_FLOAT = float(np.log(MIN_POS_FLOAT))


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


def _p_value_from_z(z_stat):
    z_stat = float(z_stat)
    log_sf = float(stats.norm.logsf(abs(z_stat)))
    log_p = float(np.log(2.0) + log_sf)
    log_p_clamped = float(max(log_p, MIN_LOG_FLOAT))
    p_val = float(np.exp(log_p_clamped))
    log10_p = float(log_p_clamped / np.log(10.0))
    return p_val, log10_p


def fisher_z(rho):
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(np.arctanh(rho))


def inv_fisher_z(z):
    return float(np.tanh(float(z)))


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


def compute_survey_effect(df, z_min=8.0, dust_positive_only=True):
    df = df.copy()

    if 'z_phot' not in df.columns:
        return None

    df = df[df['z_phot'] > float(z_min)].copy()

    required = ['gamma_t', 'dust']
    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=required)

    if dust_positive_only:
        df = df[df['dust'].astype(float) > 0]

    if len(df) < 10:
        return None

    rho, _ = stats.spearmanr(df['gamma_t'].astype(float).to_numpy(), df['dust'].astype(float).to_numpy())
    n = int(len(df))

    z = fisher_z(rho)
    var = float(1.0 / max(1.0, n - 3.0))
    se = float(np.sqrt(var))

    ci_z = (float(z - 1.96 * se), float(z + 1.96 * se))
    ci_r = (inv_fisher_z(ci_z[0]), inv_fisher_z(ci_z[1]))

    return {
        'rho': _safe_float(rho),
        'n': int(n),
        'fisher_z': _safe_float(z),
        'var_z': _safe_float(var),
        'se_z': _safe_float(se),
        'ci_rho_lower': _safe_float(ci_r[0]),
        'ci_rho_upper': _safe_float(ci_r[1]),
    }


def fixed_effects_meta(effects):
    names = list(effects.keys())
    z_vals = np.array([effects[n]['fisher_z'] for n in names], dtype=float)
    var = np.array([effects[n]['var_z'] for n in names], dtype=float)

    w = 1.0 / var
    z_fixed = float(np.sum(w * z_vals) / np.sum(w))
    se_fixed = float(np.sqrt(1.0 / np.sum(w)))

    ci = (inv_fisher_z(z_fixed - 1.96 * se_fixed), inv_fisher_z(z_fixed + 1.96 * se_fixed))
    z_stat = float(z_fixed / se_fixed)
    p, log10_p = _p_value_from_z(z_stat)

    weights = {n: float(wi) for n, wi in zip(names, w)}
    weight_share = {n: float(wi / np.sum(w)) for n, wi in zip(names, w)}

    return {
        'k': int(len(names)),
        'n_total': int(sum(effects[n]['n'] for n in names)),
        'z_fixed': _safe_float(z_fixed),
        'rho_fixed': _safe_float(inv_fisher_z(z_fixed)),
        'se_z_fixed': _safe_float(se_fixed),
        'ci_rho_lower': _safe_float(ci[0]),
        'ci_rho_upper': _safe_float(ci[1]),
        'z_stat': _safe_float(z_stat),
        'p': _safe_float(p),
        'log10_p': _safe_float(log10_p),
        'weights': weights,
        'weight_share': weight_share,
    }


def heterogeneity(effects, fixed):
    names = list(effects.keys())
    z_vals = np.array([effects[n]['fisher_z'] for n in names], dtype=float)
    var = np.array([effects[n]['var_z'] for n in names], dtype=float)

    w = 1.0 / var
    z_fixed = float(fixed['z_fixed'])

    Q = float(np.sum(w * (z_vals - z_fixed) ** 2))
    df = int(len(names) - 1)
    p_Q = float(stats.chi2.sf(Q, df)) if df > 0 else None

    I2 = 0.0
    if Q > 0 and df > 0:
        I2 = float(max(0.0, (Q - df) / Q))

    C = float(np.sum(w) - (np.sum(w**2) / np.sum(w))) if np.sum(w) > 0 else 0.0
    tau2 = 0.0
    if C > 0 and df > 0:
        tau2 = float(max(0.0, (Q - df) / C))

    return {
        'Q': _safe_float(Q),
        'df': int(df),
        'p_Q': _safe_float(p_Q),
        'I2': _safe_float(I2),
        'I2_pct': _safe_float(I2 * 100.0),
        'tau2': _safe_float(tau2),
    }


def random_effects_meta(effects, tau2):
    names = list(effects.keys())
    z_vals = np.array([effects[n]['fisher_z'] for n in names], dtype=float)
    var = np.array([effects[n]['var_z'] for n in names], dtype=float)

    w_re = 1.0 / (var + float(tau2))
    z_re = float(np.sum(w_re * z_vals) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))

    ci = (inv_fisher_z(z_re - 1.96 * se_re), inv_fisher_z(z_re + 1.96 * se_re))
    z_stat = float(z_re / se_re)
    p, log10_p = _p_value_from_z(z_stat)

    weights = {n: float(wi) for n, wi in zip(names, w_re)}
    weight_share = {n: float(wi / np.sum(w_re)) for n, wi in zip(names, w_re)}

    pred_se = float(np.sqrt(se_re**2 + float(tau2)))
    pred_ci = (inv_fisher_z(z_re - 1.96 * pred_se), inv_fisher_z(z_re + 1.96 * pred_se))

    return {
        'k': int(len(names)),
        'n_total': int(sum(effects[n]['n'] for n in names)),
        'z_random': _safe_float(z_re),
        'rho_random': _safe_float(inv_fisher_z(z_re)),
        'se_z_random': _safe_float(se_re),
        'ci_rho_lower': _safe_float(ci[0]),
        'ci_rho_upper': _safe_float(ci[1]),
        'z_stat': _safe_float(z_stat),
        'p': _safe_float(p),
        'log10_p': _safe_float(log10_p),
        'weights': weights,
        'weight_share': weight_share,
        'prediction_interval_rho_95': [_safe_float(pred_ci[0]), _safe_float(pred_ci[1])],
    }


def leave_one_out(effects):
    names = list(effects.keys())
    loo = {}
    for drop in names:
        subset = {k: v for k, v in effects.items() if k != drop}
        if len(subset) < 2:
            loo[drop] = None
            continue
        fe = fixed_effects_meta(subset)
        het = heterogeneity(subset, fe)
        re = random_effects_meta(subset, het['tau2'])
        loo[drop] = {
            'fixed_effects': fe,
            'heterogeneity': het,
            'random_effects': re,
        }
    return loo


def make_figure(effects, fixed, random, out_path: Path):
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

        ax.axvline(0.0, color='0.6', lw=1)

        for i in range(len(rows)):
            ax.plot(xs[i], y[i], 'o', color='steelblue' if i < len(rows) - 2 else 'darkorange')
            ax.hlines(y[i], lo[i], hi[i], color='steelblue' if i < len(rows) - 2 else 'darkorange', lw=2)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Spearman ρ (dust vs Γ_t)')
        ax.set_title('Cross-survey meta-analysis (z > 8, dust > 0)')

        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return True
    except Exception:
        return False


def main():
    print_status(f"STEP {STEP_NUM}: Random-effects meta-analysis + LOO", "TITLE")

    config = {
        'z_min': 8.0,
        'dust_positive_only': True,
    }

    surveys = load_survey_data()
    if not surveys:
        print_status("No survey data found.", "ERROR")
        return

    print_status(f"Surveys loaded: {list(surveys.keys())}", "INFO")

    effects = {}
    for name, df in surveys.items():
        df_std = ensure_gamma_t(df)
        eff = compute_survey_effect(df_std, z_min=float(config['z_min']), dust_positive_only=bool(config['dust_positive_only']))
        if eff is None:
            continue
        effects[name] = eff

    if len(effects) < 2:
        print_status("Insufficient surveys with valid effects (need >= 2).", "ERROR")
        return

    fe = fixed_effects_meta(effects)
    het = heterogeneity(effects, fe)
    re = random_effects_meta(effects, het['tau2'])
    loo = leave_one_out(effects)

    influence = {}
    for name, e in effects.items():
        influence[name] = {
            'n': int(e['n']),
            'rho': _safe_float(e['rho']),
            'rho_fixed': _safe_float(fe['rho_fixed']),
            'rho_random': _safe_float(re['rho_random']),
            'delta_rho_fixed_minus_study': _safe_float(float(fe['rho_fixed']) - float(e['rho'])),
            'delta_rho_random_minus_study': _safe_float(float(re['rho_random']) - float(e['rho'])),
            'weight_share_fixed': _safe_float(fe['weight_share'].get(name)),
            'weight_share_random': _safe_float(re['weight_share'].get(name)),
            'loo_rho_random': _safe_float(loo.get(name, {}).get('random_effects', {}).get('rho_random')) if loo.get(name) else None,
            'loo_delta_rho_random': (
                _safe_float(float(loo[name]['random_effects']['rho_random']) - float(re['rho_random']))
                if loo.get(name) and loo[name].get('random_effects') and re.get('rho_random') is not None
                else None
            ),
        }

    results = {
        'config': config,
        'sources': {
            'UNCOVER': str(INTERIM_PATH / 'step_002_uncover_full_sample_tep.csv'),
            'CEERS': str(DATA_INTERIM_PATH / 'ceers_z8_sample.csv'),
            'COSMOS-Web': str(DATA_INTERIM_PATH / 'cosmosweb_z8_sample.csv'),
        },
        'survey_effects': effects,
        'fixed_effects': fe,
        'heterogeneity': het,
        'random_effects': re,
        'leave_one_out': loo,
        'influence': influence,
    }

    fig_path = FIGURES_PATH / 'figure_091_random_effects_meta_loo.png'
    wrote = make_figure(effects, fe, re, fig_path)
    results['figure'] = {
        'path': str(fig_path),
        'written': bool(wrote),
    }

    out_json = OUTPUT_PATH / 'step_091_random_effects_meta_loo.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, allow_nan=False, default=safe_json_default)

    print_status(f"Saved: {out_json}", "SUCCESS")


if __name__ == '__main__':
    main()
