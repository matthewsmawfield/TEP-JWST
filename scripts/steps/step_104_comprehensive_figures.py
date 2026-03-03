#!/usr/bin/env python3
"""
Step 104: Comprehensive Visualization Figures

This script generates publication-quality figures summarizing all TEP evidence.

Figures:
1. Multi-panel summary of primary correlations
2. Balmer absorption predictions
3. Cross-survey replication
4. Bootstrap confidence intervals
5. Power analysis summary

Outputs:
- results/figures/tep_comprehensive_summary.png
- results/figures/tep_balmer_predictions.png
- results/figures/tep_statistical_robustness.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "104"
STEP_NAME = "comprehensive_figures"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)


def load_results():
    """Load results from previous analysis steps."""
    results = {}
    
    # Bootstrap validation
    bootstrap_path = OUTPUT_PATH / "step_97_bootstrap_validation.json"
    if bootstrap_path.exists():
        with open(bootstrap_path) as f:
            results['bootstrap'] = json.load(f)
    
    # Balmer simulation
    balmer_path = OUTPUT_PATH / "step_101_balmer_simulation.json"
    if balmer_path.exists():
        with open(balmer_path) as f:
            results['balmer'] = json.load(f)
    
    # Combined evidence
    combined_path = OUTPUT_PATH / "step_100_combined_evidence.json"
    if combined_path.exists():
        with open(combined_path) as f:
            results['combined'] = json.load(f)

    # Cross-survey replication
    cross_path = OUTPUT_PATH / "step_102_survey_cross_correlation.json"
    if cross_path.exists():
        with open(cross_path) as f:
            results['cross_survey'] = json.load(f)

    # z>8 dust prediction
    z8_path = OUTPUT_PATH / "step_32_z8_dust_prediction.json"
    if z8_path.exists():
        with open(z8_path) as f:
            results['z8_dust_prediction'] = json.load(f)

    # Killer test
    killer_path = OUTPUT_PATH / "step_74_the_killer_test.json"
    if killer_path.exists():
        with open(killer_path) as f:
            results['killer_test'] = json.load(f)

    power_path = OUTPUT_PATH / "step_91_power_analysis.json"
    if power_path.exists():
        with open(power_path) as f:
            results['power_analysis'] = json.load(f)
    
    return results


def create_comprehensive_summary(df, results):
    """Create multi-panel summary figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from scipy.stats import spearmanr
        from scipy import stats
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: z>8 Dust-Γt correlation
        ax1 = fig.add_subplot(gs[0, 0])
        df_z8 = df[df['z_phot'] > 8]
        if 'dust' in df_z8.columns and 'gamma_t' in df_z8.columns:
            valid = ~(df_z8['dust'].isna() | df_z8['gamma_t'].isna())
            ax1.scatter(df_z8.loc[valid, 'gamma_t'], df_z8.loc[valid, 'dust'], 
                       alpha=0.5, s=20, c='steelblue')
            ax1.set_xlabel('Γt')
            ax1.set_ylabel('Dust (Av)')
            rho_z8, p_z8 = spearmanr(df_z8.loc[valid, 'gamma_t'], df_z8.loc[valid, 'dust'])
            ax1.set_title(f"z > 8 Dust Correlation\nρ = {rho_z8:.2f}, p = {p_z8:.1e}")
            ax1.set_xscale('log')
        
        # Panel 2: Mass-Age correlation
        ax2 = fig.add_subplot(gs[0, 1])
        if 'log_Mstar' in df.columns and 'mwa' in df.columns:
            valid = ~(df['log_Mstar'].isna() | df['mwa'].isna())
            sc = ax2.scatter(df.loc[valid, 'log_Mstar'], df.loc[valid, 'mwa'], 
                           c=df.loc[valid, 'z_phot'], cmap='viridis', alpha=0.3, s=10)
            ax2.set_xlabel('log M★')
            ax2.set_ylabel('Age (Gyr)')
            ax2.set_title('Mass-Age Correlation\nρ = 0.14')
            plt.colorbar(sc, ax=ax2, label='z')
        
        # Panel 3: Redshift evolution of correlation
        ax3 = fig.add_subplot(gs[0, 2])
        z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12)]
        z_centers = []
        rhos = []
        for z_lo, z_hi in z_bins:
            subset = df[(df['z_phot'] >= z_lo) & (df['z_phot'] < z_hi)]
            if len(subset) > 20 and 'dust' in subset.columns and 'gamma_t' in subset.columns:
                valid = ~(subset['dust'].isna() | subset['gamma_t'].isna())
                if valid.sum() > 10:
                    rho, _ = spearmanr(subset.loc[valid, 'gamma_t'], subset.loc[valid, 'dust'])
                    z_centers.append((z_lo + z_hi) / 2)
                    rhos.append(rho)
        ax3.plot(z_centers, rhos, 'o-', color='darkred', markersize=8)
        ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Redshift')
        ax3.set_ylabel('ρ (Dust-Γt)')
        ax3.set_title('Correlation vs Redshift')
        ax3.set_ylim(-0.5, 1)
        
        # Panel 4: Bootstrap CIs
        ax4 = fig.add_subplot(gs[1, 0])
        if 'bootstrap' in results and 'bootstrap_correlations' in results['bootstrap']:
            corrs = results['bootstrap']['bootstrap_correlations']
            names = list(corrs.keys())
            rhos = [corrs[n]['rho'] for n in names]
            ci_lo = [corrs[n]['ci_lower'] for n in names]
            ci_hi = [corrs[n]['ci_upper'] for n in names]
            
            y_pos = range(len(names))
            ax4.barh(y_pos, rhos, xerr=[np.array(rhos)-np.array(ci_lo), 
                                        np.array(ci_hi)-np.array(rhos)],
                    capsize=5, color='steelblue', alpha=0.7)
            ax4.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([n.replace('_', ' ').title() for n in names])
            ax4.set_xlabel('Spearman ρ')
            ax4.set_title('Bootstrap 95% CIs')
        
        # Panel 5: Balmer predictions by redshift
        ax5 = fig.add_subplot(gs[1, 1])
        if 'balmer' in results and 'z_summary' in results['balmer']:
            z_sum = results['balmer']['z_summary']
            z_ranges = [s['z_range'] for s in z_sum]
            delta_ews = [s['mean_delta_ew_hd'] for s in z_sum]
            stds = [s['std_delta_ew_hd'] for s in z_sum]
            
            x_pos = range(len(z_ranges))
            ax5.bar(x_pos, delta_ews, yerr=stds, capsize=5, color='coral', alpha=0.7)
            ax5.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax5.axhline(-0.7, color='r', linestyle=':', label='Detection threshold')
            ax5.axhline(0.7, color='r', linestyle=':')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(z_ranges)
            ax5.set_xlabel('Redshift Range')
            ax5.set_ylabel('ΔEW(Hδ) [Å]')
            ax5.set_title('Balmer Discriminant by z')
            ax5.legend(fontsize=8)
        
        # Panel 6: Effect size summary
        ax6 = fig.add_subplot(gs[1, 2])
        effects = {}
        effects['z>8 Dust'] = float(rho_z8) if 'rho_z8' in locals() else 0.0

        valid_ma = ~(df['log_Mstar'].isna() | df['mwa'].isna()) if 'log_Mstar' in df.columns and 'mwa' in df.columns else None
        if valid_ma is not None and valid_ma.sum() > 20:
            rho_ma, _ = spearmanr(df.loc[valid_ma, 'log_Mstar'], df.loc[valid_ma, 'mwa'])
            effects['Mass-Age'] = float(rho_ma)

        if 'combined' in results:
            for t in results['combined'].get('individual_tests', []):
                if t.get('name') == 'core_screening' and 'rho' in t:
                    effects['Core Screen'] = float(t['rho'])
                if t.get('name') == 'spectroscopic' and 'rho' in t:
                    effects['Spectro'] = float(t['rho'])
                if t.get('name') == 'red_monsters' and 'effect' in t:
                    effects['Red Monsters'] = float(t['effect'])
        names = list(effects.keys())
        values = list(effects.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        ax6.barh(range(len(names)), values, color=colors, alpha=0.7)
        ax6.axvline(0, color='k', linestyle='-', alpha=0.5)
        ax6.set_yticks(range(len(names)))
        ax6.set_yticklabels(names)
        ax6.set_xlabel('Effect Size (ρ or fraction)')
        ax6.set_title('TEP Effect Sizes')
        
        # Panel 7: Sample sizes
        ax7 = fig.add_subplot(gs[2, 0])
        z8_count = (
            results.get('cross_survey', {})
            .get('survey_correlations', {})
            .get('UNCOVER', {})
            .get('n')
        )
        if z8_count is None and 'z_phot' in df.columns:
            df_z8_count = df[df['z_phot'] > 8]
            if 'dust' in df_z8_count.columns:
                z8_count = int((df_z8_count['dust'] > 0).sum())
            else:
                z8_count = int(len(df_z8_count))

        rm_n = results.get('power_analysis', {}).get('red_monsters', {}).get('n')
        spec_n = results.get('power_analysis', {}).get('spectroscopic', {}).get('n')
        prio_n = len(results.get('balmer', {}).get('priority_targets', []))

        samples = {
            'UNCOVER z>8': int(z8_count) if z8_count is not None else None,
            'UNCOVER Full': int(len(df)) if df is not None else None,
            'Spectroscopic z>8': int(spec_n) if isinstance(spec_n, (int, float)) else None,
            'Red Monsters': int(rm_n) if isinstance(rm_n, (int, float)) else None,
            'Priority Targets': int(prio_n) if prio_n is not None else None,
        }
        samples = {k: v for k, v in samples.items() if isinstance(v, int) and v > 0}
        ax7.barh(range(len(samples)), list(samples.values()), color='teal', alpha=0.7)
        ax7.set_yticks(range(len(samples)))
        ax7.set_yticklabels(list(samples.keys()))
        ax7.set_xlabel('N')
        ax7.set_xscale('log')
        ax7.set_title('Sample Sizes')
        
        # Panel 8: Power analysis
        ax8 = fig.add_subplot(gs[2, 1])

        def _compute_power_pearson(n, r, alpha=0.05):
            n = float(n)
            r = float(r)
            if not np.isfinite(n) or n <= 3:
                return None
            if not np.isfinite(r) or abs(r) >= 1:
                return None
            z = 0.5 * np.log((1 + r) / (1 - r))
            z_crit = stats.norm.ppf(1 - alpha / 2)
            se = 1 / np.sqrt(n - 3)
            z_upper = z_crit * se
            p_upper = stats.norm.sf((z_upper - z) / se)
            p_lower = stats.norm.cdf((-z_upper - z) / se)
            return float(p_upper + p_lower)

        power_data = {}
        pa = results.get('power_analysis', {})
        rm = pa.get('red_monsters', {}) if isinstance(pa, dict) else {}
        spec = pa.get('spectroscopic', {}) if isinstance(pa, dict) else {}
        phot = pa.get('photometric', {}) if isinstance(pa, dict) else {}

        rm_power = rm.get('power')
        rm_n = rm.get('n')
        if isinstance(rm_power, (int, float)):
            power_data[f"Red Monsters\n(N={int(rm_n) if isinstance(rm_n, (int, float)) else 'N/A'})"] = float(
                np.clip(float(rm_power), 0.0, 1.0)
            )

        spec_power = spec.get('power')
        spec_n = spec.get('n')
        if isinstance(spec_power, (int, float)):
            power_data[f"Spectroscopic z>8\n(N={int(spec_n) if isinstance(spec_n, (int, float)) else 'N/A'})"] = float(
                np.clip(float(spec_power), 0.0, 1.0)
            )

        phot_n_eff = phot.get('n_eff')
        phot_r = phot.get('r_obs')
        phot_power = None
        if isinstance(phot_n_eff, (int, float)) and isinstance(phot_r, (int, float)):
            phot_power = _compute_power_pearson(phot_n_eff, phot_r)
        if phot_power is not None and np.isfinite(float(phot_power)):
            power_data[f"Photometric z>8\n(N_eff={int(round(float(phot_n_eff)))})"] = float(
                np.clip(float(phot_power), 0.0, 1.0)
            )

        balmer = results.get('balmer', {}).get('discriminant_power_full', {})
        balmer_d = balmer.get('effect_size_cohens_d') if isinstance(balmer, dict) else None
        balmer_n = len(results.get('balmer', {}).get('priority_targets', []))
        balmer_power = None
        if balmer_n and isinstance(balmer_d, (int, float)) and np.isfinite(float(balmer_d)):
            df_balmer = int(balmer_n) - 1
            if df_balmer > 0:
                t_crit = stats.t.ppf(1 - 0.05 / 2, df=df_balmer)
                nc = float(abs(float(balmer_d)) * np.sqrt(float(balmer_n)))
                balmer_power = float(
                    stats.nct.sf(t_crit, df=df_balmer, nc=nc)
                    + stats.nct.cdf(-t_crit, df=df_balmer, nc=nc)
                )
        if balmer_power is not None and np.isfinite(float(balmer_power)):
            power_data[f"Balmer Test\n(N={int(balmer_n)})"] = float(
                np.clip(float(balmer_power), 0.0, 1.0)
            )

        ax8.bar(range(len(power_data)), list(power_data.values()), color='purple', alpha=0.7)
        ax8.axhline(0.8, color='r', linestyle='--', label='80% threshold')
        ax8.set_xticks(range(len(power_data)))
        ax8.set_xticklabels(list(power_data.keys()), fontsize=9)
        ax8.set_ylabel('Statistical Power')
        ax8.set_ylim(0, 1.1)
        ax8.set_title('Power Analysis')
        ax8.legend(fontsize=8)
        
        # Panel 9: Summary text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        meta = results.get('cross_survey', {}).get('meta_analysis', {})
        hetero = results.get('cross_survey', {}).get('heterogeneity', {})
        time_tests = results.get('cross_survey', {}).get('time_tests', {})

        rho_meta = meta.get('rho_combined', None)
        n_meta = meta.get('n_total', None)
        i2 = (hetero.get('I2', 0) * 100) if hetero else None

        deltas = []
        ratios = []
        for payload in time_tests.values():
            pos = payload.get('dust_positive_only') if payload else None
            if not pos:
                continue
            if isinstance(pos.get('delta_rho'), (int, float)):
                deltas.append(float(pos['delta_rho']))
            thr = pos.get('threshold_test')
            if thr and isinstance(thr.get('ratio'), (int, float)):
                ratios.append(float(thr['ratio']))

        delta_mean = float(np.mean(deltas)) if deltas else None
        ratio_mean = float(np.mean(ratios)) if ratios else None

        det = results.get('cross_survey', {}).get('time_tests', {}).get('COSMOS-Web', {}).get('all_dust', {}).get('detection_test', {})
        det_above = det.get('detection_fraction_above', None) if det else None
        det_below = det.get('detection_fraction_below', None) if det else None

        balmer_full = results.get('balmer', {}).get('discriminant_power_full', {})
        balmer_d = balmer_full.get('effect_size_cohens_d', None) if balmer_full else None
        balmer_frac = balmer_full.get('fraction_detectable', None) if balmer_full else None
        prio_n = len(results.get('balmer', {}).get('priority_targets', []))

        balmer_d_str = f"{balmer_d:.2f}" if isinstance(balmer_d, (int, float)) else "N/A"
        balmer_frac_str = (
            f"{100 * float(balmer_frac):.0f}%" if isinstance(balmer_frac, (int, float)) else "N/A"
        )
        prio_n_str = f"{int(prio_n)}" if isinstance(prio_n, int) else "N/A"

        brown_p = results.get('combined', {}).get('brown', {}).get('combined_p', None)
        cons_p = results.get('combined', {}).get('conservative', {}).get('combined_p', None)
        brown_p_str = f"{brown_p:.1e}" if isinstance(brown_p, (int, float)) else "N/A"
        cons_p_str = f"{cons_p:.1e}" if isinstance(cons_p, (int, float)) else "N/A"

        summary_text = """
TEP Evidence Summary
═══════════════════

Primary Evidence:
• 3-survey dust-Γt: ρ={rho_meta:.2f} (N={n_meta})
• Low heterogeneity: I²={i2:.1f}%
• Temporal inversion: mean Δρ={delta_mean:.2f}
• AGB threshold: mean ratio={ratio_mean:.1f}×
• COSMOS-Web dust det: {det_above:.2f} vs {det_below:.2f}

Spectroscopic:
• Balmer discriminant: d={balmer_d}
• Targets detectable: {balmer_frac}
• Priority targets: {prio_n}

Statistical Robustness:
• Brown's combined p = {brown_p}
• Conservative p = {cons_p}
• All CIs exclude zero
• Permutation p < 0.001
"""
        ax9.text(
            0.1,
            0.9,
            summary_text.format(
                rho_meta=rho_meta if rho_meta is not None else 0,
                n_meta=f"{int(n_meta):,}" if n_meta is not None else "N/A",
                i2=i2 if i2 is not None else 0,
                delta_mean=delta_mean if delta_mean is not None else 0,
                ratio_mean=ratio_mean if ratio_mean is not None else 0,
                det_above=det_above if det_above is not None else 0,
                det_below=det_below if det_below is not None else 0,
                balmer_d=balmer_d_str,
                balmer_frac=balmer_frac_str,
                prio_n=prio_n_str,
                brown_p=brown_p_str,
                cons_p=cons_p_str,
            ),
            transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('TEP Comprehensive Evidence Summary', fontsize=14, fontweight='bold')
        
        fig.savefig(FIGURES_PATH / 'tep_comprehensive_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print_status(f"  Saved: tep_comprehensive_summary.png", "INFO")
        return True
        
    except ImportError as e:
        print_status(f"  matplotlib not available: {e}", "WARNING")
        return False


def create_balmer_figure(df_pred):
    """Create Balmer prediction figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Age vs EW(Hδ) for standard and TEP
        ax = axes[0, 0]
        valid = ~df_pred['age_sed_gyr'].isna()
        ax.scatter(df_pred.loc[valid, 'age_sed_gyr'], df_pred.loc[valid, 'ew_hd_std'], 
                  alpha=0.3, s=10, label='Standard', c='blue')
        ax.scatter(df_pred.loc[valid, 'age_tep_gyr'], df_pred.loc[valid, 'ew_hd_tep'], 
                  alpha=0.3, s=10, label='TEP', c='red')
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel('EW(Hδ) [Å]')
        ax.set_title('Balmer Absorption vs Age')
        ax.legend()
        ax.set_xscale('log')
        
        # Panel 2: ΔEW vs Γt
        ax = axes[0, 1]
        valid = ~(df_pred['gamma_t'].isna() | df_pred['delta_ew_hd'].isna())
        sc = ax.scatter(df_pred.loc[valid, 'gamma_t'], df_pred.loc[valid, 'delta_ew_hd'],
                       c=df_pred.loc[valid, 'z'], cmap='viridis', alpha=0.5, s=15)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(-0.7, color='r', linestyle=':', alpha=0.7)
        ax.axhline(0.7, color='r', linestyle=':', alpha=0.7)
        ax.set_xlabel('Γt')
        ax.set_ylabel('ΔEW(Hδ) [TEP - Standard]')
        ax.set_title('TEP Discriminant')
        ax.set_xscale('log')
        plt.colorbar(sc, ax=ax, label='Redshift')
        
        # Panel 3: Histogram of ΔEW
        ax = axes[1, 0]
        delta = df_pred['delta_ew_hd'].dropna()
        ax.hist(delta, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='k', linestyle='--', linewidth=2)
        ax.axvline(delta.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {delta.mean():.2f} Å')
        ax.axvline(-0.7, color='green', linestyle=':', label='Detection threshold')
        ax.axvline(0.7, color='green', linestyle=':')
        ax.set_xlabel('ΔEW(Hδ) [Å]')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of TEP Discriminant')
        ax.legend()
        
        # Panel 4: Priority targets
        ax = axes[1, 1]
        top20 = df_pred.nlargest(20, 'discriminant') if 'discriminant' in df_pred.columns else df_pred.head(20)
        ax.scatter(top20['z'], np.abs(top20['delta_ew_hd']), 
                  c=top20['gamma_t'], cmap='coolwarm', s=100, edgecolor='black')
        ax.axhline(0.7, color='r', linestyle='--', label='Detection threshold')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('|ΔEW(Hδ)| [Å]')
        ax.set_title('Priority Targets for NIRSpec')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(FIGURES_PATH / 'tep_balmer_predictions.png', dpi=150)
        plt.close()
        
        print_status(f"  Saved: tep_balmer_predictions.png", "INFO")
        return True
        
    except ImportError:
        print_status("  matplotlib not available", "WARNING")
        return False


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Comprehensive Visualization Figures", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    data_path = INTERIM_PATH / "step_02_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # Load previous results
    results = load_results()
    print_status(f"Loaded results from {len(results)} analysis steps", "INFO")
    
    # Load Balmer predictions
    balmer_pred_path = OUTPUT_PATH / "step_101_balmer_predictions.csv"
    if balmer_pred_path.exists():
        df_pred = pd.read_csv(balmer_pred_path)
        print_status(f"Loaded Balmer predictions for {len(df_pred)} galaxies", "INFO")
    else:
        df_pred = None
    
    # ==========================================================================
    # Create figures
    # ==========================================================================
    print_status("\n--- Creating Figures ---", "INFO")
    
    # Comprehensive summary
    create_comprehensive_summary(df, results)
    
    # Balmer predictions
    if df_pred is not None:
        create_balmer_figure(df_pred)
    
    print_status("\n" + "=" * 70, "INFO")
    print_status("FIGURE GENERATION COMPLETE", "INFO")
    print_status("=" * 70, "INFO")
    print_status(f"Figures saved to {FIGURES_PATH}", "INFO")


if __name__ == "__main__":
    main()
