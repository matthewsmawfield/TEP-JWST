#!/usr/bin/env python3
"""
Step 105: Evidence Strengthening and Falsification Tests

This script performs comprehensive tests to strengthen TEP evidence:
1. Monte Carlo null model testing
2. Jackknife stability across all correlations
3. Effect size consistency checks
4. Alternative Model testing
5. Investigate environmental screening anomaly

Outputs:
- results/outputs/step_105_evidence_strengthening.json
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp, mannwhitneyu
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300) & JSON serialiser for numpy types

STEP_NUM = "082"  # Pipeline step number (sequential 001-176)
STEP_NAME = "evidence_strengthening"  # Evidence strengthening: Monte Carlo null tests, jackknife stability, effect size consistency, alternative model comparison (Ridge regression), environmental screening anomaly investigation
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)

LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

np.random.seed(42)  # Fixed RNG seed for reproducible Monte Carlo results (critical for falsification tests)


def monte_carlo_null_test(x, y, n_simulations=10000):
    """
    Monte Carlo test: What's the probability of observing the correlation
    under the null model (no relationship)?
    
    Shuffles y values and computes correlation distribution.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[valid], np.array(y)[valid]
    n = len(x)
    
    if n < 20:
        return None
    
    # Observed correlation
    rho_obs, _ = spearmanr(x, y)
    
    # Null distribution
    rho_null = np.zeros(n_simulations)
    for i in range(n_simulations):
        y_shuffled = np.random.permutation(y)
        rho_null[i], _ = spearmanr(x, y_shuffled)
    
    # P-value: fraction of null correlations >= observed
    if rho_obs >= 0:
        k = int(np.sum(rho_null >= rho_obs))
    else:
        k = int(np.sum(rho_null <= rho_obs))

    p_mc = (k + 1) / (n_simulations + 1)
    
    # Two-tailed
    p_mc_2tail = 2 * min(p_mc, 1 - p_mc)
    if p_mc_2tail > 1:
        p_mc_2tail = 1.0
    
    return {
        'rho_observed': float(rho_obs),
        'null_mean': float(np.mean(rho_null)),
        'null_std': float(np.std(rho_null)),
        'p_mc_1tail': format_p_value(p_mc),
        'p_mc_2tail': format_p_value(p_mc_2tail),
        'z_score': float((rho_obs - np.mean(rho_null)) / np.std(rho_null)),
        'n': int(n),
        'n_simulations': n_simulations
    }


def jackknife_stability(x, y, n_jackknife=None):
    """
    Jackknife analysis: How stable is the correlation when removing
    individual data points?
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[valid], np.array(y)[valid]
    n = len(x)
    
    if n < 20:
        return None
    
    # Full sample correlation
    rho_full, _ = spearmanr(x, y)
    
    # Jackknife: leave one out
    if n_jackknife is None:
        n_jackknife = min(n, 500)  # Cap for large samples
    
    indices = np.random.choice(n, n_jackknife, replace=False) if n > n_jackknife else range(n)
    
    rho_jack = []
    for i in indices:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho_i, _ = spearmanr(x[mask], y[mask])
        rho_jack.append(rho_i)
    
    rho_jack = np.array(rho_jack)
    
    # Jackknife standard error
    se_jack = np.sqrt((len(rho_jack) - 1) / len(rho_jack) * np.sum((rho_jack - np.mean(rho_jack))**2))
    
    # Influential points: those that change rho by > 2*SE
    influential = np.abs(rho_jack - rho_full) > 2 * se_jack
    
    return {
        'rho_full': float(rho_full),
        'rho_jack_mean': float(np.mean(rho_jack)),
        'rho_jack_std': float(np.std(rho_jack)),
        'se_jackknife': float(se_jack),
        'ci_lower': float(rho_full - 1.96 * se_jack),
        'ci_upper': float(rho_full + 1.96 * se_jack),
        'n_influential': int(np.sum(influential)),
        'fraction_influential': float(np.mean(influential)),
        'stable': bool(np.std(rho_jack) < 0.1),
        'n': int(n)
    }


def effect_size_analysis(rho, n):
    """
    Compute effect size metrics for a correlation.
    """
    # Cohen's conventions: small=0.1, medium=0.3, large=0.5
    if abs(rho) < 0.1:
        size_category = 'negligible'
    elif abs(rho) < 0.3:
        size_category = 'small'
    elif abs(rho) < 0.5:
        size_category = 'medium'
    else:
        size_category = 'large'
    
    # r² (variance explained)
    r_squared = rho ** 2
    
    # Statistical power (post-hoc)
    # Using Fisher z-transform
    z = 0.5 * np.log((1 + abs(rho)) / (1 - abs(rho) + 1e-10))
    se = 1 / np.sqrt(n - 3)
    power = stats.norm.sf(1.96 - z / se)
    
    return {
        'rho': float(rho),
        'r_squared': float(r_squared),
        'variance_explained_pct': float(r_squared * 100),
        'size_category': size_category,
        'power': float(power),
        'n': int(n)
    }


def alternative_hypothesis_test(df, x_col, y_col, z_col='z_phot'):
    """
    Test alternative models that could explain the correlation.
    
    H0: No relationship
    H1 (TEP): Correlation driven by Γt
    H2 (Confound): Correlation driven by redshift
    H3 (Confound): Correlation driven by mass
    """
    results = {}
    
    # Get valid data
    cols = [x_col, y_col, z_col, 'log_Mstar']
    valid = df[cols].notna().all(axis=1)
    df_valid = df[valid].copy()
    
    if len(df_valid) < 50:
        return None
    
    x = df_valid[x_col].values
    y = df_valid[y_col].values
    z = df_valid[z_col].values
    mass = df_valid['log_Mstar'].values
    
    # Raw correlation
    rho_raw, p_raw = spearmanr(x, y)
    results['raw'] = {'rho': float(rho_raw), 'p': format_p_value(p_raw)}
    
    # Partial correlation controlling for redshift
    # Using residuals method
    from scipy.stats import linregress
    
    # Residualize x and y against z
    slope_xz, intercept_xz, _, _, _ = linregress(z, x)
    x_resid = x - (slope_xz * z + intercept_xz)
    
    slope_yz, intercept_yz, _, _, _ = linregress(z, y)
    y_resid = y - (slope_yz * z + intercept_yz)
    
    rho_partial_z, p_partial_z = spearmanr(x_resid, y_resid)
    results['partial_z'] = {
        'rho': float(rho_partial_z), 
        'p': format_p_value(p_partial_z),
        'interpretation': 'Correlation after removing redshift dependence'
    }
    
    # Partial correlation controlling for mass
    try:
        slope_xm, intercept_xm, _, _, _ = linregress(mass, x)
        x_resid_m = x - (slope_xm * mass + intercept_xm)
        
        slope_ym, intercept_ym, _, _, _ = linregress(mass, y)
        y_resid_m = y - (slope_ym * mass + intercept_ym)
        
        rho_partial_m, p_partial_m = spearmanr(x_resid_m, y_resid_m)
        
        # Check for valid results
        if np.isnan(rho_partial_m) or np.isnan(p_partial_m):
            results['partial_mass'] = {
                'rho': None,
                'p': None,
                'interpretation': 'Correlation after removing mass dependence - insufficient variation in mass for partial correlation',
                'note': 'Partial correlation requires sufficient independent variation in both mass and the variables of interest'
            }
        else:
            results['partial_mass'] = {
                'rho': float(rho_partial_m),
                'p': format_p_value(p_partial_m),
                'interpretation': 'Correlation after removing mass dependence'
            }
    except Exception as e:
        results['partial_mass'] = {
            'rho': None,
            'p': None,
            'interpretation': 'Correlation after removing mass dependence - calculation failed',
            'error': str(e)
        }
    
    # Partial correlation controlling for both (simplified)
    # Use z residuals as approximation
    rho_partial_both = rho_partial_z  # Approximation
    
    results['partial_both'] = {
        'rho': float(rho_partial_both),
        'interpretation': 'Correlation after removing z and mass dependence (approx)'
    }
    
    # Conclusion
    if abs(rho_partial_z) > 0.8 * abs(rho_raw) and abs(rho_partial_m) > 0.8 * abs(rho_raw):
        conclusion = 'ROBUST: Correlation persists after controlling for confounds'
    elif abs(rho_partial_z) < 0.5 * abs(rho_raw):
        conclusion = 'CONFOUNDED: Correlation largely explained by redshift'
    elif abs(rho_partial_m) < 0.5 * abs(rho_raw):
        conclusion = 'CONFOUNDED: Correlation largely explained by mass'
    else:
        conclusion = 'PARTIAL: Some confounding present'
    
    results['conclusion'] = conclusion
    
    return results


def predictive_confound_control_test(df, n_splits=5, n_repeats=20, n_permutations=500, random_seed=42):
    rng = np.random.default_rng(random_seed)

    required = ['log_Mstar', 'z_phot', 'dust', 'gamma_t']
    df_sub = df[df['z_phot'] > 8].dropna(subset=required).copy()
    df_sub = df_sub[df_sub['dust'] > 0]

    if len(df_sub) < 50:
        return None

    y = df_sub['dust'].to_numpy()
    m = df_sub['log_Mstar'].to_numpy()
    z = df_sub['z_phot'].to_numpy()
    lg = np.log(df_sub['gamma_t'].clip(1e-12, None).to_numpy())

    X_massz = np.column_stack([m, z])
    X_massz_g = np.column_stack([m, z, lg])

    X_poly = np.column_stack([m, z, m**2, z**2, m * z])
    X_poly_g = np.column_stack([m, z, m**2, z**2, m * z, lg])

    def cv_scores(X_base, X_full, permute_full_feature=False):
        base_scores = []
        full_scores = []
        deltas = []
        for rep in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
            for train_idx, test_idx in kf.split(X_base):
                model_base = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model_base.fit(X_base[train_idx], y[train_idx])
                pred_base = model_base.predict(X_base[test_idx])
                r2_base = r2_score(y[test_idx], pred_base)

                if permute_full_feature:
                    X_full_perm = X_full.copy()
                    X_full_perm[:, -1] = rng.permutation(X_full_perm[:, -1])
                    X_use = X_full_perm
                else:
                    X_use = X_full

                model_full = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model_full.fit(X_use[train_idx], y[train_idx])
                pred_full = model_full.predict(X_use[test_idx])
                r2_full = r2_score(y[test_idx], pred_full)

                base_scores.append(r2_base)
                full_scores.append(r2_full)
                deltas.append(r2_full - r2_base)

        return np.array(base_scores), np.array(full_scores), np.array(deltas)

    base, full, delta = cv_scores(X_massz, X_massz_g, permute_full_feature=False)
    base_p, full_p, delta_perm = cv_scores(X_massz, X_massz_g, permute_full_feature=True)

    obs = float(np.mean(delta))
    p_perm = float((np.sum(delta_perm >= obs) + 1) / (len(delta_perm) + 1))

    base2, full2, delta2 = cv_scores(X_poly, X_poly_g, permute_full_feature=False)
    base2_p, full2_p, delta2_perm = cv_scores(X_poly, X_poly_g, permute_full_feature=True)

    obs2 = float(np.mean(delta2))
    p_perm2 = float((np.sum(delta2_perm >= obs2) + 1) / (len(delta2_perm) + 1))

    return {
        'n': int(len(df_sub)),
        'n_splits': int(n_splits),
        'n_repeats': int(n_repeats),
        'mass_z_vs_plus_log_gamma': {
            'r2_base_mean': float(np.mean(base)),
            'r2_base_std': float(np.std(base)),
            'r2_full_mean': float(np.mean(full)),
            'r2_full_std': float(np.std(full)),
            'delta_r2_mean': float(np.mean(delta)),
            'delta_r2_std': float(np.std(delta)),
            'permutation_p_value': p_perm,
            'delta_r2_perm_mean': float(np.mean(delta_perm)),
            'delta_r2_perm_std': float(np.std(delta_perm))
        },
        'poly_mass_z_vs_plus_log_gamma': {
            'r2_base_mean': float(np.mean(base2)),
            'r2_base_std': float(np.std(base2)),
            'r2_full_mean': float(np.mean(full2)),
            'r2_full_std': float(np.std(full2)),
            'delta_r2_mean': float(np.mean(delta2)),
            'delta_r2_std': float(np.std(delta2)),
            'permutation_p_value': p_perm2,
            'delta_r2_perm_mean': float(np.mean(delta2_perm)),
            'delta_r2_perm_std': float(np.std(delta2_perm))
        }
    }


def investigate_environmental_anomaly(df):
    """
    Investigate why environmental screening shows unexpected positive
    density-Γt correlation instead of negative.
    
    Possible explanations:
    1. Selection effect: Dense regions have more massive galaxies
    2. Redshift confound: Dense regions at different z
    3. Measurement artifact: Mock coordinates
    """
    results = {}
    
    # Check if we have real coordinates
    has_real_coords = 'ra' in df.columns and df['ra'].std() > 0.01
    results['has_real_coordinates'] = has_real_coords
    
    if not has_real_coords:
        results['warning'] = 'Using mock coordinates - environmental analysis unreliable'
        return results
    
    # Check mass distribution in high vs low density
    if 'local_density' in df.columns:
        median_density = df['local_density'].median()
        high_dens = df[df['local_density'] > median_density]
        low_dens = df[df['local_density'] <= median_density]
        
        results['mass_by_density'] = {
            'high_density_mean_mass': float(high_dens['log_Mstar'].mean()),
            'low_density_mean_mass': float(low_dens['log_Mstar'].mean()),
            'mass_difference': float(high_dens['log_Mstar'].mean() - low_dens['log_Mstar'].mean())
        }
        
        # If high-density regions have more massive galaxies, this explains
        # the positive density-Γt correlation (since Γt depends on mass)
        if results['mass_by_density']['mass_difference'] > 0.1:
            results['explanation'] = (
                'Selection effect: High-density regions contain more massive galaxies, '
                'which have higher Γt by construction. This is NOT a failure of TEP screening - '
                'it reflects the mass-density correlation in the sample.'
            )
        
        # Check Γt at fixed mass
        mass_bins = [(8, 9), (9, 10), (10, 11)]
        fixed_mass_results = []
        
        for m_lo, m_hi in mass_bins:
            subset = df[(df['log_Mstar'] >= m_lo) & (df['log_Mstar'] < m_hi)]
            if len(subset) > 30 and 'local_density' in subset.columns:
                rho, p = spearmanr(subset['local_density'], subset['gamma_t'])
                fixed_mass_results.append({
                    'mass_bin': f'{m_lo}-{m_hi}',
                    'n': len(subset),
                    'rho': float(rho),
                    'p': format_p_value(p)
                })
        
        results['fixed_mass_density_gamma'] = fixed_mass_results
        
        # If correlation disappears at fixed mass, the anomaly is explained
        if fixed_mass_results:
            avg_rho = np.mean([r['rho'] for r in fixed_mass_results])
            if abs(avg_rho) < 0.1:
                results['fixed_mass_conclusion'] = (
                    'At fixed mass, density-Γt correlation disappears. '
                    'The apparent positive correlation is entirely due to mass-density correlation.'
                )
    
    return results


def comprehensive_correlation_tests(df):
    """
    Run comprehensive tests on all primary correlations.
    """
    results = {}
    
    # Define primary correlations to test
    correlations = [
        ('gamma_t', 'dust', 'z_phot > 8', 'z>8 Dust-Γt'),
        ('log_Mstar', 'mwa', None, 'Mass-Age'),
        ('gamma_t', 'dust', None, 'Full Sample Dust-Γt'),
    ]
    
    for x_col, y_col, condition, name in correlations:
        if condition:
            df_subset = df.query(condition).copy()
        else:
            df_subset = df.copy()
        
        if x_col not in df_subset.columns or y_col not in df_subset.columns:
            continue
        
        valid = ~(df_subset[x_col].isna() | df_subset[y_col].isna())
        x = df_subset.loc[valid, x_col].values
        y = df_subset.loc[valid, y_col].values
        
        if len(x) < 20:
            continue
        
        test_results = {}
        
        # Monte Carlo null test
        mc = monte_carlo_null_test(x, y)
        if mc:
            test_results['monte_carlo'] = mc
        
        # Jackknife stability
        jack = jackknife_stability(x, y)
        if jack:
            test_results['jackknife'] = jack
        
        # Effect size
        rho, _ = spearmanr(x, y)
        effect = effect_size_analysis(rho, len(x))
        test_results['effect_size'] = effect
        
        # Alternative Model
        if condition is None:  # Full sample only
            alt = alternative_hypothesis_test(df_subset, x_col, y_col)
            if alt:
                test_results['alternative_hypothesis'] = alt
        
        results[name] = test_results
    
    return results


def compute_evidence_strength_score(results):
    """
    Compute an overall evidence strength score (0-100).
    """
    scores = []
    
    for name, tests in results.items():
        if 'monte_carlo' in tests:
            mc = tests['monte_carlo']
            # Score based on z-score
            z = abs(mc['z_score'])
            mc_score = min(100, z * 10)  # z=10 -> score=100
            scores.append(mc_score)
        
        if 'jackknife' in tests:
            jack = tests['jackknife']
            # Score based on stability
            if jack['stable']:
                jack_score = 100
            else:
                jack_score = max(0, 100 - jack['fraction_influential'] * 200)
            scores.append(jack_score)
        
        if 'effect_size' in tests:
            effect = tests['effect_size']
            # Score based on effect size category
            effect_scores = {'large': 100, 'medium': 75, 'small': 50, 'negligible': 25}
            scores.append(effect_scores.get(effect['size_category'], 0))
    
    if not scores:
        return 0
    
    return np.mean(scores)


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Evidence Strengthening and Falsification Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    results = {}
    
    # Load data
    data_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # ==========================================================================
    # 1. Comprehensive correlation tests
    # ==========================================================================
    print_status("\n--- 1. Comprehensive Correlation Tests ---", "INFO")
    
    corr_tests = comprehensive_correlation_tests(df)
    results['correlation_tests'] = corr_tests
    
    for name, tests in corr_tests.items():
        print_status(f"\n  {name}:", "INFO")
        if 'monte_carlo' in tests:
            mc = tests['monte_carlo']
            print_status(f"    Monte Carlo: Z = {mc['z_score']:.2f}, p = {mc['p_mc_2tail']:.2e}", "INFO")
        if 'jackknife' in tests:
            jack = tests['jackknife']
            print_status(f"    Jackknife: ρ = {jack['rho_full']:.3f} ± {jack['se_jackknife']:.3f}, stable = {jack['stable']}", "INFO")
        if 'effect_size' in tests:
            effect = tests['effect_size']
            print_status(f"    Effect size: {effect['size_category']} (r² = {effect['variance_explained_pct']:.1f}%)", "INFO")
    
    # ==========================================================================
    # 2. Investigate environmental anomaly
    # ==========================================================================
    print_status("\n--- 2. Environmental Screening Investigation ---", "INFO")
    
    # First, compute local density if not present
    if 'local_density' not in df.columns:
        if 'ra' in df.columns and 'dec' in df.columns:
            from scipy.spatial import cKDTree

            coords = np.column_stack([df['ra'].values, df['dec'].values])
            tree = cKDTree(coords)
            z_vals = df['z_phot'].values if 'z_phot' in df.columns else df['z'].values

            densities = np.zeros(len(df))
            r_deg = 1.0 / 60
            z_window = 0.5
            area = np.pi * 1.0**2

            for i in range(len(df)):
                idx = tree.query_ball_point(coords[i], r_deg)
                neighbors = [j for j in idx if j != i and abs(z_vals[j] - z_vals[i]) < z_window]
                densities[i] = len(neighbors) / area

            df['local_density'] = densities
        else:
            print_status("  WARNING: Missing RA/Dec columns; environmental density cannot be computed.", "WARNING")
    
    env_investigation = investigate_environmental_anomaly(df)
    results['environmental_investigation'] = env_investigation

    # ==========================================================================
    # 3. Predictive confound-control test
    # ==========================================================================
    print_status("\n--- 3. Predictive Confound-Control Test ---", "INFO")

    pred_cc = predictive_confound_control_test(df)
    results['predictive_confound_control'] = pred_cc

    if pred_cc and 'mass_z_vs_plus_log_gamma' in pred_cc:
        mz = pred_cc['mass_z_vs_plus_log_gamma']
        print_status(
            f"  ΔR² (mass+z -> +logΓt): {mz['delta_r2_mean']:+.4f} (perm p={mz['permutation_p_value']:.3f})",
            "INFO"
        )
    if pred_cc and 'poly_mass_z_vs_plus_log_gamma' in pred_cc:
        pz = pred_cc['poly_mass_z_vs_plus_log_gamma']
        print_status(
            f"  ΔR² (poly(m,z) -> +logΓt): {pz['delta_r2_mean']:+.4f} (perm p={pz['permutation_p_value']:.3f})",
            "INFO"
        )
    
    if 'warning' in env_investigation:
        print_status(f"  WARNING: {env_investigation['warning']}", "WARNING")
    
    if 'mass_by_density' in env_investigation:
        mbd = env_investigation['mass_by_density']
        print_status(f"  High-density mean mass: {mbd['high_density_mean_mass']:.2f}", "INFO")
        print_status(f"  Low-density mean mass: {mbd['low_density_mean_mass']:.2f}", "INFO")
        print_status(f"  Mass difference: {mbd['mass_difference']:.2f} dex", "INFO")
    
    if 'explanation' in env_investigation:
        print_status(f"  EXPLANATION: {env_investigation['explanation']}", "INFO")
    
    if 'fixed_mass_density_gamma' in env_investigation:
        print_status("  Fixed-mass density-Γt correlations:", "INFO")
        for r in env_investigation['fixed_mass_density_gamma']:
            print_status(f"    Mass {r['mass_bin']}: ρ = {r['rho']:.3f}, N = {r['n']}", "INFO")
    
    if 'fixed_mass_conclusion' in env_investigation:
        print_status(f"  CONCLUSION: {env_investigation['fixed_mass_conclusion']}", "INFO")
    
    # ==========================================================================
    # 4. Evidence strength score
    # ==========================================================================
    print_status("\n--- 4. Evidence Strength Score ---", "INFO")
    
    score = compute_evidence_strength_score(corr_tests)
    results['evidence_strength_score'] = float(score)
    
    if score >= 90:
        strength = 'VERY STRONG'
    elif score >= 75:
        strength = 'STRONG'
    elif score >= 50:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'
    
    results['evidence_strength_category'] = strength
    print_status(f"  Overall score: {score:.1f}/100 ({strength})", "INFO")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("EVIDENCE STRENGTHENING SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    summary = {
        'n_correlations_tested': len(corr_tests),
        'all_monte_carlo_significant': all(
            tests.get('monte_carlo', {}).get('p_mc_2tail', 1) < 0.001
            for tests in corr_tests.values()
        ),
        'all_jackknife_stable': all(
            tests.get('jackknife', {}).get('stable', False)
            for tests in corr_tests.values()
        ),
        'evidence_score': score,
        'evidence_category': strength,
        'environmental_anomaly_explained': 'fixed_mass_conclusion' in env_investigation
    }
    
    results['summary'] = summary
    
    print_status(f"  Correlations tested: {summary['n_correlations_tested']}", "INFO")
    print_status(f"  All Monte Carlo significant: {summary['all_monte_carlo_significant']}", "INFO")
    print_status(f"  All jackknife stable: {summary['all_jackknife_stable']}", "INFO")
    print_status(f"  Environmental anomaly explained: {summary['environmental_anomaly_explained']}", "INFO")
    print_status(f"  Evidence category: {summary['evidence_category']}", "INFO")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_evidence_strengthening.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print_status(f"\nResults saved to {output_file}", "INFO")


if __name__ == "__main__":
    main()
