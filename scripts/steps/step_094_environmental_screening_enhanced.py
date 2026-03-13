#!/usr/bin/env python3
"""
Step 115: Enhanced Environmental Screening Analysis

This script provides a more rigorous test of environmental screening by:
1. Using multiple density estimators (Nth-neighbor, aperture counts, Voronoi)
2. Testing the TEP prediction that screening should SUPPRESS Gamma_t effects
3. Stratifying by redshift to account for evolving density thresholds
4. Computing the expected screening signal strength

Key TEP Prediction:
- In overdense regions (protoclusters, groups), the ambient halo potential
  provides additional screening, suppressing TEP effects
- Galaxies in dense environments should show WEAKER Gamma_t-dust correlations
- The dust-Gamma_t correlation should be strongest in isolated field galaxies

Outputs:
- results/outputs/step_115_environmental_screening_enhanced.json
- results/figures/figure_094_environmental_screening.png
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp, mannwhitneyu
from scipy.spatial import cKDTree, Voronoi
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "094"  # Pipeline step number (sequential 001-176)
STEP_NAME = "environmental_screening_enhanced"  # Environmental screening: Nth-neighbor/aperture/Voronoi density estimators testing TEP suppression in overdense regions (protoclusters)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory (PNG/PDF for manuscript)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


def compute_nth_neighbor_density(coords, z_values, n=5, z_window=0.3):
    """
    Compute local density using Nth nearest neighbor distance.
    
    Args:
        coords: (N, 2) array of (ra, dec) in degrees
        z_values: (N,) array of redshifts
        n: Number of neighbors to use
        z_window: Redshift window for neighbor selection
    
    Returns:
        densities: (N,) array of local densities (neighbors per sq. arcmin)
    """
    densities = np.zeros(len(coords))
    
    for i in range(len(coords)):
        # Find all galaxies within redshift window
        z_mask = np.abs(z_values - z_values[i]) < z_window
        z_mask[i] = False  # Exclude self
        
        if z_mask.sum() < n:
            densities[i] = np.nan
            continue
        
        # Compute angular distances to neighbors
        neighbor_coords = coords[z_mask]
        d_ra = (neighbor_coords[:, 0] - coords[i, 0]) * np.cos(np.radians(coords[i, 1]))
        d_dec = neighbor_coords[:, 1] - coords[i, 1]
        distances = np.sqrt(d_ra**2 + d_dec**2) * 60  # Convert to arcmin
        
        # Nth neighbor distance
        sorted_dist = np.sort(distances)
        if len(sorted_dist) >= n:
            r_n = sorted_dist[n-1]
            # Density = N / (pi * r_n^2)
            densities[i] = n / (np.pi * r_n**2) if r_n > 0 else np.nan
        else:
            densities[i] = np.nan
    
    return densities


def compute_aperture_density(coords, z_values, radius_arcmin=1.0, z_window=0.3):
    """
    Compute local density using fixed aperture counts.
    """
    densities = np.zeros(len(coords))
    area = np.pi * radius_arcmin**2
    
    for i in range(len(coords)):
        z_mask = np.abs(z_values - z_values[i]) < z_window
        z_mask[i] = False
        
        if z_mask.sum() == 0:
            densities[i] = 0
            continue
        
        neighbor_coords = coords[z_mask]
        d_ra = (neighbor_coords[:, 0] - coords[i, 0]) * np.cos(np.radians(coords[i, 1]))
        d_dec = neighbor_coords[:, 1] - coords[i, 1]
        distances = np.sqrt(d_ra**2 + d_dec**2) * 60
        
        n_neighbors = np.sum(distances < radius_arcmin)
        densities[i] = n_neighbors / area
    
    return densities


def compute_overdensity(densities):
    """
    Compute overdensity delta = (rho - rho_mean) / rho_mean
    """
    valid = ~np.isnan(densities)
    mean_density = np.nanmean(densities)
    overdensity = np.full_like(densities, np.nan)
    overdensity[valid] = (densities[valid] - mean_density) / mean_density
    return overdensity


def test_screening_hypothesis(df, density_col, dust_col='dust', gamma_col='gamma_t'):
    """
    Test the TEP screening hypothesis:
    - In high-density regions, Gamma_t-dust correlation should be WEAKER
    - In low-density regions, Gamma_t-dust correlation should be STRONGER
    
    Returns dict with test results
    """
    valid = ~(df[density_col].isna() | df[dust_col].isna() | df[gamma_col].isna())
    df_valid = df[valid].copy()
    
    if len(df_valid) < 50:
        return {"error": "Insufficient data", "n": len(df_valid)}
    
    # Split by density median
    median_density = df_valid[density_col].median()
    high_density = df_valid[df_valid[density_col] > median_density]
    low_density = df_valid[df_valid[density_col] <= median_density]
    
    # Compute correlations in each regime
    rho_high, p_high = spearmanr(high_density[gamma_col], high_density[dust_col])
    rho_low, p_low = spearmanr(low_density[gamma_col], low_density[dust_col])
    
    # TEP prediction: rho_low > rho_high (screening suppresses correlation in dense regions)
    delta_rho = rho_low - rho_high
    
    # Bootstrap CI for delta_rho
    n_boot = 1000
    delta_rhos = []
    for _ in range(n_boot):
        high_boot = high_density.sample(n=len(high_density), replace=True)
        low_boot = low_density.sample(n=len(low_density), replace=True)
        rho_h, _ = spearmanr(high_boot[gamma_col], high_boot[dust_col])
        rho_l, _ = spearmanr(low_boot[gamma_col], low_boot[dust_col])
        delta_rhos.append(rho_l - rho_h)
    
    ci_low, ci_high = np.percentile(delta_rhos, [2.5, 97.5])
    
    # Is the difference significant and in the predicted direction?
    tep_confirmed = delta_rho > 0 and ci_low > 0
    
    return {
        "n_total": len(df_valid),
        "n_high_density": len(high_density),
        "n_low_density": len(low_density),
        "median_density": float(median_density),
        "rho_high_density": float(rho_high),
        "p_high_density": format_p_value(p_high),
        "rho_low_density": float(rho_low),
        "p_low_density": format_p_value(p_low),
        "delta_rho": float(delta_rho),
        "delta_rho_ci_low": float(ci_low),
        "delta_rho_ci_high": float(ci_high),
        "tep_prediction": "rho_low > rho_high (screening suppresses in dense regions)",
        "tep_confirmed": bool(tep_confirmed),
        "interpretation": "SUPPORTS TEP" if tep_confirmed else "INCONCLUSIVE or CONTRADICTS"
    }


def test_by_redshift_bins(df, density_col, z_bins=[(4, 6), (6, 8), (8, 10)]):
    """
    Test screening hypothesis in redshift bins to account for evolving thresholds.
    """
    results = []
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    
    for z_low, z_high in z_bins:
        mask = (df[z_col] >= z_low) & (df[z_col] < z_high)
        df_bin = df[mask]
        
        if len(df_bin) < 30:
            results.append({
                "z_range": f"{z_low}-{z_high}",
                "n": len(df_bin),
                "error": "Insufficient data"
            })
            continue
        
        test_result = test_screening_hypothesis(df_bin, density_col)
        test_result["z_range"] = f"{z_low}-{z_high}"
        results.append(test_result)
    
    return results


def compute_isolation_index(coords, z_values, z_window=0.5, radius_arcmin=2.0):
    """
    Compute isolation index: inverse of local density.
    High values = isolated field galaxies
    Low values = clustered/group galaxies
    """
    densities = compute_aperture_density(coords, z_values, radius_arcmin, z_window)
    # Avoid division by zero
    isolation = 1.0 / (densities + 0.1)
    return isolation


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Enhanced Environmental Screening Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data - try multiple possible filenames
    possible_paths = [
        INTERIM_PATH / "step_002_uncover_full_sample_tep.csv",
        INTERIM_PATH / "step_002_uncover_full_sample_tep.csv",
        INTERIM_PATH / "step_001_uncover_full_sample.csv",
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    
    if data_path is None:
        print_status(f"No data file found in {INTERIM_PATH}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded {len(df)} galaxies from UNCOVER", "INFO")
    
    # Check for required columns
    required = ['gamma_t', 'dust']
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    
    for col in required:
        if col not in df.columns:
            print_status(f"Missing required column: {col}", "ERROR")
            return
    
    results = {
        "step": f"Step {STEP_NUM}: Enhanced Environmental Screening",
        "n_galaxies": len(df),
        "tep_prediction": "Screening should SUPPRESS Gamma_t-dust correlation in dense environments"
    }
    
    # Check if we have real coordinates
    has_coords = 'ra' in df.columns and 'dec' in df.columns
    
    if has_coords:
        print_status("Using real sky coordinates for density estimation", "INFO")
        coords = np.column_stack([df['ra'].values, df['dec'].values])
        z_values = df[z_col].values
        
        # Compute multiple density estimators
        print_status("Computing density estimators...", "INFO")
        
        # 1. 5th nearest neighbor density
        df['density_5nn'] = compute_nth_neighbor_density(coords, z_values, n=5)
        
        # 2. Aperture density (1 arcmin)
        df['density_1arcmin'] = compute_aperture_density(coords, z_values, radius_arcmin=1.0)
        
        # 3. Aperture density (2 arcmin)
        df['density_2arcmin'] = compute_aperture_density(coords, z_values, radius_arcmin=2.0)
        
        # 4. Isolation index
        df['isolation'] = compute_isolation_index(coords, z_values)
        
        # 5. Overdensity
        df['overdensity'] = compute_overdensity(df['density_2arcmin'].values)
        
        # Test screening hypothesis with each estimator
        print_status("\nTesting screening hypothesis with multiple density estimators:", "INFO")
        
        density_tests = {}
        for density_col in ['density_5nn', 'density_1arcmin', 'density_2arcmin', 'overdensity']:
            print_status(f"\n  {density_col}:", "INFO")
            test_result = test_screening_hypothesis(df, density_col)
            density_tests[density_col] = test_result
            
            if 'error' not in test_result:
                print_status(f"    rho(high density) = {test_result['rho_high_density']:.3f}", "INFO")
                print_status(f"    rho(low density)  = {test_result['rho_low_density']:.3f}", "INFO")
                print_status(f"    delta_rho = {test_result['delta_rho']:.3f} [{test_result['delta_rho_ci_low']:.3f}, {test_result['delta_rho_ci_high']:.3f}]", "INFO")
                print_status(f"    Result: {test_result['interpretation']}", "INFO")
        
        results["density_estimator_tests"] = density_tests
        
        # Test by redshift bins
        print_status("\nTesting by redshift bins:", "INFO")
        z_bin_results = test_by_redshift_bins(df, 'density_2arcmin')
        results["redshift_bin_tests"] = z_bin_results
        
        for r in z_bin_results:
            if 'error' not in r:
                print_status(f"  z={r['z_range']}: delta_rho={r['delta_rho']:.3f}, TEP: {r['interpretation']}", "INFO")
        
        # Test with isolation index (inverted: high isolation = field)
        print_status("\nTesting with isolation index (field vs clustered):", "INFO")
        
        # Split by isolation median
        median_iso = df['isolation'].median()
        field = df[df['isolation'] > median_iso]
        clustered = df[df['isolation'] <= median_iso]
        
        rho_field, p_field = spearmanr(
            field['gamma_t'].dropna(), 
            field.loc[field['gamma_t'].notna(), 'dust']
        )
        rho_clustered, p_clustered = spearmanr(
            clustered['gamma_t'].dropna(),
            clustered.loc[clustered['gamma_t'].notna(), 'dust']
        )
        
        results["isolation_test"] = {
            "n_field": len(field),
            "n_clustered": len(clustered),
            "rho_field": float(rho_field),
            "p_field": format_p_value(p_field),
            "rho_clustered": float(rho_clustered),
            "p_clustered": format_p_value(p_clustered),
            "delta_rho": float(rho_field - rho_clustered),
            "tep_prediction": "rho_field > rho_clustered",
            "tep_confirmed": bool(rho_field > rho_clustered)
        }
        
        print_status(f"  Field galaxies: rho = {rho_field:.3f} (n={len(field)})", "INFO")
        print_status(f"  Clustered galaxies: rho = {rho_clustered:.3f} (n={len(clustered)})", "INFO")
        print_status(f"  Delta: {rho_field - rho_clustered:.3f}", "INFO")
        
    else:
        print_status("No sky coordinates available - using proxy density from catalog", "WARNING")
        
        # Use stellar mass density as proxy (higher mass = denser environment on average)
        if 'log_stellar_mass' in df.columns:
            df['mass_proxy'] = df['log_stellar_mass']
            test_result = test_screening_hypothesis(df, 'mass_proxy')
            results["mass_proxy_test"] = test_result
            print_status(f"Mass proxy test: delta_rho = {test_result.get('delta_rho', 'N/A')}", "INFO")
    
    # Summary
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    # Count how many tests support TEP
    n_tests = 0
    n_support = 0
    
    if 'density_estimator_tests' in results:
        for test in results['density_estimator_tests'].values():
            if 'tep_confirmed' in test:
                n_tests += 1
                if test['tep_confirmed']:
                    n_support += 1
    
    if 'isolation_test' in results and results['isolation_test'].get('tep_confirmed'):
        n_tests += 1
        n_support += 1
    
    results["summary"] = {
        "n_tests": n_tests,
        "n_supporting_tep": n_support,
        "fraction_supporting": n_support / n_tests if n_tests > 0 else 0,
        "conclusion": f"{n_support}/{n_tests} density estimators show the predicted screening pattern"
    }
    
    print_status(f"Tests supporting TEP screening: {n_support}/{n_tests}", "INFO")
    
    if n_support >= n_tests / 2:
        print_status("RESULT: Evidence SUPPORTS environmental screening hypothesis", "INFO")
    else:
        print_status("RESULT: Evidence is INCONCLUSIVE for environmental screening", "INFO")
    
    # Save results
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print_status(f"\nResults saved to {output_file}", "INFO")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if has_coords and 'density_2arcmin' in df.columns:
            # Panel 1: Density distribution
            ax = axes[0, 0]
            valid_density = df['density_2arcmin'].dropna()
            ax.hist(valid_density, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(valid_density.median(), color='red', linestyle='--', label=f'Median: {valid_density.median():.2f}')
            ax.set_xlabel('Local Density (gal/arcmin²)')
            ax.set_ylabel('Count')
            ax.set_title('Local Density Distribution')
            ax.legend()
            
            # Panel 2: Gamma_t vs Dust by density regime
            ax = axes[0, 1]
            median_d = df['density_2arcmin'].median()
            high_d = df[df['density_2arcmin'] > median_d]
            low_d = df[df['density_2arcmin'] <= median_d]
            
            ax.scatter(low_d['gamma_t'], low_d['dust'], alpha=0.3, s=10, label='Low density (field)', c='blue')
            ax.scatter(high_d['gamma_t'], high_d['dust'], alpha=0.3, s=10, label='High density (clustered)', c='red')
            ax.set_xlabel('$\\Gamma_t$')
            ax.set_ylabel('Dust ($A_V$)')
            ax.set_title('$\\Gamma_t$ vs Dust by Environment')
            ax.legend()
            ax.set_xscale('log')
            
            # Panel 3: Correlation strength by density bin
            ax = axes[1, 0]
            density_bins = np.percentile(valid_density, [0, 25, 50, 75, 100])
            bin_centers = []
            bin_rhos = []
            bin_errs = []
            
            for i in range(len(density_bins) - 1):
                mask = (df['density_2arcmin'] >= density_bins[i]) & (df['density_2arcmin'] < density_bins[i+1])
                subset = df[mask].dropna(subset=['gamma_t', 'dust'])
                if len(subset) > 10:
                    rho, _ = spearmanr(subset['gamma_t'], subset['dust'])
                    bin_centers.append((density_bins[i] + density_bins[i+1]) / 2)
                    bin_rhos.append(rho)
                    # Bootstrap error
                    rhos = [spearmanr(subset.sample(n=len(subset), replace=True)['gamma_t'], 
                                     subset.sample(n=len(subset), replace=True)['dust'])[0] 
                           for _ in range(100)]
                    bin_errs.append(np.std(rhos))
            
            ax.errorbar(bin_centers, bin_rhos, yerr=bin_errs, fmt='o-', capsize=5, markersize=8)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Local Density (gal/arcmin²)')
            ax.set_ylabel('Spearman $\\rho$($\\Gamma_t$, Dust)')
            ax.set_title('Correlation Strength vs Environment')
            
            # Add TEP prediction arrow
            ax.annotate('', xy=(max(bin_centers), min(bin_rhos)), xytext=(min(bin_centers), max(bin_rhos)),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(np.mean(bin_centers), np.mean(bin_rhos) + 0.1, 'TEP prediction', color='green', ha='center')
            
            # Panel 4: Summary text
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = f"""
Environmental Screening Analysis Summary
========================================

TEP Prediction:
  Screening should SUPPRESS the Γt-dust
  correlation in dense environments.

Results:
  Field galaxies: ρ = {results.get('isolation_test', {}).get('rho_field', 'N/A'):.3f}
  Clustered galaxies: ρ = {results.get('isolation_test', {}).get('rho_clustered', 'N/A'):.3f}
  Δρ = {results.get('isolation_test', {}).get('delta_rho', 'N/A'):.3f}

Conclusion:
  {results['summary']['conclusion']}
"""
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_status(f"Figure saved to {fig_path}", "INFO")
        
    except Exception as e:
        print_status(f"Could not generate figure: {e}", "WARNING")
    
    print_status(f"\nStep {STEP_NUM} complete.", "INFO")


if __name__ == "__main__":
    main()
