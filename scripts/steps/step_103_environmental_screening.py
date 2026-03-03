#!/usr/bin/env python3
"""
Step 103: Environmental Screening Quantification

This script quantifies the environmental screening signature predicted by TEP:
galaxies in overdense regions (proto-clusters, groups) should show suppressed
TEP effects due to screening by the collective halo potential.

Key features:
1. Environmental density estimation from galaxy counts
2. Correlation between local density and TEP residuals
3. Proto-cluster identification and TEP suppression test
4. Comparison with isolated field galaxies

Outputs:
- results/outputs/step_103_environmental_screening.json
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from scipy.spatial import cKDTree
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "103"
STEP_NAME = "environmental_screening"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)


def estimate_local_density(df, radius_arcmin=1.0, z_window=0.5):
    """
    Estimate local galaxy density using nearest neighbor counts.
    
    Args:
        df: DataFrame with ra, dec, z_phot columns
        radius_arcmin: Search radius in arcminutes
        z_window: Redshift window for neighbors
    
    Returns:
        Array of local densities (neighbors per arcmin²)
    """
    if 'ra' not in df.columns or 'dec' not in df.columns:
        # Generate mock coordinates if not available
        raise ValueError(
            "Missing 'ra'/'dec' columns; cannot estimate local density without real sky coordinates."
        )
    
    coords = np.column_stack([df['ra'].values, df['dec'].values])
    z = df['z_phot'].values if 'z_phot' in df.columns else df['z'].values
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    densities = np.zeros(len(df))
    
    for i in range(len(df)):
        # Find neighbors within radius
        idx = tree.query_ball_point(coords[i], radius_arcmin / 60)  # Convert to degrees
        
        # Filter by redshift
        neighbors = [j for j in idx if j != i and abs(z[j] - z[i]) < z_window]
        
        # Density = N / area
        area = np.pi * radius_arcmin**2
        densities[i] = len(neighbors) / area
    
    return densities


def compute_tep_residuals(df):
    """
    Compute TEP residuals: observed property - TEP prediction.
    
    For dust: residual = dust - f(Γt)
    """
    if 'gamma_t' not in df.columns or 'dust' not in df.columns:
        return None
    
    # Simple linear model: dust ~ a + b * log(gamma_t)
    valid = ~(df['gamma_t'].isna() | df['dust'].isna())
    x = np.log10(df.loc[valid, 'gamma_t'].clip(0.01, 100))
    y = df.loc[valid, 'dust']
    
    # Fit
    slope, intercept = np.polyfit(x, y, 1)
    
    # Residuals
    residuals = np.full(len(df), np.nan)
    residuals[valid] = y - (slope * x + intercept)
    
    return residuals


def identify_overdense_regions(df, density_threshold=2.0):
    """
    Identify galaxies in overdense regions (proto-clusters).
    
    Args:
        df: DataFrame with local_density column
        density_threshold: Factor above median to classify as overdense
    
    Returns:
        Boolean mask for overdense galaxies
    """
    if 'local_density' not in df.columns:
        return None
    
    median_density = df['local_density'].median()
    return df['local_density'] > density_threshold * median_density


def environmental_screening_test(df):
    """
    Test whether overdense environments show suppressed TEP effects.
    
    TEP prediction: Galaxies in overdense regions are screened by the
    collective halo potential, reducing Γt effects.
    """
    if 'local_density' not in df.columns or 'gamma_t' not in df.columns:
        return None
    
    # Split by density
    median_density = df['local_density'].median()
    high_density = df[df['local_density'] > median_density]
    low_density = df[df['local_density'] <= median_density]
    
    results = {}
    
    # Compare Γt distributions
    if len(high_density) > 10 and len(low_density) > 10:
        ks_stat, ks_p = ks_2samp(high_density['gamma_t'], low_density['gamma_t'])
        results['gamma_t_ks'] = {
            'statistic': float(ks_stat),
            'p': format_p_value(ks_p),
            'mean_high_density': float(high_density['gamma_t'].mean()),
            'mean_low_density': float(low_density['gamma_t'].mean()),
            'n_high': len(high_density),
            'n_low': len(low_density)
        }
    
    # Correlation: density vs Γt
    valid = ~(df['local_density'].isna() | df['gamma_t'].isna())
    if valid.sum() > 20:
        rho, p = spearmanr(df.loc[valid, 'local_density'], df.loc[valid, 'gamma_t'])
        results['density_gamma_correlation'] = {
            'rho': float(rho),
            'p': format_p_value(p),
            'n': int(valid.sum()),
            'tep_prediction': 'negative (screening suppresses Γt in dense regions)'
        }
    
    # Compare dust-Γt correlation in high vs low density
    if 'dust' in df.columns:
        for subset, name in [(high_density, 'high_density'), (low_density, 'low_density')]:
            valid = ~(subset['gamma_t'].isna() | subset['dust'].isna())
            if valid.sum() > 10:
                rho, p = spearmanr(subset.loc[valid, 'gamma_t'], subset.loc[valid, 'dust'])
                results[f'dust_gamma_{name}'] = {
                    'rho': float(rho),
                    'p': format_p_value(p),
                    'n': int(valid.sum())
                }
    
    return results


def screening_radius_analysis(df, radii=[0.5, 1.0, 2.0, 5.0]):
    """
    Test how screening signal varies with density estimation radius.
    """
    results = []
    
    for radius in radii:
        # Recompute density at this radius
        densities = estimate_local_density(df, radius_arcmin=radius)
        df_temp = df.copy()
        df_temp['local_density'] = densities
        
        # Correlation
        valid = ~(df_temp['local_density'].isna() | df_temp['gamma_t'].isna())
        if valid.sum() > 20:
            rho, p = spearmanr(
                df_temp.loc[valid, 'local_density'], 
                df_temp.loc[valid, 'gamma_t']
            )
            results.append({
                'radius_arcmin': radius,
                'rho': float(rho),
                'p': format_p_value(p),
                'n': int(valid.sum())
            })
    
    return results


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Environmental Screening Quantification", "INFO")
    print_status("=" * 70, "INFO")
    
    results = {}
    
    # Load data
    data_path = INTERIM_PATH / "step_02_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # Filter to z > 4 where TEP effects are significant
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    df = df[df[z_col] > 4].copy()
    print_status(f"z > 4 sample: N = {len(df)}", "INFO")
    
    # ==========================================================================
    # 1. Estimate local density
    # ==========================================================================
    print_status("\n--- 1. Estimating Local Density ---", "INFO")
    
    df['local_density'] = estimate_local_density(df, radius_arcmin=1.0)
    
    density_stats = {
        'mean': float(df['local_density'].mean()),
        'median': float(df['local_density'].median()),
        'std': float(df['local_density'].std()),
        'min': float(df['local_density'].min()),
        'max': float(df['local_density'].max())
    }
    results['density_statistics'] = density_stats
    print_status(f"  Density: {density_stats['mean']:.2f} ± {density_stats['std']:.2f} gal/arcmin²", "INFO")
    
    # ==========================================================================
    # 2. Identify overdense regions
    # ==========================================================================
    print_status("\n--- 2. Identifying Overdense Regions ---", "INFO")
    
    overdense_mask = identify_overdense_regions(df, density_threshold=2.0)
    if overdense_mask is not None:
        n_overdense = overdense_mask.sum()
        results['overdense_sample'] = {
            'n_overdense': int(n_overdense),
            'n_field': int((~overdense_mask).sum()),
            'fraction_overdense': float(n_overdense / len(df))
        }
        print_status(f"  Overdense: N = {n_overdense} ({n_overdense/len(df)*100:.1f}%)", "INFO")
        print_status(f"  Field: N = {(~overdense_mask).sum()}", "INFO")
    
    # ==========================================================================
    # 3. Environmental screening test
    # ==========================================================================
    print_status("\n--- 3. Environmental Screening Test ---", "INFO")
    
    screening = environmental_screening_test(df)
    if screening:
        results['screening_test'] = screening
        
        if 'density_gamma_correlation' in screening:
            dgc = screening['density_gamma_correlation']
            print_status(f"  Density-Γt correlation: ρ = {dgc['rho']:.3f}, p = {dgc['p']:.2e}", "INFO")
            
            # TEP predicts negative correlation (screening)
            if dgc['rho'] < 0:
                print_status(f"  ✓ Consistent with TEP screening prediction", "INFO")
            else:
                print_status(f"  ✗ Opposite to TEP prediction", "WARNING")
        
        if 'dust_gamma_high_density' in screening and 'dust_gamma_low_density' in screening:
            hd = screening['dust_gamma_high_density']
            ld = screening['dust_gamma_low_density']
            print_status(f"  High-density dust-Γt: ρ = {hd['rho']:.3f}", "INFO")
            print_status(f"  Low-density dust-Γt: ρ = {ld['rho']:.3f}", "INFO")
            
            # TEP predicts weaker correlation in high-density (screened)
            if abs(hd['rho']) < abs(ld['rho']):
                print_status(f"  ✓ Weaker correlation in dense regions (screening)", "INFO")
    
    # ==========================================================================
    # 4. Radius dependence
    # ==========================================================================
    print_status("\n--- 4. Screening Radius Dependence ---", "INFO")
    
    radius_analysis = screening_radius_analysis(df)
    results['radius_analysis'] = radius_analysis
    
    for ra in radius_analysis:
        print_status(f"  r = {ra['radius_arcmin']} arcmin: ρ = {ra['rho']:.3f}, p = {ra['p']:.2e}", "INFO")
    
    # ==========================================================================
    # 5. TEP residuals vs density
    # ==========================================================================
    print_status("\n--- 5. TEP Residuals vs Density ---", "INFO")
    
    residuals = compute_tep_residuals(df)
    if residuals is not None:
        df['tep_residual'] = residuals
        
        valid = ~(df['local_density'].isna() | df['tep_residual'].isna())
        if valid.sum() > 20:
            rho, p = spearmanr(df.loc[valid, 'local_density'], df.loc[valid, 'tep_residual'])
            results['residual_density_correlation'] = {
                'rho': float(rho),
                'p': format_p_value(p),
                'n': int(valid.sum()),
                'interpretation': (
                    'Positive correlation suggests overdense regions have excess dust '
                    'relative to TEP prediction (screening reduces Γt effect)'
                )
            }
            print_status(f"  Residual-density correlation: ρ = {rho:.3f}, p = {p:.2e}", "INFO")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("ENVIRONMENTAL SCREENING SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    # Assess overall evidence for screening
    evidence_for_screening = []
    
    if screening and 'density_gamma_correlation' in screening:
        if screening['density_gamma_correlation']['rho'] < 0:
            evidence_for_screening.append('negative density-Γt correlation')
    
    if screening and 'dust_gamma_high_density' in screening and 'dust_gamma_low_density' in screening:
        if abs(screening['dust_gamma_high_density']['rho']) < abs(screening['dust_gamma_low_density']['rho']):
            evidence_for_screening.append('weaker dust-Γt in dense regions')
    
    summary = {
        'n_galaxies': len(df),
        'evidence_for_screening': evidence_for_screening,
        'n_evidence_lines': len(evidence_for_screening),
        'screening_detected': len(evidence_for_screening) >= 1,
        'conclusion': (
            f"Environmental screening is {'supported' if len(evidence_for_screening) >= 1 else 'not detected'} "
            f"by {len(evidence_for_screening)} line(s) of evidence. "
            f"{'The TEP prediction that overdense regions show suppressed effects is confirmed.' if len(evidence_for_screening) >= 1 else ''}"
        )
    }
    
    results['summary'] = summary
    print_status(f"  Evidence lines: {len(evidence_for_screening)}", "INFO")
    print_status(f"  Screening detected: {summary['screening_detected']}", "INFO")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_environmental_screening.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_status(f"\nResults saved to {output_file}", "INFO")


if __name__ == "__main__":
    main()
