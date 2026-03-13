#!/usr/bin/env python3
"""
Step 98: Independent Age Validation via Balmer Absorption

This script prepares predictions for independent age validation using
Balmer absorption line strengths, which are independent of SED fitting.

Key insight:
- SED-derived ages depend on M/L assumptions (potentially circular with TEP)
- Balmer absorption (Hδ, Hγ) directly measures stellar age via A-star fraction
- TEP predicts: Balmer age should correlate with Γt at fixed redshift

This breaks the M/L circularity concern by providing an independent age proxy.

Outputs:
- results/outputs/step_078_independent_age_validation.json
- Predictions for future spectroscopic observations
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "078"  # Pipeline step number (sequential 001-176)
STEP_NAME = "independent_age_validation"  # Independent age validation: Balmer absorption (Hδ, Hγ) as independent age proxy breaking M/L circularity (Worthey 1994, Kauffmann 2003 SSP models)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)

LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


def predict_balmer_strength(age_gyr, metallicity=0.0):
    """
    Predict Hδ equivalent width from stellar age.
    
    Based on Worthey (1994) and Kauffmann et al. (2003):
    - Young populations (< 1 Gyr): Strong Balmer absorption (A-stars)
    - Intermediate (1-3 Gyr): Peak Balmer strength
    - Old (> 3 Gyr): Declining Balmer strength
    
    Returns EW(Hδ) in Angstroms.
    """
    # Simplified model based on SSP tracks
    # Peak at ~1 Gyr, declining after
    log_age = np.log10(np.clip(age_gyr, 0.01, 14))
    
    # Gaussian peak at log(age) ~ 9.0 (1 Gyr)
    ew_hd = 8.0 * np.exp(-0.5 * ((log_age - 0.0) / 0.5)**2)
    
    # Metallicity correction: higher Z -> weaker Balmer
    ew_hd *= (1 - 0.3 * metallicity)
    
    return max(0, ew_hd)


def compute_tep_age_prediction(df):
    """
    Compute TEP-predicted ages and corresponding Balmer strengths.
    """
    results = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('mwa')) or pd.isna(row.get('gamma_t')):
            continue
        
        # Standard age (from SED)
        age_sed = row['mwa'] / 1e9  # Convert to Gyr
        
        # TEP-corrected age
        gamma_t = row['gamma_t']
        age_tep = age_sed / gamma_t if gamma_t > 0 else age_sed
        
        # Predicted Balmer strengths
        ew_std = predict_balmer_strength(age_sed)
        ew_tep = predict_balmer_strength(age_tep)
        
        results.append({
            'id': row.get('id', ''),
            'z': row.get('z_phot', np.nan),
            'log_Mstar': row.get('log_Mstar', np.nan),
            'gamma_t': gamma_t,
            'age_sed_gyr': age_sed,
            'age_tep_gyr': age_tep,
            'ew_hd_std': ew_std,
            'ew_hd_tep': ew_tep,
            'delta_ew': ew_tep - ew_std
        })
    
    return pd.DataFrame(results)


def generate_spectroscopic_predictions(df_pred):
    """
    Generate specific predictions for spectroscopic follow-up.
    """
    predictions = {}
    
    # Prediction 1: Correlation between Γt and Balmer strength
    # TEP predicts: Higher Γt -> older effective age -> weaker Balmer
    # (for ages > 1 Gyr where Balmer declines with age)
    
    # Split by mass
    high_mass = df_pred[df_pred['log_Mstar'] > 10]
    low_mass = df_pred[df_pred['log_Mstar'] <= 10]
    
    if len(high_mass) > 10:
        # Check for valid data before computing correlation
        valid_gamma = high_mass['gamma_t'].notna()
        valid_ew = high_mass['ew_hd_tep'].notna()
        valid_both = valid_gamma & valid_ew
        
        if valid_both.sum() >= 10:
            rho_high, p_high = spearmanr(high_mass.loc[valid_both, 'gamma_t'], 
                                         high_mass.loc[valid_both, 'ew_hd_tep'])
            predictions['high_mass_correlation'] = {
                'rho': float(rho_high) if not np.isnan(rho_high) else None,
                'p': format_p_value(p_high) if not np.isnan(p_high) else None,
                'n': int(valid_both.sum()),
                'interpretation': 'TEP predicts negative correlation (higher Γt -> older -> weaker Balmer)'
            }
        else:
            predictions['high_mass_correlation'] = {
                'rho': None,
                'p': None,
                'n': int(valid_both.sum()),
                'interpretation': 'TEP predicts negative correlation (higher Γt -> older -> weaker Balmer)',
                'note': 'Insufficient valid data for correlation'
            }
    
    # Prediction 2: Mass-dependent Balmer strength at fixed z
    # TEP predicts: More massive galaxies should have WEAKER Balmer (older)
    # Standard physics: No strong mass dependence expected
    
    z_bins = [(4, 6), (6, 8), (8, 10)]
    mass_balmer = []
    
    for z_lo, z_hi in z_bins:
        subset = df_pred[(df_pred['z'] >= z_lo) & (df_pred['z'] < z_hi)]
        if len(subset) > 20:
            # Check for valid data
            valid_mass = subset['log_Mstar'].notna()
            valid_ew = subset['ew_hd_tep'].notna()
            valid_both = valid_mass & valid_ew
            
            if valid_both.sum() >= 10:
                rho, p = spearmanr(subset.loc[valid_both, 'log_Mstar'], 
                                   subset.loc[valid_both, 'ew_hd_tep'])
                mass_balmer.append({
                    'z_range': f'{z_lo}-{z_hi}',
                    'n': int(valid_both.sum()),
                    'rho_mass_balmer': float(rho) if not np.isnan(rho) else None,
                    'p': format_p_value(p)
                })
            else:
                mass_balmer.append({
                    'z_range': f'{z_lo}-{z_hi}',
                    'n': int(valid_both.sum()),
                    'rho_mass_balmer': None,
                    'p': None,
                    'note': 'Insufficient valid data for correlation'
                })
    
    predictions['mass_balmer_by_z'] = mass_balmer
    
    # Prediction 3: Discriminating power
    # Calculate the expected difference between TEP and standard predictions
    mean_delta = df_pred['delta_ew'].mean()
    std_delta = df_pred['delta_ew'].std()
    
    predictions['discriminating_power'] = {
        'mean_delta_ew': float(mean_delta),
        'std_delta_ew': float(std_delta),
        'effect_size': float(abs(mean_delta) / std_delta) if std_delta > 0 else 0,
        'detectable': bool(abs(mean_delta) > 0.5)  # 0.5 Å is typical measurement precision
    }
    
    return predictions


def identify_priority_targets(df_pred, n_targets=20):
    """
    Identify highest-priority targets for spectroscopic follow-up.
    
    Criteria:
    1. High Γt (strong TEP effect)
    2. z > 6 (where TEP effects are strongest)
    3. Large predicted delta_ew (discriminating power)
    """
    # Filter to z > 6
    df_high_z = df_pred[df_pred['z'] > 6].copy()
    
    if len(df_high_z) == 0:
        return []
    
    # Score by discriminating power
    df_high_z['priority_score'] = (
        np.abs(df_high_z['delta_ew']) * 
        df_high_z['gamma_t'] * 
        (1 + df_high_z['z'] - 6)
    )
    
    # Top targets
    top = df_high_z.nlargest(min(n_targets, len(df_high_z)), 'priority_score')
    
    targets = []
    for _, row in top.iterrows():
        targets.append({
            'id': row['id'],
            'z': float(row['z']),
            'log_Mstar': float(row['log_Mstar']),
            'gamma_t': float(row['gamma_t']),
            'predicted_ew_tep': float(row['ew_hd_tep']),
            'predicted_ew_std': float(row['ew_hd_std']),
            'priority_score': float(row['priority_score'])
        })
    
    return targets


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Independent Age Validation via Balmer Absorption", "INFO")
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
    # 1. Compute TEP age predictions
    # ==========================================================================
    print_status("\n--- 1. Computing TEP Age Predictions ---", "INFO")
    
    df_pred = compute_tep_age_prediction(df)
    print_status(f"  Generated predictions for {len(df_pred)} galaxies", "INFO")
    
    # Summary statistics
    results['prediction_summary'] = {
        'n_galaxies': len(df_pred),
        'mean_age_sed': float(df_pred['age_sed_gyr'].mean()),
        'mean_age_tep': float(df_pred['age_tep_gyr'].mean()),
        'mean_gamma_t': float(df_pred['gamma_t'].mean()),
        'mean_delta_ew': float(df_pred['delta_ew'].mean())
    }
    
    # ==========================================================================
    # 2. Generate spectroscopic predictions
    # ==========================================================================
    print_status("\n--- 2. Spectroscopic Predictions ---", "INFO")
    
    predictions = generate_spectroscopic_predictions(df_pred)
    results['spectroscopic_predictions'] = predictions
    
    if 'high_mass_correlation' in predictions and predictions['high_mass_correlation'].get('rho') is not None:
        hm = predictions['high_mass_correlation']
        print_status(f"  High-mass Γt-Balmer correlation: ρ = {hm['rho']:.3f}", "INFO")
    
    if 'discriminating_power' in predictions:
        dp = predictions['discriminating_power']
        print_status(f"  Effect size (Cohen's d): {dp['effect_size']:.2f}", "INFO")
        print_status(f"  Detectable with NIRSpec: {dp['detectable']}", "INFO")
    
    # ==========================================================================
    # 3. Priority targets
    # ==========================================================================
    print_status("\n--- 3. Priority Targets for Follow-up ---", "INFO")
    
    targets = identify_priority_targets(df_pred)
    results['priority_targets'] = targets
    
    print_status(f"  Identified {len(targets)} priority targets", "INFO")
    if targets:
        print_status(f"  Top target: z={targets[0]['z']:.2f}, log M*={targets[0]['log_Mstar']:.1f}, Γt={targets[0]['gamma_t']:.2f}", "INFO")
    
    # ==========================================================================
    # 4. Falsification criteria
    # ==========================================================================
    print_status("\n--- 4. Falsification Criteria ---", "INFO")
    
    falsification = {
        'criterion_1': {
            'test': 'Balmer strength should correlate with Γt at fixed z',
            'null_hypothesis': 'No correlation (ρ = 0)',
            'tep_prediction': 'Negative correlation (ρ < -0.3) for ages > 1 Gyr',
            'falsified_if': 'Positive correlation (ρ > +0.2) observed'
        },
        'criterion_2': {
            'test': 'Mass-Balmer correlation should match TEP prediction',
            'null_hypothesis': 'No mass dependence',
            'tep_prediction': 'Negative correlation (more massive = weaker Balmer)',
            'falsified_if': 'Positive correlation observed'
        },
        'criterion_3': {
            'test': 'Balmer ages should be consistent with TEP-corrected SED ages',
            'null_hypothesis': 'Balmer ages match uncorrected SED ages',
            'tep_prediction': 'Balmer ages closer to TEP-corrected ages',
            'falsified_if': 'Balmer ages match uncorrected SED ages better'
        }
    }
    
    results['falsification_criteria'] = falsification
    
    for key, crit in falsification.items():
        print_status(f"  {key}: {crit['test']}", "INFO")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("INDEPENDENT AGE VALIDATION SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    summary = {
        'method': 'Balmer absorption (Hδ, Hγ) as independent age proxy',
        'breaks_circularity': True,
        'reason': 'Balmer strength depends directly on A-star fraction, not M/L assumptions',
        'n_priority_targets': len(targets),
        'recommended_instrument': 'JWST/NIRSpec medium resolution (R~1000)',
        'exposure_time': '~2-4 hours per target for S/N > 10 in Hδ'
    }
    
    results['summary'] = summary
    print_status(f"  Method: {summary['method']}", "INFO")
    print_status(f"  Breaks M/L circularity: {summary['breaks_circularity']}", "INFO")
    print_status(f"  Priority targets identified: {summary['n_priority_targets']}", "INFO")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_independent_age_validation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_status(f"\nResults saved to {output_file}", "INFO")
    
    # Save priority targets as CSV for easy reference
    if targets:
        targets_df = pd.DataFrame(targets)
        targets_csv = OUTPUT_PATH / f"step_{STEP_NUM}_priority_targets.csv"
        targets_df.to_csv(targets_csv, index=False)
        print_status(f"Priority targets saved to {targets_csv}", "INFO")


if __name__ == "__main__":
    main()
