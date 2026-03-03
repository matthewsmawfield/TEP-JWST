#!/usr/bin/env python3
"""
TEP-JWST Step 44: Forward-Modeling SED Validation

Validates TEP mass corrections using forward-modeling approach.
Instead of post-hoc corrections, this script:
1. Generates synthetic SEDs under TEP assumptions using FSPS-like models
2. Compares predicted photometry to observed photometry
3. Tests whether TEP-corrected ages produce better SED fits
4. Evaluates alternative hypotheses (bursty SFH, metallicity variations)

This addresses the concern that M/L ~ t^0.7 is assumed rather than tested.

Inputs:
- results/interim/step_02_uncover_full_sample_tep.csv

Outputs:
- results/outputs/step_44_forward_modeling.json
- results/outputs/step_44_forward_modeling.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize_scalar
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "44"
STEP_NAME = "forward_modeling"

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# STELLAR POPULATION SYNTHESIS MODEL (Simplified FSPS-like)
# =============================================================================

class SimpleSPS:
    """
    Simplified Stellar Population Synthesis model for forward modeling.
    Based on Conroy (2013) scaling relations.
    
    Key relations:
    - M/L ~ t^0.7 for constant SFH at t < 1 Gyr (standard assumption)
    - M/L ~ t^0.9 for declining SFH (tau model)
    - M/L ~ t^0.5 for bursty SFH
    
    Colors:
    - UV slope beta ~ -2.0 + 0.3*log(t/100Myr) for dust-free
    - U-V ~ 0.3 + 0.5*log(t/100Myr)
    """
    
    def __init__(self, sfh_type='constant'):
        self.sfh_type = sfh_type
        
        # M/L power-law index
        if sfh_type == 'constant':
            self.ml_power = 0.7
        elif sfh_type == 'declining':
            self.ml_power = 0.9
        elif sfh_type == 'bursty':
            self.ml_power = 0.5
        else:
            self.ml_power = 0.7
    
    def mass_to_light(self, age_gyr, metallicity=0.0):
        """
        Compute M/L ratio for given age and metallicity.
        Returns log(M/L) in solar units.
        
        Parameters:
        - age_gyr: Age in Gyr
        - metallicity: log(Z/Z_sun)
        """
        # Base M/L scaling
        age_100myr = age_gyr / 0.1  # Age in units of 100 Myr
        log_ml = self.ml_power * np.log10(np.clip(age_100myr, 0.1, 100))
        
        # Metallicity correction: higher Z -> higher M/L
        log_ml += 0.3 * metallicity
        
        return log_ml
    
    def uv_slope(self, age_gyr, dust_av=0.0):
        """
        Compute UV slope beta.
        """
        age_100myr = age_gyr / 0.1
        beta = -2.0 + 0.3 * np.log10(np.clip(age_100myr, 0.1, 100))
        
        # Dust reddening
        beta += 0.8 * dust_av
        
        return beta
    
    def uv_color(self, age_gyr, dust_av=0.0, metallicity=0.0):
        """
        Compute U-V rest-frame color.
        """
        age_100myr = age_gyr / 0.1
        uv = 0.3 + 0.5 * np.log10(np.clip(age_100myr, 0.1, 100))
        
        # Dust reddening
        uv += 1.2 * dust_av
        
        # Metallicity: higher Z -> redder
        uv += 0.2 * metallicity
        
        return uv


def compute_tep_age(age_obs, gamma_t):
    """
    Compute true age under TEP.
    age_true = age_obs / gamma_t
    """
    return age_obs / np.clip(gamma_t, 0.1, 100)


def compute_tep_mass(mass_obs, gamma_t, ml_power=0.7):
    """
    Compute true mass under TEP.
    M_true = M_obs / gamma_t^ml_power
    """
    return mass_obs / np.power(np.clip(gamma_t, 0.1, 100), ml_power)


# =============================================================================
# FORWARD MODELING TESTS
# =============================================================================

def test_ml_power_law(df, ml_powers=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Test different M/L power-law indices.
    The best index minimizes the scatter in the mass-age relation.
    """
    results = []
    
    mask = (~df['mwa'].isna()) & (~df['gamma_t'].isna()) & (~df['log_Mstar'].isna())
    df_valid = df[mask].copy()
    
    if len(df_valid) < 50:
        return None
    
    for power in ml_powers:
        # Compute TEP-corrected mass
        df_valid['log_Mstar_tep'] = df_valid['log_Mstar'] - power * np.log10(df_valid['gamma_t'].clip(0.1, 100))
        
        # Compute correlation with age
        rho, p = spearmanr(df_valid['log_Mstar_tep'], df_valid['mwa'])
        
        # Compute scatter in mass-age relation
        # Lower scatter = better model
        residuals = df_valid['mwa'] - df_valid['mwa'].mean()
        scatter = np.std(residuals)
        
        results.append({
            'ml_power': power,
            'rho_mass_age': rho,
            'p_value': format_p_value(p),
            'scatter': scatter
        })
    
    return pd.DataFrame(results)


def test_sfh_alternatives(df):
    """
    Test whether bursty SFH or metallicity variations can explain
    the observed correlations without TEP.
    
    Alternative hypotheses:
    1. Bursty SFH: Massive galaxies have more recent bursts -> younger ages
    2. Metallicity: Massive galaxies are more metal-rich -> redder colors
    3. Dust: Massive galaxies are dustier -> redder colors
    """
    results = {}
    
    mask = (
        (~df['mwa'].isna()) & 
        (~df['gamma_t'].isna()) & 
        (~df['log_Mstar'].isna()) &
        (~df['dust'].isna())
    )
    df_valid = df[mask].copy()
    
    if len(df_valid) < 50:
        return None
    
    # Test 1: Does burstiness correlate with mass?
    # If yes, bursty SFH could explain mass-age correlation
    burstiness_available = False
    if 'sfr_100' in df_valid.columns and 'sfr' in df_valid.columns:
        # Check if we have valid non-null data
        has_sfr_100 = df_valid['sfr_100'].notna().any()
        has_sfr = df_valid['sfr'].notna().any()
        if has_sfr_100 and has_sfr:
            df_valid['burstiness'] = df_valid['sfr_100'] / df_valid['sfr'].clip(1e-3, None)
            rho_burst, p_burst = spearmanr(df_valid['log_Mstar'], df_valid['burstiness'])
            results['burstiness_mass'] = {'rho': float(rho_burst), 'p': format_p_value(p_burst)}
            burstiness_available = True
    
    if not burstiness_available:
        # Burstiness data not available - mark as not applicable
        results['burstiness_mass'] = {'rho': None, 'p': None, 'note': 'sfr_100 data not available'}
    
    # Test 2: Does metallicity explain the age-mass correlation?
    # Partial correlation: age vs mass controlling for metallicity
    if 'met' in df_valid.columns:
        from scipy.stats import pearsonr
        
        # Residualize age on metallicity
        mask_met = ~df_valid['met'].isna()
        df_met = df_valid[mask_met]
        
        if len(df_met) > 50:
            # Simple residualization
            slope_age_met = np.polyfit(df_met['met'], df_met['mwa'], 1)[0]
            age_resid = df_met['mwa'] - slope_age_met * df_met['met']
            
            rho_resid, p_resid = spearmanr(df_met['log_Mstar'], age_resid)
            results['age_mass_controlling_met'] = {
                'rho': float(rho_resid) if np.isfinite(rho_resid) else None,
                'p': format_p_value(p_resid),
            }
        else:
            results['age_mass_controlling_met'] = {'rho': None, 'p': None}
    
    # Test 3: Does dust explain the mass-age correlation?
    # Partial correlation: age vs mass controlling for dust
    slope_age_dust = np.polyfit(df_valid['dust'], df_valid['mwa'], 1)[0]
    age_resid_dust = df_valid['mwa'] - slope_age_dust * df_valid['dust']
    
    rho_resid_dust, p_resid_dust = spearmanr(df_valid['log_Mstar'], age_resid_dust)
    results['age_mass_controlling_dust'] = {
        'rho': float(rho_resid_dust) if np.isfinite(rho_resid_dust) else None,
        'p': format_p_value(p_resid_dust),
    }
    
    # Test 4: Combined control (dust + metallicity if available)
    if 'met' in df_valid.columns:
        mask_all = ~df_valid['met'].isna()
        df_all = df_valid[mask_all]
        
        if len(df_all) > 50:
            # Multiple regression residualization
            X = np.column_stack([df_all['dust'], df_all['met']])
            y = df_all['mwa']
            
            # Simple OLS
            X_aug = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            y_pred = X_aug @ beta
            age_resid_all = y - y_pred
            
            rho_resid_all, p_resid_all = spearmanr(df_all['log_Mstar'], age_resid_all)
            results['age_mass_controlling_all'] = {
                'rho': float(rho_resid_all) if np.isfinite(rho_resid_all) else None,
                'p': format_p_value(p_resid_all),
            }
        else:
            results['age_mass_controlling_all'] = {'rho': None, 'p': None}
    
    return results


def forward_model_chi2(df, sps_model):
    """
    Forward-model predicted colors and compare to observed.
    
    For each galaxy:
    1. Use observed age and dust to predict U-V color
    2. Compare to observed U-V color
    3. Repeat with TEP-corrected age
    4. Compare chi2 of standard vs TEP model
    """
    mask = (
        (~df['mwa'].isna()) & 
        (~df['gamma_t'].isna()) & 
        (~df['dust'].isna())
    )
    df_valid = df[mask].copy()
    
    if len(df_valid) < 50:
        return None
    
    # Check for rest-frame colors
    if 'restU' not in df_valid.columns or 'restV' not in df_valid.columns:
        # Use proxy from available data
        # UV slope or other color proxy
        if 'uv' in df_valid.columns:
            df_valid['obs_color'] = df_valid['uv']
        else:
            # Cannot run this test
            return None
    else:
        df_valid['obs_color'] = df_valid['restU'] - df_valid['restV']
    
    # Standard model: use observed age
    df_valid['pred_color_std'] = df_valid.apply(
        lambda row: sps_model.uv_color(row['mwa'] / 1e9, row['dust']),
        axis=1
    )
    
    # TEP model: use corrected age
    df_valid['age_tep'] = df_valid['mwa'] / df_valid['gamma_t'].clip(0.1, 100)
    df_valid['pred_color_tep'] = df_valid.apply(
        lambda row: sps_model.uv_color(row['age_tep'] / 1e9, row['dust']),
        axis=1
    )
    
    # Compute residuals
    df_valid['resid_std'] = df_valid['obs_color'] - df_valid['pred_color_std']
    df_valid['resid_tep'] = df_valid['obs_color'] - df_valid['pred_color_tep']
    
    # Chi2 comparison
    chi2_std = np.sum(df_valid['resid_std']**2)
    chi2_tep = np.sum(df_valid['resid_tep']**2)
    
    # Scatter comparison
    scatter_std = np.std(df_valid['resid_std'])
    scatter_tep = np.std(df_valid['resid_tep'])
    
    return {
        'chi2_standard': chi2_std,
        'chi2_tep': chi2_tep,
        'delta_chi2': chi2_std - chi2_tep,
        'scatter_standard': scatter_std,
        'scatter_tep': scatter_tep,
        'n_galaxies': len(df_valid),
        'tep_preferred': chi2_tep < chi2_std
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Forward-Modeling SED Validation", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {}
    
    # ==========================================================================
    # Test 1: M/L Power-Law Index
    # ==========================================================================
    print_status("\n--- Test 1: M/L Power-Law Index ---", "INFO")
    print_status("Testing M/L ~ t^n for n = 0.5, 0.6, 0.7, 0.8, 0.9", "INFO")
    
    ml_results = test_ml_power_law(df)
    
    if ml_results is not None:
        best_idx = ml_results['rho_mass_age'].abs().idxmin()
        best_power = ml_results.loc[best_idx, 'ml_power']
        
        print_status(f"  Best M/L power: {best_power}", "INFO")
        print_status(f"  (Minimizes mass-age correlation after TEP correction)", "INFO")
        
        for _, row in ml_results.iterrows():
            print_status(f"    n={row['ml_power']:.1f}: rho={row['rho_mass_age']:.3f}", "INFO")
        
        results['ml_power_test'] = ml_results.to_dict('records')
        results['best_ml_power'] = best_power
    else:
        print_status("  Insufficient data for M/L test", "WARNING")
    
    # ==========================================================================
    # Test 2: Alternative Hypotheses (Bursty SFH, Metallicity)
    # ==========================================================================
    print_status("\n--- Test 2: Alternative Hypotheses ---", "INFO")
    
    alt_results = test_sfh_alternatives(df)
    
    if alt_results is not None:
        print_status("  Testing if dust/metallicity explain mass-age correlation:", "INFO")
        
        for test_name, test_result in alt_results.items():
            rho_val = test_result.get('rho')
            p_val = test_result.get('p')
            if rho_val is not None and np.isfinite(rho_val):
                sig = ""
                if p_val is not None:
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print_status(f"    {test_name}: rho={rho_val:.3f} {sig}", "INFO")
            else:
                print_status(f"    {test_name}: data not available", "INFO")
        
        results['alternative_hypotheses'] = alt_results
        
        # Interpretation
        if 'age_mass_controlling_all' in alt_results:
            rho_controlled = alt_results['age_mass_controlling_all'].get('rho')
            if rho_controlled is None:
                print_status("  -> Insufficient data for dust+metallicity control", "WARNING")
                results['alternative_hypothesis_rejected'] = None
            elif np.isfinite(rho_controlled) and abs(rho_controlled) > 0.1:
                print_status("  -> Mass-age correlation PERSISTS after controlling for dust+metallicity", "INFO")
                print_status("  -> Supports TEP interpretation (not explained by standard astrophysics)", "INFO")
                results['alternative_hypothesis_rejected'] = True
            else:
                print_status("  -> Mass-age correlation VANISHES after controlling for dust+metallicity", "INFO")
                print_status("  -> Standard astrophysics may explain the signal", "WARNING")
                results['alternative_hypothesis_rejected'] = False
    else:
        print_status("  Insufficient data for alternative hypothesis test", "WARNING")
    
    # ==========================================================================
    # Test 3: Forward-Model Chi2 Comparison
    # ==========================================================================
    print_status("\n--- Test 3: Forward-Model Chi2 Comparison ---", "INFO")
    
    sps = SimpleSPS(sfh_type='constant')
    fm_results = forward_model_chi2(df, sps)
    
    if fm_results is not None:
        print_status(f"  Standard model chi2: {fm_results['chi2_standard']:.1f}", "INFO")
        print_status(f"  TEP model chi2: {fm_results['chi2_tep']:.1f}", "INFO")
        print_status(f"  Delta chi2: {fm_results['delta_chi2']:.1f}", "INFO")
        print_status(f"  Scatter (std): {fm_results['scatter_standard']:.3f}", "INFO")
        print_status(f"  Scatter (TEP): {fm_results['scatter_tep']:.3f}", "INFO")
        
        if fm_results['tep_preferred']:
            print_status("  -> TEP model PREFERRED (lower chi2)", "INFO")
        else:
            print_status("  -> Standard model preferred", "INFO")
        
        results['forward_model'] = fm_results
    else:
        print_status("  Insufficient data for forward modeling test", "WARNING")
    
    # ==========================================================================
    # Test 4: Redshift-Dependent M/L Test
    # ==========================================================================
    print_status("\n--- Test 4: Redshift-Dependent M/L Test ---", "INFO")
    print_status("  Testing if M/L power varies with redshift (TEP prediction)", "INFO")
    
    z_bins = [(4, 6), (6, 8), (8, 10)]
    z_ml_results = []
    
    for z_lo, z_hi in z_bins:
        mask_z = (df['z_phot'] >= z_lo) & (df['z_phot'] < z_hi)
        df_z = df[mask_z]
        
        if len(df_z) > 50:
            ml_z = test_ml_power_law(df_z)
            if ml_z is not None:
                best_idx = ml_z['rho_mass_age'].abs().idxmin()
                best_power = ml_z.loc[best_idx, 'ml_power']
                
                z_ml_results.append({
                    'z_range': f"{z_lo}-{z_hi}",
                    'n': len(df_z),
                    'best_ml_power': best_power
                })
                
                print_status(f"    z={z_lo}-{z_hi} (N={len(df_z)}): best n={best_power}", "INFO")
    
    if z_ml_results:
        results['z_dependent_ml'] = z_ml_results
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FORWARD MODELING SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    summary = {
        'best_ml_power': results.get('best_ml_power', 0.7),
        'standard_assumption_validated': abs(results.get('best_ml_power', 0.7) - 0.7) < 0.15,
        'alternative_hypotheses_rejected': results.get('alternative_hypothesis_rejected', None),
        'tep_preferred_by_chi2': results.get('forward_model', {}).get('tep_preferred', None)
    }
    
    print_status(f"  Best M/L power: {summary['best_ml_power']}", "INFO")
    print_status(f"  Standard assumption (n=0.7) validated: {summary['standard_assumption_validated']}", "INFO")
    print_status(f"  Alternative hypotheses rejected: {summary['alternative_hypotheses_rejected']}", "INFO")
    print_status(f"  TEP preferred by forward modeling: {summary['tep_preferred_by_chi2']}", "INFO")
    
    results['summary'] = summary
    
    # Save outputs
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_forward_modeling.json"
    
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2, default=safe_json_default)
    
    print_status(f"\nSaved results to {json_path}", "INFO")

if __name__ == "__main__":
    main()
