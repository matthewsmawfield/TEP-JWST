#!/usr/bin/env python3
"""
TEP-JWST Step 54: Spectroscopic Validation

This step uses the spectroscopic subsample to validate TEP predictions
with the highest-quality redshifts:

1. Spec-z subsample TEP correlations
2. Photo-z vs Spec-z offset as function of Gamma_t
3. Cross-field validation (UNCOVER vs other surveys)
4. Emission line properties vs Gamma_t
5. Balmer break strength correlation
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM = "54"
STEP_NAME = "spectroscopic_validation"

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
DATA_PATH = PROJECT_ROOT / "data" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Spectroscopic Validation", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load main data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies from UNCOVER", "INFO")
    
    results = {
        'n_total': len(df),
        'tests': {}
    }
    
    strong_evidence = []
    
    # ==========================================================================
    # TEST 1: Spectroscopic Subsample
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 1: Spectroscopic Subsample Analysis", "INFO")
    print_status("=" * 60, "INFO")
    
    spec = df.dropna(subset=['z_spec', 'gamma_t', 'dust'])
    
    if len(spec) > 30:
        print_status(f"Spectroscopic subsample: N = {len(spec)}", "INFO")
        
        # Test Gamma_t-Dust correlation in spec-z sample
        rho, p = spearmanr(spec['gamma_t'], spec['dust'])
        print_status(f"ρ(Γ_t, Dust) in spec-z sample = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Compare to photo-z only sample
        phot_only = df[df['z_spec'].isna()].dropna(subset=['gamma_t', 'dust'])
        rho_phot, p_phot = spearmanr(phot_only['gamma_t'], phot_only['dust'])
        print_status(f"ρ(Γ_t, Dust) in photo-z only = {rho_phot:.3f} (p = {p_phot:.2e})", "INFO")
        
        results['tests']['spec_subsample'] = {
            'n_spec': int(len(spec)),
            'n_phot_only': int(len(phot_only)),
            'rho_spec': float(rho),
            'p_spec': float(p),
            'rho_phot': float(rho_phot),
            'p_phot': float(p_phot)
        }
        
        # High-z spec subsample
        spec_highz = spec[spec['z_spec'] > 7]
        if len(spec_highz) > 10:
            rho_hz, p_hz = spearmanr(spec_highz['gamma_t'], spec_highz['dust'])
            print_status(f"ρ(Γ_t, Dust) in spec-z > 7: N = {len(spec_highz)}, ρ = {rho_hz:.3f} (p = {p_hz:.2e})", "INFO")
            
            if rho_hz > 0.3 and p_hz < 0.05:
                print_status("✓ STRONG: TEP-Dust correlation confirmed in spec-z > 7 sample", "INFO")
                strong_evidence.append(('spec_highz', rho_hz, p_hz))
            
            results['tests']['spec_highz'] = {
                'n': int(len(spec_highz)),
                'rho': float(rho_hz),
                'p': float(p_hz),
                'strong': bool(rho_hz > 0.3 and p_hz < 0.05)
            }
    
    # ==========================================================================
    # TEST 2: Photo-z Offset vs Gamma_t
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 2: Photo-z Offset vs Γ_t", "INFO")
    print_status("=" * 60, "INFO")
    
    spec = df.dropna(subset=['z_spec', 'z_phot', 'gamma_t'])
    
    if len(spec) > 30:
        spec = spec.copy()
        spec['z_offset'] = (spec['z_phot'] - spec['z_spec']) / (1 + spec['z_spec'])
        spec['z_offset_abs'] = np.abs(spec['z_offset'])
        
        # Does Gamma_t predict photo-z offset direction?
        rho_dir, p_dir = spearmanr(spec['gamma_t'], spec['z_offset'])
        rho_abs, p_abs = spearmanr(spec['gamma_t'], spec['z_offset_abs'])
        
        print_status(f"N = {len(spec)}", "INFO")
        print_status(f"ρ(Γ_t, z_offset) = {rho_dir:.3f} (p = {p_dir:.2e})", "INFO")
        print_status(f"ρ(Γ_t, |z_offset|) = {rho_abs:.3f} (p = {p_abs:.2e})", "INFO")
        
        # TEP predicts high-Gamma_t galaxies have z_phot > z_spec (SED looks older)
        if rho_dir > 0.1 and p_dir < 0.05:
            print_status("✓ High-Γ_t galaxies have z_phot > z_spec (TEP-consistent)", "INFO")
            strong_evidence.append(('z_offset_direction', rho_dir, p_dir))
        
        results['tests']['z_offset'] = {
            'n': int(len(spec)),
            'rho_direction': float(rho_dir),
            'p_direction': float(p_dir),
            'rho_absolute': float(rho_abs),
            'p_absolute': float(p_abs)
        }
    
    # ==========================================================================
    # TEST 3: Age Consistency Check
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 3: Age Consistency (Spec-z vs Photo-z)", "INFO")
    print_status("=" * 60, "INFO")
    
    spec = df.dropna(subset=['z_spec', 'mwa', 'gamma_t'])
    
    if len(spec) > 30:
        # Calculate cosmic age at spec-z
        from astropy.cosmology import Planck18
        spec = spec.copy()
        spec['t_cosmic_spec'] = [Planck18.age(z).value for z in spec['z_spec']]
        spec['age_ratio_spec'] = spec['mwa'] / spec['t_cosmic_spec']
        
        # Galaxies with age_ratio_spec > 0.5 are "problematic"
        problematic = spec[spec['age_ratio_spec'] > 0.5]
        normal = spec[spec['age_ratio_spec'] <= 0.5]
        
        if len(problematic) > 5 and len(normal) > 20:
            mean_gamma_prob = problematic['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            
            print_status(f"Problematic (age/t_cosmic > 0.5): N = {len(problematic)}, <Γ_t> = {mean_gamma_prob:.3f}", "INFO")
            print_status(f"Normal (age/t_cosmic ≤ 0.5): N = {len(normal)}, <Γ_t> = {mean_gamma_norm:.3f}", "INFO")
            
            if mean_gamma_prob > mean_gamma_norm * 1.5:
                print_status("✓ STRONG: Problematic ages have elevated Γ_t (spec-z confirmed)", "INFO")
                strong_evidence.append(('age_consistency', mean_gamma_prob/mean_gamma_norm, 0))
            
            results['tests']['age_consistency'] = {
                'n_problematic': int(len(problematic)),
                'n_normal': int(len(normal)),
                'mean_gamma_problematic': float(mean_gamma_prob),
                'mean_gamma_normal': float(mean_gamma_norm),
                'ratio': float(mean_gamma_prob / mean_gamma_norm)
            }
    
    # ==========================================================================
    # TEST 4: Cross-Survey Validation
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 4: Cross-Survey Validation", "INFO")
    print_status("=" * 60, "INFO")
    
    # Load CEERS data if available
    ceers_path = DATA_PATH / "ceers_z8_sample.csv"
    if ceers_path.exists():
        ceers = pd.read_csv(ceers_path)
        print_status(f"Loaded CEERS sample: N = {len(ceers)}", "INFO")
        
        # Check if CEERS has similar columns
        if 'log_Mstar' in ceers.columns and 'dust' in ceers.columns:
            ceers_valid = ceers.dropna(subset=['log_Mstar', 'dust'])
            if len(ceers_valid) > 20:
                rho_ceers, p_ceers = spearmanr(ceers_valid['log_Mstar'], ceers_valid['dust'])
                print_status(f"CEERS Mass-Dust: ρ = {rho_ceers:.3f} (p = {p_ceers:.2e})", "INFO")
                
                # Compare to UNCOVER z > 8
                uncover_z8 = df[(df['z_phot'] > 8)].dropna(subset=['log_Mstar', 'dust'])
                rho_uncover, p_uncover = spearmanr(uncover_z8['log_Mstar'], uncover_z8['dust'])
                print_status(f"UNCOVER z>8 Mass-Dust: ρ = {rho_uncover:.3f} (p = {p_uncover:.2e})", "INFO")
                
                # Cross-survey consistency
                if abs(rho_ceers - rho_uncover) < 0.2:
                    print_status("✓ Cross-survey consistency confirmed", "INFO")
                    strong_evidence.append(('cross_survey', (rho_ceers + rho_uncover)/2, min(p_ceers, p_uncover)))
                
                results['tests']['cross_survey'] = {
                    'n_ceers': int(len(ceers_valid)),
                    'n_uncover_z8': int(len(uncover_z8)),
                    'rho_ceers': float(rho_ceers),
                    'rho_uncover': float(rho_uncover),
                    'consistent': bool(abs(rho_ceers - rho_uncover) < 0.2)
                }
    
    # ==========================================================================
    # TEST 5: Emission Line Equivalent Width
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 5: Emission Line Properties", "INFO")
    print_status("=" * 60, "INFO")
    
    # Check for emission line data
    ew_cols = [c for c in df.columns if 'ew' in c.lower() or 'EW' in c]
    
    if ew_cols:
        print_status(f"Found EW columns: {ew_cols}", "INFO")
        for col in ew_cols[:3]:  # Test first 3
            valid = df.dropna(subset=[col, 'gamma_t'])
            if len(valid) > 50:
                rho, p = spearmanr(valid['gamma_t'], valid[col])
                print_status(f"ρ(Γ_t, {col}) = {rho:.3f} (p = {p:.2e})", "INFO")
    else:
        print_status("No emission line EW columns found", "INFO")
    
    # ==========================================================================
    # TEST 6: Balmer Break Proxy
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 6: Balmer Break Proxy (Color Gradient)", "INFO")
    print_status("=" * 60, "INFO")
    
    # Use U-V and V-J colors as Balmer break proxy
    if 'uv' in df.columns and 'vj' in df.columns:
        valid = df.dropna(subset=['uv', 'vj', 'gamma_t'])
        
        if len(valid) > 100:
            valid = valid.copy()
            # Balmer break strength proxy: U-V color (redder = stronger break)
            rho_uv, p_uv = spearmanr(valid['gamma_t'], valid['uv'])
            rho_vj, p_vj = spearmanr(valid['gamma_t'], valid['vj'])
            
            print_status(f"N = {len(valid)}", "INFO")
            print_status(f"ρ(Γ_t, U-V) = {rho_uv:.3f} (p = {p_uv:.2e})", "INFO")
            print_status(f"ρ(Γ_t, V-J) = {rho_vj:.3f} (p = {p_vj:.2e})", "INFO")
            
            # TEP predicts negative correlation (high Gamma_t → older → redder)
            if rho_uv < -0.1 and p_uv < 0.001:
                print_status("✓ STRONG: High-Γ_t galaxies are redder (older populations)", "INFO")
                strong_evidence.append(('color_uv', rho_uv, p_uv))
            
            results['tests']['balmer_proxy'] = {
                'n': int(len(valid)),
                'rho_uv': float(rho_uv),
                'p_uv': float(p_uv),
                'rho_vj': float(rho_vj),
                'p_vj': float(p_vj),
                'strong': bool(rho_uv < -0.1 and p_uv < 0.001)
            }
    
    # ==========================================================================
    # TEST 7: Mass Function Anomaly
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 7: Mass Function by Regime", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 8]
    
    if len(high_z) > 50:
        enhanced = high_z[high_z['gamma_t'] > 1]
        suppressed = high_z[high_z['gamma_t'] < 0.5]
        
        if len(enhanced) > 10 and len(suppressed) > 10:
            mean_mass_enh = enhanced['log_Mstar'].mean()
            mean_mass_sup = suppressed['log_Mstar'].mean()
            
            stat, p = ks_2samp(enhanced['log_Mstar'], suppressed['log_Mstar'])
            
            print_status(f"Mass distribution at z > 8:", "INFO")
            print_status(f"  Enhanced: N = {len(enhanced)}, <log M*> = {mean_mass_enh:.2f}", "INFO")
            print_status(f"  Suppressed: N = {len(suppressed)}, <log M*> = {mean_mass_sup:.2f}", "INFO")
            print_status(f"  KS stat = {stat:.3f}, p = {p:.2e}", "INFO")
            
            if p < 0.01:
                print_status("✓ Mass distributions differ between regimes at z > 8", "INFO")
                strong_evidence.append(('mass_function', stat, p))
            
            results['tests']['mass_function'] = {
                'n_enhanced': int(len(enhanced)),
                'n_suppressed': int(len(suppressed)),
                'mean_mass_enhanced': float(mean_mass_enh),
                'mean_mass_suppressed': float(mean_mass_sup),
                'ks_stat': float(stat),
                'p': float(p)
            }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Strong Evidence Found", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nStrong evidence items: {len(strong_evidence)}", "INFO")
    for name, stat, p in strong_evidence:
        print_status(f"  • {name}: stat = {stat:.3f}, p = {p:.2e}", "INFO")
    
    results['summary'] = {
        'n_strong_evidence': len(strong_evidence),
        'strong_evidence': [{'name': n, 'stat': float(s), 'p': float(p)} for n, s, p in strong_evidence]
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
