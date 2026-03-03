#!/usr/bin/env python3
"""
TEP-JWST Step 73: Deep Spectroscopic Validation

The ULTIMATE test: Do galaxies with SPECTROSCOPIC redshifts show the same TEP effect?

This is critical because:
1. Photo-z errors could create spurious correlations
2. Spec-z removes this systematic entirely
3. If TEP holds in spec-z sample, it's NOT a photo-z artifact

We will:
1. Load all available spectroscopic data
2. Match to the main UNCOVER catalog
3. Test if the Gamma_t-Dust correlation holds
4. Compare spec-z and photo-z samples
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM = "73"
STEP_NAME = "spectroscopic_validation"

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
DATA_PATH = PROJECT_ROOT / "data" / "interim"
RAW_PATH = PROJECT_ROOT / "data" / "raw"
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
    print_status(f"STEP {STEP_NUM}: Deep Spectroscopic Validation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nThe ULTIMATE test: Does TEP hold in spectroscopic sample?\n", "INFO")
    
    # Load main catalog
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded main catalog: N = {len(df)}", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'spectroscopic': {}
    }
    
    # ==========================================================================
    # SECTION 1: Check for Spectroscopic Data in Main Catalog
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SECTION 1: Spectroscopic Data in Main Catalog", "INFO")
    print_status("=" * 70, "INFO")
    
    # Check for spec-z columns
    spec_cols = [col for col in df.columns if 'spec' in col.lower() or 'zspec' in col.lower()]
    print_status(f"Spectroscopic columns: {spec_cols}", "INFO")
    
    # Check z_spec column
    if 'z_spec' in df.columns:
        spec_available = df['z_spec'].notna()
        n_spec = spec_available.sum()
        print_status(f"Galaxies with z_spec: N = {n_spec}", "INFO")
        
        if n_spec > 0:
            spec_sample = df[spec_available].copy()
            
            # Distribution of spec-z
            print_status(f"z_spec range: {spec_sample['z_spec'].min():.2f} - {spec_sample['z_spec'].max():.2f}", "INFO")
            
            # High-z spec sample
            spec_high_z = spec_sample[spec_sample['z_spec'] > 7]
            print_status(f"Spec-z > 7: N = {len(spec_high_z)}", "INFO")
            
            spec_z8 = spec_sample[spec_sample['z_spec'] > 8]
            print_status(f"Spec-z > 8: N = {len(spec_z8)}", "INFO")
    
    # ==========================================================================
    # SECTION 2: Load Combined Spectroscopic Catalog
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SECTION 2: Combined Spectroscopic Catalog", "INFO")
    print_status("=" * 70, "INFO")
    
    spec_path = DATA_PATH / "combined_spectroscopic_catalog.csv"
    if spec_path.exists():
        spec_cat = pd.read_csv(spec_path)
        print_status(f"Loaded combined spec catalog: N = {len(spec_cat)}", "INFO")
        print_status(f"Columns: {list(spec_cat.columns)}", "INFO")
        
        # Check for high-z
        if 'z_spec' in spec_cat.columns:
            high_z_spec = spec_cat[spec_cat['z_spec'] > 7]
            print_status(f"Spec-z > 7 in combined catalog: N = {len(high_z_spec)}", "INFO")
    else:
        print_status("Combined spectroscopic catalog not found", "INFO")
    
    # ==========================================================================
    # SECTION 3: Test TEP in Spectroscopic Sample
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SECTION 3: TEP Test in Spectroscopic Sample", "INFO")
    print_status("=" * 70, "INFO")
    
    # Use galaxies with z_spec in main catalog
    if 'z_spec' in df.columns:
        spec_sample = df[df['z_spec'].notna()].copy()
        
        # For spec-z galaxies, use z_spec instead of z_phot for Gamma_t calculation
        # But we already have gamma_t calculated with z_phot
        # Let's check if the correlation holds
        
        spec_valid = spec_sample.dropna(subset=['gamma_t', 'dust', 'z_spec'])
        
        if len(spec_valid) > 20:
            # Test at different z thresholds
            z_thresholds = [5, 6, 7, 8]
            
            for z_thresh in z_thresholds:
                subset = spec_valid[spec_valid['z_spec'] > z_thresh]
                if len(subset) > 10:
                    rho, p = spearmanr(subset['gamma_t'], subset['dust'])
                    sig = "✓" if rho > 0.2 and p < 0.1 else ""
                    print_status(f"Spec-z > {z_thresh}: N = {len(subset)}, ρ(Γ_t, Dust) = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
                    
                    results['spectroscopic'][f'z_gt_{z_thresh}'] = {
                        'n': int(len(subset)),
                        'rho': float(rho),
                        'p': float(p)
                    }
    
    # ==========================================================================
    # SECTION 4: Compare Spec-z vs Photo-z Samples
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SECTION 4: Spec-z vs Photo-z Comparison", "INFO")
    print_status("=" * 70, "INFO")
    
    if 'z_spec' in df.columns:
        # Split into spec-z and photo-z only samples
        has_spec = df['z_spec'].notna()
        
        spec_only = df[has_spec].dropna(subset=['gamma_t', 'dust'])
        phot_only = df[~has_spec].dropna(subset=['gamma_t', 'dust'])
        
        # High-z subsets
        spec_high = spec_only[spec_only['z_phot'] > 7]
        phot_high = phot_only[phot_only['z_phot'] > 7]
        
        if len(spec_high) > 10 and len(phot_high) > 10:
            rho_spec, p_spec = spearmanr(spec_high['gamma_t'], spec_high['dust'])
            rho_phot, p_phot = spearmanr(phot_high['gamma_t'], phot_high['dust'])
            
            print_status(f"Spec-z sample (z > 7): N = {len(spec_high)}, ρ = {rho_spec:.3f}", "INFO")
            print_status(f"Photo-z sample (z > 7): N = {len(phot_high)}, ρ = {rho_phot:.3f}", "INFO")
            print_status(f"Difference: Δρ = {abs(rho_spec - rho_phot):.3f}", "INFO")
            
            if abs(rho_spec - rho_phot) < 0.2:
                print_status("\n✓ CONSISTENT: Spec-z and photo-z samples show similar correlation", "INFO")
            
            results['spectroscopic']['comparison'] = {
                'rho_spec': float(rho_spec),
                'rho_phot': float(rho_phot),
                'difference': float(abs(rho_spec - rho_phot))
            }
    
    # ==========================================================================
    # SECTION 5: Photo-z Quality Check
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SECTION 5: Photo-z Quality Check", "INFO")
    print_status("=" * 70, "INFO")
    
    # For galaxies with both spec-z and photo-z, check agreement
    if 'z_spec' in df.columns:
        both = df[df['z_spec'].notna()].copy()
        both['z_diff'] = both['z_phot'] - both['z_spec']
        both['z_diff_frac'] = both['z_diff'] / (1 + both['z_spec'])
        
        print_status(f"Photo-z quality (N = {len(both)}):", "INFO")
        print_status(f"  Mean Δz/(1+z): {both['z_diff_frac'].mean():.4f}", "INFO")
        print_status(f"  Std Δz/(1+z): {both['z_diff_frac'].std():.4f}", "INFO")
        print_status(f"  Outlier rate (|Δz|/(1+z) > 0.15): {(both['z_diff_frac'].abs() > 0.15).mean()*100:.1f}%", "INFO")
        
        # Does photo-z quality affect the correlation?
        good_photoz = both[both['z_diff_frac'].abs() < 0.1]
        poor_photoz = both[both['z_diff_frac'].abs() >= 0.1]
        
        if len(good_photoz) > 10 and len(poor_photoz) > 5:
            rho_good, _ = spearmanr(good_photoz['gamma_t'], good_photoz['dust'])
            rho_poor, _ = spearmanr(poor_photoz['gamma_t'], poor_photoz['dust'])
            
            print_status(f"\nGood photo-z (|Δz|/(1+z) < 0.1): ρ = {rho_good:.3f} (N = {len(good_photoz)})", "INFO")
            print_status(f"Poor photo-z (|Δz|/(1+z) ≥ 0.1): ρ = {rho_poor:.3f} (N = {len(poor_photoz)})", "INFO")
            
            results['spectroscopic']['photoz_quality'] = {
                'mean_dz': float(both['z_diff_frac'].mean()),
                'std_dz': float(both['z_diff_frac'].std()),
                'rho_good': float(rho_good),
                'rho_poor': float(rho_poor)
            }
    
    # ==========================================================================
    # SECTION 6: The Ultimate Spectroscopic Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SECTION 6: The Ultimate Spectroscopic Test", "INFO")
    print_status("=" * 70, "INFO")
    
    # Recalculate Gamma_t using SPEC-Z for galaxies that have it
    if 'z_spec' in df.columns:
        spec_sample = df[df['z_spec'].notna()].copy()
        
        # Recalculate Gamma_t with spec-z
        alpha = 0.58
        spec_sample['log_Mref_spec'] = 12.0 - 1.5 * np.log10(1 + spec_sample['z_spec'])
        spec_sample['gamma_t_spec'] = np.exp(alpha * (spec_sample['log_Mh'] - spec_sample['log_Mref_spec']))
        
        # Test correlation with spec-z derived Gamma_t
        spec_valid = spec_sample.dropna(subset=['gamma_t_spec', 'dust', 'z_spec'])
        spec_high = spec_valid[spec_valid['z_spec'] > 7]
        
        if len(spec_high) > 10:
            rho_spec_gamma, p_spec_gamma = spearmanr(spec_high['gamma_t_spec'], spec_high['dust'])
            
            print_status(f"Using Γ_t calculated from SPEC-Z:", "INFO")
            print_status(f"  N = {len(spec_high)}", "INFO")
            print_status(f"  ρ(Γ_t_spec, Dust) = {rho_spec_gamma:.3f} (p = {p_spec_gamma:.2e})", "INFO")
            
            if rho_spec_gamma > 0.2 and p_spec_gamma < 0.1:
                print_status("\n✓ ULTIMATE VALIDATION: TEP holds with spec-z derived Gamma_t", "INFO")
            
            results['spectroscopic']['ultimate_test'] = {
                'n': int(len(spec_high)),
                'rho': float(rho_spec_gamma),
                'p': float(p_spec_gamma)
            }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Spectroscopic Validation", "INFO")
    print_status("=" * 70, "INFO")
    
    if results['spectroscopic']:
        print_status("\nKey findings:", "INFO")
        
        if 'ultimate_test' in results['spectroscopic']:
            ut = results['spectroscopic']['ultimate_test']
            print_status(f"  • Ultimate test (spec-z Γ_t): ρ = {ut['rho']:.3f} (N = {ut['n']})", "INFO")
        
        if 'comparison' in results['spectroscopic']:
            comp = results['spectroscopic']['comparison']
            print_status(f"  • Spec-z vs Photo-z difference: Δρ = {comp['difference']:.3f}", "INFO")
        
        if 'photoz_quality' in results['spectroscopic']:
            pq = results['spectroscopic']['photoz_quality']
            print_status(f"  • Photo-z scatter: σ(Δz/(1+z)) = {pq['std_dz']:.4f}", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
