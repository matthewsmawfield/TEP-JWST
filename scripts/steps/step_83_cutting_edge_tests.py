#!/usr/bin/env python3
"""
TEP-JWST Step 83: Cutting-Edge Tests from Latest Literature

Based on latest JWST findings:

1. OVERMASSIVE BLACK HOLES: BH-to-stellar mass ratios are 10-100x higher than local.
   TEP predicts: Differential time dilation between BH and stellar halo.

2. DISK MORPHOLOGY: Unexpectedly high fraction of disk galaxies at high-z.
   TEP predicts: More effective time allows disk formation.

3. PHOTON BUDGET CRISIS: Too many ionizing photons for reionization timeline.
   TEP predicts: Effective time is longer, so more photon production.

4. NITROGEN ENRICHMENT: Anomalous N/O ratios at high-z.
   TEP predicts: More effective time for chemical enrichment.

5. PROTO-GLOBULAR CLUSTERS: Massive star clusters at high-z.
   TEP predicts: More effective time for cluster formation.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM = "83"
STEP_NAME = "cutting_edge_tests"

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
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
    print_status(f"STEP {STEP_NUM}: Cutting-Edge Tests from Latest Literature", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'cutting_edge': {}
    }
    
    # ==========================================================================
    # TEST 1: Overmassive Black Hole Analog
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Overmassive Black Hole Analog", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: BH-to-stellar mass ratios are 10-100x higher than local.", "INFO")
    print_status("TEP predicts: High-Gamma_t galaxies have inflated stellar masses.\n", "INFO")
    
    # Use stellar-to-halo mass ratio as proxy for BH-to-stellar mass
    # High SMHM could indicate overmassive stellar component (like overmassive BH)
    valid = df[df['z_phot'] > 7].dropna(subset=['log_Mstar', 'log_Mh', 'gamma_t'])
    
    if len(valid) > 50:
        valid = valid.copy()
        valid['smhm'] = valid['log_Mstar'] - valid['log_Mh']
        
        rho, p = spearmanr(valid['gamma_t'], valid['smhm'])
        print_status(f"ρ(Γ_t, SMHM) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # High SMHM galaxies
        smhm_thresh = valid['smhm'].quantile(0.9)
        high_smhm = valid[valid['smhm'] > smhm_thresh]
        low_smhm = valid[valid['smhm'] < valid['smhm'].quantile(0.5)]
        
        if len(high_smhm) > 5 and len(low_smhm) > 10:
            mean_gamma_high = high_smhm['gamma_t'].mean()
            mean_gamma_low = low_smhm['gamma_t'].mean()
            
            print_status(f"High SMHM (top 10%): <Γ_t> = {mean_gamma_high:.2f}", "INFO")
            print_status(f"Low SMHM (bottom 50%): <Γ_t> = {mean_gamma_low:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_high/mean_gamma_low:.1f}×", "INFO")
            
            if mean_gamma_high > mean_gamma_low * 2:
                print_status("✓ High SMHM galaxies have elevated Γ_t", "INFO")
            
            results['cutting_edge']['overmassive_bh'] = {
                'rho': float(rho),
                'ratio': float(mean_gamma_high / mean_gamma_low)
            }
    
    # ==========================================================================
    # TEST 2: Disk Morphology Proxy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Disk Morphology Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: Unexpectedly high fraction of disk galaxies at high-z.", "INFO")
    print_status("TEP predicts: More effective time allows disk settling.\n", "INFO")
    
    # Use burstiness as proxy for disk (smooth SFH = disk-like)
    valid = df[df['z_phot'] > 6].dropna(subset=['sfr10', 'sfr100', 'gamma_t'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0)].copy()
    
    if len(valid) > 50:
        valid['burstiness'] = np.log10(valid['sfr10'] / valid['sfr100'])
        
        # "Disk-like" = smooth SFH = low burstiness
        disk_like = valid[valid['burstiness'] < 0]  # Declining SFR
        irregular = valid[valid['burstiness'] > 0.3]  # Bursty
        
        if len(disk_like) > 10 and len(irregular) > 10:
            mean_gamma_disk = disk_like['gamma_t'].mean()
            mean_gamma_irreg = irregular['gamma_t'].mean()
            
            print_status(f"'Disk-like' (smooth SFH): <Γ_t> = {mean_gamma_disk:.2f} (N = {len(disk_like)})", "INFO")
            print_status(f"'Irregular' (bursty SFH): <Γ_t> = {mean_gamma_irreg:.2f} (N = {len(irregular)})", "INFO")
            
            if mean_gamma_disk > mean_gamma_irreg:
                print_status("✓ 'Disk-like' galaxies have higher Γ_t", "INFO")
            
            results['cutting_edge']['disk_morphology'] = {
                'mean_gamma_disk': float(mean_gamma_disk),
                'mean_gamma_irregular': float(mean_gamma_irreg)
            }
    
    # ==========================================================================
    # TEST 3: Photon Budget Proxy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Photon Budget Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: Too many ionizing photons for reionization timeline.", "INFO")
    print_status("TEP predicts: Effective time is longer, so more photon production.\n", "INFO")
    
    # Use SFR as proxy for ionizing photon production
    valid = df[df['z_phot'] > 8].dropna(subset=['sfr100', 'gamma_t', 't_eff'])
    valid = valid[valid['sfr100'] > 0].copy()
    
    if len(valid) > 50:
        # Total photon production ~ SFR × t_eff
        valid['photon_proxy'] = valid['sfr100'] * valid['t_eff']
        
        # Compare to cosmic time expectation
        from astropy.cosmology import Planck18
        valid['t_cosmic'] = [Planck18.age(z).value for z in valid['z_phot']]
        valid['photon_cosmic'] = valid['sfr100'] * valid['t_cosmic']
        
        # Ratio of TEP to cosmic photon production
        valid['photon_ratio'] = valid['photon_proxy'] / valid['photon_cosmic']
        
        mean_ratio = valid['photon_ratio'].mean()
        print_status(f"Mean photon production ratio (TEP/cosmic): {mean_ratio:.2f}×", "INFO")
        
        # High Gamma_t galaxies
        high_gamma = valid[valid['gamma_t'] > 1]
        if len(high_gamma) > 10:
            mean_ratio_high = high_gamma['photon_ratio'].mean()
            print_status(f"High Γ_t galaxies: photon ratio = {mean_ratio_high:.2f}×", "INFO")
        
        results['cutting_edge']['photon_budget'] = {
            'mean_ratio': float(mean_ratio)
        }
    
    # ==========================================================================
    # TEST 4: Nitrogen Enrichment Proxy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Nitrogen Enrichment Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: Anomalous N/O ratios at high-z.", "INFO")
    print_status("TEP predicts: More effective time for chemical enrichment.\n", "INFO")
    
    # Use metallicity as proxy for N enrichment
    valid = df[df['z_phot'] > 7].dropna(subset=['met', 'gamma_t', 't_eff'])
    
    if len(valid) > 50:
        # Correlation between t_eff and metallicity
        rho_teff, p_teff = spearmanr(valid['t_eff'], valid['met'])
        
        # Compare to t_cosmic
        from astropy.cosmology import Planck18
        valid = valid.copy()
        valid['t_cosmic'] = [Planck18.age(z).value for z in valid['z_phot']]
        rho_cosmic, p_cosmic = spearmanr(valid['t_cosmic'], valid['met'])
        
        print_status(f"ρ(t_cosmic, Metallicity) = {rho_cosmic:.3f} (p = {p_cosmic:.2e})", "INFO")
        print_status(f"ρ(t_eff, Metallicity) = {rho_teff:.3f} (p = {p_teff:.2e})", "INFO")
        print_status(f"Improvement: Δρ = {rho_teff - rho_cosmic:+.3f}", "INFO")
        
        if rho_teff > rho_cosmic:
            print_status("✓ t_eff predicts metallicity better than t_cosmic", "INFO")
        
        results['cutting_edge']['nitrogen_enrichment'] = {
            'rho_cosmic': float(rho_cosmic),
            'rho_teff': float(rho_teff),
            'improvement': float(rho_teff - rho_cosmic)
        }
    
    # ==========================================================================
    # TEST 5: Proto-Globular Cluster Proxy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Proto-Globular Cluster Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: Massive star clusters at high-z.", "INFO")
    print_status("TEP predicts: More effective time for cluster formation.\n", "INFO")
    
    # Use high mass + compact (high dust) as proxy for proto-GC hosts
    valid = df[df['z_phot'] > 6].dropna(subset=['log_Mstar', 'dust', 'gamma_t'])
    
    if len(valid) > 50:
        # Proto-GC hosts: high mass + high dust (compact, evolved)
        mass_thresh = valid['log_Mstar'].quantile(0.75)
        dust_thresh = valid['dust'].quantile(0.75)
        
        proto_gc = valid[(valid['log_Mstar'] > mass_thresh) & (valid['dust'] > dust_thresh)]
        normal = valid.drop(proto_gc.index)
        
        if len(proto_gc) > 5 and len(normal) > 10:
            mean_gamma_pgc = proto_gc['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            
            print_status(f"Proto-GC hosts (high mass + high dust): N = {len(proto_gc)}", "INFO")
            print_status(f"<Γ_t> proto-GC: {mean_gamma_pgc:.2f}", "INFO")
            print_status(f"<Γ_t> normal: {mean_gamma_norm:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_pgc/mean_gamma_norm:.1f}×", "INFO")
            
            if mean_gamma_pgc > mean_gamma_norm * 3:
                print_status("✓ Proto-GC hosts have elevated Γ_t", "INFO")
            
            results['cutting_edge']['proto_gc'] = {
                'n': int(len(proto_gc)),
                'mean_gamma_pgc': float(mean_gamma_pgc),
                'mean_gamma_normal': float(mean_gamma_norm),
                'ratio': float(mean_gamma_pgc / mean_gamma_norm)
            }
    
    # ==========================================================================
    # TEST 6: The "Impossible" Reionization Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: The 'Impossible' Reionization Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("At z > 10, galaxies should be too young to produce enough photons.", "INFO")
    print_status("TEP predicts: Effective time allows sufficient photon production.\n", "INFO")
    
    z10 = df[df['z_phot'] > 10].dropna(subset=['sfr100', 'gamma_t', 't_eff'])
    z10 = z10[z10['sfr100'] > 0]
    
    if len(z10) > 10:
        # At z > 10, t_cosmic < 450 Myr
        # But t_eff can be much longer
        
        mean_teff = z10['t_eff'].mean()
        mean_gamma = z10['gamma_t'].mean()
        
        print_status(f"z > 10 galaxies: N = {len(z10)}", "INFO")
        print_status(f"<t_eff> = {mean_teff:.3f} Gyr", "INFO")
        print_status(f"<Γ_t> = {mean_gamma:.2f}", "INFO")
        
        # How many have t_eff > 0.5 Gyr (enough for significant enrichment)?
        n_sufficient = (z10['t_eff'] > 0.5).sum()
        print_status(f"Galaxies with t_eff > 0.5 Gyr: {n_sufficient}/{len(z10)} ({n_sufficient/len(z10)*100:.1f}%)", "INFO")
        
        results['cutting_edge']['reionization'] = {
            'n_z10': int(len(z10)),
            'mean_teff': float(mean_teff),
            'mean_gamma': float(mean_gamma),
            'n_sufficient': int(n_sufficient)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Cutting-Edge Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status("\nKey findings from cutting-edge literature:", "INFO")
    
    for test_name, test_data in results['cutting_edge'].items():
        if 'ratio' in test_data:
            print_status(f"  • {test_name}: {test_data['ratio']:.1f}× elevation", "INFO")
        elif 'improvement' in test_data:
            print_status(f"  • {test_name}: Δρ = {test_data['improvement']:+.3f}", "INFO")
        elif 'mean_ratio' in test_data:
            print_status(f"  • {test_name}: {test_data['mean_ratio']:.2f}× ratio", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
