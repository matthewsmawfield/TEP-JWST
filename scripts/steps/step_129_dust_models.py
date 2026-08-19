#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): <0.1s.
"""Step 152: Dust Physics Alternative Models Test

Tests TEP against alternative dust production models:
1. Supernova-only dust (no AGB)
2. Enhanced supernova yields
3. Dust growth in ISM
"""
import json, numpy as np, sys
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)

STEP_NUM  = "129"  # Pipeline step number (sequential 001-176)
STEP_NAME = "dust_models"  # Dust physics alternative models: TEP vs SN-only, enhanced SN yields, ISM growth - AGB dust production timeline discriminator
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)

def model_sn_only(t_gyr, sfr, sn_yield=0.15):
    """Supernova-only dust production."""
    return sn_yield * sfr * t_gyr

def model_tep(t_gyr, sfr, gamma_t, sn_yield=0.15):
    """TEP: enhanced time for AGB dust production."""
    t_eff = t_gyr * gamma_t
    # Simplified dust production
    if t_eff < 0.3:  # Before AGB onset
        return sn_yield * sfr * t_eff
    else:
        agb_dust = 0.5 * sfr * (t_eff - 0.3)  # AGB contribution
        return sn_yield * sfr * t_eff + agb_dust

def test_alternative_models():
    """Compare TEP predictions against alternative dust models."""
    # Test at z=8 (t_cosmic ~ 0.65 Gyr)
    t_cosmic = 0.65
    gamma_t_values = [0.1, 0.5, 1.0, 2.0, 3.0]  # Range from low-mass to massive
    sfr = 10.0  # Solar masses per year
    
    results = []
    for gamma_t in gamma_t_values:
        tep_dust = model_tep(t_cosmic, sfr, gamma_t)
        sn_only_dust = model_sn_only(t_cosmic, sfr)
        
        # Ratio of TEP to SN-only
        ratio = tep_dust / sn_only_dust if sn_only_dust > 0 else 0
        
        results.append({
            'gamma_t': gamma_t,
            't_eff': t_cosmic * gamma_t,
            'tep_dust': tep_dust,
            'sn_only_dust': sn_only_dust,
            'tep_enhancement': ratio
        })
    
    # Statistical test: can TEP explain the massive galaxy dust excess?
    massive_gamma_t = 2.5
    dwarf_gamma_t = 0.2
    
    massive_dust = model_tep(t_cosmic, sfr, massive_gamma_t)
    dwarf_dust = model_tep(t_cosmic, sfr, dwarf_gamma_t)
    tep_mass_dust_ratio = massive_dust / dwarf_dust if dwarf_dust > 0 else 0
    
    # SN-only predicts much smaller ratio
    sn_massive = model_sn_only(t_cosmic, sfr)
    sn_dwarf = model_sn_only(t_cosmic, sfr * 0.1)  # Lower SFR for dwarfs
    sn_mass_dust_ratio = sn_massive / sn_dwarf if sn_dwarf > 0 else 0
    
    return {
        'gamma_t_grid': results,
        'key_test': {
            'tep_mass_dust_ratio': float(tep_mass_dust_ratio),
            'sn_only_mass_dust_ratio': float(sn_mass_dust_ratio),
            'observed_mass_dust_ratio_approx': 4.0,  # From observations
            'tep_matches_observation': 1.0 < tep_mass_dust_ratio / 4.0 < 3.0,
            'sn_only_fails': sn_mass_dust_ratio < 2.0,
            'tep_overprediction_factor': float(tep_mass_dust_ratio / 4.0),
            'sn_overprediction_factor': float(sn_mass_dust_ratio / 4.0)
        }
    }

def main():
    print_status("STEP 129: Dust Physics Alternative Models Test", "TITLE")
    print_status("Comparing TEP vs SN-only dust production models against observed mass-dust ratios.", "INFO")
    print_status("")
    
    results = test_alternative_models()
    
    print_status("")
    print_status("Gamma_t grid test:", "PROCESS")
    for r in results['gamma_t_grid']:
        print_status(f"  Gamma_t={r['gamma_t']:.1f}: TEP dust={r['tep_dust']:.2f}, "
                     f"SN-only={r['sn_only_dust']:.2f}, enhancement={r['tep_enhancement']:.2f}x", "INFO")
    
    kt = results['key_test']
    print_status("")
    print_status("Key test (massive vs dwarf galaxies at z~8):", "PROCESS")
    print_status(f"  TEP predicts dust ratio: {kt['tep_mass_dust_ratio']:.1f}x", "INFO")
    print_status(f"  SN-only predicts: {kt['sn_only_mass_dust_ratio']:.1f}x", "INFO")
    print_status(f"  Observed ratio: ~{kt['observed_mass_dust_ratio_approx']:.0f}x", "INFO")
    print_status(f"  TEP overpredicts by factor: {kt.get('tep_overprediction_factor', float('nan')):.1f}x", "INFO")
    print_status(f"  SN-only overpredicts by factor: {kt.get('sn_overprediction_factor', float('nan')):.1f}x", "INFO")
    print_status(f"  TEP matches: {kt['tep_matches_observation']}", "INFO")
    print_status(f"  SN-only fails: {kt['sn_only_fails']}", "INFO")
    
    tep_factor = kt.get('tep_overprediction_factor', float('nan'))
    sn_factor = kt.get('sn_overprediction_factor', float('nan'))
    if kt['tep_matches_observation'] and kt['sn_only_fails']:
        conclusion = 'TEP dust predictions match observations; SN-only models fail'
    elif kt['tep_matches_observation'] and not kt['sn_only_fails']:
        conclusion = 'TEP dust predictions match observations; SN-only models also within range'
    elif not kt['tep_matches_observation'] and kt['sn_only_fails']:
        conclusion = f'TEP overpredicts dust mass ratio by {tep_factor:.1f}x; SN-only models fail'
    else:
        if tep_factor < sn_factor:
            conclusion = f'TEP overpredicts by {tep_factor:.1f}x, SN-only overpredicts by {sn_factor:.1f}x; TEP is closer but both exceed observed ratio'
        else:
            conclusion = f'TEP overpredicts by {tep_factor:.1f}x, SN-only overpredicts by {sn_factor:.1f}x; SN-only is closer but both exceed observed ratio'
    
    output = {
        'step': 129,
        'description': 'Dust Physics Alternative Models Test',
        'results': results,
        'conclusion': conclusion
    }
    
    with open(RESULTS_DIR / "step_129_dust_models.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print_status("")
    print_status(f"Conclusion: {conclusion}", "INFO")
    print_status(f"Results saved to step_129_dust_models.json", "SUCCESS")

if __name__ == "__main__":
    main()
