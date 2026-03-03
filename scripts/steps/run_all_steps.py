#!/usr/bin/env python3
"""
TEP-JWST: Run All Analysis Steps

This script runs the complete reproducible analysis pipeline for the
manuscript build. The authoritative step registry is the STEPS list
defined below.

Usage:
    python scripts/steps/run_all_steps.py
"""

import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEPS_DIR = PROJECT_ROOT / "scripts" / "steps"
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"

# =============================================================================
# FULL PIPELINE: 163 analysis steps
# Includes core analysis, replication, robustness, falsification, and
# advanced discriminating tests used by the current manuscript build.
# =============================================================================
STEPS = [
    # =========================================================================
    # PHASE I: CORE PIPELINE (steps 001-008)
    # Data loading, TEP model, and seven threads of evidence
    # =========================================================================
    "step_001_uncover_load.py",                     # Data loading and quality cuts
    "step_002_tep_model.py",                        # TEP model and Γ_t computation
    "step_003_first_principles.py",                 # First-principles derivation
    "step_004_thread1_z7_inversion.py",             # Thread 1: z>7 mass-sSFR inversion
    "step_005_thread2_4_partial_correlations.py",   # Threads 2-4: Partial correlations
    "step_006_thread5_z8_dust.py",                  # Thread 5: z>8 dust anomaly
    "step_007_thread6_7_coherence.py",              # Threads 6-7: Coherence tests
    "step_008_summary.py",                          # Seven threads summary

    # =========================================================================
    # PHASE II: CROSS-DOMAIN VALIDATION (steps 009-013)
    # Testing TEP predictions across different astrophysical domains
    # =========================================================================
    "step_009_holographic_synthesis.py",             # Cross-paper consistency
    "step_010_sn_ia_mass_step.py",                  # SN Ia mass step prediction
    "step_011_mw_gc_gradient.py",                   # MW GC screening test
    "step_012_sn_ia_extended.py",                   # Extended SN Ia analysis
    "step_013_trgb_cepheid.py",                     # TRGB-Cepheid offset

    # =========================================================================
    # PHASE III: JWST EXTENDED ANALYSIS (steps 014-019)
    # Deep analysis of JWST high-z galaxy properties
    # =========================================================================
    "step_014_jwst_uv_slope.py",                    # UV slope analysis
    "step_015_jwst_impossible_galaxies.py",         # Impossible galaxies resolution
    "step_016_robustness_tests.py",                 # Robustness and systematics
    "step_017_ml_ratio.py",                         # Mass-to-light ratio
    "step_018_assembly_time.py",                    # Assembly time
    "step_019_chi2_analysis.py",                    # χ² correlation analysis

    # =========================================================================
    # PHASE IV: STATISTICAL SYNTHESIS (steps 020-024)
    # Parameter validation and cosmological implications
    # =========================================================================
    "step_020_parameter_validation.py",             # α₀ = 0.58 validation
    "step_021_scatter_reduction.py",                # Scatter reduction
    "step_022_extreme_population.py",               # Extreme population analysis
    "step_023_self_consistency.py",                  # Self-consistency tests
    "step_024_cosmological_implications.py",        # Cosmological implications

    # =========================================================================
    # PHASE V: VALIDATION (steps 025-028)
    # Cross-sample and multi-angle validation
    # =========================================================================
    "step_025_cross_sample_validation.py",          # Cross-sample validation
    "step_026_prediction_alignment.py",             # Prediction-observation alignment
    "step_027_multi_angle_validation.py",           # Multi-angle validation
    "step_028_chi2_diagnostic.py",                  # χ² as TEP diagnostic

    # =========================================================================
    # PHASE VI: INDEPENDENT REPLICATION (steps 029-035)
    # Cross-survey validation with CEERS, COSMOS-Web, spectroscopy
    # =========================================================================
    "step_029_independent_validation.py",           # Out-of-sample validation
    "step_030_z8_dust_prediction.py",               # z>8 dust quantitative tests
    "step_031_ceers_download.py",                   # CEERS DR1 catalog download
    "step_032_ceers_replication.py",                # CEERS independent replication
    "step_033_cosmosweb_download.py",               # COSMOS-Web DR1 catalog download
    "step_034_cosmosweb_replication.py",            # COSMOS-Web independent replication
    "step_035_spectroscopic_validation.py",         # Spectroscopic validation (PRIMARY)

    # =========================================================================
    # PHASE VII: SCREENING & SENSITIVITY (steps 036-038)
    # Core screening, resolved gradients, parameter sensitivity
    # =========================================================================
    "step_036_spectroscopic_refinement.py",         # Simpson's paradox & bin-normalized
    "step_037_resolved_gradients.py",               # Resolved core screening
    "step_038_sensitivity_analysis.py",             # Parameter sensitivity sweep

    # =========================================================================
    # PHASE VIII: BLACK HOLE & POPULATION ANALYSIS (steps 039-043)
    # LRD, Blue Monsters, and mass sensitivity
    # =========================================================================
    "step_039_overmassive_bh.py",                   # Overmassive Black Hole / LRD
    "step_040_mass_sensitivity.py",                 # Mass reduction sensitivity
    "step_041_forward_modeling.py",                 # Forward-modeling SED validation
    "step_042_lrd_population.py",                   # LRD population analysis
    "step_043_blue_monsters.py",                    # Blue Monster TEP analysis

    # =========================================================================
    # PHASE IX: DIAGNOSTIC TESTS (steps 044-053)
    # Multi-property diagnostics and independent tests
    # =========================================================================
    "step_044_metallicity_age_decoupling.py",       # Metallicity-age decoupling
    "step_045_sfr_age_consistency.py",              # SFR-age consistency
    "step_046_multi_diagnostic.py",                 # Multi-diagnostic evidence
    "step_047_independent_tests.py",                # Independent tests
    "step_048_advanced_diagnostics.py",             # Advanced diagnostics
    "step_049_prediction_tests.py",                 # TEP prediction tests
    "step_050_critical_evidence.py",                # Critical evidence tests
    "step_051_temporal_coherence.py",               # Temporal coherence
    "step_052_mass_discrepancy.py",                 # Mass discrepancy
    "step_053_tep_discriminant.py",                 # TEP discriminant

    # =========================================================================
    # PHASE X: FALSIFICATION & ADVERSARIAL (steps 054-058)
    # Tests designed to challenge and falsify TEP
    # =========================================================================
    "step_054_unique_predictions.py",               # Unique predictions
    "step_055_cross_domain.py",                     # Cross-domain consistency
    "step_056_out_of_box.py",                       # Out-of-box tests
    "step_057_adversarial_tests.py",                # Adversarial tests
    "step_058_creative_tests.py",                   # Creative tests

    # =========================================================================
    # PHASE XI: ADVANCED PHYSICS (steps 059-065)
    # Killer tests and extreme case analysis
    # =========================================================================
    "step_059_ultimate_missing_piece.py",           # Ultimate missing piece
    "step_060_the_killer_test.py",                  # The killer test (t_eff vs t_cosmic)
    "step_061_extreme_cases.py",                    # Extreme cases
    "step_062_predictive_power.py",                 # Predictive power
    "step_063_correlation_deep_dive.py",            # Correlation deep dive
    "step_064_web_inspired_tests.py",               # Web-inspired tests
    "step_065_literature_tests.py",                 # Literature tests

    # =========================================================================
    # PHASE XII: OBSERVATIONAL CONSTRAINTS (steps 066-071)
    # Literature tests, compactness, binary pulsars, BBN
    # =========================================================================
    "step_066_final_literature_tests.py",           # Final literature tests
    "step_067_cutting_edge_tests.py",               # Cutting edge tests
    "step_068_observational_signatures.py",         # Observational signatures
    "step_069_compactness_verification.py",         # Compactness verification (LRDs)
    "step_070_binary_pulsar_constraints.py",        # Binary pulsar screening verification
    "step_071_bbn_analysis.py",                     # BBN compatibility analysis

    # =========================================================================
    # PHASE XIII: STATISTICAL ROBUSTNESS (steps 072-088)
    # Bootstrap, Bayesian, power analysis, comprehensive synthesis
    # =========================================================================
    "step_072_sign_paradox_check.py",               # Theoretical scalar proper-time sign check
    "step_073_growth_factor.py",                    # Linear growth and σ₈ constraint
    "step_074_z67_tep_prediction.py",               # z=6-7 dip dust physics prediction
    "step_075_bayesian_model_comparison.py",        # Bayesian model comparison (Savage-Dickey)
    "step_076_ml_scaling_justification.py",         # M/L scaling theoretical justification
    "step_077_bootstrap_validation.py",             # Bootstrap CIs and permutation tests
    "step_078_independent_age_validation.py",       # Balmer absorption age validation
    "step_079_ml_cross_validation.py",              # K-fold cross-validation of M/L index
    "step_080_balmer_simulation.py",                # Balmer absorption line simulation
    "step_081_survey_cross_correlation.py",         # Multi-survey meta-analysis
    "step_082_evidence_strengthening.py",           # Monte Carlo evidence strengthening
    "step_083_falsification_battery.py",            # Comprehensive falsification battery
    "step_084_effect_size_meta.py",                 # Effect size meta-analysis
    "step_085_time_lens_map.py",                    # Time-lens map
    "step_086_posterior_predictive.py",             # Posterior predictive checks
    "step_087_emission_line_tests.py",              # Emission line tests
    "step_088_final_comprehensive.py",              # FINAL COMPREHENSIVE SUMMARY

    # =========================================================================
    # PHASE XIV: THRESHOLD & PERMUTATION TESTS (steps 089-096)
    # Discovery scans, holdout validation, enhanced screening
    # =========================================================================
    "step_089_teff_threshold_scan.py",              # AGB-timescale threshold scan (discovery)
    "step_090_permutation_battery.py",              # Empirical permutation p-values
    "step_091_random_effects_meta_loo.py",          # Random-effects meta + leave-one-out influence
    "step_092_mass_matched_confirmation.py",        # Mass-matched/stratified confirmation
    "step_093_teff_threshold_holdout.py",           # Holdout (leave-one-survey-out) threshold validation
    "step_094_environmental_screening_enhanced.py", # Enhanced environmental screening (multiple density estimators)
    "step_095_lrd_core_halo_mass.py",               # LRD core-halo mass derivation from resolved photometry
    "step_096_balmer_target_list.py",               # Balmer absorption target list for JWST proposals

    # =========================================================================
    # PHASE XV: MODEL COMPARISON & PREDICTIONS (steps 097-118)
    # Alternative models, alpha recovery, predictions, significance
    # =========================================================================
    "step_097_money_plot.py",                       # TEP predictions vs observations summary figure
    "step_098_alternative_model_comparison.py",     # Alternative model comparison (AIC/BIC)
    "step_099_alpha_evolution_test.py",             # Alpha(z) evolution test
    "step_100_multi_domain_model_comparison.py",    # Multi-domain model comparison
    "step_101_alpha_recovery_multi_observable.py",  # Multi-observable α₀ recovery
    "step_102_literature_anomaly_resolution.py",    # Literature anomaly resolution table
    "step_103_zphot_error_propagation.py",          # Photometric redshift error propagation
    "step_104_cosmic_variance.py",                  # Cosmic variance quantification
    "step_105_morphology_tep.py",                   # Morphology-TEP correlation
    "step_106_emission_line_diagnostic.py",         # Emission line diagnostic predictions
    "step_107_sn_rate_prediction.py",               # Time-domain SN rate prediction
    "step_108_jwst_cycle4_targets.py",              # JWST Cycle 4 observing proposal targets
    "step_109_lcdm_tension_quantification.py",      # ΛCDM tension quantification
    "step_110_gw_timing_prediction.py",             # Gravitational wave timing prediction
    "step_111_power_analysis.py",                   # Power analysis for key tests
    "step_112_scalar_tensor_constraints.py",        # Scalar-tensor constraint comparison
    "step_113_agn_feedback_discriminant.py",        # AGN feedback discriminant
    "step_114_imf_constraint.py",                   # IMF constraint from TEP
    "step_115_euclid_roman_predictions.py",         # Euclid and Roman predictions
    "step_116_literature_spectroscopic_ages.py",    # Literature spectroscopic age compilation
    "step_117_dynamical_mass_comparison.py",        # Quantitative M*/M_dyn analysis
    "step_118_neff_corrected_significance.py",      # N_eff-corrected combined significance

    # =========================================================================
    # PHASE XVI: THEORETICAL CONSISTENCY (steps 119-126)
    # Causality, screening, multi-tracer, modified gravity
    # =========================================================================
    "step_119_blind_validation.py",                 # Blind validation protocol
    "step_120_screening_transition_profile.py",     # Screening transition profile
    "step_121_extended_modified_gravity_comparison.py", # Extended modified gravity comparison
    "step_122_causality_verification.py",           # Causality constraint verification
    "step_123_alpha0_error_budget.py",              # Alpha_0 systematic error propagation
    "step_124_timespace_coupling.py",               # Time-space coupling consistency test
    "step_125_multitracer_consistency.py",          # Multi-tracer redshift consistency
    "step_126_screening_scale.py",                  # Screening length scale self-consistency

    # =========================================================================
    # PHASE XVII: ADVANCED MODEL DISCRIMINATION (steps 127-134)
    # IMF, selection MC, dust models, AGN, LRD, Hubble connection
    # =========================================================================
    "step_127_imf_discrimination.py",               # IMF vs TEP discrimination power
    "step_128_selection_mc.py",                     # Selection function Monte Carlo
    "step_129_dust_models.py",                      # Dust physics alternative models
    "step_130_cross_survey_systematics.py",         # Cross-survey systematic budget
    "step_131_agn_power.py",                        # AGN feedback discrimination power
    "step_132_lrd_validation.py",                   # LRD case study validation
    "step_133_hubble_connection.py",                # Hubble tension connection
    "step_134_prediction_errors.py",                # TEP prediction error budget

    # =========================================================================
    # PHASE XVIII: DISCRIMINATING TESTS (steps 135-148)
    # Steiger Z-tests, adversarial ML, phase boundaries, SMF, SFRD
    # =========================================================================
    "step_135_scale_dependent_growth.py",           # Scale-dependent growth factor
    "step_136_functional_form_test.py",             # Functional form discrimination test
    "step_137_cross_survey_generalization.py",      # Cross-survey generalization test
    "step_138_environmental_screening_steiger.py",  # Environmental screening Steiger Z-test
    "step_139_colour_gradient_steiger.py",          # Colour-gradient Steiger Z-test (t_eff vs M*)
    "step_140_evidence_tier_summary.py",            # Evidence tier summary (Tier 1/2/3 hierarchy)
    "step_141_nonlinear_aic.py",                    # Non-linear AIC: step-function t_eff vs M* (ΔAIC=-23)
    "step_142_lrd_mbh_mstar_prediction.py",         # LRD M_BH/M_* quantitative prediction vs observation
    "step_143_mass_proxy_breaker.py",               # Mass-proxy degeneracy breaker (3 independent tests)
    "step_144_adversarial_ml_attack.py",            # Adversarial ML attack: GBR/RF vs Gamma_t + cross-survey + CMI
    "step_145_phase_boundary_activation.py",        # AGB dust phase boundary + activation curve fit
    "step_146_stellar_mass_function_resolution.py", # SMF crisis resolution: TEP corrects impossible masses at z>7
    "step_147_cosmic_sfrd_correction.py",           # Cosmic SFRD correction: reduces z>8 excess from 11× to 2.6× ΛCDM
    "step_148_mass_independent_proxy.py",           # Mass-independent potential-depth proxy tests (5 tests)

    # =========================================================================
    # PHASE XIX: MULTI-DATASET INGESTION (steps 149-163)
    # JADES, DJA, UNCOVER DR4, COSMOS2025, manuscript QA
    # =========================================================================
    "step_149_jades_dr4_ingestion.py",              # JADES DR4 spectroscopic catalog (2,858 spec-z)
    "step_150_dja_nirspec_merged.py",               # DJA NIRSpec Merged v4.4 (19,445 grade-3 sources)
    "step_151_dja_ceers_crossmatch.py",             # DJA spec-z × CEERS+UNCOVER SED (776 z>5)
    "step_152_uncover_dr4_full_sps.py",             # UNCOVER DR4 full SPS MegaScience 20-band
    "step_153_cosmos2025_sed_analysis.py",          # COSMOS2025 LePHARE SED (37,965 z>4)
    "step_154_jades_emission_lines.py",             # JADES DR4 emission lines (ionization signal)
    "step_155_jades_dr5_morphology.py",             # JADES DR5 morphology × DR4 spec-z
    "step_156_dja_gds_morphology.py",               # DJA GDS morphology × spec-z
    "step_157_cosmos2025_ssfr_inversion.py",        # COSMOS2025 sSFR inversion (Steiger Z=6.37)
    "step_158_dja_balmer_decrement.py",             # DJA NIRSpec Balmer decrement (N=2925)
    "step_159_mass_measurement_bias.py",            # TEP mass bias analysis
    "step_160_manuscript_consistency_check.py",     # Automated manuscript↔JSON consistency checks
    "step_161_multi_dataset_l1_combination.py",     # Multi-dataset L1 Fisher combination (5 fields)
    "step_162_l1_l3_independence.py",               # L1-L3 independence test
    "step_163_external_dataset_registry.py",        # External-dataset shortlist + ingestion registry
]

def _json_output_state():
    state = {}
    if not OUTPUTS_DIR.exists():
        return state
    for path in OUTPUTS_DIR.rglob("*.json"):
        try:
            st = path.stat()
        except OSError:
            continue
        state[path] = (st.st_mtime_ns, st.st_size)
    return state


def _is_p_value_key(key: str) -> bool:
    k = key.lower()
    if "log10" in k or "formatted" in k:
        return False
    if k in {"p", "p_value", "pval", "pvalue"}:
        return True
    if k.startswith("p_"):
        return True
    if k.endswith("_p") or k.endswith("_p_value") or k.endswith("_pvalue"):
        return True
    return False


def _validate_json_obj(obj, problems, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                problems.append(("non_finite", new_path, v))
            if _is_p_value_key(k) and isinstance(v, (int, float)) and v == 0:
                problems.append(("p_value_zero", new_path, v))
            _validate_json_obj(v, problems, new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                problems.append(("non_finite", new_path, v))
            _validate_json_obj(v, problems, new_path)


def _validate_json_file(json_path: Path):
    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        return [("json_load_error", "", str(e))]

    problems = []
    _validate_json_obj(data, problems)
    return problems

def run_step(script_name):
    """Run a single step script."""
    before_state = _json_output_state()
    script_path = STEPS_DIR / script_name
    print(f"\n{'='*70}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with return code {result.returncode}")
        return False

    after_state = _json_output_state()
    changed_json = [
        p for p, sig in after_state.items()
        if sig != before_state.get(p)
    ]
    for json_path in sorted(changed_json):
        problems = _validate_json_file(json_path)
        if problems:
            print(f"\nERROR: Output guardrail failed for {json_path.relative_to(PROJECT_ROOT)}")
            for kind, loc, val in problems[:20]:
                print(f"  - {kind}: {loc} = {val}")
            if len(problems) > 20:
                print(f"  - truncated: {len(problems) - 20} more")
            return False
    
    return True

def main():
    print("="*70)
    print("TEP-JWST: REPRODUCIBLE ANALYSIS PIPELINE")
    print("="*70)
    print()
    print("This pipeline tests the seven threads of TEP evidence:")
    print("  1. z > 7 Mass-sSFR Inversion")
    print("  2. Γ_t vs Age Ratio (partial)")
    print("  3. Γ_t vs Metallicity (partial)")
    print("  4. Γ_t vs Dust (partial)")
    print("  5. z > 8 Dust Anomaly")
    print("  6. Age-Metallicity Coherence")
    print("  7. Multi-Property Split")
    print()
    print("Plus holographic synthesis (cross-paper consistency):")
    print("  - Proto-GC ages (Sparkler system)")
    print("  - Spectroscopic confirmations")
    print("  - TEP-H0/TEP-COS consistency check")
    print()
    
    success_count = 0
    for step in STEPS:
        if run_step(step):
            success_count += 1
        else:
            print(f"\nPipeline stopped at {step}")
            break
    
    print()
    print("="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Steps completed: {success_count}/{len(STEPS)}")
    print()
    print("Results saved to: results/outputs/")
    print(f"  - step_XX_*.json ({len(STEPS)} output files)")
    print()
    print("Logs saved to: logs/")
    print(f"  - step_XX_*.log ({len(STEPS)} log files)")

if __name__ == "__main__":
    main()
