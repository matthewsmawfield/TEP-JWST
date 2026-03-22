#!/usr/bin/env python3
"""
TEP-JWST: Run All Analysis Steps

This script runs the complete reproducible analysis pipeline for the
manuscript build. The authoritative step registry is the STEPS list
defined below. Runtime estimates shown at startup and before each step
are loaded from `results/outputs/pipeline_summary.json`, i.e. the most
recent successful full canonical run.

Usage:
    python scripts/steps/run_all_steps.py
"""

import datetime
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEPS_DIR = PROJECT_ROOT / "scripts" / "steps"
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"

# =============================================================================
# FULL PIPELINE: 158 analysis steps
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
    "step_015_jwst_impossible_galaxies.py",         # Anomalous galaxies resolution
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
    # Functional form tests and extreme case analysis
    # =========================================================================
    "step_059_ultimate_missing_piece.py",           # Ultimate missing piece
    "step_060_functional_form_discrimination.py",   # The functional form discrimination test (t_eff vs t_cosmic)
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
    "step_075_bayesian_model_comparison.py",        # Bayesian model comparison (Savage-Dickey)
    "step_076_ml_scaling_justification.py",         # M/L scaling theoretical justification
    "step_077_bootstrap_validation.py",             # Bootstrap CIs and permutation tests
    "step_078_independent_age_validation.py",       # Balmer absorption age validation
    "step_079_ml_cross_validation.py",              # K-fold cross-validation of M/L index
    "step_081_survey_cross_correlation.py",         # Multi-survey meta-analysis
    "step_083_falsification_battery.py",            # Comprehensive falsification battery
    "step_084_effect_size_meta.py",                 # Effect size meta-analysis
    "step_085_time_lens_map.py",                    # Time-lens map
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

    # =========================================================================
    # PHASE XV: MODEL COMPARISON & PREDICTIONS (steps 097-118)
    # Alternative models, alpha recovery, predictions, significance
    # =========================================================================
    "step_098_alternative_model_comparison.py",     # Alternative model comparison (AIC/BIC)
    "step_099_alpha_evolution_test.py",             # Alpha(z) evolution test
    "step_100_multi_domain_model_comparison.py",    # Multi-domain model comparison
    "step_101_alpha_recovery_multi_observable.py",  # Multi-observable α₀ recovery
    "step_102_literature_anomaly_resolution.py",    # Literature anomaly resolution table
    "step_103_zphot_error_propagation.py",          # Photometric redshift error propagation
    "step_104_cosmic_variance.py",                  # Cosmic variance quantification
    "step_105_morphology_tep.py",                   # Morphology-TEP correlation
    "step_109_lcdm_tension_quantification.py",      # ΛCDM tension quantification
    "step_111_power_analysis.py",                   # Power analysis for key tests
    "step_112_scalar_tensor_constraints.py",        # Scalar-tensor constraint comparison
    "step_114_imf_constraint.py",                   # IMF constraint from TEP
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
    "step_141_nonlinear_aic.py",                    # Non-linear AIC: step-function t_eff vs M* (ΔAIC≈-5)
    "step_143_mass_proxy_breaker.py",               # Mass-proxy degeneracy breaker (3 independent tests)
    "step_144_adversarial_ml_attack.py",            # Adversarial ML attack: GBR/RF vs Gamma_t + cross-survey + CMI
    "step_145_phase_boundary_activation.py",        # AGB dust phase boundary + activation curve fit
    "step_146_stellar_mass_function_resolution.py", # SMF crisis resolution: TEP corrects anomalous masses at z>7
    "step_147_cosmic_sfrd_correction.py",           # Cosmic SFRD correction: reduces z>8 excess from 11× to 2.6× ΛCDM
    "step_148_mass_independent_proxy.py",           # Mass-independent potential-depth proxy tests (5 tests)

    # =========================================================================
    # PHASE XIX: MULTI-DATASET INGESTION, LATE AUDITS, AND FINAL SYNTHESIS
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
    "step_161_multi_dataset_l1_combination.py",     # Multi-dataset L1 Fisher combination (5 fields)
    "step_162_l1_l3_independence.py",               # L1-L3 independence test
    "step_163_external_dataset_registry.py",        # External-dataset shortlist + ingestion registry
    "step_164_uncover_z9_null_audit.py",            # UNCOVER z=9-12 null branch audit
    "step_165_uncover_z9_reddening_stack.py",       # UNCOVER z=9-12 stacked reddening surrogate
    "step_166_jades_z9_beta_contrast.py",           # JADES z=9-12 UV-slope contrast companion
    "step_167_protocluster_switch.py",              # Protocluster switch sign-reversal test
    "step_168_gradient_sign_reversal.py",           # Resolved gradient sign-reversal test
    "step_169_dja_sigma_pilot.py",                    # DJA pilot sigma extraction from public spec.fits URLs
    "step_171_sigma_kinematic_expansion.py",         # Sigma-based kinematic expansion (mass-circularity breaker)
    "step_170_kinematic_decisive_test.py",           # Kinematic decisive test + federated package (reads step_171)
    "step_173_cosmic_sfrd_correction.py",           # Cosmic SFRD correction (Table 16)
    "step_174_smf_mass_threshold_counts.py",        # SMF mass threshold counts (Table 15)
    "step_176_nested_bayesian_evidence.py",         # Nested Bayesian model comparison (dynesty)
    "step_140_evidence_tier_summary.py",            # Final evidence synthesis after late-ingestion outputs
    "step_160_manuscript_consistency_check.py",     # Automated manuscript↔JSON consistency checks
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


def _safe_read_json(json_path: Path):
    try:
        return json.loads(json_path.read_text())
    except Exception:
        return None


def _format_run_at_label(run_at: str | None) -> str | None:
    if not run_at:
        return None
    try:
        dt = datetime.datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    except ValueError:
        return run_at
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _load_last_run_estimates():
    summary = _safe_read_json(OUTPUTS_DIR / "pipeline_summary.json")
    if not summary or summary.get("status") != "PASS":
        return None

    total_elapsed_s = summary.get("total_elapsed_s")
    if not isinstance(total_elapsed_s, (int, float)):
        return None

    step_elapsed_s = {}
    for step in summary.get("steps", []):
        name = step.get("name")
        elapsed_s = step.get("elapsed_s")
        if isinstance(name, str) and isinstance(elapsed_s, (int, float)):
            step_elapsed_s[name] = float(elapsed_s)

    if not step_elapsed_s:
        return None

    total_steps = summary.get("total_steps")
    return {
        "run_at": summary.get("run_at"),
        "status": summary.get("status"),
        "total_steps": total_steps if isinstance(total_steps, int) else len(step_elapsed_s),
        "total_elapsed_s": float(total_elapsed_s),
        "step_elapsed_s": step_elapsed_s,
    }


def _scientific_reproducibility_snapshot():
    snapshot = {
        "execution_status_note": (
            "PASS means the step executed successfully and JSON guardrails passed; "
            "scientific branches may still be live, skipped, missing, or reference-only."
        )
    }

    evidence = _safe_read_json(OUTPUTS_DIR / "step_140_evidence_tier_summary.json")
    if evidence:
        final = evidence.get("final_synthesis", {})
        availability = final.get("availability", {})
        lines = final.get("lines_of_evidence", {})
        snapshot["primary_lines"] = {
            "live": availability.get("live_primary_lines"),
            "skipped": availability.get("skipped_primary_lines"),
            "missing": availability.get("missing_primary_lines"),
        }
        snapshot["line_statuses"] = {
            "L1": lines.get("L1_dust_replication", {}).get("status"),
            "L2": lines.get("L2_core_screening", {}).get("status"),
            "L3": lines.get("L3_ssfr_inversion", {}).get("status"),
            "L4": lines.get("L4_dynamical_mass", {}).get("status"),
            "L5": lines.get("L5_kinematic_decisive", {}).get("status"),
        }
        snapshot["claim_hierarchy"] = final.get("claim_hierarchy")
        verdict = final.get("verdict", {})
        snapshot["headline_primary_result"] = verdict.get("headline_primary_result")
        snapshot["headline_direct_test"] = verdict.get("headline_direct_test")
        snapshot["supplementary_mixed_branches"] = verdict.get("supplementary_mixed_branches", [])

    external_catalog_branches = {}
    for key, filename in {
        "step_150_dja_nirspec_merged": "step_150_dja_nirspec_merged.json",
        "step_158_dja_balmer_decrement": "step_158_dja_balmer_decrement.json",
    }.items():
        data = _safe_read_json(OUTPUTS_DIR / filename)
        if not data:
            continue
        external_catalog_branches[key] = {
            "status": data.get("status"),
            "reproducible_dja_available": data.get("reproducible_dja_available"),
            "manuscript_table_reproduced": data.get("manuscript_table_reproduced"),
        }

    if external_catalog_branches:
        snapshot["external_catalog_branches"] = external_catalog_branches
        snapshot["reference_only_branches"] = sum(
            1
            for branch in external_catalog_branches.values()
            if branch.get("status") == "SUCCESS_REFERENCE_ONLY"
        )

    return snapshot

# ---------------------------------------------------------------------------
# Step result record
# ---------------------------------------------------------------------------

class StepResult:
    def __init__(self, name, status, elapsed_s, returncode=0, problems=None):
        self.name       = name
        self.status     = status          # "PASS" | "FAIL" | "GUARDRAIL"
        self.elapsed_s  = elapsed_s
        self.returncode = returncode
        self.problems   = problems or []


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 0.1:
        return "<0.1s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    m, s = divmod(total_seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _boxed_line(text: str) -> str:
    return f"║  {text:<66}║"


# ---------------------------------------------------------------------------
# Run a single step
# ---------------------------------------------------------------------------

def run_step(script_name: str, step_idx: int, total: int, estimated_s: float | None = None) -> StepResult:
    """Run a single step script and return a StepResult."""
    before_state = _json_output_state()
    script_path  = STEPS_DIR / script_name
    phase_label  = f"[{step_idx:>3}/{total}]"
    pre_elapsed = 0.0

    bar_width = 50
    progress  = int(bar_width * step_idx / total)
    bar       = "█" * progress + "░" * (bar_width - progress)

    print(f"\n{'═'*70}")
    print(f" {phase_label}  {script_name}")
    print(f" Progress: [{bar}] {100*step_idx//total:3d}%")
    if estimated_s is not None:
        print(f" Estimated time from last full run: {_fmt_elapsed(estimated_s)}")
    print(f"{'═'*70}\n")

    if script_name == "step_160_manuscript_consistency_check.py":
        print(" Rebuilding site and generated markdown before step_160...")
        pre_t0 = time.perf_counter()
        build_result = subprocess.run(
            ["npm", "--prefix", str(PROJECT_ROOT / "site"), "run", "build"],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )
        pre_elapsed = time.perf_counter() - pre_t0
        if build_result.returncode != 0:
            print(f"\n✗  FAILED  site build before {script_name}  (rc={build_result.returncode}, {_fmt_elapsed(pre_elapsed)})")
            return StepResult(script_name, "FAIL", pre_elapsed, build_result.returncode)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    elapsed = pre_elapsed + (time.perf_counter() - t0)
    estimate_suffix = f"; est {_fmt_elapsed(estimated_s)}" if estimated_s is not None else ""

    if result.returncode != 0:
        print(f"\n✗  FAILED  {script_name}  (rc={result.returncode}, {_fmt_elapsed(elapsed)}{estimate_suffix})")
        return StepResult(script_name, "FAIL", elapsed, result.returncode)

    # --- JSON guardrail check ---
    after_state  = _json_output_state()
    changed_json = [p for p, sig in after_state.items() if sig != before_state.get(p)]
    all_problems = []
    for json_path in sorted(changed_json):
        problems = _validate_json_file(json_path)
        if problems:
            rel = json_path.relative_to(PROJECT_ROOT)
            print(f"\n✗  GUARDRAIL  {rel}")
            for kind, loc, val in problems[:10]:
                print(f"     {kind}: {loc} = {val}")
            if len(problems) > 10:
                print(f"     … {len(problems) - 10} more")
            all_problems.extend(problems)

    if all_problems:
        return StepResult(script_name, "GUARDRAIL", elapsed, result.returncode, all_problems)

    print(f"\n✓  PASS  {script_name}  ({_fmt_elapsed(elapsed)}{estimate_suffix})")
    return StepResult(script_name, "PASS", elapsed)


# ---------------------------------------------------------------------------
# Pipeline summary helpers
# ---------------------------------------------------------------------------

def _print_summary_table(results: list[StepResult], total_elapsed: float, scientific_state: dict | None = None):
    """Print a formatted summary table of all step results."""
    n_pass      = sum(1 for r in results if r.status == "PASS")
    n_fail      = sum(1 for r in results if r.status == "FAIL")
    n_guardrail = sum(1 for r in results if r.status == "GUARDRAIL")
    n_total     = len(results)

    col_w = 50
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  TEP-JWST PIPELINE SUMMARY" + " " * 41 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Step':<{col_w}}  {'Status':<10}  {'Time':>6}  ║")
    print("╠" + "═" * 68 + "╣")

    for r in results:
        icon = "✓" if r.status == "PASS" else "✗"
        stat = r.status
        name = r.name[:col_w]
        t    = _fmt_elapsed(r.elapsed_s)
        print(f"║  {icon} {name:<{col_w-2}}  {stat:<10}  {t:>6}  ║")

    print("╠" + "═" * 68 + "╣")
    pct = 100 * n_pass // n_total if n_total else 0
    print(f"║  PASS: {n_pass}/{n_total} ({pct}%)   FAIL: {n_fail}   GUARDRAIL: {n_guardrail}" + " " * (30 - len(str(n_pass)) - len(str(n_total))) + "  ║")
    print(f"║  Total elapsed: {_fmt_elapsed(total_elapsed):<52}  ║")
    print("╚" + "═" * 68 + "╝")

    if scientific_state:
        print("  Note: PASS means execution + JSON guardrails, not necessarily live scientific reproduction.")
        primary = scientific_state.get("primary_lines")
        if primary:
            print(
                "  Scientific state: "
                f"live primary lines={primary.get('live')}  "
                f"skipped={primary.get('skipped')}  "
                f"missing={primary.get('missing')}"
            )
        line_statuses = scientific_state.get("line_statuses")
        if line_statuses:
            print(
                "  Line statuses: "
                f"L1={line_statuses.get('L1')}  "
                f"L2={line_statuses.get('L2')}  "
                f"L3={line_statuses.get('L3')}  "
                f"L4={line_statuses.get('L4')}  "
                f"L5={line_statuses.get('L5')}"
            )
        headline_primary_result = scientific_state.get("headline_primary_result")
        if headline_primary_result:
            print(f"  Headline primary result: {headline_primary_result}")
        headline_direct_test = scientific_state.get("headline_direct_test")
        if headline_direct_test:
            print(f"  Headline direct test: {headline_direct_test}")
        supplementary_mixed_branches = scientific_state.get("supplementary_mixed_branches")
        if supplementary_mixed_branches:
            print(
                "  Supplementary or mixed branches: "
                + ", ".join(supplementary_mixed_branches)
            )
        if "reference_only_branches" in scientific_state:
            print(
                "  External catalog branches marked reference-only: "
                f"{scientific_state['reference_only_branches']}"
            )


def _write_pipeline_summary(results: list[StepResult], total_elapsed: float, scientific_state: dict | None = None):
    """Write a machine-readable pipeline summary JSON."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    n_pass      = sum(1 for r in results if r.status == "PASS")
    n_fail      = sum(1 for r in results if r.status == "FAIL")
    n_guardrail = sum(1 for r in results if r.status == "GUARDRAIL")

    payload = {
        "pipeline":       "TEP-JWST reproducible analysis",
        "run_at":         datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_steps":    len(results),
        "n_pass":         n_pass,
        "n_fail":         n_fail,
        "n_guardrail":    n_guardrail,
        "total_elapsed_s": round(total_elapsed, 1),
        "status":         "PASS" if n_fail == 0 and n_guardrail == 0 else "FAIL",
        "execution_status_note": (
            "PASS means the step executed successfully and JSON guardrails passed; "
            "scientific branches may still be live, skipped, missing, or reference-only."
        ),
        "scientific_reproducibility": scientific_state,
        "steps": [
            {
                "name":      r.name,
                "status":    r.status,
                "elapsed_s": round(r.elapsed_s, 2),
                "returncode": r.returncode,
            }
            for r in results
        ],
    }

    out = OUTPUTS_DIR / "pipeline_summary.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Pipeline summary written to: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()
    runtime_estimates = _load_last_run_estimates()
    step_estimates = runtime_estimates.get("step_elapsed_s", {}) if runtime_estimates else {}
    estimated_total = runtime_estimates.get("total_elapsed_s") if runtime_estimates else None
    estimate_source = _format_run_at_label(runtime_estimates.get("run_at")) if runtime_estimates else None

    print("╔" + "═" * 68 + "╗")
    print(_boxed_line("TEP-JWST: REPRODUCIBLE ANALYSIS PIPELINE"))
    print(_boxed_line("Temporal Equivalence Principle — JWST High-z Galaxy Analysis"))
    print("╠" + "═" * 68 + "╣")
    print(_boxed_line("Evidence lanes surfaced by the pipeline:"))
    print(_boxed_line("  L1. Dust–Γ_t correlation (live photometric line)"))
    print(_boxed_line("  L2. Resolved core screening (needs CIRC_CONV inputs)"))
    print(_boxed_line("  L3. Mass–sSFR inversion at z > 7"))
    print(_boxed_line("  L4. Dynamical mass consistency (regime-level live test)"))
    print("╠" + "═" * 68 + "╣")
    print(_boxed_line(f"Steps to run: {len(STEPS)}"))
    if estimated_total is not None:
        print(_boxed_line(f"Est. full runtime from last full run: {_fmt_elapsed(estimated_total)}"))
    if estimate_source:
        print(_boxed_line(f"Estimate source: {estimate_source}"))
    print(_boxed_line(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print("╚" + "═" * 68 + "╝\n")

    results: list[StepResult] = []
    failures = []

    for i, step in enumerate(STEPS, start=1):
        r = run_step(step, i, len(STEPS), step_estimates.get(step))
        results.append(r)
        if r.status != "PASS":
            failures.append(r)

    total_elapsed = time.perf_counter() - wall_start
    scientific_state = _scientific_reproducibility_snapshot()
    _print_summary_table(results, total_elapsed, scientific_state)
    _write_pipeline_summary(results, total_elapsed, scientific_state)

    if failures:
        print(f"\n  ⚠  {len(failures)} step(s) did not PASS:")
        for r in failures:
            print(f"       ✗  {r.name}  [{r.status}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
