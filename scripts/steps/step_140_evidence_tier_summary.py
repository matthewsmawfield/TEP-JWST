#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
TEP-JWST Step 140: Final Synthesis

This step compiles ALL tests from the entire exploration into a
comprehensive final summary with statistics.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging

STEP_NUM = "140"  # Pipeline step number
STEP_NAME = "evidence_tier_summary"  # Used in log / output filenames

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_json_optional(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Final Synthesis", "INFO")
    print_status("=" * 70, "INFO")
    
    data_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    n_total = None
    if data_path.exists():
        df = pd.read_csv(data_path)
        n_total = int(len(df))
        print_status(f"\nLoaded N = {n_total} galaxies", "INFO")
    else:
        print_status("Input catalog not found; continuing with output-only synthesis.", "INFO")

    s161 = load_json_optional(OUTPUT_PATH / "step_161_multi_dataset_l1_combination.json")
    s004 = load_json_optional(OUTPUT_PATH / "step_004_thread1_z7_inversion.json")
    s037 = load_json_optional(OUTPUT_PATH / "step_037_resolved_gradients.json")
    s139 = load_json_optional(OUTPUT_PATH / "step_139_colour_gradient_steiger.json")
    s117 = load_json_optional(OUTPUT_PATH / "step_117_dynamical_mass_comparison.json")
    s170 = load_json_optional(OUTPUT_PATH / "step_170_kinematic_decisive_test.json")
    s143 = load_json_optional(OUTPUT_PATH / "step_143_mass_proxy_breaker.json")
    s144 = load_json_optional(OUTPUT_PATH / "step_144_adversarial_ml_attack.json")
    s162 = load_json_optional(OUTPUT_PATH / "step_162_l1_l3_independence.json")
    s137 = load_json_optional(OUTPUT_PATH / "step_137_cross_survey_generalization.json")
    s138 = load_json_optional(OUTPUT_PATH / "step_138_environmental_screening_steiger.json")
    s141 = load_json_optional(OUTPUT_PATH / "step_141_nonlinear_aic.json")
    s148 = load_json_optional(OUTPUT_PATH / "step_148_mass_independent_proxy.json")
    s146 = load_json_optional(OUTPUT_PATH / "step_146_stellar_mass_function_resolution.json")
    s147 = load_json_optional(OUTPUT_PATH / "step_147_cosmic_sfrd_correction.json")
    s155 = load_json_optional(OUTPUT_PATH / "step_155_jades_dr5_morphology.json")
    s157 = load_json_optional(OUTPUT_PATH / "step_157_cosmos2025_ssfr_inversion.json")
    s159 = load_json_optional(OUTPUT_PATH / "step_159_mass_measurement_bias.json")
    s168 = load_json_optional(OUTPUT_PATH / "step_168_gradient_sign_reversal.json")
    s150 = load_json_optional(OUTPUT_PATH / "step_150_dja_nirspec_merged.json")
    s158 = load_json_optional(OUTPUT_PATH / "step_158_dja_balmer_decrement.json")
    s164 = load_json_optional(OUTPUT_PATH / "step_164_uncover_z9_null_audit.json")
    s169 = load_json_optional(OUTPUT_PATH / "step_169_dja_sigma_pilot.json")
    s176 = load_json_optional(OUTPUT_PATH / "step_176_nested_bayesian_evidence.json")

    results = {
        'n_total': n_total,
        'final_synthesis': {}
    }
    lines_of_evidence = {}

    if s161:
        l1_summary = s161.get('summary', {})
        l1_primary = s161.get('conservative_photometric_only', {})
        l1_all = s161.get('fisher_combination', {})
        lines_of_evidence['L1_dust_replication'] = {
            'status': 'live',
            'n_datasets': l1_primary.get('n_datasets', l1_all.get('n_datasets')),
            'headline_z': l1_primary.get('z_sigma', l1_summary.get('conservative_z', l1_summary.get('headline_z'))),
            'headline_p': l1_primary.get('p', l1_summary.get('headline_p')),
            'supplementary_all_dataset_n': l1_all.get('n_datasets'),
            'supplementary_all_dataset_z': l1_all.get('z_sigma'),
            'supplementary_all_dataset_p': l1_all.get('p'),
            'conservative_z': l1_summary.get('conservative_z'),
            'key_result': l1_summary.get('key_result'),
        }
    else:
        lines_of_evidence['L1_dust_replication'] = {
            'status': 'missing',
            'reason': 'step_161 output not available'
        }

    l2_reason = None
    if s037 and s037.get('status') == 'skipped':
        l2_reason = s037.get('reason')
    if s139 and s139.get('status') == 'skipped':
        l2_reason = s139.get('reason')
    if s037 and s037.get('status') != 'skipped' and s037.get('rho_mass_grad') is not None:
        l2_headline = s155.get('headline', {}) if s155 else {}
        l2_results = s155.get('results', {}) if s155 else {}
        l2_supportive_keys = l2_headline.get('supportive_partial_keys') or []
        l2_strongest_key = l2_headline.get('strongest_partial_key')
        l2_strongest_result = l2_results.get(l2_strongest_key, {}) if l2_strongest_key else {}
        l2_supportive_results = {
            key: {
                'rho_partial_mass_z': l2_results.get(key, {}).get('rho_partial_mass_z'),
                'p_partial_mass_z': l2_results.get(key, {}).get('p_partial_mass_z'),
                'N': l2_results.get(key, {}).get('N'),
            }
            for key in l2_supportive_keys
        }
        s168_primary = s168.get('primary_contrast', {}) if s168 else {}
        lines_of_evidence['L2_core_screening'] = {
            'status': 'live',
            'rho_mass_grad': s037.get('rho_mass_grad'),
            'p_mass_grad': s037.get('p_mass_grad'),
            'n': s037.get('n'),
            'rho_gamma_grad': (
                s139.get('raw_correlations', {}).get('rho_gradient_gamma_t') if s139 else None
            ),
            'p_gamma_grad': (
                s139.get('raw_correlations', {}).get('p_gradient_gamma_t') if s139 else None
            ),
            'gradient_beta_debias_used': (
                s139.get('partial_correlations', {}).get('beta_debias_used') if s139 else None
            ),
            'gradient_beta_debias_source': (
                s139.get('partial_correlations', {}).get('beta_debias_source') if s139 else None
            ),
            'gradient_partial_rho_gamma_given_mass_z': (
                s139.get('partial_correlations', {}).get('rho_gradient_gamma_t_given_mass_z') if s139 else None
            ),
            'gradient_partial_p_gamma_given_mass_z': (
                s139.get('partial_correlations', {}).get('p_gradient_gamma_t_given_mass_z') if s139 else None
            ),
            'gradient_partial_rho_gamma_given_debiased_mass_z': (
                s139.get('partial_correlations', {}).get('rho_gradient_gamma_t_given_debiased_mass_z') if s139 else None
            ),
            'gradient_partial_p_gamma_given_debiased_mass_z': (
                s139.get('partial_correlations', {}).get('p_gradient_gamma_t_given_debiased_mass_z') if s139 else None
            ),
            'gradient_partial_preferred_mass_control': (
                s139.get('partial_correlations', {}).get('preferred_gamma_gradient_mass_control') if s139 else None
            ),
            'raw_colour_gradient_tep_confirmed': (
                s139.get('summary', {}).get('tep_confirmed') if s139 else None
            ),
            'gradient_sign_test_assessment': s168.get('assessment') if s168 else None,
            'gradient_sign_test_primary_selection': s168.get('primary_selection') if s168 else None,
            'gradient_sign_test_primary_negative_fraction_high': s168_primary.get('negative_fraction_high'),
            'gradient_sign_test_primary_negative_fraction_low': s168_primary.get('negative_fraction_low'),
            'gradient_sign_test_primary_fisher_p': s168_primary.get('fisher_p_one_sided'),
            'gradient_sign_test_primary_delta_mean': s168_primary.get('delta_mean_high_minus_low'),
            'structural_support_status': (
                l2_headline.get('conclusion') if s155 else None
            ),
            'structural_supportive_keys': l2_supportive_keys,
            'structural_supportive_partial_count': (
                l2_headline.get('n_structural_proxies_supportive_after_mass_z_control') if s155 else None
            ),
            'structural_preferred_sample': s155.get('preferred_sample') if s155 else None,
            'structural_n_matched': s155.get('n_matched') if s155 else None,
            'structural_n_with_mass': s155.get('n_with_mass') if s155 else None,
            'structural_strongest_partial_key': l2_strongest_key,
            'structural_strongest_partial_rho': l2_strongest_result.get('rho_partial_mass_z'),
            'structural_strongest_partial_p': l2_strongest_result.get('p_partial_mass_z'),
            'structural_strongest_partial_N': l2_strongest_result.get('N'),
            'structural_supportive_results': l2_supportive_results,
        }
    else:
        lines_of_evidence['L2_core_screening'] = {
            'status': 'skipped',
            'reason': l2_reason or 'resolved gradient outputs not available'
        }

    if s004:
        lines_of_evidence['L3_ssfr_inversion'] = {
            'status': 'live',
            'delta_rho': s004.get('delta_rho'),
            'delta_ci_95': s004.get('delta_ci_95'),
            'high_z_rho': s004.get('high_z', {}).get('rho'),
            'high_z_p': s004.get('high_z', {}).get('p_value'),
            'n_high_z': s004.get('high_z', {}).get('n'),
            'significant': s004.get('significant'),
            'external_blank_field_assessment': (
                s157.get('external_replication_summary', {}).get('assessment') if s157 else None
            ),
            'external_blank_field_matched_bin_rho_debiased': (
                s157.get('external_replication_summary', {})
                .get('primary_matched_bin', {})
                .get('partial_rho_debiased_mass')
                if s157 else None
            ),
            'external_blank_field_ultrahighz_rho_debiased': (
                s157.get('external_replication_summary', {})
                .get('ultrahighz_sensitivity_bin', {})
                .get('partial_rho_debiased_mass')
                if s157 else None
            ),
            'external_blank_field_ultrahighz_reference_mass_reweighted_rho_debiased': (
                s157.get('external_replication_summary', {})
                .get('ultrahighz_sensitivity_bin', {})
                .get('reference_mass_reweighted_partial_rho_debiased_mass')
                if s157 else None
            ),
        }
    else:
        lines_of_evidence['L3_ssfr_inversion'] = {
            'status': 'missing',
            'reason': 'step_004 output not available'
        }

    if s117:
        pub = s117.get('published_tension_resolution', {})
        supplementary_direct = s117.get('supplementary_direct_object_level', {})
        supplementary_direct_table = supplementary_direct.get('kinematic_table', {})
        supplementary_direct_summary = supplementary_direct.get('object_level_summary', {})
        supplementary_direct_upper_limits = supplementary_direct.get('upper_limit_summary', {})
        dja_branch = (
            s170.get('federated_direct_kinematic_package', {}).get('branches', {}).get('dja_sigma_balmer_pilot', {})
            if s170 else {}
        )
        dja_branch_present = bool(dja_branch)
        dja_branch_counted = bool(dja_branch_present and dja_branch.get('counts_toward_supportive_tally', True))
        sigma_surface_allowed = bool(s169) and (dja_branch_present or s170 is None)
        sigma_pilot = s169.get('pilot_balmer_sigma_test', {}).get('quality_screened', {}) if sigma_surface_allowed else {}
        sigma_fit_summary = s169.get('fit_summary', {}).get('quality_screened', {}) if sigma_surface_allowed else {}
        lines_of_evidence['L4_dynamical_mass'] = {
            'status': (
                'direct_object_level_kinematics'
                if s117.get('direct_kinematic_measurements_used') else
                'derived_from_real_data'
            ),
            'analysis_class': s117.get('analysis_class'),
            'direct_kinematic_measurements_used': s117.get('direct_kinematic_measurements_used'),
            'published_excess_dex': pub.get('published_excess_dex'),
            'tep_reduction_dex': pub.get('tep_reduction_dex'),
            'resolved': pub.get('resolved'),
            'n_kinematic_regime': s117.get('n_kinematic_regime'),
            'object_level_beta_bootstrap': s117.get('object_level_beta_bootstrap'),
            'supplementary_direct_object_level_available': s117.get('supplementary_direct_object_level_available', False),
            'supplementary_direct_n_objects': supplementary_direct.get('n_kinematic_regime'),
            'supplementary_direct_n_objects_exact_mdyn': supplementary_direct_table.get('n_objects_exact_mdyn'),
            'supplementary_direct_n_objects_upper_limit_only': supplementary_direct_table.get('n_objects_upper_limit_only'),
            'supplementary_direct_mean_observed_excess_dex': supplementary_direct_summary.get('mean_observed_excess_dex'),
            'supplementary_direct_mean_corrected_excess_dex': supplementary_direct_summary.get('mean_corrected_excess_dex'),
            'supplementary_direct_mean_excess_metrics_basis': supplementary_direct_summary.get('mean_excess_metrics_basis'),
            'supplementary_direct_resolution_fraction_among_anomalous': supplementary_direct_summary.get('resolution_fraction_among_anomalous'),
            'supplementary_direct_upper_limit_mean_observed_excess_lower_bound_dex': supplementary_direct_upper_limits.get('mean_observed_excess_lower_bound_dex'),
            'supplementary_direct_upper_limit_mean_corrected_excess_lower_bound_dex': supplementary_direct_upper_limits.get('mean_corrected_excess_lower_bound_dex'),
            'sigma_pilot_status': s169.get('status') if sigma_surface_allowed else None,
            'sigma_pilot_assessment': s169.get('assessment') if sigma_surface_allowed else None,
            'sigma_pilot_counted_toward_supportive_tally': dja_branch_counted if sigma_surface_allowed and s170 else None,
            'sigma_pilot_quality_fit_n': sigma_fit_summary.get('n_success'),
            'sigma_pilot_quality_balmer_n': sigma_pilot.get('n'),
            'sigma_pilot_quality_partial_rho_sigma_given_mass_z': sigma_pilot.get('partial_sigma_given_mass_z', {}).get('rho'),
            'sigma_pilot_quality_partial_p_sigma_given_mass_z': sigma_pilot.get('partial_sigma_given_mass_z', {}).get('p'),
            'sigma_pilot_quality_partial_rho_mass_given_sigma_z': sigma_pilot.get('partial_mass_given_sigma_z', {}).get('rho'),
            'sigma_pilot_quality_partial_p_mass_given_sigma_z': sigma_pilot.get('partial_mass_given_sigma_z', {}).get('p'),
            'sigma_pilot_quality_raw_rho_sigma_vs_balmer': sigma_pilot.get('raw_sigma_vs_balmer', {}).get('rho'),
            'sigma_pilot_quality_raw_p_sigma_vs_balmer': sigma_pilot.get('raw_sigma_vs_balmer', {}).get('p'),
            'sigma_pilot_resolution_model': (
                'grating_fallback_pilot'
                if sigma_fit_summary.get('resolution_source_counts') else None
            ),
        }
    else:
        lines_of_evidence['L4_dynamical_mass'] = {
            'status': 'missing',
            'reason': 'step_117 output not available'
        }

    
    if s170:
        direct_package = s170.get('federated_direct_kinematic_package', {})
        direct_package_summary = direct_package.get('summary', {})
        direct_package_branches = direct_package.get('branches', {})
        l5_sigma_pilot = direct_package_branches.get('dja_sigma_balmer_pilot', {})
        l5_object_level = direct_package_branches.get('object_level_mass_anomaly_resolution', {})
        l5_same_regime = direct_package_branches.get('same_regime_literature_kinematics', {})
        l5_sigma_expansion = direct_package_branches.get('sigma_kinematic_expansion', {})
        lines_of_evidence['L5_kinematic_decisive'] = {
            'status': 'live',
            'assessment': s170.get('assessment'),
            'federated_assessment': s170.get('federated_assessment'),
            'rho_mstar_age': s170.get('results', {}).get('rho_mstar_age'),
            'p_mstar_age': s170.get('results', {}).get('p_mstar_age'),
            'rho_gamma_dyn_age': s170.get('results', {}).get('rho_gamma_dyn_age'),
            'p_gamma_dyn_age': s170.get('results', {}).get('p_gamma_dyn_age'),
            'rho_tep_age': s170.get('results', {}).get('rho_tep_age'),
            'p_tep_age': s170.get('results', {}).get('p_tep_age'),
            'partial_rho_gamma_dyn_age_given_z': s170.get('results', {}).get('partial_rho_gamma_dyn_age_given_z'),
            'p_partial_gamma_dyn_age_given_z': s170.get('results', {}).get('p_partial_gamma_dyn_age_given_z'),
            'partial_rho_mstar_age_given_z': s170.get('results', {}).get('partial_rho_mstar_age_given_z'),
            'p_partial_mstar_age_given_z': s170.get('results', {}).get('p_partial_mstar_age_given_z'),
            'partial_rho_gamma_dyn_age_given_mstar_z': s170.get('results', {}).get('partial_rho_gamma_dyn_age_given_mstar_z'),
            'p_partial_gamma_dyn_age_given_mstar_z': s170.get('results', {}).get('p_partial_gamma_dyn_age_given_mstar_z'),
            'partial_rho_mstar_age_given_gamma_dyn_z': s170.get('results', {}).get('partial_rho_mstar_age_given_gamma_dyn_z'),
            'p_partial_mstar_age_given_gamma_dyn_z': s170.get('results', {}).get('p_partial_mstar_age_given_gamma_dyn_z'),
            'delta_partial_rho_gamma_minus_mstar_given_z': s170.get('results', {}).get('delta_partial_rho_gamma_minus_mstar_given_z'),
            'delta_partial_rho_gamma_minus_mstar_given_competitor_z': s170.get('results', {}).get('delta_partial_rho_gamma_minus_mstar_given_competitor_z'),
            'steiger_gamma_dyn_vs_mstar_given_z': s170.get('results', {}).get('steiger_gamma_dyn_vs_mstar_given_z'),
            'steiger_gamma_dyn_vs_mstar_given_z_bootstrap': s170.get('results', {}).get('steiger_gamma_dyn_vs_mstar_given_z_bootstrap'),
            'published_uncertainty_coverage': s170.get('robustness', {}).get('published_uncertainty_coverage'),
            'uncertainty_monte_carlo': s170.get('robustness', {}).get('uncertainty_monte_carlo'),
            'direct_kinematic_package_branch_count': direct_package_summary.get('n_available_branches'),
            'direct_kinematic_package_counted_branch_count': direct_package_summary.get('n_counted_branches'),
            'direct_kinematic_package_contextual_branch_count': direct_package_summary.get('n_contextual_branches'),
            'direct_kinematic_package_supportive_branch_count': direct_package_summary.get('n_supportive_branches'),
            'direct_kinematic_package_primary_branch_label': direct_package_summary.get('primary_branch_label'),
            'direct_kinematic_package_available_branch_labels': direct_package_summary.get('available_branch_labels'),
            'direct_kinematic_package_counted_branch_labels': direct_package_summary.get('counted_branch_labels'),
            'direct_kinematic_package_contextual_branch_labels': direct_package_summary.get('contextual_branch_labels'),
            'direct_kinematic_package_supportive_branch_labels': direct_package_summary.get('supportive_branch_labels'),
            'direct_kinematic_package_all_available_branches_supportive': direct_package_summary.get('all_available_branches_supportive'),
            'direct_kinematic_package_caveat': direct_package_summary.get('caveat'),
            'auxiliary_direct_object_level_assessment': l5_object_level.get('assessment'),
            'auxiliary_direct_object_level_n_objects_total': l5_object_level.get('n_objects_total'),
            'auxiliary_direct_object_level_n_objects_exact_mdyn': l5_object_level.get('n_objects_exact_mdyn'),
            'auxiliary_direct_object_level_n_objects_upper_limit_only': l5_object_level.get('n_objects_upper_limit_only'),
            'auxiliary_direct_object_level_mean_observed_excess_dex': l5_object_level.get('mean_observed_excess_dex'),
            'auxiliary_direct_object_level_mean_corrected_excess_dex': l5_object_level.get('mean_corrected_excess_dex'),
            'auxiliary_direct_object_level_resolution_fraction_among_anomalous': l5_object_level.get('resolution_fraction_among_anomalous'),
            'auxiliary_sigma_pilot_assessment': l5_sigma_pilot.get('assessment'),
            'auxiliary_sigma_pilot_counts_toward_supportive_tally': l5_sigma_pilot.get('counts_toward_supportive_tally'),
            'auxiliary_sigma_pilot_n_quality_screened': l5_sigma_pilot.get('n_quality_screened'),
            'auxiliary_sigma_pilot_n_with_balmer': l5_sigma_pilot.get('n_with_balmer'),
            'auxiliary_sigma_pilot_partial_rho_sigma_given_mass_z': l5_sigma_pilot.get('results', {}).get('partial_rho_direct_outcome_given_mass_z'),
            'auxiliary_sigma_pilot_partial_p_sigma_given_mass_z': l5_sigma_pilot.get('results', {}).get('p_partial_direct_outcome_given_mass_z'),
            'auxiliary_sigma_pilot_partial_rho_mass_given_sigma_z': l5_sigma_pilot.get('results', {}).get('partial_rho_mass_outcome_given_direct_z'),
            'auxiliary_sigma_pilot_partial_p_mass_given_sigma_z': l5_sigma_pilot.get('results', {}).get('p_partial_mass_outcome_given_direct_z'),
            'auxiliary_sigma_pilot_uses_fallback_resolution_model': l5_sigma_pilot.get('uses_fallback_resolution_model'),
            'auxiliary_sigma_pilot_robustness_gate': l5_sigma_pilot.get('robustness_gate'),
            'contextual_same_regime_assessment': l5_same_regime.get('assessment'),
            'contextual_same_regime_n_objects_total': l5_same_regime.get('n_objects_total'),
            'contextual_same_regime_n_objects_exact_mdyn': l5_same_regime.get('n_objects_exact_mdyn'),
            'contextual_same_regime_n_objects_upper_limit_only': l5_same_regime.get('n_objects_upper_limit_only'),
            'contextual_same_regime_z_min': l5_same_regime.get('z_min'),
            'contextual_same_regime_z_max': l5_same_regime.get('z_max'),
            'contextual_same_regime_fraction_exact_mdyn_gt_mstar': l5_same_regime.get('fraction_exact_mdyn_gt_mstar'),
            'contextual_same_regime_source_breakdown': l5_same_regime.get('source_breakdown'),
            'sigma_expansion_assessment': l5_sigma_expansion.get('assessment'),
            'sigma_expansion_supportive': l5_sigma_expansion.get('supportive'),
            'sigma_expansion_counts_toward_supportive_tally': l5_sigma_expansion.get('counts_toward_supportive_tally'),
            'sigma_expansion_n_objects_total': l5_sigma_expansion.get('n_objects_total'),
            'sigma_expansion_n_sources': l5_sigma_expansion.get('n_sources'),
            'sigma_expansion_z_min': l5_sigma_expansion.get('z_min'),
            'sigma_expansion_z_max': l5_sigma_expansion.get('z_max'),
            'sigma_expansion_T3_partial_rho': l5_sigma_expansion.get('T3_partial_rho_gamma_mstar_given_sigma_z'),
            'sigma_expansion_T3_p': l5_sigma_expansion.get('T3_p_partial_gamma_mstar_given_sigma_z'),
            'sigma_expansion_T3_ci_95': l5_sigma_expansion.get('T3_ci_95'),
            'sigma_expansion_T5_highz_n': l5_sigma_expansion.get('T5_highz_n'),
            'sigma_expansion_T5_highz_partial_rho': l5_sigma_expansion.get('T5_highz_partial_rho_gamma'),
            'sigma_expansion_T5_highz_p': l5_sigma_expansion.get('T5_highz_p_gamma'),
            'sigma_expansion_source_breakdown': l5_sigma_expansion.get('source_paper_breakdown'),
        }
    else:
        lines_of_evidence['L5_kinematic_decisive'] = {
            'status': 'missing',
            'reason': 'step_170 output not available'
        }

    auxiliary_checks = {}
    if s143:
        t1 = s143.get('test_1_environment_density', {})
        t2 = s143.get('test_2_double_residual', {}).get('z_gt_8', {})
        t3 = s143.get('test_3_shuffled_mass', {}).get('z_gt_8', {})
        auxiliary_checks['mass_proxy_breaker'] = {
            'status': 'live',
            'environment_density_rho': t1.get('partial_density_dust_given_mass_z', {}).get('rho'),
            'environment_density_p': t1.get('partial_density_dust_given_mass_z', {}).get('p'),
            'double_residual_poly_rho': t2.get('polynomial_double_residual', {}).get('rho'),
            'double_residual_poly_p': t2.get('polynomial_double_residual', {}).get('p'),
            'double_residual_lowess_rho': t2.get('lowess_double_residual', {}).get('rho'),
            'double_residual_lowess_p': t2.get('lowess_double_residual', {}).get('p'),
            'partial_rank_rho': t2.get('partial_spearman_mass_z', {}).get('rho'),
            'partial_rank_p': t2.get('partial_spearman_mass_z', {}).get('p'),
            'shuffled_mass_z_score': t3.get('z_score'),
            'shuffled_mass_p': t3.get('p_empirical'),
            'degeneracy_state': s143.get('overall_verdict', {}).get('degeneracy_state'),
            'limitations': s143.get('overall_verdict', {}).get('limitations'),
            'overall_conclusion': s143.get('overall_verdict', {}).get('conclusion'),
        }

    if s148:
        headline = s148.get('headline', {})
        auxiliary_checks['mass_independent_proxy_suite'] = {
            'status': 'live',
            'z_gt_8_tests_passing': headline.get('z_gt_8_tests_passing'),
            'z_gt_8_tests_total': headline.get('z_gt_8_tests_total'),
            'z_gt_8_tests_with_dust_association': headline.get('z_gt_8_tests_with_dust_association'),
            'z_gt_8_tests_with_gamma_link': headline.get('z_gt_8_tests_with_gamma_link'),
            'conclusion': headline.get('conclusion'),
            'honest_caveat': headline.get('honest_caveat'),
        }

    if s144:
        cross_pairs = [
            v for v in s144.get('test_2_cross_survey', {}).values()
            if isinstance(v, dict) and 'mass_z_poly' in v and 'mass_z_poly+gamma_t' in v
        ]
        cross_rho_lifts = [
            v.get('mass_z_poly+gamma_t', {}).get('rho', 0.0) - v.get('mass_z_poly', {}).get('rho', 0.0)
            for v in cross_pairs
        ]
        z8_info = s144.get('test_3_information_theory', {}).get('z_gt_8', {})
        perm = s144.get('test_1_within_survey', {}).get('permutation_importance', {})
        within_lift = s144.get('test_1_within_survey', {}).get('gamma_t_lift', {})
        auxiliary_checks['adversarial_information'] = {
            'status': 'live',
            'within_survey_delta_r2': within_lift.get('delta_R2'),
            'within_survey_assessment': within_lift.get('assessment'),
            'permutation_mean_r2_drop': perm.get('mean_R2_drop'),
            'permutation_std_r2_drop': perm.get('std_R2_drop'),
            'cross_survey_positive_pairs': int(sum(lift > 0 for lift in cross_rho_lifts)),
            'cross_survey_pairs_total': int(len(cross_rho_lifts)),
            'cross_survey_mean_delta_rho': float(np.mean(cross_rho_lifts)) if cross_rho_lifts else None,
            'z_gt_8_cmi': z8_info.get('I_gamma_dust_given_mass_z'),
            'z_gt_8_cmi_z_score': z8_info.get('z_score'),
            'z_gt_8_information_asymmetry': z8_info.get('information_asymmetry'),
        }

    if s162:
        conc = s162.get('concordance', {})
        external_ref = conc.get('external_cepheid_reference', conc.get('cepheid_reference', {}))
        auxiliary_checks['kappa_gal_concordance'] = {
            'status': 'live',
            'kappa_gal_weighted_mean': conc.get('kappa_gal_weighted_mean'),
            'kappa_gal_weighted_sigma': conc.get('kappa_gal_weighted_sigma'),
            'jwst_recovered_kappa_gal': conc.get('jwst_recovered_kappa_gal', conc.get('kappa_gal_weighted_mean')),
            'jwst_recovered_sigma': conc.get('jwst_recovered_sigma', conc.get('kappa_gal_weighted_sigma')),
            'p_concordance': conc.get('p_concordance'),
            'max_pairwise_tension_sigma': conc.get('max_pairwise_tension_sigma'),
            'cepheid_tension_sigma': conc.get('cepheid_tension_sigma'),
            'external_cepheid_tension_sigma': conc.get('external_cepheid_tension_sigma', conc.get('cepheid_tension_sigma')),
            'external_cepheid_kappa_gal': external_ref.get('kappa_gal'),
            'external_cepheid_sigma': external_ref.get('sigma'),
            'verdict': conc.get('verdict'),
        }

    if s137:
        summary = s137.get('summary', {})
        auxiliary_checks['cross_survey_generalization'] = {
            'status': 'live',
            'n_tests': summary.get('n_tests'),
            'mean_teff_rho_advantage': summary.get('mean_teff_rho_advantage'),
            'teff_rho_wins': summary.get('teff_rho_wins'),
            'mean_poly_r2_drop': summary.get('mean_poly_r2_drop'),
            'interpretation': summary.get('interpretation'),
        }

    if s138:
        full_sample = s138.get('full_sample', {})
        z_gt_8 = s138.get('z_gt_8', {})
        summary = s138.get('summary', {})
        auxiliary_checks['environmental_screening'] = {
            'status': 'live',
            'full_sample_delta_rho': full_sample.get('delta_rho'),
            'full_sample_z': full_sample.get('Z_fisher_independent'),
            'full_sample_p': full_sample.get('p_fisher_independent'),
            'z_gt_8_delta_rho': z_gt_8.get('delta_rho'),
            'z_gt_8_z': z_gt_8.get('Z_fisher_independent'),
            'z_gt_8_p': z_gt_8.get('p_fisher_independent'),
            'z_gt_8_tep_confirmed': z_gt_8.get('tep_confirmed'),
            'n_bins_confirmed': summary.get('n_tep_confirmed'),
            'n_bins_tested': summary.get('n_tests'),
            'overall_conclusion': summary.get('conclusion'),
        }

    if s141 and s141.get('analysis_class') == 'real_data_model_comparison':
        key = s141.get('key_comparisons', {})
        auxiliary_checks['functional_form_discrimination'] = {
            'status': 'live',
            'step_teff_vs_step_mass_delta_aic': key.get('step_teff_vs_step_mass_delta_aic'),
            'step_teff_vs_step_mass_delta_bic': key.get('step_teff_vs_step_mass_delta_bic'),
            'step_teff_wins_aic': key.get('step_teff_wins_aic'),
            'step_teff_wins_bic': key.get('step_teff_wins_bic'),
            'best_model_aic': key.get('best_model_aic'),
            'best_model_bic': key.get('best_model_bic'),
        }

    if s159:
        beta07 = s159.get('suppression_by_beta', {}).get('beta_0.7', {})
        uncover_corr = s159.get('key_result_corrections', {}).get('UNCOVER_dust_z4_10', {})
        cosmos_corr = s159.get('key_result_corrections', {}).get('COSMOS2025_dust_z9_13', {})
        auxiliary_checks['mass_measurement_bias'] = {
            'status': 'live',
            'beta_ml': s159.get('beta_ml'),
            'beta_empirical_from_l4': s159.get('beta_empirical_from_L4'),
            'rho_true_unbiased': s159.get('rho_true_unbiased'),
            'rho_partial_obs_beta07': beta07.get('rho_partial_obs'),
            'suppression_fraction_beta07': beta07.get('suppression_fraction'),
            'correction_factor_beta07': s159.get('correction_factor_beta07'),
            'uncover_corrected_rho': uncover_corr.get('rho_corrected_estimate_bounded'),
            'cosmos_z9_13_corrected_rho': cosmos_corr.get('rho_corrected_estimate_bounded'),
        }

    if s155:
        headline = s155.get('headline', {})
        auxiliary_checks['jades_dr5_morphology'] = {
            'status': 'live',
            'n_matched': s155.get('n_matched'),
            'n_with_muv': s155.get('n_with_muv'),
            'supportive_partial_count': headline.get('n_structural_proxies_supportive_after_mass_z_control'),
            'supportive_partial_keys': headline.get('supportive_partial_keys'),
            'strongest_partial_key': headline.get('strongest_partial_key'),
            'conclusion': headline.get('conclusion'),
        }

    if s157:
        ext = s157.get('external_replication_summary', {})
        auxiliary_checks['cosmos2025_blank_field_l3'] = {
            'status': 'live',
            'beta_debias_used': s157.get('beta_debias_used'),
            'beta_debias_source': s157.get('beta_debias_source'),
            'n_supportive_bins_after_debias': ext.get('n_supportive_bins_after_debias'),
            'n_highz_bins_tested': ext.get('n_highz_bins_tested'),
            'primary_matched_bin': ext.get('primary_matched_bin'),
            'ultrahighz_sensitivity_bin': ext.get('ultrahighz_sensitivity_bin'),
            'assessment': ext.get('assessment'),
        }

    if s164:
        auxiliary_checks['uncover_z9_null_audit'] = {
            'status': 'live',
            'assessment': s164.get('diagnosis', {}).get('assessment'),
            'flags': s164.get('diagnosis', {}).get('flags'),
        }

    if s176:
        joint_bf = s176.get('joint_bayes_factors', {})
        residual_bf = s176.get('residual_space_bayes_factors', {})
        auxiliary_checks['nested_bayesian_model_comparison'] = {
            'status': 'live',
            'sample_size': s176.get('sample_size'),
            'n_observables': s176.get('n_observables'),
            'observables': s176.get('observables'),
            'joint_key_finding': s176.get('key_finding', {}).get('statement'),
            'joint_mean_ln_BF': s176.get('joint_summary', {}).get('mean_ln_BF'),
            'joint_mean_log10_BF': s176.get('joint_summary', {}).get('mean_log10_BF'),
            'joint_hardest_alternative': s176.get('joint_summary', {}).get('hardest_alternative'),
            'joint_hardest_ln_BF': s176.get('joint_summary', {}).get('hardest_ln_BF'),
            'joint_n_strong_for_TEP': s176.get('joint_summary', {}).get('n_strong_for_TEP'),
            'joint_n_favour_alternative': s176.get('joint_summary', {}).get('n_favour_alternative'),
            'joint_ln_BF_standard_physics': joint_bf.get('Standard_Physics', {}).get('ln_BF_TEP_vs_alt'),
            'joint_ln_BF_bursty_sf': joint_bf.get('Bursty_SF', {}).get('ln_BF_TEP_vs_alt'),
            'joint_ln_BF_varying_imf': joint_bf.get('Varying_IMF', {}).get('ln_BF_TEP_vs_alt'),
            'joint_ln_BF_agn_feedback': joint_bf.get('AGN_Feedback', {}).get('ln_BF_TEP_vs_alt'),
            'residual_key_finding': s176.get('residual_space_key_finding', {}).get('statement'),
            'residual_mean_ln_BF': s176.get('residual_space_summary', {}).get('mean_ln_BF'),
            'residual_mean_log10_BF': s176.get('residual_space_summary', {}).get('mean_log10_BF'),
            'residual_n_decisive_for_TEP': s176.get('residual_space_summary', {}).get('n_decisive_for_TEP'),
            'residual_n_favour_alternative': s176.get('residual_space_summary', {}).get('n_favour_alternative'),
            'residual_ln_BF_null': residual_bf.get('Residual_Null', {}).get('ln_BF_TEP_vs_alt'),
            'residual_ln_BF_constrained_agn': residual_bf.get('Constrained_AGN', {}).get('ln_BF_TEP_vs_alt'),
        }

    derived_consequences = {}
    if s146:
        labbe = s146.get('labbe_resolution', {})
        derived_consequences['stellar_mass_function_resolution'] = {
            'status': 'live',
            'mean_resolution': s146.get('mean_resolution'),
            'labbe_n_impossible': labbe.get('n_impossible'),
            'labbe_n_resolved_tep': labbe.get('n_resolved_tep'),
            'labbe_resolution_frac': labbe.get('resolution_frac'),
            'conclusion': s146.get('conclusion'),
        }

    if s147:
        summary = s147.get('summary', {})
        derived_consequences['cosmic_sfrd_correction'] = {
            'status': 'live',
            'z_gt_8_mean_reduction_pct': summary.get('z_gt_8_mean_reduction_pct'),
            'z_gt_8_mean_excess_obs': summary.get('z_gt_8_mean_excess_obs'),
            'z_gt_8_mean_excess_corr': summary.get('z_gt_8_mean_excess_corr'),
            'overall_mean_reduction_pct': summary.get('overall_mean_reduction_pct'),
        }

    reference_only_branches = {}
    for branch_name, data in {
        'dja_nirspec_merged': s150,
        'dja_balmer_decrement': s158,
    }.items():
        if data and data.get('status') == 'SUCCESS_REFERENCE_ONLY':
            reference_only_branches[branch_name] = {
                'status': data.get('status'),
                'reproducible_available': data.get('reproducible_dja_available'),
                'manuscript_table_reproduced': data.get('manuscript_table_reproduced'),
            }

    primary_line_keys = [
        'L1_dust_replication',
        'L3_ssfr_inversion',
    ]
    live_primary_lines = sum(1 for key in primary_line_keys if lines_of_evidence.get(key, {}).get('status') == 'live')
    skipped_primary_lines = sum(1 for key in primary_line_keys if lines_of_evidence.get(key, {}).get('status') == 'skipped')
    missing_primary_lines = sum(1 for key in primary_line_keys if lines_of_evidence.get(key, {}).get('status') == 'missing')

    results['final_synthesis']['lines_of_evidence'] = lines_of_evidence
    results['final_synthesis']['auxiliary_checks'] = auxiliary_checks
    results['final_synthesis']['availability'] = {
        'live_primary_lines': live_primary_lines,
        'skipped_primary_lines': skipped_primary_lines,
        'missing_primary_lines': missing_primary_lines,
    }
    if derived_consequences:
        results['final_synthesis']['derived_consequences'] = derived_consequences
    results['final_synthesis']['claim_hierarchy'] = {
        'headline_primary_result': 'L1 dust-Gamma_t replication across three independent JWST surveys remains the primary large-sample JWST result',
        'headline_direct_test': 'L5 JWST-SUSPENSE direct kinematic comparison materially narrows the photometric mass-circularity objection',
        'primary_live_lines': ['L1_dust_replication', 'L3_ssfr_inversion'],
        'direct_kinematic_tests': [
            key for key in ['L5_kinematic_decisive']
            if lines_of_evidence.get(key, {}).get('status') == 'live'
        ],
        'ancillary_indications': ['L2_core_screening'],
        'derived_regime_comparisons': [
            key for key in ['L4_dynamical_mass']
            if key in lines_of_evidence
        ],
        'robustness_checks': [
            key for key in [
                'mass_proxy_breaker',
                'mass_independent_proxy_suite',
                'adversarial_information',
                'mass_measurement_bias',
                'cross_survey_generalization',
                'functional_form_discrimination',
                'kappa_gal_concordance',
                'jades_dr5_morphology',
                'cosmos2025_blank_field_l3',
                'uncover_z9_null_audit',
            ]
            if key in auxiliary_checks
        ],
        'supplementary_or_mixed_branches': [
            key for key in ['environmental_screening']
            if key in auxiliary_checks
            or key in lines_of_evidence
        ],
        'reference_only_branches': reference_only_branches,
    }

    print_status("\n" + "=" * 70, "INFO")
    print_status("LIVE SYNTHESIS", "INFO")
    print_status("=" * 70, "INFO")
    print_status(f"  Live primary lines: {live_primary_lines}", "INFO")
    print_status(f"  Skipped primary lines: {skipped_primary_lines}", "INFO")
    print_status(f"  Missing primary lines: {missing_primary_lines}", "INFO")

    if lines_of_evidence['L1_dust_replication']['status'] == 'live':
        print_status(
            f"  L1: Fisher z = {lines_of_evidence['L1_dust_replication']['headline_z']:.2f}σ across "
            f"{lines_of_evidence['L1_dust_replication']['n_datasets']} datasets",
            "INFO",
        )
    if lines_of_evidence['L2_core_screening']['status'] != 'live':
        print_status(f"  L2: {lines_of_evidence['L2_core_screening']['reason']}", "INFO")
    if lines_of_evidence['L3_ssfr_inversion']['status'] == 'live':
        print_status(
            f"  L3: Δρ = {lines_of_evidence['L3_ssfr_inversion']['delta_rho']:.3f}, "
            f"high-z ρ = {lines_of_evidence['L3_ssfr_inversion']['high_z_rho']:.3f}",
            "INFO",
        )
    if lines_of_evidence['L4_dynamical_mass']['status'] in {'derived_from_real_data', 'direct_object_level_kinematics'}:
        print_status(
            f"  L4 ({lines_of_evidence['L4_dynamical_mass']['status']}): published excess {lines_of_evidence['L4_dynamical_mass']['published_excess_dex']:.3f} dex, "
            f"TEP reduction {lines_of_evidence['L4_dynamical_mass']['tep_reduction_dex']:.3f} dex",
            "INFO",
        )
    
    if 'L5_kinematic_decisive' in lines_of_evidence and lines_of_evidence['L5_kinematic_decisive']['status'] == 'live':
        l5 = lines_of_evidence['L5_kinematic_decisive']
        steiger = l5.get('steiger_gamma_dyn_vs_mstar_given_z') or {}
        print_status(
            f"  L5 (live kinematics): {l5['assessment']}; "
            f"ρ(Age, M_star|z)={l5['partial_rho_mstar_age_given_z']:.3f} (p={l5['p_partial_mstar_age_given_z']:.3e}) vs "
            f"ρ(Age, Γ_dyn|z)={l5['partial_rho_gamma_dyn_age_given_z']:.3f} (p={l5['p_partial_gamma_dyn_age_given_z']:.3e}); "
            f"ρ(Age, Γ_dyn|M_star,z)={l5['partial_rho_gamma_dyn_age_given_mstar_z']:.3f} vs "
            f"ρ(Age, M_star|Γ_dyn,z)={l5['partial_rho_mstar_age_given_gamma_dyn_z']:.3f}; "
            f"Steiger Z={(steiger.get('z_stat_gamma_dyn_better_than_mstar') if steiger.get('z_stat_gamma_dyn_better_than_mstar') is not None else float('nan')):.3f}",
            "INFO"
        )

    if 'kappa_gal_concordance' in auxiliary_checks:
        print_status(
            f"  κ_gal concordance: JWST {auxiliary_checks['kappa_gal_concordance']['jwst_recovered_kappa_gal']:.3e} ± "
            f"{auxiliary_checks['kappa_gal_concordance']['jwst_recovered_sigma']:.3e}; "
            f"external Cepheid prior {auxiliary_checks['kappa_gal_concordance']['external_cepheid_kappa_gal']:.3e} ± "
            f"{auxiliary_checks['kappa_gal_concordance']['external_cepheid_sigma']:.3e} mag; "
            f"agreement = {auxiliary_checks['kappa_gal_concordance']['external_cepheid_tension_sigma']:.2f}σ "
            f"(p = {auxiliary_checks['kappa_gal_concordance']['p_concordance']:.3f})",
            "INFO",
        )
    if 'cross_survey_generalization' in auxiliary_checks:
        print_status(
            f"  Cross-survey generalization: mean Δρ = {auxiliary_checks['cross_survey_generalization']['mean_teff_rho_advantage']:.3f}; "
            f"mean polynomial R² drop = {auxiliary_checks['cross_survey_generalization']['mean_poly_r2_drop']:.3f}",
            "INFO",
        )
    if 'adversarial_information' in auxiliary_checks:
        print_status(
            f"  Adversarial information: z>8 CMI z = {auxiliary_checks['adversarial_information']['z_gt_8_cmi_z_score']:.2f}; "
            f"mean cross-survey Δρ = {auxiliary_checks['adversarial_information']['cross_survey_mean_delta_rho']:.3f}",
            "INFO",
        )
    if 'functional_form_discrimination' in auxiliary_checks:
        print_status(
            f"  Functional-form discrimination: step ΔAIC = {auxiliary_checks['functional_form_discrimination']['step_teff_vs_step_mass_delta_aic']:.2f}",
            "INFO",
        )
    if 'mass_measurement_bias' in auxiliary_checks:
        print_status(
            f"  Mass-bias attenuation: ρ_true ≈ {auxiliary_checks['mass_measurement_bias']['rho_true_unbiased']:.3f}; "
            f"β=0.7 observed ρ ≈ {auxiliary_checks['mass_measurement_bias']['rho_partial_obs_beta07']:.3f}",
            "INFO",
        )
    if 'jades_dr5_morphology' in auxiliary_checks:
        print_status(
            f"  JADES DR5 morphology: supportive partials = {auxiliary_checks['jades_dr5_morphology']['supportive_partial_count']}; "
            f"strongest = {auxiliary_checks['jades_dr5_morphology']['strongest_partial_key']}",
            "INFO",
        )
    if 'cosmos2025_blank_field_l3' in auxiliary_checks:
        print_status(
            f"  COSMOS2025 blank-field L3: {auxiliary_checks['cosmos2025_blank_field_l3']['assessment']}",
            "INFO",
        )
    if 'uncover_z9_null_audit' in auxiliary_checks:
        print_status(
            f"  UNCOVER z=9-12 audit: {auxiliary_checks['uncover_z9_null_audit']['assessment']}",
            "INFO",
        )
    if 'environmental_screening' in auxiliary_checks:
        print_status(
            f"  Environmental screening: full-sample Δρ = {auxiliary_checks['environmental_screening']['full_sample_delta_rho']:.3f}; "
            f"z>8 Δρ = {auxiliary_checks['environmental_screening']['z_gt_8_delta_rho']:.3f} "
            f"(p = {auxiliary_checks['environmental_screening']['z_gt_8_p']:.3f})",
            "INFO",
        )

    results['final_synthesis']['verdict'] = {
        'status': 'live_summary',
        'live_primary_lines': live_primary_lines,
        'skipped_primary_lines': skipped_primary_lines,
        'missing_primary_lines': missing_primary_lines,
        'headline_primary_result': 'L1 dust-Gamma_t replication across three independent JWST surveys',
        'headline_direct_test': 'L5 JWST-SUSPENSE direct kinematic comparison materially narrows the photometric mass-circularity objection',
        'supporting_primary_lines': ['L3_ssfr_inversion'],
        'ancillary_indication': 'L2_core_screening',
        'derived_regime_comparison': 'L4_dynamical_mass' if 'L4_dynamical_mass' in lines_of_evidence else None,
        'direct_kinematic_test': (
            'L5_kinematic_decisive'
            if lines_of_evidence.get('L5_kinematic_decisive', {}).get('status') == 'live'
            else None
        ),
        'supplementary_mixed_branches': [
            key for key in ['environmental_screening']
            if key in lines_of_evidence or key in auxiliary_checks
        ],
        'reference_only_branch_count': len(reference_only_branches),
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
