#!/usr/bin/env python3
"""
Pipeline cleanup: map old step names → canonical 3-digit names from run_all_steps.py.
Run from project root: python scripts/steps/_reorganize_steps.py
"""
import shutil
from pathlib import Path

STEPS_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = STEPS_DIR / "_archive"
ARCHIVE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL 163 steps (as expected by run_all_steps.py)
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL = [
    "step_001_uncover_load.py",
    "step_002_tep_model.py",
    "step_003_first_principles.py",
    "step_004_thread1_z7_inversion.py",
    "step_005_thread2_4_partial_correlations.py",
    "step_006_thread5_z8_dust.py",
    "step_007_thread6_7_coherence.py",
    "step_008_summary.py",
    "step_009_holographic_synthesis.py",
    "step_010_sn_ia_mass_step.py",
    "step_011_mw_gc_gradient.py",
    "step_012_sn_ia_extended.py",
    "step_013_trgb_cepheid.py",
    "step_014_jwst_uv_slope.py",
    "step_015_jwst_impossible_galaxies.py",
    "step_016_robustness_tests.py",
    "step_017_ml_ratio.py",
    "step_018_assembly_time.py",
    "step_019_chi2_analysis.py",
    "step_020_parameter_validation.py",
    "step_021_scatter_reduction.py",
    "step_022_extreme_population.py",
    "step_023_self_consistency.py",
    "step_024_cosmological_implications.py",
    "step_025_cross_sample_validation.py",
    "step_026_prediction_alignment.py",
    "step_027_multi_angle_validation.py",
    "step_028_chi2_diagnostic.py",
    "step_029_independent_validation.py",
    "step_030_z8_dust_prediction.py",
    "step_031_ceers_download.py",
    "step_032_ceers_replication.py",
    "step_033_cosmosweb_download.py",
    "step_034_cosmosweb_replication.py",
    "step_035_spectroscopic_validation.py",
    "step_036_spectroscopic_refinement.py",
    "step_037_resolved_gradients.py",
    "step_038_sensitivity_analysis.py",
    "step_039_overmassive_bh.py",
    "step_040_mass_sensitivity.py",
    "step_041_forward_modeling.py",
    "step_042_lrd_population.py",
    "step_043_blue_monsters.py",
    "step_044_metallicity_age_decoupling.py",
    "step_045_sfr_age_consistency.py",
    "step_046_multi_diagnostic.py",
    "step_047_independent_tests.py",
    "step_048_advanced_diagnostics.py",
    "step_049_prediction_tests.py",
    "step_050_critical_evidence.py",
    "step_051_temporal_coherence.py",
    "step_052_mass_discrepancy.py",
    "step_053_tep_discriminant.py",
    "step_054_unique_predictions.py",
    "step_055_cross_domain.py",
    "step_056_out_of_box.py",
    "step_057_adversarial_tests.py",
    "step_058_creative_tests.py",
    "step_059_ultimate_missing_piece.py",
    "step_060_the_functional_form_discrimination.py",
    "step_061_extreme_cases.py",
    "step_062_predictive_power.py",
    "step_063_correlation_deep_dive.py",
    "step_064_web_inspired_tests.py",
    "step_065_literature_tests.py",
    "step_066_final_literature_tests.py",
    "step_067_cutting_edge_tests.py",
    "step_068_observational_signatures.py",
    "step_069_compactness_verification.py",
    "step_070_binary_pulsar_constraints.py",
    "step_071_bbn_analysis.py",
    "step_072_sign_paradox_check.py",
    "step_073_growth_factor.py",
    "step_074_z67_tep_prediction.py",
    "step_075_bayesian_model_comparison.py",
    "step_076_ml_scaling_justification.py",
    "step_077_bootstrap_validation.py",
    "step_078_independent_age_validation.py",
    "step_079_ml_cross_validation.py",
    "step_080_balmer_simulation.py",
    "step_081_survey_cross_correlation.py",
    "step_082_evidence_strengthening.py",
    "step_083_falsification_battery.py",
    "step_084_effect_size_meta.py",
    "step_085_time_lens_map.py",
    "step_086_posterior_predictive.py",
    "step_087_emission_line_tests.py",
    "step_088_final_comprehensive.py",
    "step_089_teff_threshold_scan.py",
    "step_090_permutation_battery.py",
    "step_091_random_effects_meta_loo.py",
    "step_092_mass_matched_confirmation.py",
    "step_093_teff_threshold_holdout.py",
    "step_094_environmental_screening_enhanced.py",
    "step_095_lrd_core_halo_mass.py",
    "step_096_balmer_target_list.py",
    "step_097_money_plot.py",
    "step_098_alternative_model_comparison.py",
    "step_099_alpha_evolution_test.py",
    "step_100_multi_domain_model_comparison.py",
    "step_101_alpha_recovery_multi_observable.py",
    "step_102_literature_anomaly_resolution.py",
    "step_103_zphot_error_propagation.py",
    "step_104_cosmic_variance.py",
    "step_105_morphology_tep.py",
    "step_106_emission_line_diagnostic.py",
    "step_107_sn_rate_prediction.py",
    "step_108_jwst_cycle4_targets.py",
    "step_109_lcdm_tension_quantification.py",
    "step_110_gw_timing_prediction.py",
    "step_111_power_analysis.py",
    "step_112_scalar_tensor_constraints.py",
    "step_113_agn_feedback_discriminant.py",
    "step_114_imf_constraint.py",
    "step_115_euclid_roman_predictions.py",
    "step_116_literature_spectroscopic_ages.py",
    "step_117_dynamical_mass_comparison.py",
    "step_118_neff_corrected_significance.py",
    "step_119_blind_validation.py",
    "step_120_screening_transition_profile.py",
    "step_121_extended_modified_gravity_comparison.py",
    "step_122_causality_verification.py",
    "step_123_alpha0_error_budget.py",
    "step_124_timespace_coupling.py",
    "step_125_multitracer_consistency.py",
    "step_126_screening_scale.py",
    "step_127_imf_discrimination.py",
    "step_128_selection_mc.py",
    "step_129_dust_models.py",
    "step_130_cross_survey_systematics.py",
    "step_131_agn_power.py",
    "step_132_lrd_validation.py",
    "step_133_hubble_connection.py",
    "step_134_prediction_errors.py",
    "step_135_scale_dependent_growth.py",
    "step_136_functional_form_test.py",
    "step_137_cross_survey_generalization.py",
    "step_138_environmental_screening_steiger.py",
    "step_139_colour_gradient_steiger.py",
    "step_140_evidence_tier_summary.py",
    "step_141_nonlinear_aic.py",
    "step_142_lrd_mbh_mstar_prediction.py",
    "step_143_mass_proxy_breaker.py",
    "step_144_adversarial_ml_attack.py",
    "step_145_phase_boundary_activation.py",
    "step_146_stellar_mass_function_resolution.py",
    "step_147_cosmic_sfrd_correction.py",
    "step_148_mass_independent_proxy.py",
    "step_149_jades_dr4_ingestion.py",
    "step_150_dja_nirspec_merged.py",
    "step_151_dja_ceers_crossmatch.py",
    "step_152_uncover_dr4_full_sps.py",
    "step_153_cosmos2025_sed_analysis.py",
    "step_154_jades_emission_lines.py",
    "step_155_jades_dr5_morphology.py",
    "step_156_dja_gds_morphology.py",
    "step_157_cosmos2025_ssfr_inversion.py",
    "step_158_dja_balmer_decrement.py",
    "step_159_mass_measurement_bias.py",
    "step_160_manuscript_consistency_check.py",
    "step_161_multi_dataset_l1_combination.py",
    "step_162_l1_l3_independence.py",
    "step_163_external_dataset_registry.py",
]

CANONICAL_SET = set(CANONICAL)

# ─────────────────────────────────────────────────────────────────────────────
# RENAME MAP: old_name → canonical_name
# ─────────────────────────────────────────────────────────────────────────────
RENAMES = {
    # Phase I duplicates (already have 3-digit versions → archive old 2-digit)
    "step_00_first_principles.py":          None,   # dup of step_003
    "step_01_uncover_load.py":              None,   # dup of step_001
    "step_02_tep_model.py":                 None,   # dup of step_002
    "step_03_thread1_z7_inversion.py":      None,   # dup of step_004
    "step_04_thread2_4_partial_correlations.py": None,  # dup of step_005
    "step_05_thread5_z8_dust.py":           None,   # dup of step_006
    # Phase I → canonical
    "step_06_thread6_7_coherence.py":       "step_007_thread6_7_coherence.py",
    "step_07_summary.py":                   "step_008_summary.py",
    "step_08_holographic_synthesis.py":     "step_009_holographic_synthesis.py",
    "step_09_sn_ia_mass_step.py":           "step_010_sn_ia_mass_step.py",
    "step_10_mw_gc_gradient.py":            "step_011_mw_gc_gradient.py",
    "step_10_thread6_7_coherence.py":       None,   # superseded dup
    "step_11_sn_ia_extended.py":            "step_012_sn_ia_extended.py",
    "step_11_summary.py":                   None,   # superseded dup
    "step_12_trgb_cepheid.py":              "step_013_trgb_cepheid.py",
    "step_12_holographic_synthesis.py":     None,   # superseded dup
    "step_12_z8_dust_theory.py":            None,   # old exploratory
    "step_13_jwst_uv_slope.py":             "step_014_jwst_uv_slope.py",
    "step_13_extended_analysis.py":         None,   # old exploratory
    "step_13_sn_ia_mass_step.py":           None,   # superseded dup
    "step_14_jwst_impossible_galaxies.py":  "step_015_jwst_impossible_galaxies.py",
    "step_14_first_principles_tests.py":    None,   # old exploratory
    "step_14_mw_gc_gradient.py":            None,   # superseded dup
    "step_15_robustness_tests.py":          "step_016_robustness_tests.py",
    "step_15_harmonic_resolution.py":       None,   # old exploratory
    "step_15_sn_ia_extended.py":            None,   # superseded dup
    "step_16_ml_ratio.py":                  "step_017_ml_ratio.py",
    "step_16_chi2_diagnostic.py":           None,   # superseded dup
    "step_16_trgb_cepheid.py":              None,   # superseded dup
    "step_17_assembly_time.py":             "step_018_assembly_time.py",
    "step_17_extended_tests.py":            None,   # old exploratory
    "step_17_jwst_uv_slope.py":             None,   # superseded dup
    "step_18_chi2_analysis.py":             "step_019_chi2_analysis.py",
    "step_18_inlay_synthesis.py":           None,   # old exploratory
    "step_18_jwst_impossible_galaxies.py":  None,   # superseded dup
    "step_19_combined_significance.py":     None,   # superseded (Brown's method combined)
    "step_19_golden_veins.py":              None,   # old exploratory
    "step_19_mortar_binding.py":            None,   # old exploratory
    # Gen1 (very old single-digit names)
    "step_1_data_ingestion.py":             None,
    "step_2_mass_age_analysis.py":          None,
    "step_3_complementary_evidence.py":     None,
    "step_3_gc_age_analysis.py":            None,
    "step_3_process_real_data.py":          None,
    "step_4_acorn_analysis.py":             None,
    "step_4_labbe_analysis.py":             None,
    "step_5_uncover_load.py":               None,
    "step_6_tep_model.py":                  None,
    "step_7_thread1_z7_inversion.py":       None,
    "step_8_thread2_4_partial_correlations.py": None,
    "step_9_thread5_z8_dust.py":            None,
    # Phase IV onwards
    "step_20_parameter_validation.py":      "step_020_parameter_validation.py",
    "step_20_rising_ocean.py":              None,   # old exploratory
    "step_20_tendrils.py":                  None,   # old exploratory
    "step_21_scatter_reduction.py":         "step_021_scatter_reduction.py",
    "step_21_molten_bronze.py":             None,
    "step_21_river_healing.py":             None,
    "step_22_extreme_population.py":        "step_022_extreme_population.py",
    "step_22_liquid_light.py":              None,
    "step_22_root_system.py":               None,
    "step_23_self_consistency.py":          "step_023_self_consistency.py",
    "step_23_morning_dew.py":               None,
    "step_23_transparent_sphere.py":        None,
    "step_24_cosmological_implications.py": "step_024_cosmological_implications.py",
    "step_24_birth_of_star.py":             None,
    "step_24_ignition.py":                  None,
    "step_25_cross_sample_validation.py":   "step_025_cross_sample_validation.py",
    "step_25_first_breath.py":              None,
    "step_25_golden_spike.py":              None,
    "step_26_prediction_alignment.py":      "step_026_prediction_alignment.py",
    "step_26_golden_capstone.py":           None,
    "step_26_great_thaw.py":               None,
    "step_27_multi_angle_validation.py":    "step_027_multi_angle_validation.py",
    "step_27_holy_grail.py":               None,
    "step_27_sunrise.py":                   None,
    "step_28_chi2_diagnostic.py":           "step_028_chi2_diagnostic.py",
    "step_28_cosmic_eye.py":                None,
    "step_28_final_piece.py":               None,
    "step_29_final_synthesis.py":           None,   # exploratory
    "step_29_totality.py":                  None,
    "step_30_diamond.py":                   None,   # old exploratory
    "step_30_model_comparison.py":          None,   # superseded by step_030
    "step_31_independent_validation.py":    "step_029_independent_validation.py",
    "step_32_z8_dust_prediction.py":        None,   # superseded by step_030
    "step_33_ceers_replication.py":         "step_032_ceers_replication.py",
    "step_34_ceers_download.py":            "step_031_ceers_download.py",
    "step_35_cosmosweb_download.py":        "step_033_cosmosweb_download.py",
    "step_36_cosmosweb_replication.py":     "step_034_cosmosweb_replication.py",
    "step_37_spectroscopic_validation.py":  None,   # superseded by step_035
    "step_37b_combine_spectroscopic_data.py": None, # superseded
    "step_37c_spectroscopic_refinement.py": "step_036_spectroscopic_refinement.py",
    "step_38_resolved_gradients.py":        None,   # superseded by step_037
    "step_38_z67_dip_forensics.py":         None,   # exploratory
    "step_39_environment.py":              None,
    "step_39_environment_screening.py":    None,    # superseded by step_094
    "step_40_sensitivity_analysis.py":     "step_038_sensitivity_analysis.py",
    "step_41_overmassive_bh.py":           "step_039_overmassive_bh.py",
    "step_42_mass_sensitivity.py":         "step_040_mass_sensitivity.py",
    "step_43_selection_bias.py":           None,    # exploratory
    "step_44_forward_modeling.py":         "step_041_forward_modeling.py",
    "step_45_posterior_predictive.py":     "step_086_posterior_predictive.py",
    "step_46_lrd_population.py":           "step_042_lrd_population.py",
    "step_46_lrd_population_analysis.py":  None,    # superseded dup
    "step_47_blue_monsters_tep.py":        "step_043_blue_monsters.py",
    "step_47_blue_monsters.py":            None,    # superseded dup
    "step_48_redshift_gradient_test.py":   None,    # exploratory
    "step_49_metallicity_age_decoupling.py": "step_044_metallicity_age_decoupling.py",
    "step_49_selection_effects_investigation.py": None,  # exploratory
    "step_50_sfr_age_consistency.py":      "step_045_sfr_age_consistency.py",
    "step_50_compactness_paradox.py":      None,
    "step_51_multi_diagnostic.py":         "step_046_multi_diagnostic.py",
    "step_52_independent_tests.py":        "step_047_independent_tests.py",
    "step_52_independent_dust.py":         None,    # exploratory
    "step_53_dust_yield_comparison.py":    None,    # exploratory
    "step_53_deep_evidence.py":            None,
    "step_53b_sensitivity.py":             None,
    "step_53c_yield_requirement.py":       None,
    "step_54_spectroscopic_validation.py": None,    # superseded by step_035
    "step_55_advanced_diagnostics.py":     "step_048_advanced_diagnostics.py",
    "step_56_prediction_tests.py":         "step_049_prediction_tests.py",
    "step_57_critical_evidence.py":        None,    # superseded dup
    "step_57_decisive_evidence.py":        "step_050_critical_evidence.py",
    "step_57_critical_signal.py":              None,    # superseded dup
    "step_58_temporal_coherence.py":       "step_051_temporal_coherence.py",
    "step_59_mass_discrepancy.py":         "step_052_mass_discrepancy.py",
    "step_60_deep_signatures.py":          None,    # exploratory
    "step_61_tep_discriminant.py":         "step_053_tep_discriminant.py",
    "step_62_falsification_tests.py":      None,    # superseded by step_083
    "step_63_ultimate_synthesis.py":       None,    # exploratory
    "step_64_unique_predictions.py":       "step_054_unique_predictions.py",
    "step_65_cross_domain.py":             "step_055_cross_domain.py",
    "step_66_final_evidence.py":           None,    # exploratory
    "step_67_out_of_box.py":              "step_056_out_of_box.py",
    "step_68_adversarial_tests.py":        "step_057_adversarial_tests.py",
    "step_69_creative_tests.py":           "step_058_creative_tests.py",
    "step_70_deep_physics.py":             None,    # exploratory
    "step_71_final_synthesis.py":          None,    # exploratory
    "step_72_ultimate_missing_piece.py":   "step_059_ultimate_missing_piece.py",
    "step_73_spectroscopic_validation.py": None,    # superseded
    "step_74_the_functional_form_discrimination.py":          "step_060_the_functional_form_discrimination.py",
    "step_75_deeper_physics.py":           None,    # exploratory
    "step_76_extreme_cases.py":            "step_061_extreme_cases.py",
    "step_77_predictive_power.py":         "step_062_predictive_power.py",
    "step_78_correlation_deep_dive.py":    "step_063_correlation_deep_dive.py",
    "step_79_comprehensive_summary.py":    None,    # superseded
    "step_80_web_inspired_tests.py":       "step_064_web_inspired_tests.py",
    "step_81_literature_tests.py":         "step_065_literature_tests.py",
    "step_82_final_literature_tests.py":   "step_066_final_literature_tests.py",
    "step_83_cutting_edge_tests.py":       "step_067_cutting_edge_tests.py",
    "step_84_emission_line_tests.py":      "step_087_emission_line_tests.py",
    "step_85_final_comprehensive.py":      "step_088_final_comprehensive.py",
    "step_86_observational_signatures.py": "step_068_observational_signatures.py",
    "step_87_compactness_verification.py": "step_069_compactness_verification.py",
    "step_88_bbn_analysis.py":             None,    # superseded dup (keep step_89)
    "step_88_binary_pulsar_constraints.py":"step_070_binary_pulsar_constraints.py",
    "step_89_bbn_analysis.py":             "step_071_bbn_analysis.py",
    "step_90_sign_paradox_check.py":       "step_072_sign_paradox_check.py",
    "step_91_power_analysis.py":           None,    # superseded by step_111
    "step_92_growth_factor.py":            "step_073_growth_factor.py",
    "step_93_z67_tep_prediction.py":       "step_074_z67_tep_prediction.py",
    "step_94_mass_circularity_break.py":   None,    # superseded by step_143
    "step_95_bayesian_model_comparison.py":"step_075_bayesian_model_comparison.py",
    "step_96_ml_scaling_justification.py": "step_076_ml_scaling_justification.py",
    "step_97_bootstrap_validation.py":     "step_077_bootstrap_validation.py",
    "step_98_independent_age_validation.py":"step_078_independent_age_validation.py",
    "step_99_ml_cross_validation.py":      "step_079_ml_cross_validation.py",
    # 3-digit old → canonical (renumber conflicts)
    "step_100_combined_evidence.py":       "step_100_multi_domain_model_comparison.py",
    "step_101_balmer_simulation.py":       "step_080_balmer_simulation.py",
    "step_102_survey_cross_correlation.py":"step_081_survey_cross_correlation.py",
    "step_103_environmental_screening.py": None,    # superseded by step_094
    "step_104_comprehensive_figures.py":   None,    # superseded by step_085
    "step_105_evidence_strengthening.py":  "step_082_evidence_strengthening.py",
    "step_106_falsification_battery.py":   "step_083_falsification_battery.py",
    "step_107_effect_size_meta.py":        "step_084_effect_size_meta.py",
    "step_108_comprehensive_evidence_report.py": None,  # old dup
    "step_109_time_lens_map.py":           None,    # superseded by step_085
    "step_110_teff_threshold_scan.py":     "step_089_teff_threshold_scan.py",
    "step_111_permutation_battery.py":     "step_090_permutation_battery.py",
    "step_112_random_effects_meta_loo.py": "step_091_random_effects_meta_loo.py",
    "step_113_mass_matched_confirmation.py":"step_092_mass_matched_confirmation.py",
    "step_114_teff_threshold_holdout.py":  "step_093_teff_threshold_holdout.py",
    "step_115_environmental_screening_enhanced.py": "step_094_environmental_screening_enhanced.py",
    "step_116_lrd_core_halo_mass.py":      "step_095_lrd_core_halo_mass.py",
    "step_117_balmer_target_list.py":      "step_096_balmer_target_list.py",
    "step_118_independence_corrected_significance.py": None,  # superseded by step_118_neff
    "step_119_money_plot.py":              "step_097_money_plot.py",
    "step_120_alternative_model_comparison.py": "step_098_alternative_model_comparison.py",
    "step_121_alpha_evolution_test.py":    "step_099_alpha_evolution_test.py",
    "step_123_alpha_recovery_multi_observable.py": "step_101_alpha_recovery_multi_observable.py",
    "step_126_selection_function.py":      "step_128_selection_mc.py",
    "step_127_cosmic_variance.py":         "step_104_cosmic_variance.py",
    "step_128_morphology_tep.py":          "step_105_morphology_tep.py",
    "step_129_emission_line_diagnostic.py":"step_106_emission_line_diagnostic.py",
    "step_130_sn_rate_prediction.py":      "step_107_sn_rate_prediction.py",
    "step_132_lcdm_tension_quantification.py": "step_109_lcdm_tension_quantification.py",
    "step_133_gw_timing_prediction.py":    "step_110_gw_timing_prediction.py",
    "step_134_power_analysis.py":          "step_111_power_analysis.py",
    "step_135_scalar_tensor_constraints.py":"step_112_scalar_tensor_constraints.py",
    "step_136_agn_feedback_discriminant.py":"step_113_agn_feedback_discriminant.py",
    "step_137_imf_constraint.py":          "step_114_imf_constraint.py",
    "step_139_literature_spectroscopic_ages.py": "step_116_literature_spectroscopic_ages.py",
    "step_140_dynamical_mass_comparison.py":"step_117_dynamical_mass_comparison.py",
    "step_141_neff_corrected_significance.py": "step_118_neff_corrected_significance.py",
    "step_142_blind_validation.py":        "step_119_blind_validation.py",
    "step_143_screening_transition_profile.py": "step_120_screening_transition_profile.py",
    "step_144_extended_modified_gravity_comparison.py": "step_121_extended_modified_gravity_comparison.py",
    "step_146_alpha0_error_budget.py":     "step_123_alpha0_error_budget.py",
    "step_147_timespace_coupling.py":      "step_124_timespace_coupling.py",
    "step_149_screening_scale.py":         "step_126_screening_scale.py",
    "step_152_dust_models.py":             "step_129_dust_models.py",
    # These OLD high-numbered files are superseded by the existing canonical versions
    "step_158_scale_dependent_growth.py":  None,    # dup of step_135
    "step_160_functional_form_test.py":    None,    # dup of step_136
    "step_161_cross_survey_generalization.py": None, # dup of step_137
    "step_162_environmental_screening_steiger.py": "step_138_environmental_screening_steiger.py",
    "step_163_colour_gradient_steiger.py": "step_139_colour_gradient_steiger.py",
    "step_167_mass_proxy_breaker.py":      None,    # dup of step_143
    "step_170_adversarial_ml_attack.py":   None,    # dup of step_144
    "step_171_phase_boundary_activation.py": None,  # dup of step_145
    "step_173_cosmic_sfrd_correction.py":  None,    # dup of step_147
    "step_173_tep_concordance.py":         None,    # exploratory
    "step_177_jades_dr4_ingestion.py":     "step_149_jades_dr4_ingestion.py",
    "step_180_uncover_dr4_full_sps.py":    "step_152_uncover_dr4_full_sps.py",
    "step_183_jades_dr5_morphology.py":    "step_155_jades_dr5_morphology.py",
    "step_187_mass_measurement_bias.py":   "step_159_mass_measurement_bias.py",
    "step_188_manuscript_consistency_check.py": "step_160_manuscript_consistency_check.py",
    "step_189_multi_dataset_l1_combination.py": "step_161_multi_dataset_l1_combination.py",
}

def main():
    print("=" * 70)
    print("TEP-JWST Pipeline Reorganisation")
    print("=" * 70)

    renames_done = []
    archived = []
    conflicts = []
    already_canonical = []

    for old_name, new_name in RENAMES.items():
        old_path = STEPS_DIR / old_name
        if not old_path.exists():
            continue  # already gone

        if new_name is None:
            # Archive it
            dest = ARCHIVE_DIR / old_name
            if dest.exists():
                dest = ARCHIVE_DIR / (old_name + ".bak")
            shutil.move(str(old_path), str(dest))
            archived.append(old_name)
        else:
            new_path = STEPS_DIR / new_name
            if new_path.exists():
                # Canonical already exists — archive the old version
                dest = ARCHIVE_DIR / old_name
                if dest.exists():
                    dest = ARCHIVE_DIR / (old_name + ".bak")
                shutil.move(str(old_path), str(dest))
                archived.append(f"{old_name} (canonical exists)")
            else:
                shutil.move(str(old_path), str(new_path))
                renames_done.append(f"{old_name} → {new_name}")

    print(f"\n✓ Renamed: {len(renames_done)} files")
    for r in renames_done:
        print(f"  {r}")

    print(f"\n✓ Archived: {len(archived)} files")

    # ── Report missing canonical files ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("MISSING CANONICAL FILES (need stub creation)")
    print("=" * 70)
    missing = [f for f in CANONICAL if not (STEPS_DIR / f).exists()]
    for f in missing:
        print(f"  MISSING: {f}")
    print(f"\nTotal missing: {len(missing)} / {len(CANONICAL)}")

    # ── Unaccounted-for files ────────────────────────────────────────────────
    all_step_files = set(p.name for p in STEPS_DIR.glob("step_*.py"))
    known = CANONICAL_SET | set(RENAMES.keys()) | {"_reorganize_steps.py"}
    unaccounted = sorted(all_step_files - known - CANONICAL_SET)
    if unaccounted:
        print("\nUNACCOUNTED FILES (not in canonical list or rename map):")
        for f in unaccounted:
            print(f"  {f}")

if __name__ == "__main__":
    main()
