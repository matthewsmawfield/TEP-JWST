"""
step_160_manuscript_consistency_check.py

Automated manuscript-to-JSON consistency checks.

Purpose:
- Catch drift between manuscript claims and pipeline outputs.
- Catch formatting regressions in critical summary tables.
- Fail fast in run_all_steps when key consistency checks break.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM  = "160"
STEP_NAME = "manuscript_consistency_check"
LOGS_PATH = PROJECT_ROOT / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

ROOT = PROJECT_ROOT
OUTPUT = ROOT / "results/outputs/step_160_manuscript_consistency_check.json"
RUN_ALL = ROOT / "scripts/steps/run_all_steps.py"
GENERATED_ROOT_MARKDOWN = ROOT / "13manuscript-tep-jwst.md"

FILES = {
    "abstract": ROOT / "site/components/1_abstract.html",
    "introduction": ROOT / "site/components/2_introduction.html",
    "results": ROOT / "site/components/4_results.html",
    "discussion": ROOT / "site/components/5_discussion.html",
    "conclusion": ROOT / "site/components/6_conclusion.html",
    "supplementary": ROOT / "site/components/8_appendix.html",
    "supplementary_discussion": ROOT / "site/components/9_supplementary.html",
}

JSONS = {
    "step_037": ROOT / "results/outputs/step_037_resolved_gradients.json",
    "step_076": ROOT / "results/outputs/step_076_ml_scaling_justification.json",
    "step_079": ROOT / "results/outputs/step_079_ml_cross_validation.json",
    "step_117": ROOT / "results/outputs/step_117_dynamical_mass_comparison.json",
    "step_138": ROOT / "results/outputs/step_138_environmental_screening_steiger.json",
    "step_140": ROOT / "results/outputs/step_140_evidence_tier_summary.json",
    "step_143": ROOT / "results/outputs/step_143_mass_proxy_breaker.json",
    "step_157": ROOT / "results/outputs/step_157_cosmos2025_ssfr_inversion.json",
    "step_158": ROOT / "results/outputs/step_158_dja_balmer_decrement.json",
    "step_159": ROOT / "results/outputs/step_159_mass_measurement_bias.json",
}


def count_registered_steps(run_all_text: str) -> int:
    # Count quoted step script entries in STEPS list.
    return len(re.findall(r'"step_\d+[^"\n]*\.py"', run_all_text))


def extract_first_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1))


def check_close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def contains_any(texts: dict[str, str], snippets: list[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, text in texts.items():
        matched = [snippet for snippet in snippets if snippet in text]
        if matched:
            hits[name] = matched
    return hits


def main() -> None:
    texts = {k: p.read_text() for k, p in FILES.items()}
    combined_text = "\n".join(texts.values())
    generated_root_markdown_text = GENERATED_ROOT_MARKDOWN.read_text() if GENERATED_ROOT_MARKDOWN.exists() else ""
    run_all_text = RUN_ALL.read_text()
    j037 = json.loads(JSONS["step_037"].read_text())
    j076 = json.loads(JSONS["step_076"].read_text())
    j079 = json.loads(JSONS["step_079"].read_text())
    j117 = json.loads(JSONS["step_117"].read_text())
    j138 = json.loads(JSONS["step_138"].read_text())
    j140 = json.loads(JSONS["step_140"].read_text())
    j143 = json.loads(JSONS["step_143"].read_text())
    j185 = json.loads(JSONS["step_157"].read_text())
    j186 = json.loads(JSONS["step_158"].read_text())
    j187 = json.loads(JSONS["step_159"].read_text())

    checks = []

    # 1) Conclusion reproducibility wording should stay reader-facing
    stale_conclusion_step_phrases = [
        "pipeline comprises",
        "pipeline steps",
        "step-indexed JSON outputs",
        "Pipeline repository:",
    ]
    stale_conclusion_hits = [
        snippet for snippet in stale_conclusion_step_phrases
        if snippet in texts["conclusion"]
    ]
    required_conclusion_tokens = [
        "end-to-end analysis code",
        "archived outputs",
        "Analysis repository:",
    ]
    conclusion_token_presence = {
        token: (token in texts["conclusion"])
        for token in required_conclusion_tokens
    }
    checks.append({
        "name": "conclusion_reproducibility_wording_is_reader_facing",
        "expected": {
            "stale_step_phrases_absent": True,
            "required_tokens_present": True,
        },
        "found": {
            "stale_step_phrases": stale_conclusion_hits,
            "required_tokens_present": conclusion_token_presence,
        },
        "pass": (
            not stale_conclusion_hits
            and all(conclusion_token_presence.values())
        ),
    })

    # 2) New-results heading/list count consistency
    heading_ok = "Six new cross-dataset results" in texts["results"] or "Seven new cross-dataset results" in texts["results"] or "Eight new cross-dataset results" in texts["results"]
    list_match = re.search(
        r"(?:Six|Seven|Eight) new cross-dataset results[\s\S]*?<ol>([\s\S]*?)</ol>",
        texts["results"],
    )
    li_count = len(re.findall(r"<li>", list_match.group(1))) if list_match else 0
    checks.append({
        "name": "new_results_heading_and_list_count",
        "expected": [6, 7, 8],
        "found": li_count,
        "pass": heading_ok and (li_count in [6, 7, 8]),
    })

    # 2b) Canonical registry should include the new audit step before final synthesis
    step_164_token = '"step_164_uncover_z9_null_audit.py"'
    step_140_token = '"step_140_evidence_tier_summary.py"'
    has_step_164 = step_164_token in run_all_text
    has_step_140 = step_140_token in run_all_text
    step_164_before_140 = (
        has_step_164 and has_step_140
        and run_all_text.index(step_164_token) < run_all_text.index(step_140_token)
    )
    checks.append({
        "name": "canonical_registry_includes_late_audit_before_final_synthesis",
        "expected": {
            "step_164_present": True,
            "step_140_present": True,
            "step_164_before_step_140": True,
        },
        "found": {
            "step_164_present": has_step_164,
            "step_140_present": has_step_140,
            "step_164_before_step_140": step_164_before_140,
            "registered_steps": count_registered_steps(run_all_text),
        },
        "pass": has_step_164 and has_step_140 and step_164_before_140,
    })

    # 3) Known math-format regression guard
    bad_token = "$N = 1{,}283$–2{,}971$"
    checks.append({
        "name": "table13_n_range_math_delimiter_regression",
        "expected": "bad token absent",
        "found": bad_token in texts["results"],
        "pass": bad_token not in texts["results"],
    })

    # 3b) Results-section numbering and guide references must stay monotonic after the kinematic insertion
    numbering_stale_tokens = [
        "<h3>3.2 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations</h3>",
        "<h3>3.10 TEP Predictions vs Observations Summary</h3>",
        "<h4>3.10.1 Adversarial Tests</h4>",
        "<h4>3.10.2 Falsification Battery</h4>",
        "<h3>3.11 Strategy for Kinematic Validation</h3>",
    ]
    numbering_hits = contains_any(
        {
            "introduction": texts["introduction"],
            "results": texts["results"],
        },
        numbering_stale_tokens,
    )
    numbering_expected_tokens = {
        "intro_stage1_ref_updated": "Stage 1: The zero-parameter prediction (§3.2)." in texts["introduction"],
        "intro_stage2_ref_updated": "Stage 2: Two primary empirical lines, one ancillary spatial indication, one derived regime-level comparison, and one direct kinematic test (§3.0–3.10)." in texts["introduction"],
        "results_3_3_uncover": "<h3>3.3 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations</h3>" in texts["results"],
        "results_3_9_summary": "<h3>3.9 TEP Predictions vs Observations Summary</h3>" in texts["results"],
        "results_3_9_1_adversarial": "<h4>3.9.1 Adversarial Tests</h4>" in texts["results"],
        "results_3_9_2_falsification": "<h4>3.9.2 Falsification Battery</h4>" in texts["results"],
        "results_3_10_strategy": "<h3>3.10 Strategy for Kinematic Validation</h3>" in texts["results"],
    }
    checks.append({
        "name": "results_section_numbering_and_reader_guide_refs_are_synced",
        "expected": {
            "stale_numbering_tokens_absent": True,
            "updated_numbering_tokens_present": True,
        },
        "found": {
            "stale_numbering_token_hits": numbering_hits,
            "updated_numbering_token_presence": numbering_expected_tokens,
        },
        "pass": (
            not numbering_hits
            and all(numbering_expected_tokens.values())
        ),
    })

    # 4) L2 live/skipped state must agree with manuscript wording
    l2_status = (
        j140.get("final_synthesis", {})
        .get("lines_of_evidence", {})
        .get("L2_core_screening", {})
        .get("status")
    )
    live_primary_lines = (
        j140.get("final_synthesis", {})
        .get("availability", {})
        .get("live_primary_lines")
    )
    skipped_primary_lines = (
        j140.get("final_synthesis", {})
        .get("availability", {})
        .get("skipped_primary_lines")
    )
    stale_four_line_phrases = [
        "Stage 2: The Four Independent Lines of Evidence",
        "Four independent lines of evidence survive rigorous mass-control checks",
        "Four independent lines of evidence address this concern",
        "The four independent lines (L1–L4):",
        "4 independent lines; $N = 4{,}726$; 3 surveys; $z = 4$–$10$",
    ]
    stale_reader_workflow_phrases = [
        "present workspace",
        "current workspace",
        "not regenerated",
        "live reproducible",
        "live three-line package",
        "three live lines",
        "historical secondary check",
        "pending restoration",
        "step_117",
        "step_148",
    ]
    stale_l2_hits = contains_any(texts, stale_four_line_phrases + stale_reader_workflow_phrases)
    expected_ancillary_l2 = all(
        token in combined_text
        for token in ["ancillary", "resolved-gradient"]
    )
    checks.append({
        "name": "l2_skipped_state_matches_manuscript_wording",
        "expected": {
            "step_037_status": ["skipped", None, "SUCCESS"],
            "step_140_L2_status": ["skipped", "live"],
            "live_primary_lines": 2,
            "skipped_primary_lines": 0,
            "stale_four_line_and_workflow_phrases_absent": True,
            "ancillary_l2_language_present": True,
        },
        "found": {
            "step_037_status": j037.get("status"),
            "step_140_L2_status": l2_status,
            "live_primary_lines": live_primary_lines,
            "skipped_primary_lines": skipped_primary_lines,
            "stale_four_line_or_workflow_phrase_hits": stale_l2_hits,
            "ancillary_l2_language_present": expected_ancillary_l2,
        },
        "pass": (
            j037.get("status") in {"skipped", None, "SUCCESS"}
            and l2_status in {"skipped", "live"}
            and live_primary_lines == 2
            and skipped_primary_lines == 0
            and not stale_l2_hits
            and expected_ancillary_l2
        ),
    })

    # 4b) Evidence-line labels must match the live pipeline hierarchy everywhere
    claim_hierarchy = j140.get("final_synthesis", {}).get("claim_hierarchy", {})
    direct_tests = claim_hierarchy.get("direct_kinematic_tests", [])
    hierarchy_stale_phrases = [
        "SUSPENSE kinematic comparison (L1)",
        "current SUSPENSE kinematic comparison (L1)",
        "The direct kinematic comparison (L1)",
        "L1. Direct Kinematic Comparison",
        "The L1 kinematic test",
        "L4. Dust–",
        "L4. Dust-",
    ]
    hierarchy_hits = contains_any(
        {
            "introduction": texts["introduction"],
            "results": texts["results"],
            "discussion": texts["discussion"],
            "conclusion": texts["conclusion"],
            "generated_root_markdown": generated_root_markdown_text,
        },
        hierarchy_stale_phrases,
    )
    hierarchy_expected_tokens = {
        "claim_hierarchy_headline_primary_is_l1_dust": claim_hierarchy.get("headline_primary_result", "").startswith("L1 dust-Gamma_t"),
        "claim_hierarchy_direct_test_is_l5": direct_tests == ["L5_kinematic_decisive"],
        "introduction_l1_dust": "L1. Dust–" in texts["introduction"],
        "introduction_l5_direct_test": "L5. Direct kinematic test:" in texts["introduction"],
        "results_l1_dust": "L1. Dust–" in texts["results"],
        "results_l5_direct": "3.9 Direct Kinematic Decisive Test" in texts["results"],
        "discussion_l5_direct": "SUSPENSE kinematic comparison (L5)" in texts["discussion"],
        "conclusion_l5_direct": "SUSPENSE kinematic comparison (L5)" in texts["conclusion"],
        "generated_root_l5_direct": "SUSPENSE kinematic comparison (L5)" in generated_root_markdown_text,
    }
    checks.append({
        "name": "evidence_hierarchy_labels_match_pipeline_summary",
        "expected": {
            "l1_remains_dust": True,
            "l5_is_direct_kinematic_test": True,
            "stale_hierarchy_phrases_absent": True,
            "manuscript_and_generated_markdown_tokens_present": True,
        },
        "found": {
            "claim_hierarchy_direct_tests": direct_tests,
            "stale_hierarchy_phrase_hits": hierarchy_hits,
            "expected_token_presence": hierarchy_expected_tokens,
        },
        "pass": (
            not hierarchy_hits
            and all(hierarchy_expected_tokens.values())
        ),
    })

    # 5) alpha0 provenance wording must distinguish external prior from JWST recovery
    stale_alpha0_phrases = [
        "local-universe metric coupling ($\\alpha_0 = 0.548$)",
        "The coupling constant $\\alpha_0 = 0.548 \\pm 0.010$ is calibrated entirely from local Cepheid observations",
        "the coupling constant $\\alpha_0 = 0.548 \\pm 0.010$ is derived from $N = 29$ SH0ES Cepheid hosts",
    ]
    alpha0_hits = contains_any(texts, stale_alpha0_phrases)
    external_alpha = (
        j140.get("final_synthesis", {})
        .get("auxiliary_checks", {})
        .get("alpha0_concordance", {})
        .get("external_cepheid_alpha0")
    )
    external_sigma = (
        j140.get("final_synthesis", {})
        .get("auxiliary_checks", {})
        .get("alpha0_concordance", {})
        .get("external_cepheid_sigma")
    )
    jwst_alpha = (
        j140.get("final_synthesis", {})
        .get("auxiliary_checks", {})
        .get("alpha0_concordance", {})
        .get("jwst_recovered_alpha0")
    )
    jwst_sigma = (
        j140.get("final_synthesis", {})
        .get("auxiliary_checks", {})
        .get("alpha0_concordance", {})
        .get("jwst_recovered_sigma")
    )
    alpha0_expected_tokens = {
        "external_prior": [f"{external_alpha:.2f}", f"{external_sigma:.2f}"],
        "jwst_recovery": [f"{jwst_alpha:.3f}", f"{jwst_sigma:.3f}"],
    }
    alpha0_expected_presence = {
        key: all(token in combined_text for token in tokens)
        for key, tokens in alpha0_expected_tokens.items()
    }
    checks.append({
        "name": "alpha0_external_prior_and_jwst_recovery_are_distinguished",
        "expected": {
            "stale_alpha0_phrases_absent": True,
            "external_prior_tokens_present": True,
            "jwst_recovery_tokens_present": True,
        },
        "found": {
            "stale_alpha0_phrase_hits": alpha0_hits,
            "external_prior_tokens_present": alpha0_expected_presence["external_prior"],
            "jwst_recovery_tokens_present": alpha0_expected_presence["jwst_recovery"],
        },
        "pass": (
            not alpha0_hits
            and alpha0_expected_presence["external_prior"]
            and alpha0_expected_presence["jwst_recovery"]
        ),
    })

    # 6) Environmental-screening branch must remain supplementary/mixed, not primary
    env_branch = (
        j140.get("final_synthesis", {})
        .get("claim_hierarchy", {})
        .get("supplementary_or_mixed_branches", [])
    )
    z_gt_8_env = j138.get("z_gt_8", {})
    stale_environmental_phrases = [
        "All five density estimators show the predicted pattern",
        "5/5 density estimators show the predicted screening pattern",
        "Both parts of this prediction pass",
        "Additional test (predicted null confirms coupling)",
    ]
    environmental_hits = contains_any(texts, stale_environmental_phrases)
    environmental_expected_tokens = {
        "supplementary": "supplementary" in combined_text,
        "mixed": "mixed" in combined_text,
        "z_gt_8_null_value": f"{z_gt_8_env.get('delta_rho', 0.0):.3f}" in combined_text,
    }
    checks.append({
        "name": "environmental_screening_is_supplementary_and_mixed",
        "expected": {
            "step_140_marks_environmental_screening_supplementary": True,
            "stale_environmental_screening_phrases_absent": True,
            "supplementary_and_mixed_language_present": True,
        },
        "found": {
            "step_140_supplementary_or_mixed_branches": env_branch,
            "stale_environmental_screening_phrase_hits": environmental_hits,
            "supplementary_and_mixed_language_present": environmental_expected_tokens,
        },
        "pass": (
            "environmental_screening" in env_branch
            and not environmental_hits
            and all(environmental_expected_tokens.values())
        ),
    })

    # 6b) Mass-proxy branch must avoid closure language absent direct kinematics
    mass_proxy_state = j143.get("overall_verdict", {}).get("degeneracy_state")
    mass_proxy_conclusion = j143.get("overall_verdict", {}).get("conclusion", "")
    checks.append({
        "name": "mass_proxy_breaker_language_is_non_overclaiming",
        "expected": {
            "degeneracy_state": ["narrowed_not_closed", "partially_narrowed", "unbroken"],
            "broken_absent": True,
        },
        "found": {
            "degeneracy_state": mass_proxy_state,
            "broken_in_conclusion": "BROKEN" in mass_proxy_conclusion,
        },
        "pass": (
            mass_proxy_state in {"narrowed_not_closed", "partially_narrowed", "unbroken"}
            and "BROKEN" not in mass_proxy_conclusion
        ),
    })

    # 6c) L4 status in synthesis must match actual direct-kinematic availability
    l4_status = (
        j140.get("final_synthesis", {})
        .get("lines_of_evidence", {})
        .get("L4_dynamical_mass", {})
        .get("status")
    )
    l4_direct = bool(j117.get("direct_kinematic_measurements_used", False))
    expected_l4_status = "direct_object_level_kinematics" if l4_direct else "derived_from_real_data"
    checks.append({
        "name": "l4_status_matches_direct_kinematic_availability",
        "expected": expected_l4_status,
        "found": {
            "step_117_direct_kinematic_measurements_used": l4_direct,
            "step_140_l4_status": l4_status,
        },
        "pass": l4_status == expected_l4_status,
    })

    # 6d) Derived L4 wording must stay non-overclaiming and the generated root manuscript must be rebuilt
    stale_l4_overclaim_phrases = [
        "definitively breaks the photometric mass circularity",
        "proves that the scale factor tracks real gravitational dynamics",
    ]
    derived_l4_hits = contains_any(
        {
            "conclusion": texts["conclusion"],
            "generated_root_markdown": generated_root_markdown_text,
        },
        stale_l4_overclaim_phrases,
    )
    derived_l4_token_presence = {
        "conclusion": {
            "materially_narrows_phrase": "materially narrows the photometric mass-circularity objection" in texts["conclusion"],
            "derived_regime_phrase": "derived regime-level comparison rather than a primary empirical line" in texts["conclusion"],
        },
        "generated_root_markdown_exists": GENERATED_ROOT_MARKDOWN.exists(),
        "generated_root_markdown": {
            "materially_narrows_phrase": "materially narrows the photometric mass-circularity objection" in generated_root_markdown_text,
            "five_object_phrase": "five-object direct literature ingestion" in generated_root_markdown_text,
            "upper_limit_phrase": "conservative upper-limit row" in generated_root_markdown_text,
        },
    }
    checks.append({
        "name": "derived_l4_wording_is_non_overclaiming_and_generated_markdown_is_synced",
        "expected": (
            {
                "stale_overclaim_phrases_absent": True,
                "updated_conclusion_tokens_present": True,
                "generated_root_markdown_exists": True,
                "generated_root_markdown_tokens_present": True,
            }
            if expected_l4_status == "derived_from_real_data"
            else "check skipped when L4 is not in derived-regime mode"
        ),
        "found": {
            "step_140_l4_status": l4_status,
            "stale_overclaim_phrase_hits": derived_l4_hits,
            "updated_token_presence": derived_l4_token_presence,
        },
        "pass": (
            not derived_l4_hits
            and all(derived_l4_token_presence["conclusion"].values())
            and derived_l4_token_presence["generated_root_markdown_exists"]
            and all(derived_l4_token_presence["generated_root_markdown"].values())
        ) if expected_l4_status == "derived_from_real_data" else None,
    })

    # 7) Cross-domain language must keep the non-independent-programme caveat
    stale_cross_domain_phrases = [
        "10 independent domains",
    ]
    cross_domain_hits = contains_any(texts, stale_cross_domain_phrases)
    cross_domain_expected_tokens = {
        "single_theoretical_programme": "single theoretical programme" in combined_text,
        "not_independent_verification": "not independent verification" in combined_text,
    }
    checks.append({
        "name": "cross_domain_consistency_keeps_programme_level_caveat",
        "expected": {
            "stale_independence_phrases_absent": True,
            "programme_level_caveat_present": True,
        },
        "found": {
            "stale_independence_phrase_hits": cross_domain_hits,
            "programme_level_caveat_present": cross_domain_expected_tokens,
        },
        "pass": (
            not cross_domain_hits
            and all(cross_domain_expected_tokens.values())
        ),
    })

    # 8) Stale 6.4-sigma omnibus wording should stay out of the manuscript body
    stale_omnibus_sigma_phrases = [
        "6.4\\sigma",
        "6.4σ",
    ]
    omnibus_sigma_hits = contains_any(texts, stale_omnibus_sigma_phrases)
    checks.append({
        "name": "stale_omnibus_6p4sigma_claim_absent",
        "expected": {
            "stale_omnibus_sigma_phrases_absent": True,
        },
        "found": {
            "stale_omnibus_sigma_phrase_hits": omnibus_sigma_hits,
        },
        "pass": not omnibus_sigma_hits,
    })

    # 9) Supplementary discussion must stay aligned with live canonical support text
    supp_text = texts["supplementary_discussion"]
    stale_supplementary_phrases = [
        "Forward Modeling Validation",
        "forward modeling analysis",
        "Selection Function MC",
        "0% spurious detection rate",
    ]
    supplementary_hits = contains_any({"supplementary_discussion": supp_text}, stale_supplementary_phrases)
    ml_best_n = j076.get("empirical_validation", {}).get("best_n")
    ml_cv_n = j079.get("kfold_summary", {}).get("n_mean")
    ml_cv_rho = j079.get("kfold_summary", {}).get("rho_mean")
    ml_holdout_rho = j079.get("redshift_blind", {}).get("rho_high_z_with_low_z_n")
    mass_proxy_lowess = (
        j143.get("test_2_double_residual", {})
        .get("z_gt_8", {})
        .get("lowess_double_residual", {})
        .get("rho")
    )
    mass_proxy_partial = (
        j143.get("test_2_double_residual", {})
        .get("z_gt_8", {})
        .get("partial_spearman_mass_z", {})
        .get("rho")
    )
    mass_proxy_z = (
        j143.get("test_3_shuffled_mass", {})
        .get("z_gt_8", {})
        .get("z_score")
    )
    mass_proxy_unique = (
        j143.get("test_3_shuffled_mass", {})
        .get("z_gt_8", {})
        .get("unique_fraction_from_gamma_t_form")
    )
    l2_core = j140.get("final_synthesis", {}).get("lines_of_evidence", {}).get("L2_core_screening", {})
    supplementary_expected_tokens = {
        "l2_ancillary_language": all(
            token in supp_text
            for token in [
                "ancillary",
                f"{l2_core.get('rho_mass_grad'):.3f}",
                f"{l2_core.get('gradient_partial_p_gamma_given_mass_z'):.2f}",
                f"{l2_core.get('gradient_partial_p_gamma_given_debiased_mass_z'):.2f}",
            ]
            if token not in {"nan", "None"}
        ),
        "ml_live_validation": all(
            token in supp_text for token in [
                f"{ml_best_n:.1f}",
                f"{ml_cv_n:.2f}",
                f"{ml_cv_rho:.2f}",
                f"{ml_holdout_rho:.2f}",
            ]
        ),
        "mass_proxy_breaker": all(
            token in supp_text for token in [
                f"{mass_proxy_lowess:.3f}",
                f"{mass_proxy_partial:.3f}",
                f"{mass_proxy_z:.1f}",
                f"{100.0 * mass_proxy_unique:.1f}%",
            ]
        ),
    }
    checks.append({
        "name": "supplementary_discussion_matches_live_canonical_supporting_outputs",
        "expected": {
            "stale_forward_modeling_and_selection_mc_absent": True,
            "ancillary_l2_language_present": True,
            "ml_live_validation_present": True,
            "mass_proxy_breaker_present": True,
        },
        "found": {
            "stale_supplementary_phrase_hits": supplementary_hits,
            "supporting_tokens_present": supplementary_expected_tokens,
        },
        "pass": (
            not supplementary_hits
            and all(supplementary_expected_tokens.values())
        ),
    })

    # 10) step_157 key value appears in manuscript (skip if upstream is a stub)
    s185_bin = next((row for row in j185.get("ssfr_bins", []) if row.get("label") == "z=9-13 (high-z)"), None)
    if s185_bin and s185_bin.get("partial_rho") is not None:
        s185_rho = s185_bin["partial_rho"]
        s185_token = f"{s185_rho:.3f}"
        checks.append({
            "name": "step157_z9_13_partial_rho_present",
            "expected": s185_token,
            "found": s185_token in combined_text,
            "pass": s185_token in combined_text,
        })
    else:
        checks.append({"name": "step157_z9_13_partial_rho_present", "pass": None,
                        "note": "step_157 output does not expose the expected high-z partial-rho token; check skipped"})

    # 11) step_158 key value appears in manuscript (skip if upstream is reference-only)
    if j186.get("status") == "SUCCESS_REFERENCE_ONLY":
        checks.append({"name": "step158_z_gt_2_partial_rho_present", "pass": None,
                        "note": "step_158 is reference-only in this workspace; manuscript token check skipped"})
    elif "results" in j186:
        s186_rho = j186["results"]["z_gt_2"]["rho_partial"]
        s186_token = f"{s186_rho:.3f}"
        checks.append({
            "name": "step158_z_gt_2_partial_rho_present",
            "expected": s186_token,
            "found": s186_token in texts["results"],
            "pass": s186_token in texts["results"],
        })
    else:
        checks.append({"name": "step158_z_gt_2_partial_rho_present", "pass": None,
                        "note": "step_158 output does not expose a live reproducible z>2 partial-rho token; check skipped"})

    # 12) step_159 suppression values present in discussion (skip if stub)
    if "rho_true_unbiased" in j187:
        rho_true = j187["rho_true_unbiased"]
        rho_obs_beta07 = j187["suppression_by_beta"]["beta_0.7"]["rho_partial_obs"]
        t1 = f"{rho_true:.3f}"
        t2 = f"{rho_obs_beta07:.3f}"
        checks.append({
            "name": "step159_suppression_values_present_in_discussion",
            "expected": {"rho_true": t1, "rho_obs_beta07": t2},
            "found": {"rho_true": t1 in texts["discussion"], "rho_obs_beta07": t2 in texts["discussion"]},
            "pass": (t1 in texts["discussion"] and t2 in texts["discussion"]),
        })

        # 13) step_159 bootstrap beta CI appears in discussion (rounded)
        beta_boot = j187.get("beta_empirical_bootstrap_L4")
        if beta_boot:
            b_lo = f"{beta_boot['ci_95'][0]:.3f}"
            b_hi = f"{beta_boot['ci_95'][1]:.3f}"
            checks.append({
                "name": "step159_bootstrap_beta_ci_present_in_discussion",
                "expected": {"beta_ci_low": b_lo, "beta_ci_high": b_hi},
                "found": {"beta_ci_low": b_lo in texts["discussion"], "beta_ci_high": b_hi in texts["discussion"]},
                "pass": (b_lo in texts["discussion"] and b_hi in texts["discussion"]),
            })
    else:
        checks.append({"name": "step159_suppression_values_present_in_discussion", "pass": None,
                        "note": "step_159 output lacks suppression keys; check skipped"})

    passed  = sum(1 for c in checks if c["pass"] is True)
    skipped = sum(1 for c in checks if c["pass"] is None)
    failed  = sum(1 for c in checks if c["pass"] is False)

    output = {
        "step": "step_160",
        "description": "Automated manuscript-to-JSON consistency checks for defensibility",
        "summary": {
            "checks_total": len(checks),
            "checks_passed": passed,
            "checks_skipped": skipped,
            "checks_failed": failed,
            "status": "PASS" if failed == 0 else "FAIL",
        },
        "checks": checks,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"Saved: {OUTPUT}")
    print(f"Consistency checks: {passed} passed, {skipped} skipped, {failed} failed (of {len(checks)} total)")

    if failed > 0:
        print("\nFailed checks:")
        for c in checks:
            if c["pass"] is False:
                print(f"  - {c['name']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
