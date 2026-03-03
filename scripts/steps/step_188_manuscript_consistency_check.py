"""
step_188_manuscript_consistency_check.py

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

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/outputs/step_188_manuscript_consistency_check.json"
RUN_ALL = ROOT / "scripts/steps/run_all_steps.py"

FILES = {
    "results": ROOT / "site/components/4_results.html",
    "discussion": ROOT / "site/components/5_discussion.html",
    "conclusion": ROOT / "site/components/6_conclusion.html",
}

JSONS = {
    "step_185": ROOT / "results/outputs/step_185_cosmos2025_ssfr_inversion.json",
    "step_186": ROOT / "results/outputs/step_186_dja_balmer_decrement.json",
    "step_187": ROOT / "results/outputs/step_187_mass_measurement_bias.json",
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


def main() -> None:
    texts = {k: p.read_text() for k, p in FILES.items()}
    run_all_text = RUN_ALL.read_text()
    j185 = json.loads(JSONS["step_185"].read_text())
    j186 = json.loads(JSONS["step_186"].read_text())
    j187 = json.loads(JSONS["step_187"].read_text())

    checks = []

    # 1) Pipeline step count consistency in conclusion
    n_steps = count_registered_steps(run_all_text)
    c1 = extract_first_int(r"pipeline comprises\s+(\d+)\s+steps", texts["conclusion"])
    c2 = extract_first_int(r"complete analysis code,\s+(\d+)\s+pipeline steps", texts["conclusion"])
    checks.append({
        "name": "conclusion_step_count_matches_run_all_steps",
        "expected": n_steps,
        "found": {"reproducibility": c1, "data_availability": c2},
        "pass": (c1 == n_steps and c2 == n_steps),
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

    # 3) Known math-format regression guard
    bad_token = "$N = 1{,}283$–2{,}971$"
    checks.append({
        "name": "table13_n_range_math_delimiter_regression",
        "expected": "bad token absent",
        "found": bad_token in texts["results"],
        "pass": bad_token not in texts["results"],
    })

    # 4) step_185 key value appears in manuscript
    s185_rho = j185["results"]["ssfr"]["z_9_13"]["rho_partial"]
    s185_token = f"{s185_rho:.3f}"
    checks.append({
        "name": "step185_z9_13_partial_rho_present",
        "expected": s185_token,
        "found": s185_token in texts["results"],
        "pass": s185_token in texts["results"],
    })

    # 5) step_186 key value appears in manuscript
    s186_rho = j186["results"]["z_gt_2"]["rho_partial"]
    s186_token = f"{s186_rho:.3f}"
    checks.append({
        "name": "step186_z_gt_2_partial_rho_present",
        "expected": s186_token,
        "found": s186_token in texts["results"],
        "pass": s186_token in texts["results"],
    })

    # 6) step_187 suppression values present in discussion
    rho_true = j187["rho_true_unbiased"]
    rho_obs_beta07 = j187["suppression_by_beta"]["beta_0.7"]["rho_partial_obs"]
    t1 = f"{rho_true:.3f}"
    t2 = f"{rho_obs_beta07:.3f}"
    checks.append({
        "name": "step187_suppression_values_present_in_discussion",
        "expected": {"rho_true": t1, "rho_obs_beta07": t2},
        "found": {"rho_true": t1 in texts["discussion"], "rho_obs_beta07": t2 in texts["discussion"]},
        "pass": (t1 in texts["discussion"] and t2 in texts["discussion"]),
    })

    # 7) step_187 bootstrap beta CI appears in discussion (rounded)
    beta_boot = j187.get("beta_empirical_bootstrap_L4")
    if beta_boot:
        b_lo = f"{beta_boot['ci_95'][0]:.3f}"
        b_hi = f"{beta_boot['ci_95'][1]:.3f}"
        checks.append({
            "name": "step187_bootstrap_beta_ci_present_in_discussion",
            "expected": {"beta_ci_low": b_lo, "beta_ci_high": b_hi},
            "found": {"beta_ci_low": b_lo in texts["discussion"], "beta_ci_high": b_hi in texts["discussion"]},
            "pass": (b_lo in texts["discussion"] and b_hi in texts["discussion"]),
        })

    passed = sum(1 for c in checks if c["pass"])
    failed = len(checks) - passed

    output = {
        "step": "step_188",
        "description": "Automated manuscript-to-JSON consistency checks for defensibility",
        "summary": {
            "checks_total": len(checks),
            "checks_passed": passed,
            "checks_failed": failed,
            "status": "PASS" if failed == 0 else "FAIL",
        },
        "checks": checks,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"Saved: {OUTPUT}")
    print(f"Consistency checks: {passed}/{len(checks)} passed")

    if failed > 0:
        print("\nFailed checks:")
        for c in checks:
            if not c["pass"]:
                print(f"  - {c['name']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
