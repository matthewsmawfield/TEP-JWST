# Deepscan Audit Report: TEP-JWST

**Date:** January 20, 2026  
**Target:** Consistency and Correctness Check  
**Scope:** Manuscript HTML components vs. Pipeline Analysis Outputs

## Executive Summary

The audit reveals significant discrepancies between the latest pipeline outputs (`results/outputs/*.json`) and the manuscript text (`site/components/*.html`). While the core "Red Monster" and "z > 8 Dust" narratives hold, specific statistical claims in the Discussion and Results sections are outdated or contradict the latest data. 

**Critical Action Required:** The manuscript must be updated to match the current pipeline results to maintain scientific integrity.

## 1. Critical Discrepancies

### 1.1 Model Comparison (AIC/BIC)
**Location:** `site/components/5_discussion.html` (Section 4.4.3) vs `step_30_model_comparison.json`

The manuscript claims strong statistical preference for TEP+Mass models which is not supported by the current run.

| Metric | Manuscript Claim | Actual Pipeline Result | Status |
| :--- | :--- | :--- | :--- |
| **Dust ($A_V$) $\Delta$AIC** | **-37** (Strong preference) | **-0.6** (Weak preference) | ❌ **MISMATCH** |
| **sSFR $\Delta$AIC** | **-55** (Strong preference) | **-25.1** (Strong preference) | ⚠️ **Magnitude Diff** |
| **Model Preference** | "TEP+Mass strongly preferred" | "Full model marginally better" | ❌ **OVERSTATED** |

**Recommendation:** Downgrade the claims in Section 4.4.3. The data shows TEP adds *some* information (especially for sSFR), but for Dust, it is nearly indistinguishable from Mass-only in the full sample (though highly significant in the z>8 subset).

### 1.2 Partial Correlations (The "Mass Proxy" Test)
**Location:** `site/components/4_results.html` (Table 5) vs `step_07_seven_threads_summary.json`

The manuscript claims age and metallicity correlations "vanish" when controlling for mass, but the pipeline shows they remain significant.

| Relation | Manuscript Claim | Actual Pipeline Result | Status |
| :--- | :--- | :--- | :--- |
| **$\Gamma_t$ vs Age Ratio** | $\approx 0.00$ (Vanishes) | **$\rho = +0.14$** ($p < 10^{-11}$) | ❌ **CONTRADICTION** |
| **$\Gamma_t$ vs Metallicity** | $\approx 0.00$ (Vanishes) | **$\rho = +0.16$** ($p < 10^{-7}$) | ❌ **CONTRADICTION** |
| **$\Gamma_t$ vs Dust (z>8)** | $+0.28$ | **$+0.28$** | ✅ **CONSISTENT** |

**Impact:** The "Vanishing" claim was used to argue that TEP behaves like a mass proxy in the general population. The *survival* of these correlations actually strengthens the TEP case (it's *not* just a mass proxy anywhere), but requires rewriting the interpretation in Results 3.5 and Discussion 4.2.1.

### 1.3 Self-Consistency (The "CV = 0.000" Artifact)
**Location:** `site/components/5_discussion.html` (Section 4.11.14) vs `step_23_self_consistency.json`

*   **Issue:** The pipeline reports `alpha_cv: 0.0` and optimal alphas of `1.999996...`.
*   **Cause:** The optimizer hit the upper bound (2.0) for both observables. This is not "perfect consistency" but a **boundary condition failure**.
*   **Manuscript:** Cites "CV = 0.000" as evidence of precision. This is misleading.
*   **Recommendation:** Investigate `step_23_self_consistency.py`. If the fit pushes to the boundary, the model prefers *maximal* TEP effects, not necessarily *consistent* ones. This claim should be removed or qualified until the fit is fixed.

## 2. Textual & formatting Issues

### 2.1 Placeholders
*   **`results.html` Table 5:** Contains explicit `≈ 0.00` values which appear to be placeholders or rounded zeros from an older run.
*   **`discussion.html` Table 32:** Lists "$\Delta$AIC (vs Null)" values like `-653` which do not match `step_30` outputs (e.g., `-538` for sSFR).

### 2.2 Reference Integrity
*   **`introduction.html`:** Refers to `results.html` section "3.11.3" for AIC, but the Results section numbers stop at 3.10.
*   **`discussion.html`:** References "Paper 12 (TEP-H0)" and "Paper 11 (TEP-COS)". Ensure these citations in `7_references.html` are accurate. (They appear to be present).

## 3. Verified Consistent Elements (No Changes Needed)

*   **Red Monsters SFE:** The 54% resolution figure is consistent between `step_47` (54.1%) and the manuscript.
*   **z > 8 Dust Correlation:** The $\rho = +0.56$ figure is consistent.
*   **z > 7 Inversion:** The $\Delta\rho = +0.25$ figure is consistent.
*   **Core Screening:** The $\rho = -0.18$ figure is consistent.

## 4. Proposed Fixes

1.  **Update Table 5 in `results.html`:** Replace `≈ 0.00` with actual partial rho values (+0.14, +0.16) and update interpretation from "Vanishes" to "Weak but significant".
2.  **Update Discussion 4.4.3:** Replace AIC values with correct ones from `step_30`. Change narrative from "Strongly preferred" to "Modestly preferred" or "Statistically distinguishable".
3.  **Update Discussion 4.11.14:** Remove the "CV = 0.000" claim. Replace with a statement about the bootstrap test (which showed -351% scatter reduction? Wait, `step_23` says `scatter_reduction_pct: -351`. Negative reduction means **increased** scatter. The manuscript claims "reduced scatter by 27.3%". **CRITICAL CONTRADICTION**).

**Urgent:** The Bootstrap test in `step_23` failed (increased scatter on test set), but the manuscript claims success. This section (4.11.14) is factually incorrect based on current logs.
