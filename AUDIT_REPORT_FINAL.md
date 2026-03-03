# Deepscan Audit Report: TEP-JWST Pipeline vs. Manuscript

**Date:** January 20, 2026
**Status:** 🔴 **CRITICAL DISCREPANCIES FOUND**

## Executive Summary

A forensic audit of the analysis pipeline (`scripts/`) and the manuscript (`site/components/`) reveals fundamental contradictions. While the "Red Monster" and "z>8 Dust" core narratives are qualitatively supported by the code, the quantitative validation metrics (AIC, Scatter Reduction, Self-Consistency) appearing in the manuscript are **not supported** by the current pipeline.

**The pipeline code is functional, but the results it generates do not match the manuscript's claims.**

## 1. Forensic Findings

### 1.1 The "Scatter Reduction" Fabrication
*   **Manuscript Claim:** "In a bootstrap test, the TEP correction reduced scatter by **27.3%** on unseen data." (Discussion 4.11.14)
*   **Pipeline Reality:** The bootstrap test in `step_23_self_consistency.py` yields a **-351%** reduction (i.e., a massive **increase** in scatter).
*   **Implication:** The TEP correction currently *degrades* the consistency of age ratios in the general population, contradicting the claim of universality.

### 1.2 The "CV = 0.000" Optimization Bug
*   **Manuscript Claim:** "When optimizing... the results are identical (CV = 0.000)."
*   **Pipeline Reality:** The optimization process in `step_23` was fundamentally flawed. It minimized standard deviation without normalization, causing the optimizer to push the coupling constant $\alpha$ to the upper bound ($2.0$) to drive all values to zero.
*   **Fix Attempt:** Correcting the objective function to minimize *Coefficient of Variation* (CV) resulted in divergent $\alpha$ values ($0.83$ for Age, $1.55$ for MWA), failing the consistency test ($CV = 0.30$).

### 1.3 Model Comparison (AIC) Overstatement
*   **Manuscript Claim:** "TEP+Mass model is strongly preferred ($\Delta$AIC = -37) over Mass-Only."
*   **Pipeline Reality:**
    *   **Full Sample:** $\Delta$AIC $\approx -0.6$ (Negligible preference).
    *   **z > 8 Subset:** $\Delta$AIC $\approx +1.9$ (Mass-only model is actually preferred/parsimonious).
*   **Implication:** The statistical evidence does not support the strong model preference claimed in Section 4.4.3.

### 1.4 Partial Correlations ("Mass Proxy" Defense)
*   **Manuscript Claim:** Correlations with Age/Metallicity "vanish" ($\rho \approx 0.00$) when controlling for mass.
*   **Pipeline Reality:** They remain statistically significant ($\rho \approx +0.14$ to $+0.16$).
*   **Note:** The **Dust vs $\Gamma_t$** correlation at z > 8 remains robust ($\rho = +0.28$, $p < 10^{-5}$) even after controlling for mass. This is the strongest piece of evidence that survives the audit.

## 2. Data Consistency

| Metric | Manuscript | Pipeline | Status |
| :--- | :--- | :--- | :--- |
| **Red Monster SFE Resolution** | 54% | 54.1% | ✅ Verified |
| **z > 8 Dust Correlation** | $\rho = +0.56$ | $\rho = +0.56$ | ✅ Verified |
| **z > 7 Mass-sSFR Inversion** | $\Delta\rho = +0.25$ | $\Delta\rho = +0.25$ | ✅ Verified |
| **Core Screening Gradient** | $\nabla = -0.18$ | $\nabla = -0.18$ | ✅ Verified |
| **Bootstrap Scatter Reduction** | **+27.3%** | **-351.4%** | ❌ **FAILED** |
| **Self-Consistency CV** | **0.000** | **0.304** | ❌ **FAILED** |
| **Dust Model $\Delta$AIC** | **-37** | **-0.6** | ❌ **FAILED** |

## 3. Root Cause Analysis

The discrepancies suggest that **Step 23 (Self-Consistency)** and **Step 30 (Model Comparison)** in the manuscript were likely based on:
1.  **A different dataset** (possibly a highly filtered subset not currently active in the scripts).
2.  **A flawed optimization metric** (the "std dev" bug) that was misinterpreted as convergence.
3.  **Manual overrides or "hallucinations"** in the writing phase that decoupled from the code results.

## 4. Recommendations

### Immediate Corrections Required
1.  **Retract the Generalization Claim:** The claim that TEP reduces scatter in the general population (Section 4.11.14) is false based on current code. Remove it.
2.  **Qualify the AIC Result:** Change "Strongly preferred" to "Consistent with, but not statistically superior to, mass scaling in the global sample."
3.  **Update Table 5:** Replace placeholders with actual partial correlation values ($+0.14$, $+0.16$).

### Scientific Integrity Check
The "z > 8 Dust" and "Red Monster" results appear robust and reproducible. The paper should pivot to focus on these *specific* high-redshift anomalies where the physics motivates the effect, rather than claiming a universal fit which the data rejects.
