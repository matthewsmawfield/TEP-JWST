# TEP-JWST Pipeline Audit Report

**Date:** 2026-01-16  
**Pipeline Version:** 38 steps  
**Status:** Complete audit of all logs and outputs

---

## Executive Summary

The pipeline executes 38 analysis steps testing the Temporal Equivalence Principle (TEP) against JWST high-z galaxy data. This audit identifies both **methodological strengths** and **critical weaknesses** that must be addressed for rigorous scientific defense.

---

## 1. Derivation Chain Analysis

### 1.1 Core Equation
```
Γ_t = exp[α(z) × (2/3) × (log M_h - 12) × z_factor]
```
- **α₀ = 0.58** derived from Cepheid observations (TEP-H0, Paper 12)
- Applied to JWST data at z = 4-10 with **no tuning**

### 1.2 Parameter Provenance
| Parameter | Source | Uncertainty |
|-----------|--------|-------------|
| α₀ | Cepheid P-L relation | ±0.16 (28%) |
| M_h,ref | log 12 (convention) | Fixed |
| z_ref | 5.5 | Fixed |

**STRENGTH:** Single parameter (α₀) calibrated independently from local observations.

---

## 2. Statistical Findings

### 2.1 Primary Evidence (Strong)

| Test | Result | p-value | Assessment |
|------|--------|---------|------------|
| z > 8 dust anomaly | ρ = +0.56 | 10⁻²⁴ | **Strong** |
| CEERS replication | ρ = +0.68 | 10⁻¹² | **Confirmed** |
| COSMOS-Web replication | ρ = +0.63 | 10⁻⁵⁰ | **Confirmed** |
| "Impossible" galaxy resolution | 8/8 (100%) | N/A | **Complete** |
| Combined Fisher χ² | 37.2σ | 10⁻³⁰³ | **Highly significant** |

### 2.2 Weak or Problematic Evidence

| Test | Result | p-value | Issue |
|------|--------|---------|-------|
| Age ratio vs Γ_t | ρ = -0.07 | 0.0001 | **Wrong sign** (negative, not positive) |
| Spectroscopic validation | ρ = +0.18 | 0.047 | **Marginal** (barely significant) |
| z = 8-10 bin | ρ = +0.09 | 0.12 | **Not significant** |
| SN Ia mass step (this analysis) | 0.015 mag | 0.06 | **Not significant** (1.9σ) |
| TEP model R² improvement | ΔR² = 0.006 | N/A | **Minimal** (0.6% improvement) |

---

## 3. Critical Methodological Issues

### 3.1 The Mass Proxy Problem (CRITICAL)

**Step 30 Model Comparison reveals:**
- TEP model **loses** to mass-only model in 4/4 property tests (AIC/BIC)
- MASS outperforms TEP: 4/4
- TEP outperforms MASS: 0/4

**However:** Partial correlation test shows:
- corr(Dust, Γ_t | M*) = +0.17, p = 10⁻³⁹
- The z-dependent component (α(z) = α₀√(1+z)) is non-zero

**Interpretation:** Γ_t is primarily a mass proxy, but the z-dependent enhancement provides additional explanatory power that mass alone cannot capture.

**RECOMMENDATION:** The manuscript must explicitly acknowledge that TEP's primary signal is mass-driven, with the z-dependent component providing the discriminating test.

### 3.2 The Negative Correlation Paradox

**Step 14 shows:**
- Raw correlation (age_ratio vs Γ_t): ρ = **-0.07** (negative!)
- TEP predicts: positive correlation

**Resolution attempted (Step 15):**
- After TEP correction: ρ = -0.86 (more negative)
- Interpretation: "reveals underlying downsizing"

**PROBLEM:** This is post-hoc rationalization. The raw data shows the opposite of the TEP prediction. The "resolution" requires accepting that TEP masks an even stronger anti-correlation.

**RECOMMENDATION:** This must be addressed head-on in the manuscript. The negative correlation is a genuine challenge to TEP.

### 3.3 Spectroscopic Validation Weakness

**Step 37 shows:**
- Full sample (N=122): ρ = +0.18, p = 0.047 (barely significant)
- z = 6-8 bin (N=38): ρ = +0.27, p = 0.11 (not significant)
- z > 8 bin (N=7): insufficient sample

**PROBLEM:** The spectroscopic sample is too small to provide robust validation. The p = 0.047 is marginal and would not survive multiple testing correction.

**RECOMMENDATION:** Explicitly state that spectroscopic validation is "marginal" and requires larger samples.

### 3.4 SN Ia Mass Step Discrepancy

**Step 09 shows:**
- This analysis: 0.015 ± 0.008 mag (1.9σ, not significant)
- Literature: ~0.06 mag
- TEP predicted: 0.050 mag

**PROBLEM:** The observed mass step in this analysis is 4× smaller than literature values and not statistically significant.

**RECOMMENDATION:** Investigate why this analysis differs from literature. May be due to sample selection or methodology differences.

---

## 4. Strengths of the Analysis

### 4.1 Independent Replication
The z > 8 dust anomaly replicates across three independent JWST surveys:
- UNCOVER: ρ = +0.56 (N=283)
- CEERS: ρ = +0.68 (N=82)
- COSMOS-Web: ρ = +0.63 (N=918)

Fisher z-test confirms statistical consistency (p = 0.10).

### 4.2 Zero-Parameter Prediction
The α₀ = 0.58 parameter was calibrated from Cepheid observations and applied to JWST data with no tuning. This is a genuine prediction, not a fit.

### 4.3 Complete Resolution of "Impossible" Galaxies
All 8 galaxies with age ratio > 0.5 have Γ_t > 1, and all become physically plausible after TEP correction. The most extreme case (age ratio = 0.93) reduces to 0.28.

### 4.4 Rigorous Statistical Framework
- Bootstrap confidence intervals throughout
- Fisher's combined significance
- Partial correlations controlling for confounders
- AIC/BIC model comparison

---

## 5. Recommendations for Manuscript

### 5.1 Must Address
1. **Mass proxy critique:** Explicitly acknowledge that Γ_t is primarily mass-driven; the z-dependent component is the discriminating test.
2. **Negative correlation:** Address the raw negative age_ratio-Γ_t correlation directly.
3. **Spectroscopic weakness:** State that spectroscopic validation is marginal (p = 0.047).
4. **SN Ia discrepancy:** Explain why this analysis shows smaller mass step than literature.

### 5.2 Strengthen
1. Add uncertainty propagation for α₀ = 0.58 ± 0.16 through all predictions.
2. Perform sensitivity analysis: how do conclusions change if α₀ varies within uncertainty?
3. Add blind tests: can TEP predict properties of held-out samples?

### 5.3 Avoid Overclaiming
- Do not claim "37σ significance" without noting that many tests are correlated.
- Do not claim "perfect" correlations (ρ = 1.00) without noting small sample sizes.
- Do not claim TEP "explains" the mass step when this analysis shows it's not significant.

---

## 6. Summary Statistics

| Category | Count |
|----------|-------|
| Total steps | 38 |
| Steps with significant results | 28 |
| Steps with marginal/weak results | 7 |
| Steps with null/negative results | 3 |
| Independent replications | 3 (UNCOVER, CEERS, COSMOS-Web) |

---

## 7. Conclusion

The TEP-JWST analysis provides **suggestive but not conclusive** evidence for the Temporal Equivalence Principle. The strongest evidence comes from:
1. The z > 8 dust anomaly (replicated across 3 surveys)
2. The resolution of "impossible" galaxies
3. The zero-parameter prediction from Cepheid calibration

The weakest aspects are:
1. The mass proxy problem (TEP is primarily mass-driven)
2. The negative age_ratio-Γ_t correlation
3. The marginal spectroscopic validation

**A thesis defense must address these weaknesses directly.** Rigorous derivation requires acknowledging limitations, not hiding them.

---

*Report generated by pipeline audit, 2026-01-16*
