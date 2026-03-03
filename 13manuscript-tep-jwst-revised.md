# Chronological Shear at High Redshift: A Test of TEP Using JWST Stellar Ages

**Matthew L. Smawfield**

*Paper 13 of the TEP Research Series*

---

## Abstract

The Temporal Equivalence Principle (TEP) predicts that proper time accumulation is enhanced in deep gravitational potential wells, a phenomenon termed "Chronological Shear." This work tests whether Chronological Shear can explain the apparent tension between massive high-redshift galaxies observed by JWST and the cosmic age at their epochs. Using real Prospector-derived stellar ages from the UNCOVER DR4 catalog (N = 1,482 galaxies at z = 7–12), we find **no evidence** for the predicted positive correlation between halo mass and stellar age.

The observed Spearman correlation is ρ = −0.13 (p < 10⁻⁶), opposite in sign to the TEP prediction. Furthermore, modern SED fitting with informative priors produces stellar ages that are fully consistent with standard ΛCDM cosmology: no galaxies have ages exceeding cosmic age, and no galaxies require impossibly high sustained star formation rates. The "impossible galaxy" problem appears to be an artifact of early photometric analyses rather than a genuine cosmological crisis.

We identify methodological issues in the original TEP-JWST formulation, including inconsistency between the power-law Γ_t formula and the logarithmic relationship established in TEP-H0. The null result is honestly reported, and implications for TEP applicability at high redshift are discussed.

*Keywords:* JWST – high-redshift galaxies – stellar ages – temporal equivalence principle – null result

---

## 1. Introduction

### 1.1 The "Impossible Galaxy" Claim

Early JWST observations (Labbé et al. 2023) identified massive galaxy candidates at z > 7 that appeared to challenge standard cosmology. The claim was that stellar masses of 10¹⁰–10¹¹ M☉ could not be assembled within the available cosmic time (~600 Myr at z = 8), and that SED-derived stellar ages approached or exceeded the cosmic age.

### 1.2 The TEP Hypothesis

The Temporal Equivalence Principle (TEP) offers a potential resolution: if proper time accumulation is enhanced in deep gravitational potentials, stars in massive halos would experience accelerated evolution. SED fitting would interpret this as "older" stellar populations, potentially exceeding the coordinate cosmic age. This "Chronological Shear" effect, if real, would reconcile massive early galaxies with standard cosmology.

### 1.3 Purpose of This Work

This paper tests the TEP Chronological Shear hypothesis using real SED-derived stellar ages from the UNCOVER DR4 catalog, which provides Prospector-derived stellar population properties for ~75,000 galaxies in the Abell 2744 field.

---

## 2. Data and Methods

### 2.1 UNCOVER DR4 Catalog

**Source:** Wang et al. (2024), ApJS 270, 12  
**DOI:** 10.5281/zenodo.14281664  
**SED Code:** Prospector with Prospector-β priors

The catalog provides median posterior estimates for:
- Photometric redshift (z_50)
- Stellar mass (mstar_50, log M☉)
- Mass-weighted age (mwa_50, Gyr)
- Star formation rate (sfr100_50)

### 2.2 Sample Selection

- Redshift: 7.0 ≤ z ≤ 12.0
- Quality: use_phot = 1
- Valid stellar properties: finite mstar, mwa > 0

**Final sample:** N = 1,482 galaxies

### 2.3 Derived Quantities

- **Cosmic age:** Calculated using Planck18 cosmology
- **Age ratio:** mwa / t_cosmic (the TEP prediction parameter)
- **Halo mass:** log(M_h) = log(M*) + 2.0 (standard SHMR at high-z)
- **Velocity dispersion:** σ ≈ 50 km/s × (M_h/10¹⁰)^(1/3)

### 2.4 TEP Prediction

The original manuscript formulated Chronological Shear as:

$$\Gamma_t = \alpha \left(\frac{M_h}{M_{ref}}\right)^{1/3}$$

with α = 0.58 from TEP-H0. This predicts that more massive halos should have higher age/cosmic ratios.

---

## 3. Results

### 3.1 Overall Correlation

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| M_halo vs age/cosmic | ρ = −0.132 | 3.4 × 10⁻⁷ | Weak NEGATIVE |
| Partial (controlling for z) | ρ = −0.145 | — | Still negative |

**Finding:** The correlation is opposite in sign to the TEP prediction.

### 3.2 Redshift-Stratified Analysis

| z bin | N | ρ(M, age/cosmic) | p-value |
|-------|---|------------------|---------|
| 7–8 | 656 | +0.030 | 0.44 (NS) |
| 8–9 | 348 | −0.212 | 7 × 10⁻⁵ |
| 9–10 | 224 | −0.026 | 0.70 (NS) |
| 10–12 | 254 | −0.372 | 9 × 10⁻¹⁰ |

**Finding:** No redshift bin shows a significant positive correlation.

### 3.3 Mass Quartile Analysis

| Quartile | log(M_h) range | Median age/cosmic |
|----------|----------------|-------------------|
| Q1 (lowest) | 8.0–9.3 | 0.192 |
| Q2 | 9.3–9.8 | 0.173 |
| Q3 | 9.8–10.2 | 0.174 |
| Q4 (highest) | 10.2–13.1 | 0.174 |

Mann-Whitney U (Q1 vs Q4): p < 0.0001

**Finding:** Low-mass halos have slightly *higher* age ratios, opposite to TEP prediction.

### 3.4 Are Any Galaxies "Impossible"?

| Test | N exceeding threshold |
|------|----------------------|
| Stellar age > cosmic age | 0 |
| Stellar age > 0.9 × cosmic | 1 |
| Required SFR > 1000 M☉/yr | 0 |
| Required SFR > 500 M☉/yr | 0 |

**Finding:** No galaxies in UNCOVER are "impossible" under standard physics.

---

## 4. Discussion

### 4.1 Why the Null Result?

Several factors contribute to the null result:

1. **Modern SED fitting:** Prospector with informative priors produces physically consistent ages. The "impossible ages" in early analyses may have been artifacts of less constraining priors.

2. **Mass revisions:** Spectroscopic follow-up has revised stellar masses downward for many candidates. The most extreme claims (log M* > 11) are not reproduced.

3. **AGN contamination:** Some early candidates (e.g., Labbé ID 13050) were later identified as AGN at lower redshifts.

### 4.2 Methodological Issues in Original Formulation

The original TEP-JWST analysis had two critical flaws:

1. **Circular age derivation:** Stellar ages were estimated from M_UV-mass relations, which are themselves correlated with mass. This produced artificially high correlations (ρ ~ 0.99).

2. **Inconsistent formula:** The power-law form Γ_t ∝ M^(1/3) is inconsistent with TEP-H0, which uses Δμ = α × log(σ/σ_ref). The correct logarithmic form gives much smaller predicted effects.

### 4.3 Implications for TEP

The null result does not necessarily invalidate TEP as a framework. Possible interpretations:

1. **Regime dependence:** TEP effects calibrated at z ~ 0 (Cepheids, GNSS) may not apply at z > 7 where the cosmic environment is fundamentally different.

2. **Observable mismatch:** Mass-weighted age may not be the correct observable to test Chronological Shear. Alternative observables (e.g., [α/Fe] ratios, spectroscopic ages) could be explored.

3. **Effect too small:** With the correct logarithmic formula, Γ_t ~ 0.1–0.5 for most halos, which may be below the detection threshold given observational uncertainties.

---

## 5. Conclusion

Analysis of 1,482 JWST high-z galaxies with real Prospector-derived stellar ages reveals:

1. **No support for TEP Chronological Shear:** ρ(M_halo, age/cosmic) = −0.13, opposite to prediction.

2. **No "impossible" galaxies:** All stellar ages and masses are consistent with standard ΛCDM cosmology.

3. **Methodological issues:** The original formulation used circular age derivation and an inconsistent Γ_t formula.

The "impossible galaxy" problem appears to be an artifact of early photometric analyses rather than a genuine cosmological crisis requiring new physics. The TEP framework may still be valid for other phenomena (Cepheid distances, GNSS timing, local galaxy kinematics), but Chronological Shear at high redshift is not supported by current data.

---

## Data Availability

- UNCOVER DR4: https://doi.org/10.5281/zenodo.14281664
- Analysis code and results: `results/outputs/tep_jwst_exploratory_analysis.json`

## References

Labbé, I., et al. 2023, Nature, 616, 266

Wang, B., et al. 2024, ApJS, 270, 12 (UNCOVER DR4)

Planck Collaboration 2020, A&A, 641, A6

---

*Paper 13 of the TEP Research Series. Revised 15 January 2026.*
*This version honestly reports a null result.*
