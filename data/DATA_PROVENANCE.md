# TEP-JWST Data Provenance

This document tracks all external data sources used in the TEP-JWST analysis.

**Last Updated:** 2026-04-24  
**Analysis Status:** Complete with real Prospector-derived SED ages from UNCOVER DR4

---

## PRIMARY DATA SOURCE: UNCOVER DR4 SPS Catalog

**This is the authoritative data source for TEP-JWST analysis.**

| Property | Value |
|----------|-------|
| **Source** | UNCOVER DR4 Stellar Population Catalog |
| **Reference** | Wang et al. 2024, ApJS 270, 12 |
| **DOI** | [10.5281/zenodo.14281664](https://doi.org/10.5281/zenodo.14281664) |
| **SED Fitting Code** | Prospector (Johnson et al. 2021) |
| **Local File** | `data/raw/uncover/UNCOVER_DR4_SPS_catalog.fits` |
| **Downloaded** | 2026-01-15 |
| **File Size** | 62.8 MB |

### Key Columns Used

| Column | Description | Units |
|--------|-------------|-------|
| `z_50` | Median photometric redshift | - |
| `mstar_50` | Median stellar mass | log(M☉) |
| `mwa_50` | Mass-weighted age | Gyr |
| `use_phot` | Photometry quality flag | 0/1 |

### Sample Selection

```
N_total = 74,020 sources
N_highz = 2,710 (z > 7, use_phot = 1, valid ages)
```

### Analysis Results

| Metric | Value |
|--------|-------|
| Redshift range | z = 7.0 – 19.2 |
| Stellar mass range | log(M*) = 5.97 – 11.26 |
| Mass-weighted age range | 15.8 – 444.9 Myr |
| **Mass-Age Correlation** | **ρ = -0.041, p = 0.033** |
| Interpretation | **NO positive correlation** |

### Verification Command

```bash
# Download UNCOVER DR4 SPS catalog
curl -L "https://zenodo.org/api/records/14281664/files/UNCOVER_DR4_SPS_catalog.fits/content" \
  -o data/raw/uncover/UNCOVER_DR4_SPS_catalog.fits
```

---

## 1. JADES Photometric Catalog

**Source:** JWST Advanced Deep Extragalactic Survey (JADES)  
**Reference:** Eisenstein et al. 2023, arXiv:2306.02465  
**Data Repository:** https://archive.stsci.edu/hlsp/jades

**Files Used:**
- `hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits` - GOODS-S Deep photometry (45,000+ sources)
- `hlsp_jades_jwst_nircam_goods-n_photometry_v1.0_catalog.fits` - GOODS-N photometry

**Key Columns:**
- Photometric redshifts
- Stellar masses (from SED fitting)
- Rest-frame colors

---

## 2. JADES Spectroscopic Catalog

**Source:** JADES NIRSpec Program  
**Reference:** Curtis-Lake et al. 2023, Nature Astronomy, 7, 622  
**Data Repository:** https://archive.stsci.edu/hlsp/jades

**Files Used:**
- `hlsp_jades_jwst_nirspec_*_clear-prism_line-fluxes_v1.0_catalog.fits`

**Key Columns:**
- Spectroscopic redshifts (z_spec)
- Emission line fluxes
- Line widths (for mass estimation)

---

## 3. CEERS Photometric Catalog

**Source:** Cosmic Evolution Early Release Science Survey (CEERS)  
**Reference:** Finkelstein et al. 2023, ApJL, 946, L13  
**Data Repository:** https://ceers.github.io/

**Key Papers:**
- Labbé et al. 2023, Nature, 616, 266 ("Impossible Galaxies")
- Arrabal Haro et al. 2023, Nature, 622, 707 (Spectroscopic confirmation/refutation)

**Key Columns:**
- Photometric redshifts
- Stellar masses
- SED-derived stellar ages

---

## 4. High-z Galaxy Samples (ANALYZED)

### 4.1 Labbé et al. 2023 - "Impossible Galaxies" (Reference only)

**Reference:** Nature 616, 266 (arXiv:2207.12446)  
**GitHub Repository:** https://github.com/ivolabbe/red-massive-candidates  
**Local File:** `data/raw/labbe_2023_github.tar.gz`

| Property | Value |
|----------|-------|
| N galaxies (published) | 13 |
| z range | 6.5 - 9.1 |
| Status | Reference catalog only; not used in primary analysis |

**Note:** The Labbé et al. 2023 catalog is retained as a reference. The primary Red Monster case study (step_043) uses only the 3 verified Xiao et al. (2024) FRESCO objects with published spectroscopic confirmations.

### 4.2 Hainline et al. 2023 - JADES z > 8 Catalog

**Reference:** arXiv:2306.02468  
**MAST URL:** https://archive.stsci.edu/hlsps/jades/  
**Local File:** `data/raw/JADES_z_gt_8_Candidates_Hainline_et_al.fits`

| Property | Value |
|----------|-------|
| N galaxies | 689 (after quality cuts) |
| z range | 8.0 - 14.0 |
| log(M*) range | 6.7 - 9.3 M⊙ |
| Status | ✓ ANALYZED |

**Analysis Results:**
- Spearman ρ = 0.835, p = 10⁻¹⁸¹
- TEP Γ_t range: 0.22 - 1.89
- Median Γ_t = 0.47

### 4.3 JADES Data Release 4 — NIRSpec/MSA Spectroscopic Catalog

**Reference:** D'Eugenio et al. 2025; Curtis-Lake, Cameron, Bunker et al. 2025 (Paper I); Scholtz, Carniani et al. 2025 (Paper II)
**Data URL:** https://jades.herts.ac.uk/DR4/Combined_DR4_external_v1.2.1.fits
**Local File:** `data/raw/jades_hainline/JADES_DR4_spectroscopic_catalog.fits`
**Downloaded:** 2026-02

| Property | Value |
|----------|-------|
| N total targets | 5,190 |
| N good spec-z (flags A/B) | 2,858 |
| N at z > 5 | 495 |
| N at z > 7 | 118 |
| N at z > 8 | 41 |
| z_max | 14.18 |
| Fields | GOODS-N, GOODS-S |
| Status | ✓ ANALYZED (step_149) |

**Flag definitions:**
- A: Secure redshift from multiple emission lines
- B: Secure redshift from single strong line + continuum break
- C–E: Tentative or unreliable (excluded)

**Analysis Results (step_149):**
- ρ(Γ_t, M_UV) = −0.877 (full, N=1,345), p < 10⁻³⁰⁰
- ρ(Γ_t, M_UV) = −0.998 (z>7, N=114), p = 5.6×10⁻¹⁴⁰
- ρ(Γ_t, M_UV) = −0.997 (z>8, N=40), p = 7.7×10⁻⁴⁴
- Photo-z accuracy: σ_MAD = 0.042, η_outlier = 9.7%
- 19× upgrade over prior combined spec catalog (N=147)

### 4.5 DAWN JWST Archive (DJA) NIRSpec Merged Table v4.4

**Reference:** Brammer et al. (DJA); de Graaff et al. (2024); Heintz et al. (2023)
**Data URL:** https://zenodo.org/records/15472354/files/dja_msaexp_emission_lines_v4.4.csv.gz
**Local File:** `data/raw/dja_nirspec_merged_v4.4.csv.gz`
**Downloaded:** 2026-02
**Version:** v4.4 (September 5, 2025)

| Property | Value |
|----------|-------|
| N total spectra | 80,367 |
| N grade≥3 unique sources | 19,445 |
| N at z > 5 (grade≥3) | 3,251 |
| N at z > 7 (grade≥3) | 698 |
| N at z > 8 (grade≥3) | 234 |
| N at z > 10 (grade≥3) | 34 |
| z_max | 14.08 |
| Programs covered | JADES, CEERS, RUBIES, UNCOVER, GLASS, PRIMER, 50+ others |
| Status | ✓ ANALYZED (step_150) |

**Quality grades:**
- 3: Secure redshift (multiple features or strong single line + continuum)
- 2: Tentative (excluded)
- 1/0: Uncertain/failed (excluded)

**Analysis Results (step_150):**
- ρ(Γ_t, log M*) = +0.986 (z>5 full, N=2,598), p < 10⁻³⁰⁰
- ρ(Γ_t, log M*) = +0.991 (z>7, N=552), p < 10⁻³⁰⁰
- ρ(Γ_t, log M*) = +0.992 (z>8, N=190), p = 2.5×10⁻¹⁷⁰
- Fixed-effects meta-analysis: ρ_FE = 0.980, I² = 0.78
- Photo-z accuracy: σ_MAD = 0.022, η_outlier = 13.7%

### 4.6 DJA × CEERS+UNCOVER SED Cross-match (step_151)

**Description:** Positional cross-match (0.5 arcsec) of DJA NIRSpec Merged v4.4 spec-z against CEERS DR1.0 and UNCOVER DR4 SED catalogs to obtain SED-derived physical properties for spectroscopically confirmed galaxies.
**Local Files:** `results/interim/step_151_specz_sed_crossmatch.csv`

| Property | Value |
|----------|-------|
| N matched (CEERS EGS) | 683 |
| N matched (UNCOVER A2744) | 93 |
| N total matched | 776 |
| N at z > 5 | 776 |
| N at z > 7 | 142 |
| N with Gamma_t (spec-z) | 765 |
| Status | ✓ ANALYZED (step_151) |

**Key Results:**
- ρ(Γ_t, E(B-V)) = −0.013 (z>4, N=765): **null** — no dust signal below z=7
- ρ(Γ_t, E(B-V)) = −0.357 (z>7, N=142): p=1.3×10⁻⁵ — signal emerges at z>7
- ρ(Γ_t, log SFR) = +0.575 (z>4, N=564): p=5.9×10⁻⁵¹ — consistent across full range
- CEERS photo-z accuracy for matched sources: σ_MAD=0.056, η_outlier=23%

### 4.4 Other High-z Compilations (Not Yet Used)

| Source | Reference | N galaxies | z range |
|--------|-----------|------------|--------|
| Curtis-Lake et al. 2023 | Nature Astro. 7, 622 | 4 | 10.4-13.2 |
| Robertson et al. 2023 | Nature Astro. 7, 611 | 8 | 9-12 |

---

## 5. COSMOS-Web Master Catalog

**Source:** COSMOS-Web Survey  
**Reference:** Shuntov et al. 2025  
**Local File:** `data/raw/COSMOSWeb_mastercatalog_v1_lephare.fits` (270 MB)

| Property | Value |
|----------|-------|
| SED Fitting Code | LePhare |
| Field | COSMOS |
| Status | ✓ ANALYZED (step_033, step_157, step_174) |

---

## 6. Kokorev LRD Catalog

**Source:** Little Red Dots photometric catalog  
**Reference:** Kokorev et al. 2024, ApJ, 968, 38  
**Local File:** `data/raw/kokorev_lrd_catalog_v1.1.fits` (210 KB)

| Property | Value |
|----------|-------|
| N objects | 260 |
| Status | ✓ ANALYZED (step_132) |

---

## 7. Direct Kinematic Samples

### 7.1 Primary Literature Kinematic Sample (L4)

**Local File:** `data/interim/literature_kinematic_sample.json`

| Source | Reference | N objects | z range | Type |
|--------|-----------|-----------|---------|------|
| Esdaile et al. 2021 | arXiv:2010.09738 | 4 | 3.2–3.7 | Exact M_dyn |
| Tanaka et al. 2019 | arXiv:1909.10721 | 1 | 4.01 | Upper-limit M_dyn |

**Status:** ✓ ANALYZED (step_117)

### 7.2 Same-Regime Literature Kinematic Sample (contextual)

**Local File:** `data/interim/same_regime_literature_kinematic_sample.json`

| Source | Reference | N objects | z range |
|--------|-----------|-----------|---------|
| de Graaff et al. 2024 | JADES NIRSpec | 6 | 5.5–7.4 |
| Saldana-Lopez et al. 2025 | arXiv:2501.17145 | 16 | 4.0–7.6 |
| Danhaive et al. 2025 | arXiv:2503.21863 | 33 | 4.0–7.6 |

**Status:** ✓ ANALYZED (step_170, contextual branch)

### 7.3 SUSPENSE Spectral Ages (L5)

**Source:** SUSPENSE Survey  
**Reference:** Slob et al. 2025  
**Local File:** `data/interim/suspense_kinematics_ages.json`

| Property | Value |
|----------|-------|
| N objects | 15 |
| z range | 1.0–2.5 |
| Status | ✓ ANALYZED (step_170, primary branch) |

---

## 8. Red Monsters (FRESCO)

**Source:** Xiao et al. 2024  
**Reference:** arXiv:2309.02492  
**Local File:** Hardcoded in `scripts/steps/step_043_blue_monsters.py`

| Property | Value |
|----------|-------|
| N objects | 3 (spectroscopically confirmed) |
| z range | 5.3–5.9 |
| log(M*) range | 10.5–11.1 M⊙ |
| Status | ✓ ANALYZED (step_043) |

---

## 9. Stellar Population Models

**SED Fitting Codes:**
- BAGPIPES (Carnall et al. 2018)
- Prospector (Johnson et al. 2021)
- EAZY (Brammer et al. 2008)

**Stellar Libraries:**
- FSPS (Conroy et al. 2009)
- BC03 (Bruzual & Charlot 2003)

---

## 6. Cosmological Parameters

**Reference:** Planck 2018 Results (Planck Collaboration 2020)

| Parameter | Value |
|-----------|-------|
| H₀ | 67.4 ± 0.5 km/s/Mpc |
| Ωₘ | 0.315 ± 0.007 |
| ΩΛ | 0.685 ± 0.007 |
| Age of Universe | 13.797 ± 0.023 Gyr |

**Cosmic Age at Redshift:**
- z = 10: t_cosmic ≈ 470 Myr
- z = 12: t_cosmic ≈ 370 Myr
- z = 13: t_cosmic ≈ 330 Myr

---

## 7. TEP Prediction Framework

**Source:** Paper 7 (TEP-UCD)  
**Reference:** Smawfield 2025, Zenodo (Universal Critical Density)

**Key Predictions:**
- Chronological Shear: Γ_t ∝ M^(1/3)
- Universal Critical Density: ρ_c ≈ 20 g/cm³
- Enhancement factor: α_eff ~ 10⁶-10⁷

---

## Verification Commands

```bash
# Download JADES photometry catalog
curl -O https://archive.stsci.edu/hlsps/jades/dr2/goods-s/catalogs/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits

# Run full pipeline
python scripts/run_pipeline.py --regenerate

# Verify high-z sample selection
python scripts/steps/step_001_uncover_load.py --verify
```

---

## Data Quality Requirements

| Dataset | Automated Download | VizieR Query | Manual Transcription | Status |
|---------|-------------------|--------------|---------------------|--------|
| JADES GOODS-S Photometry | ✓ | - | - | ✓ Downloaded (673 MB) |
| JADES GOODS-N Photometry | ✓ | - | - | ✓ Downloaded (818 MB) |
| Hainline+23 z>8 Catalog | ✓ | - | - | ✓ ANALYZED (689 galaxies) |
| Labbé+23 Sample | ✓ (GitHub) | - | - | ✓ ANALYZED (13 galaxies) |
| Cosmic Ages | - | - | ✓ (Planck18) | ✓ |

---

*Last updated: 2026-04-24*
