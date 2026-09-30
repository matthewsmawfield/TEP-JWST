# Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19000827.svg)](https://doi.org/10.5281/zenodo.19000827)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.7 (Kos)  
**First published:** 13 March 2026 · **Last updated:** 30 September 2026  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827)  
**Website:** [https://mlsmawfield.com/tep/jwst/](https://mlsmawfield.com/tep/jwst/)  
**Paper Series:** TEP Series: Paper 12 (JWST High-Redshift Anomalies)

## Abstract





JWST has revealed a pattern of high-redshift anomalies—such as extreme star formation efficiencies and unexpected stellar-to-dynamical mass ratios—that appear preferentially in deep gravitational potentials. This work tests whether these tensions can be resolved by relaxing the assumption of universal cosmic time. Under the Temporal Equivalence Principle (TEP)—a continuously screened two-metric framework—proper time depends on the local environment in unscreened halos. Using a prespecified magnitude-sector benchmark (κ_gal = 0.960 × 10⁶ mag, Paper 11) applied without any JWST-specific parameter refitting, this framework quantitatively accounts for the leading photometric excesses; the factor ~ 2.4 residual between the canonical benchmark and the independent empirical Cepheid-channel recovery is stated explicitly and propagated through the prior range rather than tuned away. To address mass-proxy circularity, the framework is tested against kinematic data using the JWST-SUSPENSE survey (N=15) and a broader velocity dispersion (σ)-based expansion (N=75). In the SUSPENSE sample, the dynamical mass-to-light inference response (R_ML) retains residual spectral-age information after controlling for stellar mass and redshift (ρ = +0.556, p = 0.032), whereas stellar mass loses its predictive power once R_ML is controlled (ρ = +0.031, p = 0.91) — a suggestive one-sided asymmetry at N=15, pending replication, since the direct predictor comparison remains non-significant. The broader N=75 expansion is mixed: the σ-only R_ML adds no predictive power for M_beyond σ and z (partial ρ = +0.079, p = 0.50), and the sign of the residual-z trend is construction-dependent — opposite to expectation in the absorption-line subsample only under a σ-to-halo map that embeds expanding-cosmology virial drift, while the predicted sign is recovered under a cosmology-free σ-to-depth construction. The pooled trend is in any case a between-survey composition effect rather than a uniform within-survey signal. In large-sample photometry (N = 1,254 across three surveys), dust emergence and apparent evolutionary advance align strongly with potential depth (ρ = +0.635 at z > 8), organizing more cleanly along an effective-time coordinate than raw cosmic time. A secondary photometric test confirms that R_ML carries sSFR information beyond mass and redshift (ρ(R_ML, sSFR mid M_, z) = -0.62, p = 8.2 × 10⁻²⁹), with the negative sign predicted by the TEP measurement equations ( sSFR_obs ∝ R_ML^m-n_SPS with m 8 galaxy's ambient field, and with the DR4 lens-model magnification and cluster-centric distance in the baseline (+34.2), no same-epoch density proxy adds evidence (-7.1 on the full observable set, -6.0 on the physical subset). The comparison is interpreted as channel consistency with the operative ambient-field response law, not functional-form discrimination of the instantiated magnitude-sector kernel, and is treated as supportive context alongside the kinematic comparisons and the photometric correlation structure, positioning TEP as a coherent and falsifiable organizing framework for high-redshift galaxy evolution.


## Overview

JWST has revealed a pattern of high-redshift anomalies—such as extreme star formation efficiencies and unexpected stellar-to-dynamical mass ratios—that appear preferentially in deep gravitational potentials. This repository tests whether these tensions can be resolved by relaxing the assumption of universal cosmic time. In the Temporal Equivalence Principle (TEP), a continuously screened two-metric framework, proper time depends on the local environment in unscreened halos. The framework quantitatively accounts for the leading photometric excesses using the prespecified canonical TEP magnitude-sector benchmark ($\kappa_{\rm gal} = 0.960 \times 10^6$ mag, Paper 11) applied without any JWST-specific parameter refitting.

The empirical program is organized around one primary empirical line, one secondary partial-correlation line, one ancillary spatial line, one derived regime-level line, and one direct kinematic test:

| Evidence branch | Test | Sample | Role |
|------|------|--------|------|
| L1. Primary empirical line | Dust-R_ML relation and AGB threshold behavior | Multi-survey high-z photometric catalogs | Primary large-sample empirical line |
| L2. Ancillary spatial evidence | Resolved core screening and morphology controls | JADES resolved and direct-mass morphology branches | Ancillary |
| L3. Secondary partial-correlation test | R_ML–sSFR partial correlation (sign predicted by measurement equations) | UNCOVER, CEERS, COSMOS-Web/COSMOS2025 | Secondary test |
| L4. Derived regime-level evidence | Dynamical-mass consistency under TEP mapping | Literature kinematic compilations and regime-level checks | Derived |
| L5. Direct kinematic decisive test | Spectral age versus R_ML from M_dyn, compared directly against photometric M_* under matched controls | JWST-SUSPENSE quiescent galaxies + sigma expansion | Strongest direct test of mass circularity |

The strongest direct test is a kinematic comparison using the JWST-SUSPENSE survey (L5; N = 15) and a broader sigma-based expansion (N = 75), which break mass-proxy circularity by utilizing dynamically measured masses and spectral ages. The SUSPENSE comparison shows that the dynamical R_ML predictor retains spectral-age information after stellar-mass and redshift control (rho = +0.556, p = 0.032), whereas stellar mass contributes no residual signal once R_ML is controlled. A broader (N = 75) sigma-based expansion is mixed: the primary residual-evolution test is directionally positive overall, but stratification by sigma measurement type reveals the signal is driven by emission-line sigma while absorption-line sigma (the cleaner potential tracer) is non-significant with the wrong sign.

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Consistency Test | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Synthesis | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 6** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Cross-Scale Consistency of ρ_T | [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) |
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate | [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) |
| **Paper 8** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging | [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | **TEP-JWST** (This repo) | Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | [TEP-WB](https://github.com/matthewsmawfield/TEP-WB) | Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) |
| **Paper 15** | [TEP-EFA](https://github.com/matthewsmawfield/TEP-EFA) | Temporal Equivalence Principle: Temporal Shear in the Earth Flyby Anomaly | [10.5281/zenodo.19454862](https://doi.org/10.5281/zenodo.19454862) |
| **Paper 16** | [TEP-J0437](https://github.com/matthewsmawfield/TEP-J0437) | Synchronization Holonomy in Pulsar Scintillation | [10.5281/zenodo.19454620](https://doi.org/10.5281/zenodo.19454620) |
| **Paper 17** | [TEP-LLR](https://github.com/matthewsmawfield/TEP-LLR) | Lunar Laser Ranging and the Nordtvedt Effect | [10.5281/zenodo.19446028](https://doi.org/10.5281/zenodo.19446028) |

## Repository Structure

```text
TEP-JWST/
├── data/                    # Raw and interim catalogs
├── logs/                    # Step execution logs
├── manuscripts/             # Archived manuscript snapshots
├── results/
│   ├── outputs/             # JSON/CSV analytical outputs
│   └── figures/             # Generated figures
├── scripts/
│   ├── steps/               # Canonical stepwise pipeline
│   └── utils/               # Shared analysis utilities
├── site/
│   ├── components/          # Source of truth for manuscript text
│   └── dist/                # Built site artifacts
├── 12-TEP-JWST-v0.7-Kos.md    # Generated manuscript markdown
└── README.md
```

## Installation

```bash
git clone https://github.com/matthewsmawfield/TEP-JWST.git
cd TEP-JWST
pip install -r requirements.txt
npm install --prefix site
```

## Reproduction Workflow

### 1) Full canonical pipeline

```bash
python scripts/steps/run_all_steps.py
```

## Data Sources

Primary data families include:

- JWST high-z photometric and spectroscopic catalogs (UNCOVER, JADES, CEERS, COSMOS-Web/COSMOS2025).
- FRESCO Red Monsters and related high-z massive-galaxy compilations.
- Literature kinematic samples used for dynamical-mass anchoring.
- Standard cosmology references (for consistency and guardrail checks).

See `data/DATA_PROVENANCE.md` for acquisition details and provenance notes.

## Citation

```bibtex
@article{smawfield2026jwst,
  title={Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.19000827},
  note={Preprint v0.7 (Kos)}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Open Science Statement

This is an open research preprint repository. Manuscript sources, pipeline code, and derived outputs are provided to support transparent inspection and independent reproduction.

## Contact

Email: matthew@mlsmawfield.com  
ORCID: [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
