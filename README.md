# Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19000827.svg)](https://doi.org/10.5281/zenodo.19000827)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Abstract

JWST has revealed a set of high-redshift anomalies that appear disparate in detail but share a common structure: star formation efficiencies exceeding $\Lambda$CDM limits and anomalous stellar-to-dynamical mass ratios both appear preferentially in the deepest gravitational potentials, while overmassive black holes in Little Red Dots provide a separate compact-core stress test. This work tests whether that common pattern can arise from a single violation of the isochrony axiom. In the Temporal Equivalence Principle (TEP), a continuously screened two-metric Temporal Topology framework, proper time depends on environment in unscreened halos. Using a Cepheid-calibrated response prior derived from Paper 11's $\kappa_{\rm Cep}$, transferred to the galaxy stellar-population sector through the phenomenological normalization $K_{\rm gal}$, applied directly to the potential-linear $\Gamma_t$ formula (externally calibrated response prior, with no JWST-specific refit), the framework quantitatively accounts for the leading excess under the TEP response mapping. The Little Red Dot branch is retained only as an unresolved mass-model sensitivity diagnostic, not as a primary closure result.


Here "unified resolution" means a unified TEP response-prior organization of the tested JWST anomalies; it does not imply first-principles closure of all high-redshift galaxy-formation physics.

Strongest current direct test is a kinematic comparison using the JWST-SUSPENSE survey of massive quiescent galaxies at $z = 1.2$–$2.3$ ($N = 15$). A fundamental vulnerability of evaluating TEP photometrically is mass-proxy circularity, as $\Gamma_t$ depends on the gravitational potential. By employing dynamically measured masses ($M_{\rm dyn}$) from stellar velocity dispersions and spectral ages derived from absorption features, the SUSPENSE analysis tests a dynamical-potential predictor and photometric stellar mass side by side. The central comparison shows that $\Gamma_t$ predicts spectral age more strongly than stellar mass, yielding $\rho({\rm Age}, \Gamma_t \mid z) = +0.717$ ($p = 2.62 \times 10^{-3}$) compared to $\rho({\rm Age}, M_* \mid z) = +0.493$ ($p = 0.062$). Under joint control of the competing predictor and redshift, $\Gamma_t$ retains a residual association with age, $\rho({\rm Age}, \Gamma_t \mid M_*, z) = +0.599$ ($p = 1.83 \times 10^{-2}$), whereas stellar mass contributes no residual signal once $\Gamma_t$ is controlled, $\rho({\rm Age}, M_* \mid \Gamma_t, z) = +0.025$ ($p = 0.931$). Propagating the published asymmetric uncertainties for all 15 galaxies preserves a positive $\Gamma_t$ residual in 99.7\% of Monte Carlo draws. This one-sided residual structure supports the interpretation that galaxy evolution scales more closely with gravitational potential depth than with baryonic mass alone, while the small sample and non-significant Steiger comparison keep the branch explicitly caveated.


Primary large-sample JWST evidence comes from the photometric anomalies that motivated the theory, treated here as two primary empirical lines across three surveys ($N = 1{,}283$). A key model-discriminating result is the Uniformity Paradox: dust and accelerated evolution switch on selectively with potential depth ($\rho = +0.62$ at $z > 8$). Any standard-physics explanation that adjusts a time-uniform ingredient, such as enhanced AGB yields, would predict dust to become broadly ubiquitous or to follow star formation, rather than tracking gravitational depth. The effective-time coordinate organizes this dust signal better than raw cosmic time, passing a dedicated validation battery with $\rho(t_{\rm eff}, A_V \mid t_{\rm cosmic}) = +0.600$ ($p = 5.0 \times 10^{-29}$). Similarly, the mass–sSFR relation at $z > 7$ shows a strong partial correlation with $\Gamma_t$ ($\rho = -0.49$, $p = 10^{-18}$).


The same mapping also relieves the benchmark stellar-mass-function and cosmic-SFRD excesses. A preliminary CAMB implementation is reported to remain consistent with Planck constraints ($\sigma_8$ within $0.1\sigma$). The combination of the large-sample photometric lines, the direct kinematic comparison, and the structural selectivity of the Uniformity Paradox supports TEP as a coherent, falsifiable organizing framework for the anomalies considered here.

*Keywords:* Cosmology: early universe – Galaxies: high-redshift – Galaxies: evolution – Gravitation – Scalar-tensor theories – Infrared: galaxies

![JWST Galaxy Age Resolution](site/public/image.webp)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.6 (Kos)  
**Date:** First published: 13 March 2026 · Last updated: 19 August 2026
**Status:** Preprint (Open for Collaboration)  
**DOI:** [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827)  
**Website:** [https://mlsmawfield.com/tep/jwst/](https://mlsmawfield.com/tep/jwst/)  
**Paper Series:** TEP Series: Paper 12 (High-Redshift Anomalies)

## Overview

JWST has revealed a set of high-redshift anomalies that appear disparate in detail but share a common structure: star formation efficiencies exceeding LambdaCDM limits and anomalous stellar-to-dynamical mass ratios both appear preferentially in the deepest gravitational potentials, while overmassive black holes in Little Red Dots provide a separate compact-core stress test. This repository tests whether that common pattern can arise from a single violation of the isochrony axiom. In the Temporal Equivalence Principle (TEP), a continuously screened two-metric Temporal Topology framework, proper time depends on environment in unscreened halos. Using the Observable Response Coefficient kappa = (9.6 +/- 4.0) x 10^5 mag measured by Paper 11's Cepheid analysis, applied directly to the potential-linear Gamma_t formula (externally calibrated response prior, with no JWST-specific refit), the framework quantitatively accounts for the leading excess under the TEP response mapping. The Little Red Dot branch is retained as an unresolved mass-model sensitivity diagnostic, not as a primary closure result.

The empirical program is organized around two primary empirical lines, one ancillary spatial line, one derived regime-level line, and one direct kinematic decisive test:

| Evidence branch | Test | Sample | Role |
|------|------|--------|------|
| L1. Primary empirical line | Dust-Gamma_t relation and AGB threshold behavior | Multi-survey high-z photometric catalogs | Primary large-sample empirical line |
| L2. Ancillary spatial evidence | Resolved core screening and morphology controls | JADES resolved and direct-mass morphology branches | Ancillary |
| L3. Primary empirical line | Mass-sSFR inversion across redshift | UNCOVER, CEERS, COSMOS-Web/COSMOS2025 | Second primary empirical line |
| L4. Derived regime-level evidence | Dynamical-mass consistency under TEP mapping | Literature kinematic compilations and regime-level checks | Derived |
| L5. Direct kinematic decisive test | Spectral age versus Gamma_t from M_dyn, compared directly against photometric M_* under matched controls | JWST-SUSPENSE quiescent galaxies | Strongest direct test of mass circularity |

The strongest current direct test is a kinematic comparison using the JWST-SUSPENSE survey of massive quiescent galaxies at z = 1.2-2.3 (L5; N = 15). A fundamental vulnerability of evaluating TEP photometrically is mass-proxy circularity, as Gamma_t depends on the gravitational potential. By employing dynamically measured masses (M_dyn) from stellar velocity dispersions and spectral ages derived from absorption features, the SUSPENSE analysis tests a dynamical-potential predictor and photometric stellar mass side by side. The central comparison shows that Gamma_t predicts spectral age more strongly than stellar mass, yielding rho(Age, Gamma_t | z) = +0.717 (p = 2.62e-3) compared to rho(Age, M_* | z) = +0.493 (p = 0.062). Under joint control of the competing predictor and redshift, Gamma_t retains a residual association with age, rho(Age, Gamma_t | M_*, z) = +0.599 (p = 1.83e-2), whereas stellar mass contributes no residual signal once Gamma_t is controlled, rho(Age, M_* | Gamma_t, z) = +0.025 (p = 0.931). Propagating the published asymmetric uncertainties for all 15 galaxies preserves a positive Gamma_t residual in 99.7% of Monte Carlo draws. The direct Steiger predictor-comparison remains non-significant (p = 0.197), so this branch is carried as a direct kinematic test with an explicit small-sample caveat rather than as one of the two primary large-sample lines. This one-sided residual structure supports the interpretation that galaxy evolution scales more closely with gravitational potential depth than with baryonic mass alone, and it materially narrows the photometric circularity objection.

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
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate | [10.5281/zenodo.18059251](https://doi.org/10.5281/zenodo.18059251) |
| **Paper 8** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging | [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109761](https://doi.org/10.5281/zenodo.18109761) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | **TEP-JWST** (This repo) | Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | [TEP-WB](https://github.com/matthewsmawfield/TEP-WB) | Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102062](https://doi.org/10.5281/zenodo.19102062) |
| **Paper 15** | [TEP-EFA](https://github.com/matthewsmawfield/TEP-EFA) | Temporal Equivalence Principle: Temporal Shear in the Earth Flyby Anomaly | [10.5281/zenodo.19454863](https://doi.org/10.5281/zenodo.19454863) |
| **Paper 16** | [TEP-J0437](https://github.com/matthewsmawfield/TEP-J0437) | Synchronization Holonomy in Pulsar Scintillation | [10.5281/zenodo.19454620](https://doi.org/10.5281/zenodo.19454620) |
| **Paper 17** | [TEP-LLR](https://github.com/matthewsmawfield/TEP-LLR) | Lunar Laser Ranging and the Nordtvedt Effect | [10.5281/zenodo.19446029](https://doi.org/10.5281/zenodo.19446029) |

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
├── 12-TEP-JWST-v0.6-Kos.md    # Generated manuscript markdown
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
  note={Preprint v0.6 (Kos)}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Open Science Statement

This is an open research preprint repository. Manuscript sources, pipeline code, and derived outputs are provided to support transparent inspection and independent reproduction.

## Contact

Email: matthew@mlsmawfield.com  
ORCID: [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
