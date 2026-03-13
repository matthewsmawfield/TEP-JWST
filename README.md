# The Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19000828.svg)](https://doi.org/10.5281/zenodo.19000828)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![JWST Galaxy Age Resolution](site/public/og-image.jpg)

Author: Matthew Lukin Smawfield  
Version: v0.1 (Kos)  
Date: 13 March 2026  
Status: Preprint  
DOI: [10.5281/zenodo.19000828](https://doi.org/10.5281/zenodo.19000828)  
Website: [https://matthewsmawfield.github.io/TEP-JWST/](https://matthewsmawfield.github.io/TEP-JWST/)

## Overview

JWST has revealed a set of high-redshift anomalies that appear disparate in detail but share a common structure: star formation efficiencies exceeding LambdaCDM limits, overmassive black holes, and anomalous stellar-to-dynamical mass ratios all appear preferentially in the deepest gravitational potentials. This repository tests whether that common pattern can arise from a single violation of the isochrony axiom. In the Temporal Equivalence Principle (TEP), a chameleon-screened scalar-tensor theory, proper time depends on environment in unscreened halos. Using the external Cepheid prior alpha_0 = 0.58 +/- 0.16 with no JWST retuning, the framework fully resolves the Red Monster efficiency excess and provides a physical route to differential black-hole growth in Little Red Dots.

The empirical program is organized around two primary empirical lines, one ancillary spatial line, one derived regime-level line, and one direct kinematic decisive test:

| Evidence branch | Test | Sample | Role |
|------|------|--------|------|
| L1. Primary empirical line | Dust-Gamma_t relation and AGB threshold behavior | Multi-survey high-z photometric catalogs | Primary large-sample empirical line |
| L2. Ancillary spatial evidence | Resolved core screening and morphology controls | JADES resolved and direct-mass morphology branches | Ancillary |
| L3. Primary empirical line | Mass-sSFR inversion across redshift | UNCOVER, CEERS, COSMOS-Web/COSMOS2025 | Second primary empirical line |
| L4. Derived regime-level evidence | Dynamical-mass consistency under TEP mapping | Literature kinematic compilations and regime-level checks | Derived |
| L5. Direct kinematic decisive test | Spectral age versus Gamma_t from M_dyn, compared directly against photometric M_* under matched controls | JWST-SUSPENSE quiescent galaxies | Strongest direct test of mass circularity |

The strongest current direct test is a kinematic comparison using the JWST-SUSPENSE survey of massive quiescent galaxies at z = 1.2-2.3 (L5; N = 15). A fundamental vulnerability of evaluating TEP photometrically is mass-proxy circularity, as Gamma_t depends on the gravitational potential. By employing dynamically measured masses (M_dyn) from stellar velocity dispersions and spectral ages derived from absorption features, the SUSPENSE analysis tests a dynamical-potential predictor and photometric stellar mass side by side. The central comparison shows that Gamma_t predicts spectral age more strongly than stellar mass, yielding rho(Age, Gamma_t | z) = +0.733 (p = 0.0019) compared to rho(Age, M_* | z) = +0.493 (p = 0.062). Under joint control of the competing predictor and redshift, Gamma_t retains a residual association with age, rho(Age, Gamma_t | M_*, z) = +0.624 (p = 0.0129), whereas stellar mass contributes no residual signal once Gamma_t is controlled, rho(Age, M_* | Gamma_t, z) = -0.036 (p = 0.898). Propagating the published asymmetric uncertainties for all 15 galaxies preserves a positive Gamma_t residual in 99.7% of Monte Carlo draws. This one-sided residual structure supports the interpretation that galaxy evolution scales more closely with gravitational potential depth than with baryonic mass alone, and it materially narrows the photometric circularity objection.

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
├── 13manuscript-tep-jwst.md # Generated manuscript markdown
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

### 2) Fast evidence check (L1-L4)

```bash
python scripts/run_core_evidence.py
```

### 3) Rebuild generated manuscript markdown

```bash
npm run build:markdown --prefix site
```

### 4) Manuscript consistency check

```bash
python scripts/steps/step_160_manuscript_consistency_check.py
```

## Manuscript Source of Truth

- Do not edit `13manuscript-tep-jwst.md` directly.
- Edit manuscript content in `site/components/`.
- Rebuild generated markdown via `npm run build:markdown --prefix site`.

## Data Sources

Primary data families include:

- JWST high-z photometric and spectroscopic catalogs (UNCOVER, JADES, CEERS, COSMOS-Web/COSMOS2025).
- FRESCO Red Monsters and related high-z massive-galaxy compilations.
- Literature kinematic samples used for dynamical-mass anchoring.
- Standard cosmology references (for consistency and guardrail checks).

See `data/DATA_PROVENANCE.md` for acquisition details and provenance notes.

## Methodological Caveats

- The direct kinematic comparison is currently N = 15; the strongest defensible claim is one-sided conditional asymmetry and published-uncertainty robustness, not a decisive dependent-correlation rejection.
- L2 has a strong direct-mass morphology branch, while the resolved color-gradient sign discriminator remains ancillary and not yet decisive on its own.
- The literal Gamma_t > 1 tail can be low-count in some splits; power-aware interpretations are required.
- Correlated tests on the same underlying residuals are not meta-combined as if independent.
- Claims should be tied to live pipeline outputs, not manuscript text alone.

## Citation

```bibtex
@article{smawfield2026jwst,
  title={The Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.19000828},
  note={Preprint v0.1 (Kos)}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Open Science Statement

This is an open research preprint repository. Manuscript sources, pipeline code, and derived outputs are provided to support transparent inspection and independent reproduction.

## Contact

Email: matthewsmawfield@gmail.com  
ORCID: [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
