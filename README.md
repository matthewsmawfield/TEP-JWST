# Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19000827.svg)](https://doi.org/10.5281/zenodo.19000827)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.9 (Kos)  
**First published:** 13 March 2026 · **Last updated:** 24 October 2026  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827)  
**Website:** [https://mlsmawfield.com/tep/jwst/](https://mlsmawfield.com/tep/jwst/)  
**Paper Series:** TEP Series: Paper 12 (JWST High-Redshift Anomalies)

## Abstract

<!-- Generated from site/components/1_abstract.html; update the manuscript source, then run tools/sync_current_corpus_readmes.py. -->

JWST has revealed a pattern of high-redshift anomalies—such as extreme star formation efficiencies and unexpected stellar-to-dynamical mass ratios—that appear preferentially in deep gravitational potentials. This work tests whether these tensions can be resolved by relaxing the assumption of universal cosmic time. Under the Temporal Equivalence Principle (TEP)—a continuously screened two-metric framework—proper time depends on the local environment in unscreened halos. Using a prespecified magnitude-sector benchmark ($\kappa_{\rm gal} = 0.960 \times 10⁶$ mag, Paper 11 — where distinct empirical Cepheid-channel recoveries bracket the benchmark with a midpoint of $0.943 \times 10⁶$ mag, within $1.8\%$, and Step 62 derives $|\kappa_{P²}^{\rm pred}(\lambda_0=1)| = 1.196 \times 10⁶$ mag from the Cepheid envelope density ratio $\rho_T/\langle\rho\rangle(10\text{ d}) = 8.45 \times 10⁵$ at the unit-transport benchmark $\lambda_0 = 1$ with bare coupling $|\beta_A| = 1$ fixed by Paper 0, lying at the center of the direct SH0ES $P²$-carrier fit $(1.191 \pm 0.480) \times 10⁶$ mag ($\lambda_0 = 0.995 \pm 0.401$; $0.5\%$ central-value offset)) applied without any JWST-specific parameter refitting, this framework quantitatively accounts for the leading photometric and kinematic excesses.

To address mass-proxy circularity, the framework is tested against kinematic data using the JWST-SUSPENSE survey ($N=15$) and a broader velocity dispersion ($\sigma$)-based expansion ($N=75$, Step 171). In the SUSPENSE sample, the dynamical mass-to-light inference response ($R_{\rm ML}$) retains residual spectral-age information after controlling for stellar mass and redshift ($\rho = +0.556$, $p = 0.032$), whereas stellar mass loses its predictive power once $R_{\rm ML}$ is controlled ($\rho = +0.031$, $p = 0.91$). In the broader $N=75$ velocity-dispersion expansion, evaluating the purely kinematic predictor $R_{\rm ML,\sigma}^{\rm dir}$ under the Virial-exact depth map $\Psi \propto \sigma²$ (with zero dependence on SED-fitted $M_*$ or size $R_e$) yields a strong circularity-free partial correlation controlling for redshift: $\rho(R_{\rm ML,\sigma}^{\rm dir}, M_{*,\rm obs} \mid z) = +0.463$ ($p = 2.84 \times 10^{-5}$, $N=75$), strengthening to $\rho = +0.608$ ($p = 8.43 \times 10^{-7}$) on the $N=55$ exact-$M_{\rm dyn}$ subsample and $\rho = +0.575$ ($p = 0.0080$) on the $N=20$ stellar absorption-line subsample. Under the Virial-exact depth map $\Psi \propto \sigma²$, all five physical subsamples (pooled $N=75$, absorption $N=20$, emission $N=55$, $z \ge 4$ $N=56$, and SUSPENSE $N=15$) agree in residual-$z$ sign (5/5), and holding the local $z=0$ reference at $z_{\rm ref}=0$ ($\Psi(\sigma)\sqrt{1+z}-\Psi_{\rm ref}$) yields sign agreement across all three $N \ge 15$ individual surveys plus the absorption tracer while attenuating the absorption-line residual-$z$ trend by $86.5\%$ ($+0.252 \to +0.034$).

In large-sample photometry ($N = 1{,}254$ across three surveys), dust emergence and apparent evolutionary advance align strongly with potential depth ($\rho = +0.635$ at $z > 8$), organizing more cleanly along an effective-time coordinate than raw cosmic time. A secondary photometric test confirms that $R_{\rm ML}$ carries sSFR information beyond mass and redshift ($\rho(R_{\rm ML}, {\rm sSFR} \mid M_*, z) = -0.62$, $p = 8.2 \times 10^{-29}$), with the negative sign predicted by the TEP measurement equations (${\rm sSFR}_{\rm obs} \propto R_{\rm ML}^{m-n_{\rm SPS}}$ with $m < n_{\rm SPS}$). Furthermore, a nested Bayesian model comparison of the SED observables—utilizing a joint covariance likelihood that accounts for correlated outputs (Steps 176, 179)—shows that the nested augmented model (raw mass + $z$ + $\log R_{\rm ML}$, 17 parameters, $\ln Z = -1243.8$) beats the unconditioned raw mass+$z$ baseline ($\ln Z = -1248.2$) by $\Delta\ln Z = +4.45 \pm 1.09$ ($\text{BF} = 85.4 : 1$, very strong Bayesian evidence on the Jeffreys scale for adding the nonlinear depth predictor $\log R_{\rm ML}$ to the linear mass+$z$ baseline despite the 4-parameter Occam penalty), while the direction-isolation diagnostic confirms that the $R_{\rm ML}$-aligned component carries $\Delta\ln Z = +55.9 \pm 1.0$ ($\ln{\rm BF} = +41.5$ for 9-parameter TEP-only over orthogonalized standard, and $\ln{\rm BF} = +9.07 \pm 0.93$, $\text{BF} = 8{,}720 : 1$ in symmetric conventional residual space). While an unconstrained 17-parameter empirical bilinear interaction ($\{1, M_*, z, M_*\times z\}$, $\ln Z = -1190.9$) attains a higher raw-sample likelihood by fitting Prospector grid-boundary jumps in $\log{\rm sSFR}$ and metallicity across the $204/254$ broad photo-$z$ posteriors ($\Delta z_{68} = z_{84}-z_{16} > 5.0$, median $z_{16} = 3.78$, $\chi²$ up to $297.7$), it holds no advantage on the primary dust observable ($R² = 0.524$ vs $0.529$) or $\chi²$ ($0.346$ vs $0.312$, TEP higher); on the quality-controlled clean high-$z$ posterior subset ($z_{16} \ge 7.0$, $\chi² < 50$, $N=38$), the TEP augmented model outperforms the interaction across dust ($R² = 0.822$ vs $0.817$), $\log{\rm sSFR}$ ($0.440$ vs $0.428$), and $\chi²$ ($0.378$ vs $0.367$), reversing the joint Bayesian evidence in TEP's favour by $\Delta\ln Z = +2.85$ ($+3.68$ for 9-parameter Cov-TEP over 17-parameter Cov-Interaction on all 4 observables, and $+3.77$ on the 3-observable physical subset; Step 179), while any monotone-in-$z$ bilinear polynomial is independently falsified by the narrow-mass-bin sign reversal across $\Psi = \Psi_{\rm ref}$ ($\log M_* \simeq 9.6$, 4/4 bins; Step 178). Finally, the lensing and environmental-channel audits (Steps 138, 182) demonstrate that explicitly de-magnifying stellar masses by the UNCOVER DR4 lens magnification $\mu_{\rm ml}$ sharpens the $z>8$ dust–$R_{\rm ML}$ correlation from $\rho = +0.646$ to $\rho = +0.656$ ($p = 1.30 \times 10^{-32}$, $N=254$), that the correlation holds in the unmagnified $\mu_{\rm ml} \le 1.05$ outer field at $\rho = +0.603$ ($p = 1.29 \times 10^{-15}$, $N=144$), and that while an additive linear sky-density column in the Gaussian mean model tracks foreground Abell 2744 ($z=0.308$) lensing/depth systematics ($\{M_*,z,d_{\rm A2744},\mu_{\rm ml}\}$ gaining $\Delta\ln Z = +34.2$), the physical TEP environmental screening signature—suppression of the dust–$R_{\rm ML}$ correlation slope in overdense same-epoch ($\Delta z \le 0.5$) regions ($\rho_{\rm field} = +0.712$ vs $\rho_{\rm dense} = +0.484$, $\Delta\rho = +0.227$, Fisher $Z = +2.85$, $p = 0.0044$)—survives fixed-$d_{\rm A2744}$ clustercentric control across all three terciles ($+0.269, +0.184, +0.137$) and all five mass-matched quintiles (Step 138), positioning TEP as a coherent and falsifiable organizing framework for high-redshift galaxy evolution.

Keywords: Cosmology: early universe – Galaxies: high-redshift – Galaxies: evolution – Gravitation – Scalar-tensor theories – Infrared: galaxies

Current manuscript source: [1_abstract.html](site/components/1_abstract.html).

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

The strongest direct test is a kinematic comparison using the JWST-SUSPENSE survey (L5; N = 15) and a broader sigma-based expansion (N = 75, Step 171), which break mass-proxy circularity by utilizing dynamically measured masses and spectral ages. The SUSPENSE comparison shows that the dynamical R_ML predictor retains spectral-age information after stellar-mass and redshift control (rho = +0.556, p = 0.032), whereas stellar mass contributes no residual signal once R_ML is controlled (rho = +0.031, p = 0.91). In the broader N = 75 velocity-dispersion expansion under the Virial-exact depth map Psi propto sigma^2, the purely kinematic predictor R_{ML,sigma}^dir yields a strong circularity-free partial correlation controlling for redshift: rho(R_{ML,sigma}^dir, M_{*,obs} | z) = +0.463 (p = 2.84 x 10^-5, N = 75), strengthening to rho = +0.608 (p = 8.43 x 10^-7) on the N = 55 exact-M_dyn subsample and rho = +0.575 (p = 0.0080) on the N = 20 stellar absorption-line subsample, with 5/5 subsample sign agreement.

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
| **Paper 16** | [TEP-J0437](https://github.com/matthewsmawfield/TEP-J0437) | Synchronization Holonomy in Pulsar Scintillation | [10.5281/zenodo.19454619](https://doi.org/10.5281/zenodo.19454619) |
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
├── E12-TEP-JWST-v0.9-Kos.md    # Generated manuscript markdown
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
  note={Preprint v0.9 (Kos)}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Open Science Statement

This is an open research preprint repository. Manuscript sources, pipeline code, and derived outputs are provided to support transparent inspection and independent reproduction.

## Contact

Email: matthew@mlsmawfield.com  
ORCID: [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
