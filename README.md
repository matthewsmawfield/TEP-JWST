# Temporal Shear: Resolving the High-Redshift Efficiency Crisis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18204192.svg)](https://doi.org/10.5281/zenodo.18204192)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![JWST Galaxy Age Resolution](site/public/og-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.1 (Kos)  
**Date:** 12 March 2026  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.18204192](https://doi.org/10.5281/zenodo.18204192)  
**Website:** [https://matthewsmawfield.github.io/TEP-JWST/](https://matthewsmawfield.github.io/TEP-JWST/)

## Abstract

JWST has revealed coherent anomalies at $z > 5$: star formation efficiencies exceeding $\Lambda$CDM limits, overmassive black holes, and stellar masses surpassing dynamical masses — all concentrated in the deepest gravitational potentials. This study tests whether a single violation of the isochrony axiom can account for the common systematic component of these anomalies. The Temporal Equivalence Principle (TEP), a chameleon-screened scalar-tensor theory, predicts environment-dependent proper time in unscreened halos. Using the external Cepheid prior $\alpha_0 = 0.58 \pm 0.16$ with no JWST retuning, the framework predicts a substantial reduction of the Red Monster efficiency excess ($\sim 34\%$) and provides a regime-level mechanism for the stellar-to-dynamical mass discrepancy and for differential black-hole growth in Little Red Dots. The core empirical case rests on two primary lines across three surveys ($N = 4{,}726$), supplemented by one ancillary spatial indication and one derived regime-level comparison. L1 is a dust–$\Gamma_t$ correlation at $z > 8$ ($\rho = +0.62$, $N = 1{,}283$) together with the AGB-threshold test (odds ratio 42.8, $\Delta\text{AIC} \approx -4.8$ against a mass-matched threshold). L3 is the inversion of the mass–sSFR relation at $z > 7$, which is difficult to mimic with smooth mass-measurement systematics; in UNCOVER, partial $\rho(\Gamma_t, {\rm sSFR}|{\rm dust}) = -0.49$ ($p = 10^{-18}$). L2 (resolved core screening in JADES) and L4 (a matched regime-level dynamical-mass reconciliation, supplemented by a five-object direct literature ingestion at $z = 3.2$–$4.0$ that breaks the photometric mass-proxy degeneracy) provide rigorous, physically distinct, and supportive tests. A key model-discriminating result is the Uniformity Paradox: the anomaly is not that massive galaxies at $z > 8$ are dusty, but that low-mass galaxies are not — in a pattern that tracks gravitational potential depth ($\rho = +0.56$). Any standard-physics resolution that tunes a time-uniform parameter (e.g., enhanced AGB yields) cannot reproduce this mass-dependent suppression: if dust production were maximally efficient everywhere, dust should be ubiquitous or track star formation, not potential depth. Only a mechanism that selectively suppresses effective time in shallow potentials — as TEP predicts via $\Gamma_t \ll 1$ for low-mass halos — reproduces the observed gradient without additional parameters. The effective-time coordinate organizes the dust signal better than raw cosmic time. The dedicated UNCOVER $z > 8$ validation battery independently passes all four targeted tests, including $\rho(t_{\rm eff}, A_V \mid t_{\rm cosmic}) = +0.600$ ($p = 5.0 \times 10^{-29}$), and the three photometric L1 surveys combine to $z = 24.4\sigma$. Fitted polynomials collapse cross-survey ($R^2 = -6.4$), whereas $t_{\rm eff}$ generalises without retraining ($\rho = 0.60$–$0.80$). Full CAMB Boltzmann integration remains consistent with Planck constraints ($\sigma_8$ within $0.1\sigma$). Additional consequences of the same mapping include relief of the benchmark stellar-mass-function and cosmic-SFRD excesses and a physically motivated route to rapid black-hole growth. Conditional internal concordance analyses recover $\alpha_0$ near $0.55$, consistent with the external prior. The primary remaining systematic vulnerability is photometric mass circularity ($\Gamma_t$ depends on $M_*$); this degeneracy is broken through the L4 direct kinematic branch (five objects and a robust regime-level shift where TEP removes the 0.15-dex literature dynamical-mass anomaly) and a dedicated DJA emission-line pilot ($\rho = +0.887$ after mass control), demonstrating the time-dilation signal is driven by the physical gravitational potential, not photometric scaling. Keywords: Cosmology: early universe – Galaxies: high-redshift – Galaxies: evolution – Gravitation – Scalar-tensor theories – Infrared: galaxies

## Repository Structure

- `site/components/`: HTML components comprising the manuscript sections.
- `scripts/steps/`: Analysis pipeline (163 individual steps).
- `scripts/utils/`: Shared utilities (TEP model, data loading, plotting).
- `scripts/run_core_evidence.py`: **Fast-path for L1-L4 evidence** (~5 min).
- `scripts/run_photometry_anomalies.py`: Photometry-focused pipeline.
- `scripts/run_kinematics.py`: Kinematics & dynamical masses.
- `data/raw/`: Original survey catalogs (FITS).
- `data/interim/`: Processed CSV extracts.
- `results/outputs/`: Pipeline JSON outputs.
- `results/figures/`: Generated figures.

## Lines of Evidence

| Line | Test | Correlation | Sample | Status |
|------|------|-------------|--------|--------|
| **L1** | Dust-$\Gamma_t$ (z>8) | $\rho = +0.62$ | $N=1{,}283$ | Primary |
| **L2** | Core screening / direct-mass morphology | 4 supportive structural partials; raw gradient $\rho = -0.166$ | JADES DR5 direct-mass ($N=384$) + resolved photometry ($N=277$) | Ancillary |
| **L3** | Mass-sSFR inversion | Sign flip at $z>7$ | $N=4{,}726$ | Primary |
| **L4** | Dynamical-mass comparison | $1.41 \rightarrow 0.56$ representative regime shift | Published kinematic regime | Derived |

Run `python scripts/run_core_evidence.py` to validate the live evidence hierarchy (~5 minutes).

## Data Sources

Key datasets referenced in the analysis (see `data/DATA_PROVENANCE.md` for full details and download commands):

- Xiao et al. 2024 (FRESCO Red Monsters).
- Wang et al. 2024, UNCOVER DR4 SPS catalog (doi:10.5281/zenodo.14281664).
- JADES photometric + spectroscopic catalogs: Eisenstein et al. 2023; Curtis-Lake et al. 2023; Hainline et al. 2023.
- CEERS DR1 catalog: Finkelstein et al. 2023; Cox et al. 2025.
- COSMOS-Web DR1 / COSMOS2025 catalog: Shuntov et al. 2025.
- Labbé et al. 2023 high-z sample (GitHub: https://github.com/ivolabbe/red-massive-candidates).
- Planck 2018 cosmology (Planck Collaboration 2020).
- SN Ia host-mass step measurements: Kelly et al. 2010; Sullivan et al. 2010; Brout et al. 2022.
- TRGB and Cepheid distance ladder datasets: Freedman et al. 2024; Riess et al. 2022; Kodric et al. 2018.
- Milky Way globular cluster ages and positions: VandenBerg et al. 2013; Harris 2010.

## Citation

```bibtex
@article{smawfield2026jwst,
  title={Temporal Shear: Resolving the High-Redshift Efficiency Crisis},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18204192},
  note={Preprint v0.1 (Kos)}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).
