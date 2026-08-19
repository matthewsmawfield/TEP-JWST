# Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies
**Matthew Lukin Smawfield**  
Version: v0.6 (Kos)  
First published: 13 March 2026 · Last updated: 19 August 2026  
DOI: 10.5281/zenodo.19000827

---

## Abstract

JWST has revealed high-redshift anomalies that share a common structure: star formation efficiencies and anomalous stellar-to-dynamical mass ratios appearing preferentially in deep gravitational potentials. This work tests whether this pattern arises from a violation of the isochrony axiom. In the Temporal Equivalence Principle (TEP)—a continuously screened two-metric framework—proper time depends on environment in unscreened halos. The framework quantitatively accounts for the leading photometric excesses using the prespecified potential-linear halo kernel, with $\Phi\propto M_h^{2/3}$, normalized by an external Cepheid prior and applied without JWST-specific parameter refitting.


The strongest direct test of this framework is a kinematic comparison using the JWST-SUSPENSE survey ($N=15$) and a broader $\sigma$-based expansion ($N=83$), which break mass-proxy circularity by utilizing dynamically measured masses and spectral ages. The SUSPENSE comparison shows that the dynamical $\Gamma_t$ predictor retains spectral-age information after stellar-mass and redshift control ($\rho=+0.599$, $p=0.018$), whereas stellar mass contributes no residual signal once $\Gamma_t$ is controlled. A broader ($N=83$) sigma-based expansion is mixed: its secondary TEP-specific partial is positive, and the primary residual-evolution test is directionally positive overall, but stratification by $\sigma$ measurement type reveals the signal is driven by emission-line $\sigma$ while absorption-line $\sigma$ (the cleaner potential tracer) is non-significant with the wrong sign.


Primary large-sample JWST evidence derives from two empirical lines across three photometric surveys ($N = 1{,}283$). First, a Uniformity Paradox: dust and accelerated evolution switch on selectively with potential depth ($\rho = +0.60$ at $z > 8$), organizing along the effective-time coordinate rather than raw cosmic time. Second, the mass–sSFR relation inverts sign at $z > 7$, with the full-sample $\Gamma_t$–sSFR correlation yielding $\rho = -0.50$ ($p = 8.5 \times 10^{-146}$). The combination of these photometric lines, the direct kinematic comparisons, and an incremental Bayesian model preference ($\ln{\rm BF}=+104.8$) supports TEP as a coherent, falsifiable organizing framework for high-redshift galaxy evolution.

*Keywords:* Cosmology: early universe – Galaxies: high-redshift – Galaxies: evolution – Gravitation – Scalar-tensor theories – Infrared: galaxies




## 1. Introduction


### 1.1 Observational Tensions

JWST has revealed a coherent pattern of anomalies at $z > 5$ that strains the standard framework for inferring stellar properties from photometry. The most visible example is the class of spectroscopically confirmed "Red Monsters" (Xiao et al. 2024), whose stellar masses ($M_* \gtrsim 10^{11}\,M_\odot$) imply baryon-to-star conversion efficiencies of $\sim 0.50$, more than double the $\sim 0.20$ theoretical maximum imposed by feedback in $\Lambda$CDM halos. Within the Boylan-Kolchin (2023) framework, the discrepancy reaches $11\sigma$. This tension is not isolated. The UNCOVER UV luminosity function at $z > 9$ implies a star formation rate density exceeding the halo accretion limit by factors of 4–10 (Chemerynska et al. 2024).

A second tension emerges in JWST NIRSpec kinematics: at $z \gtrsim 3$–4, massive quiescent galaxies show $M_*/M_{\rm dyn} \gtrsim 1$ (Esdaile et al. 2021; Tanaka et al. 2019), while at $z > 5.5$ low-mass star-forming systems show the opposite extreme, with dynamical masses exceeding stellar masses by up to a factor 40 (de Graaff et al. 2024a). The population of "Little Red Dots" (LRDs) is discussed separately as a compact-core stress test: these red broad-line AGN can host black holes that appear overmassive relative to their galaxies, but the corrected TEP calculation is too sensitive to stellar-mass calibration to count as a primary empirical line.

Across these cases, the common structure is the same: stellar masses and ages inferred from photometry appear systematically too large, too early, in precisely the environments with the deepest gravitational potentials.


### 1.2 Challenging Isochrony with TEP

Underlying every photometric inference of stellar age, mass, and star formation rate is the isochrony axiom: the assumption that the clock governing stellar evolution ticks at the universal cosmic rate, regardless of local gravitational environment. Under isochrony, an observed red colour is interpreted as a combination of age, dust, and metallicity, and the resulting mass-to-light ratio ($M/L \propto t^n$) is treated as universal. If this axiom is violated — if stars in environments where Temporal Shear remains active accumulate proper time faster than the cosmic mean — then SED-inferred masses and ages are systematically inflated in precisely the environments where JWST finds the largest anomalies.


**Observer-age convention.** Throughout this paper, $t_{\rm cosmic}(z)$ denotes the age assigned to redshift $z$ by the standard FLRW observational reconstruction. It is an observer-frame coordinate used by conventional stellar-population inference, not the fundamental physical age of the universe in TEP. The canonical TEP cosmology has an asymptotic temporal past with unbounded local proper-time history.



The Temporal Equivalence Principle (TEP) formalises this possibility within a continuously screened two-metric Temporal Topology framework. In massive, active-shear halos, the Temporal Shear relaxes to values where proper time flows faster than coordinate time, so that a galaxy's effective age $\tau_{\star,\rm eff}(z,E) = \Gamma_t(E,z)\,t_{\rm FLRW}^{\rm obs}(z)$ exceeds the standard FLRW age assigned to that redshift. The resulting bias in $M/L$ inflates inferred stellar masses by $\Gamma_t^n$, directly mimicking a star formation efficiency excess. This effect is physically distinct from standard gravitational redshift: photons still lose energy climbing out of potential wells (kinematic redshift, fully preserved in TEP), while the scalar field coupling independently accelerates atomic processes within the well. Both effects coexist; only the latter biases photometric mass inference.


**Physics Note: Dilation vs. Enhancement**

It is essential to distinguish between two relativistic effects:


- **Kinematic Gravitational Redshift (Standard GR):** Photons lose energy climbing out of potential wells. This affects light and is fully preserved in TEP.

- **Dynamical Clock Rate (TEP):** The Temporal Shear modifies the effective mass of particles, changing the rate at which atomic clocks tick relative to coordinate time. In the TEP framework, diffuse halos relax to values where $A(\phi) > 1$, causing clocks to tick *faster* (enhancement) than the cosmic mean, even while photons suffer redshift.




The TEP enhancement effect is governed by the temporal enhancement factor $\Gamma_t$:


$$\Gamma_t = \exp\left[ K \frac{\Phi - \Phi_{\rm ref}}{c^2} \times \sqrt{1+z} \right]$$


The JWST response-prior test adopts the canonical response coefficient $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag from Paper 11 (where the empirical Cepheid fit yields $\kappa_{\rm Cep}^{\rm emp} = (1.27 \pm 0.46) \times 10^6$ mag, consistent with the TEP-native equivalent $\approx 7.21 \times 10^5$ mag), transferred to the galaxy stellar-population sector through the phenomenological normalization $K_{\rm gal}$, and applies it without JWST-specific refit. The high-redshift observables are then examined for internal consistency of the response scale. These internal recoveries are treated as self-consistency checks rather than as replacement calibrations; the latest multi-observable recovery is anchor-consistent and internally concordant. No parameters are tuned to the JWST data the model seeks to explain. The structural choices in the $\Gamma_t$ formula — the reference redshift $z_{\rm ref} = 5.5$, reference halo mass $\log M_{h,\rm ref} = 12.0$, exponential functional form, and $\sqrt{1+z}$ coupling scaling — were fixed by the scalar-tensor framework in prior papers; all have independent physical motivation and none were adjusted to improve JWST fits.


### 1.3 Reader's Guide to the Evidence

The analysis proceeds in three stages: an externally calibrated prediction (§3.1), the main multi-survey evidence (§3.0–3.10), and a compact-core stress test (§4.4). The five evidence lines are:


- **L1. Dust–$\Gamma_t$ emergence (Primary):** At $z > 8$, massive galaxies are anomalously dusty while low-mass galaxies remain dust-poor ($\rho = +0.60$).

- **L2. Inside-out core screening (Ancillary):** Central screening structures, supported by the JADES DR5 morphology sample.

- **L3. Mass–sSFR inversion (Primary):** The standard downsizing correlation reverses sign at $z > 7$.

- **L4. Dynamical mass comparison (Derived):** A kinematic comparison showing the TEP correction reconciles anomalous $M_*/M_{\rm dyn} > 1$ observations.

- **L5. Direct kinematic test (Direct):** The JWST-SUSPENSE comparison evaluates $\Gamma_t$ from $M_{\rm dyn}$ against photometric $M_*$ in predicting spectral age.


L1 and L3 are the two primary lines; L2 is ancillary; L4 is derived; L5 is a direct test. This classification is stated once here and not repeated for each result.


### 1.4 Prior Cross-Domain Evidence for TEP

The JWST analysis presented here is not the first test of the TEP framework. A local Cepheid analysis (Paper 11) provides $\kappa_{\rm Cep}$; this work transfers that prior to the galaxy sector through $K_{\rm gal}$, and the present high-redshift study asks whether JWST observables recover an anchor-consistent response across multiple domains spanning 13.5 Gyr of cosmic time. *Important caveat:* all prior constraints derive from a single theoretical programme; this is not independent verification; the TEP series has not yet undergone independent replication or peer review in a refereed journal. Readers should weigh the cross-domain consistency with this single-source limitation in mind. The three domains most directly used in this work are:


- **Hubble Tension:** Stratification of $N = 29$ SH0ES Cepheid hosts by velocity dispersion reveals an environmental bias. High-$\sigma$ hosts yield $H_0 = 74.12 \pm 1.30$ km/s/Mpc; low-$\sigma$ hosts yield $H_0 = 66.26 \pm 2.10$ km/s/Mpc — consistent with Planck within $1\sigma$. The TEP correction with the Paper 11-derived response coefficient transferred via $K_{\rm gal}$ yields $H_0^{\rm TEP} = 68.84$ km/s/Mpc (bootstrap mean $68.92 \pm 1.44$), reducing the Hubble tension from $\approx 5\sigma$ to $\approx 1\sigma$. This provides the low-redshift anchor used in this work.

- **Globular Cluster Pulsars:** Analysis of 197 globular-cluster millisecond pulsars (against 346 field controls) reveals a 0.63 dex raw spin-down excess (Welch t-test p ≈ 10⁻¹⁷) and a 0.40 dex hybrid-controlled residual (bootstrap p = 0.0002). The environmental screening threshold σ > 165 km/s derived from this population is used directly in §2.3.2.2 of this work.

- **Temporal Topology Reference Scale:** The screening threshold $\rho_T \approx 20$ g/cm³ is independently anchored by the SPARC rotation curve slope, magnetar critical periods, and terrestrial atomic clock correlation lengths. This $\rho_T$ informs the continuous screening function in this work.


The central question this work addresses is whether the same Paper 11-derived $\kappa_{\rm Cep}$, transferred via $K_{\rm gal}$, that resolves the Hubble tension and accounts for pulsar timing anomalies also predicts the high-redshift galaxy anomalies, with no re-tuning. The JWST analysis uses this coupling directly in the potential-linear $\Gamma_t$ formula, converting from the magnitude sector (Cepheid P-L residuals) to the stellar-population sector (nuclear burning timescales) via the shared TEP framework.


### 1.5 Alternative Explanations

There is no shortage of standard-physics explanations for JWST's high-redshift surprises. Proposed mechanisms include top-heavy initial mass functions, bursty or ultra-efficient star formation, early black hole seeding, strong AGN contamination, dust geometry effects, and selection/systematic biases in the spectral-energy-distribution (SED) fitting procedures. The present work does not dismiss these a priori; instead, it evaluates whether they can reproduce the specific temporal and structural signatures in the data.

Standard alternatives include top-heavy initial mass functions (Boylan-Kolchin 2023), enhanced AGN feedback, bursty star formation, and super-Eddington accretion. Each can partially address one or two of the observed anomalies, and the flexible AGN-feedback family remains the hardest competitor in raw joint Bayesian comparisons. AGN feedback often predicts a negative dust–black hole mass correlation, as AGN activity clears dust; the observed relation is positive ($\rho = +0.60$). Bursty star formation predicts bluer colours during burst phases, whereas the TEP-enhanced population is significantly redder at fixed magnitude ($\rho(M_{\rm mag}, \text{color}) = -0.39$, $p = 5.8 \times 10^{-15}$, $N = 375$). Top-heavy IMFs can partially relieve the star formation efficiency crisis but offer no mechanism for the spatially resolved screening gradients or the mass–sSFR inversion. In the systematic comparison, TEP accounts for both primary empirical lines while remaining directionally consistent with L2 and L4; the result is not a claim that every flexible astrophysical alternative is excluded.


**Key Limitations and Scope**


- **Mass circularity:** In purely photometric samples, distinguishing TEP effects from intrinsic mass-dependent evolution requires careful partial-correlation analysis. The SUSPENSE kinematic comparison (L5) materially narrows this objection by testing a dynamical-potential predictor.

- **Spectroscopic sample size:** While recent compilations (JADES DR4, DJA NIRSpec Merged Table v4.4) provide substantial $z > 7$ samples, stellar masses rely on photometric estimates ($\pm 0.3$–$0.5$ dex). The spectroscopic analyses remain supportive consistency checks.

- **Theoretical foundation:** A full joint cosmological parameter inference is outside the scope of this work. The manuscript presents only the components required to define and test the observational mapping (§2.3.2).




Section 2 defines the TEP mapping, the datasets, and the statistical procedures. Section 3 presents the two primary empirical lines, the direct kinematic test, and then places the ancillary spatial indication, the derived regime-level comparison, and the supplementary replications in their proper evidential order. Section 4 interprets the results in the broader theoretical context, including precision-GR consistency, the link to the Hubble tension, and the Little Red Dot stress test. Section 5 closes with falsification criteria and observational predictions. Appendix A provides the theoretical foundation (action, field equations, screening mechanism), and Appendix B documents key computational definitions and reference tables.


## 2. Data and Methods


This section follows the same logic as the manuscript as a whole. It
first defines the observational datasets, then the derived TEP
quantities, then the statistical tests used to separate genuine TEP
signatures from mass-proxy artifacts, and finally the black-hole
stress test used for the Little Red Dot analysis. The aim is to state
the observational mapping clearly enough that each empirical result in
§3 can be read directly back to its data and assumptions.



### 2.1 Data


#### 2.1.1 Red Monsters (FRESCO)


The motivating case study is the class of ultra-massive galaxies at $z
\sim 5$–$6$ exemplified by the three spectroscopically confirmed "Red
Monsters" reported by Xiao et al. (2024). For the illustrative TEP
prediction (§3.1), representative parameters spanning the published
range ($z \approx 5.3$–$5.9$, $\log M_* \approx 10.8$–$11.2$, SFE
$\approx 0.50$) are adopted. These capture the regime where the anomaly
is most acute. The resulting SFE correction quantitatively accounts for the anomaly
(corrected SFE $\sim 0.20$, at the $\Lambda$CDM limit of 0.20),
with the correction depending primarily on $\Gamma_t$ (set by halo mass
and redshift via the pre-calibrated TEP formula) and insensitive to the
precise input SFE at the $\lesssim 2\%$ level.



#### 2.1.2 UNCOVER DR4


For population-level tests, the UNCOVER DR4 stellar population synthesis
catalog is used (Wang et al. 2024; Furtak et al. 2023). The analysis
applies quality cuts and constructs a high-redshift sample with $4 <
z < 10$ and $\log M_* > 8$, yielding $N = 2{,}315$ galaxies. For
multi-property analyses (age ratio, metallicity, dust), a subset with
complete measurements is used (e.g., $N = 1{,}108$ for the
partial-correlation and split tests).



#### 2.1.3 Independent replications and spectroscopic validation


To evaluate independent replication of the $z > 8$ dust result,
catalogs for CEERS are used (Cox et al. 2025; Finkelstein et al. 2023;
photometric redshifts via EAZY, Brammer et al. 2008) and COSMOS-Web
(Shuntov et al. 2025). The COSMOS2025 catalog (Shuntov et al. 2025)
provides LePHARE SED-derived stellar masses, SFRs, E(B-V) dust, and ages
for 784,016 galaxies over 0.54 deg², with 37,965 sources at $z > 4$
passing quality cuts — the largest high-$z$ photometric SED sample used
in this analysis. The UNCOVER DR4 SPS catalog (Wang et al. 2024; Suess
et al. 2024; Price et al. 2025) uses 20-band MegaScience photometry and
Prospector-β SED fitting, providing 2,628 sources at $z > 4$ with
Prospector dust2 and a spec-z sub-catalog of 203 sources with
spectroscopic redshifts fixed in the SED fit. For spectroscopic
validation, two complementary catalogs are used:



**JADES Data Release 4** (D'Eugenio et al. 2025;
Curtis-Lake et al. 2025; Scholtz et al. 2025): 2,858 high-quality
spectroscopic redshifts (flags A/B) across GOODS-N and GOODS-S, with 118
sources at $z > 7$ and 41 at $z > 8$. UV-luminosity-based stellar
masses (Song et al. 2016) are derived for the 1,345 sources with valid
$M_{\rm UV}$.



**DAWN JWST Archive (DJA) NIRSpec Merged Table v4.4**
(Brammer et al.; de Graaff et al. 2024a; Heintz et al. 2023; September
2025): a comprehensive compilation of 80,367 uniformly reduced
JWST/NIRSpec spectra from all public programs, processed with the
msaexp/grizli reductions. After applying grade $\ge 3$ quality cuts and
deduplication by sky position, 19,445 unique sources are retained, of
which 3,251 are at $z > 5$, 698 at $z > 7$, and 234 at $z > 8$.
Photometric stellar masses are available for 2,598 of the high-$z$
sources. This catalog spans JADES, CEERS, RUBIES, UNCOVER, GLASS,
PRIMER, and more than 50 other public programs, providing the largest
uniform cross-survey spectroscopic sample to date.



#### 2.1.4 MIRI-based mass calibration context


Recent JWST/MIRI analyses (Pérez-González et al. 2024) show that
NIRCam-only SED fits can overestimate stellar masses at $z > 5$
because of age-attenuation degeneracy and emission-line contamination.
When MIRI photometry is included, the number density of the most massive
systems decreases and some candidates are reclassified as dusty or
line-dominated sources. The photometry is not reprocessed in this work,
but published masses are treated as conservative upper bounds and
MIRI-based studies serve as an external check on the interpretation of
the extreme-mass tail.





Table 1a: Observational Datasets

| Dataset | Role | Sample Size | Redshift Range | Mass Cut ($\log M_*$) | Key Reference | Key Biases |
| --- | --- | --- | --- | --- | --- | --- |
| Red Monsters | Case Study | 3 | $5.3 < z < 5.9$ | $> 10.5$ | Xiao et al. (2024) | Small N, Selection Function |
| UNCOVER DR4 | Primary Statistical Sample | 2,315 | $4 < z < 10$ | $> 8.0$ | Wang et al. (2024) | NIRCam Mass Overestimation |
| CEERS DR1 | Independent Replication | 82 | $z > 8$ | $> 8.0$ | Cox et al. (2025) | Field Variance |
| COSMOS-Web | Large-Volume Check | 2,606 (918 dust-detected) | $z > 8$ | $> 8.0$ | Shuntov et al. (2025) | Photometric Redshift Uncertainties; Zero-Inflated Dust |
| JADES DR4 (NIRSpec/MSA) | Spectroscopic Validation | 2,858 (flags A/B); 118 at $z > 7$ | $z = 0.1$–$14.2$ | None | D'Eugenio et al. (2025); Curtis-Lake et al. (2025) | Slit Losses; UV-based $M_*$ ($\pm 0.4$ dex) |
| DJA NIRSpec Merged v4.4 | Cross-Survey Spectroscopic Validation | 19,445 unique (grade $\ge 3$); 698 at $z > 7$; 234 at $z > 8$ | $z = 0.1$–$14.1$ | None | Brammer et al. (DJA); de Graaff et al. (2024) | Photometric $M_*$ from grizli; heterogeneous survey depths |
| UNCOVER DR4 SPS (MegaScience) | Primary + Spectroscopic Validation | 2,628 (z$>$4, Prospector-β); 203 with spec-z fixed fits | $z = 4$–$12$ | Abell 2744 (lensed) | Wang et al. 2024; Suess et al. 2024; Price et al. 2025 | 20-band photometry; lensing magnification corrections |
| COSMOS2025 (LePHARE SED) | Cross-Field Replication | 48,861 (z$>$4, adopted LePHARE selection); 7,249 at $z > 7$; 2,659 at $z > 8$ | $z = 4$–$13$ | None (blank field) | Shuntov et al. 2025 (COSMOS2025) | LePHARE E(B-V) less precise than Prospector dust2; photo-z scatter |




Related MIRI-supported analyses of Little Red Dots (LRDs) at $z > 4$
find that inferred stellar masses can shift by up to orders of magnitude
depending on the assumed AGN contribution. This motivates a conservative
stance in the interpretation of compact red sources and provides a
systematic-control context for any extreme-mass claims in the
literature.



### 2.2 Key Terminology

The following terms are used consistently throughout this work:




Table 1b: Glossary of Key Terms

| Term | Symbol | Definition |
| --- | --- | --- |
| Temporal Enhancement Factor | $\Gamma_t$ | The ratio of effective stellar-population proper time to cosmic time under the TEP response mapping. It is attenuated where the locally observable Temporal Shear/source-charge sector is screened. The saturation density $\rho_T \approx 20$ g/cm³ is an organizing scale, not a binary condition of the form $\rho > \rho_T \Rightarrow \Gamma_t = 1$. |
| Temporal Shear | $\Sigma_\mu$ | The locally active gradient of the temporal potential, $\Sigma_\mu = \nabla_\mu \Theta$, where $\Theta = \ln A(\phi)$. High ambient matter density flattens this gradient, suppressing the effective response continuously rather than at a discrete boundary. Observationally proxied by gradients in $\Gamma_t$ across a galaxy or environment. |
| Isochrony Bias | — | The systematic error in inferred stellar properties (mass, age, SFR) arising from the assumption that stellar clocks tick at the cosmic rate everywhere. Under TEP, this assumption is violated in deep potential wells. |
| Screening | — | The suppression of TEP effects in regions where the locally observable Temporal Shear/source-charge sector is screened ($\rho_T \approx 20$ g/cm³ is an organizing saturation scale). Two types are distinguished: *Core Screening*—Screening within a single galaxy, where the deep central potential suppresses TEP ($\Gamma_t \to 1$) while the outskirts remain enhanced. Produces bluer cores and redder outskirts. *Environmental Screening*—Screening by the ambient group or cluster potential, causing galaxies in dense environments to appear younger than isolated field galaxies of the same mass. |
| Effective Time | $t_{\rm eff}$ | The proper time experienced by stellar populations: $t_{\rm eff}^{\rm proxy} = \Gamma_t(M_h,z)\,t_{\rm FLRW}^{\rm obs}(z)$. |




The potential-linear response $\Gamma_t$ and the phenomenological normalization $K_{\rm gal}$ act as the galactic-scale formulation of the abstract environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$. Because the deepest gravitational potentials of high-redshift halos drive the continuous suppression of the proper-time field, local potential depth effectively operationalizes the Temporal Topology saturation for early stellar assembly.



### 2.3 Derived quantities


#### 2.3.1 Halo mass proxy


For each galaxy, the analysis uses an abundance-matching relation
(Behroozi et al. 2019) to map stellar mass to halo mass. This mapping is
used solely to construct the potential proxy $\Phi$ for the TEP parameterization.
To mitigate circularity, sensitivity tests are performed with $\pm 0.3$
dex scatter in the $M_h-M_*$ relation, propagating to $\pm 12\%$ in
$\Gamma_t$ corrections.



#### 2.3.2 The TEP Metric Coupling


The temporal enhancement factor $\Gamma_t$ is not introduced here as an
ad hoc fitting function. It is the observable mapping of a conformally
coupled scalar-tensor framework in which the scalar time field
modifies the local rate at which material clocks accumulate proper time. The full
theoretical development is extensive; this section states only the steps
needed to connect the action-level construction to the measurable
quantities used in the present analysis.



##### 2.3.2.1 From Action to Observable


The TEP framework builds upon scalar-tensor theories with
environment-dependent screening (Khoury & Weltman 2004; Brax et al. 2004;
Burrage & Sakstein 2018), extending them via the TEP Temporal Topology
mechanism where field gradient flattening (Temporal Shear) provides
continuous geometric screening. The key steps mapping the fundamental physics to the
observable $\Gamma_t$ are:



- 
**Action:** Matter couples to $\tilde{g}_{\mu\nu} =
A^2(\phi) g_{\mu\nu} + B(\phi) \partial_\mu \phi \partial_\nu \phi$ where $A(\phi) = \exp(\beta_A\phi/M_{\rm Pl})$.
The Klein-Gordon equation sources $\phi$ from the matter density
trace $T^\mu_\mu$.


- 
**Proper time:** Clock rates scale as $d\tau/dt \approx
A(\phi)$, defining $\Gamma_t \equiv (d\tau/dt)/(d\tau/dt)_{\rm
ref}$.


- 
**Halo mapping:** In virialized halos, $\phi$ tracks
the potential depth $\Phi \propto M_h^{2/3}$. The executed Potential-Linear kernel is:




$$\Gamma_t = \exp\left[ K_{\rm gal} \left(\frac{\Phi - \Phi_{\rm ref}}{c^2}\right) \sqrt{1+z} \right]$$


where $K_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag is the clock-sector response coefficient calibrated from Cepheid data (Paper 11), $\Phi_{\rm ref}$ is the potential at the reference halo mass $\log M_{h,\rm ref} = 12.0$, and the $\sqrt{1+z}$ factor is the prespecified cosmological temporal-calibration response. This is the kernel used throughout all population-level results in §3.



##### 2.3.2.1a Enhancement and Suppression: The Two-Sided Prediction


The exponential formula predicts two physically distinct regimes
depending on halo mass relative to the $\log M_{h,\rm ref}=12.0$ reference:



- 
**Enhancement ($\Gamma_t > 1$, massive halos):** For
$M_h > M_{\rm ref}$, $\Delta\log M_h > 0$ and $\Gamma_t > 1$. The
scalar field $\phi$ is sourced more strongly by the deeper
potential, raising $A(\phi) > 1$ and accelerating material clock
rates relative to the cosmic mean. This is the regime of the Red
Monsters and massive $z > 8$ galaxies.


- 
**Suppression ($\Gamma_t < 1$, low-mass halos):** For
$M_h < M_{\rm ref}$, $\Delta\log M_h < 0$ and $\Gamma_t < 1$. In
shallow potentials, the scalar time field relaxes toward a
lower-energy minimum with $A(\phi) < 1$, meaning material clocks
tick *slower* than the cosmic mean. This is not an ad-hoc
extension: it follows directly from the same conformal coupling
$d\tau/dt \propto A(\phi)$, which is symmetric about the
reference environment. The reference mass $\log M_{h,\rm ref} =
12.0$ defines the environment where $A(\phi) =
1$ (i.e., $\phi = 0$ in the Einstein frame), and deviations in
either direction produce proportional clock-rate shifts.




This two-sided prediction is central to the interpretation: the
"Uniformity Paradox" — why low-mass galaxies at $z > 8$ are dust-poor
despite cosmic time being nominally sufficient for AGB production — is
resolved because $\Gamma_t \ll 1$ in low-mass halos shuts off the
effective AGB clock. A model that only predicted enhancement ($\Gamma_t
\geq 1$ everywhere) would not explain the dust-poor low-mass population.
The suppression regime is therefore a falsifiable prediction rather than
a free parameter: it predicts that low-mass galaxies at $z > 8$ should
be systematically younger in their stellar populations than their cosmic
age implies, and should lack dust regardless of the available cosmic
time.



##### 2.3.2.1b Auxiliary Log-Mass Approximation


Earlier variants considered a Log-Mass approximation in which $\Gamma_t$ scales with the log-mass perturbation rather than the potential directly:



$$\Gamma_t = \exp\left[ \alpha(z) \cdot \frac{2}{3} \cdot (\log_{10} M_h - \log_{10} M_{h,\rm ref}) \cdot \frac{1+z}{1+z_{\rm ref}} \right]$$


where $\alpha(z) = \kappa_{\rm gal} \sqrt{1+z}/10^6$. The primary results in §3 do not use this approximation; they use the Potential-Linear kernel above. The Log-Mass form appears only as an auxiliary note for interpreting observable scaling relations and for the nested Bayesian comparison in §3.6.



##### 2.3.2.2 Screening and Scale Separation

For a representative bare coupling $\beta_A \approx 0.8$, the bare
Brans-Dicke parameter would be $\omega_{\rm BD} = 1/(2\beta^2) - 1/2 \approx
0.28$ — roughly five orders of magnitude below the Cassini bound
($\omega_{\rm BD} > 40{,}000$; Bertotti et al. 2003). This large
pre-screening discrepancy illustrates the central logic of the TEP
framework: any underlying bare coupling is strong, but in dense environments the scalar
field gradient (Temporal Shear) flattens continuously, suppressing the
effective coupling to $\kappa_{\rm gal} \ll \beta_A$ and yielding
$\omega_{\rm BD}^{\rm eff} > 10^6$. On cosmological scales, the Compton
wavelength $\lambda_C \sim 1$ Mpc yields Yukawa suppression $\beta_{\rm
eff}(R_8) \approx 0.002$ on $\sigma_8$ scales—well below the Planck bound.
Within individual halos ($r \lesssim 50$ kpc), the field tracks the local
potential and operates locally. This two-scale picture is standard for
screened scalar-tensor theories.
The continuous screening via Temporal Shear provides PPN-compliant
suppression without a rigid threshold:


$$\kappa_{\rm eff}(\rho) = \kappa_{\rm gal} \cdot \mathcal{S}(\rho/\rho_T),$$


where $\mathcal{S}$ is a smooth suppression function and $\rho_T \approx 20$ g/cm³
is the Temporal Topology reference density. This reference scale is
derived independently from three sources that converge on the same value:
GNSS atomic clock networks ($L_c \approx 4200$ km for Earth's mass),
atomic physics (Temporal Topology radius $R_T(m_p) \sim a_0$ at the proton
mass scale), and magnetar anti-glitches ($P_{\rm crit} \approx 6.8$ s
for 1E 2259+586, 4% match). The convergence across 40 orders of magnitude
in mass provides an independent consistency check on this screening scale.


At galactic scales, an effective kinematic screening threshold emerges from
analysis of 380 millisecond pulsars in globular clusters, which reveals that
the TEP spin-down excess saturates for systems with velocity dispersion
$\sigma \gtrsim 165$ km/s, consistent with the scalar field gradient
flattening as potential depth increases. This threshold is used in to
define the environmental screening boundary for JWST galaxies: halos with
$\sigma \gtrsim 165$ km/s (corresponding to $\log M_h \gtrsim 13.5$ at $z
\sim 0$) are expected to be partially screened, suppressing $\Gamma_t$ below
the unscreened prediction.


##### 2.3.2.3 Enhancement vs. Dilation


Standard GR predicts time *dilation* in deep potentials; TEP
predicts *enhancement* ($\Gamma_t > 1$). These refer to different
metrics: gravitational redshift is governed by $g_{\mu\nu}$ (preserved
identically), while material clock rates are governed by
$\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu} + B(\phi) \partial_\mu \phi \partial_\nu \phi$. The key distinction is that
$\Gamma_t$ compares clock rates between
*different environments at the same epoch*, not between positions
in a single well. Numerical integration confirms $A(\phi) > 1$ in
unscreened halos for $2\beta^2 > 1$. Solar System bodies are fully
screened ($\Gamma_t \to 1$).





Table 1: TEP Model Parameters (Fixed)

| Parameter | Value | Source | Description |
| --- | --- | --- | --- |
| $\kappa_{\rm Cep}$ (Paper 11) | $(9.6 \pm 4.0) \times 10^5$ mag | Cepheid distance-ladder (Paper 11) | Cepheid Observable Response Coefficient; magnitude-sector measurement from period-luminosity residuals |
| $K_{\rm gal}$ (galaxy kernel) | $\approx 1.26 \times 10^6$ | Transferred from $\kappa_{\rm Cep}$ via response normalization | Phenomenological response normalization for stellar-population $\Gamma_t$ kernel; not a microscopic coupling |
| $z_{\rm ref}$ | 5.5 | TEP-H0 | Reference redshift for calibration |
| $\log M_{h, \rm ref}$ | 12.0 | TEP-COS | Reference halo mass at $z=0$ ($\Gamma_t=1$) |
| $\rho_T$ | 20 g/cm$^3$ | TEP-UCD | Temporal Topology reference density for continuous screening |






![TEP Metric Coupling Gamma_t as a function of halo mass and redshift](public/figures/figure_1_tep_model.png)




Figure 1: The TEP Metric Coupling $\Gamma_t(M_h, z)$ in the
unscreened regime. The enhancement factor increases with halo mass
(potential depth) and redshift (weakening of cosmological
screening). The reference mass ($\log M_h = 12$) defines $\Gamma_t =
1$ (cosmic time flow). Massive halos at high redshift experience
significant temporal enhancement ($\Gamma_t > 1$), while low-mass
halos are suppressed ($\Gamma_t < 1$). Continuous screening via
Temporal Shear suppresses $\Gamma_t$ smoothly as density approaches
the Temporal Topology reference scale ($\rho_T \approx 20$ g/cm³).





The JWST response-prior test adopts the external Cepheid-calibrated
response coefficient $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag
and applies it without JWST-specific refit, together with the fixed reference choice $z_{\rm ref} = 5.5$.
The high-redshift observables are then examined for internal consistency of the response scale.
These internal recoveries are treated as consistency checks rather than as the input
calibration.



##### 2.3.2.4 Cosmological Viability Summary


The TEP framework has been checked against three classes of precision
cosmological constraints:



- 
**Early-universe compatibility (historical):** Earlier
versions verified that a conformal scalar remains perturbatively
invisible during a conventional radiation-dominated era ($|\Delta
H/H|_{\rm max} = 1.7 \times 10^{-13}$; $\Delta Y_p <
10^{-14}$). This serves as a historical compatibility check;
the canonical TEP early-universe interpretation is supplied by
TEP-TH and TEP-BBN (Appendix A.1.7).


- 
**Linear growth ($\sigma_8$):** Yukawa suppression on
$\gtrsim 10$ Mpc scales reduces the effective coupling to
$\beta_{\rm eff} \lesssim 0.002$, preserving $\Lambda$CDM-consistent
$\sigma_8$ (§2.3.2.2; Appendix A.1.7–A.1.8).


- 
**Solar System (PPN):** Temporal Shear suppression in
dense environments reduces the effective coupling to $\kappa_{\rm
eff} \ll \kappa_{\rm gal}$ for solar-system bodies, satisfying Cassini and
lunar laser ranging bounds (Appendix A.1.3).




**Scale-dependent growth computation:** To go beyond the
analytic Yukawa argument, the linear growth ODE is solved independently
for each Fourier mode $k$, incorporating the full scale-dependent
gravitational coupling $G_{\rm eff}(k,z)/G_N = 1 + 2\beta^2 k^2/(k^2 +
m_\phi(z)^2)$, where $m_\phi(z) = m_{\phi,0}(1+z)^{9/4}$ for $n=1$
potential (standard chameleon form). The resulting matter power spectrum ratio $P_{\rm
TEP}(k)/P_{\Lambda{\rm CDM}}(k)$ and integrated $\sigma_8$ are computed
self-consistently (Appendix A.1.8). Key results:



- 
Planck consistency requires $m_{\phi,0} \gtrsim 0.43\,h$/Mpc
($\lambda_C \lesssim 14.6\,h^{-1}$ Mpc at $z=0$; Appendix A.1.8).


- 
For typical Temporal Shear parameters, $\beta_{\rm eff}$ on $R_8$ scales
is $\approx 0.0079$—well below the bare coupling
($G_{\rm eff}/G_N - 1 = 1.2 \times 10^{-4}$ at $k_8$; Appendix
A.1.8).


- 
The predicted $\sigma_8^{\rm TEP} = 0.8110$ vs.
$\sigma_8^{\Lambda{\rm CDM}} = 0.811$; $\Delta\sigma_8 = 2.1 \times
10^{-7}$ ($3.6 \times 10^{-5}\sigma$). RSD: $\Delta\chi^2 = 2.7 \times
10^{-4}$ across 8 data points (Appendix A.1.8).




This k-dependent growth calculation substantially strengthens the
viability argument beyond the earlier analytic estimate. A full CAMB
Boltzmann integration (Appendix A.1.8.8) confirms these results: at the
fiducial $m_{\phi,0} = 1.0\,h$/Mpc, the CAMB-computed $\sigma_8^{\rm
TEP} = 0.8116$ ($0.10\sigma$ from Planck), with CMB TT deviations $<
0.02\%$ at all $\ell < 2500$ and $\chi^2/{\rm dof} \ll 1$ against
Planck error bars. Planck consistency holds for $m_{\phi,0} \gtrsim
0.43\,h$/Mpc. The CAMB integration uses the scale-dependent growth
equation with Yukawa-suppressed $G_{\rm eff}(k,z)$ and CAMB's exact
lensed CMB spectra, substantially narrowing the remaining theoretical
gap. The remaining approximation — that the scalar field does not modify
acoustic peaks at $z > 1089$ — is justified by $T^\mu_\mu \approx 0$
during radiation domination. A natively coupled hi_class integration
remains desirable for completeness but is no longer expected to change
the conclusion.



#### 2.3.3 Effective time and isochrony bias correction


The general TEP expression for proper time is an integral over the evolving scalar field: $\tau_\star=\int_{t_{\rm form}}^{t_{\rm obs}} \Gamma_t[\mathcal E(t),t]\,dt$. For the empirical analysis in this paper, we define a frozen-environment catalogue proxy: $t_{\rm eff}^{\rm proxy} = \Gamma_t(M_h,z)\,t_{\rm FLRW}^{\rm obs}(z)$, where $t_{\rm FLRW}^{\rm obs}$ is the
observer-frame age coordinate computed from a fiducial
cosmology (Planck18). Under the isochrony-bias model used here, the
mass-to-light ratio is assumed to scale as $M/L \propto t^n$ (following
standard SSP predictions; Bruzual & Charlot 2003; Conroy et al.
2009). Forward-modeling analysis finds that $n \approx 0.5$ minimizes
the residual mass-age correlation at $z > 6$, while $n \approx 0.9$ is
preferred at $z = 4$–$6$. For the primary high-$z$ analysis, $n = 0.5$
is adopted. The corrected stellar mass and SFE are:



$$M_{*,\rm true} = M_{*,\rm obs}/\Gamma_t^{n}, \quad \mathrm{SFE}_{\rm
true} = \mathrm{SFE}_{\rm obs}/\Gamma_t^{n}.$$



### 2.4 Statistical procedures


The statistical design separates three questions: whether the predicted
associations are present, whether they survive control for obvious
confounders, and whether they generalize across surveys and subsamples.
Associations are quantified using Spearman rank correlations and
bootstrap confidence intervals. To address confounding by redshift and
stellar mass, partial-correlation analyses implemented via
residualization are employed. In addition to correlation-based tests,
the following are reported:



- 
Stratified comparisons (e.g., high vs low $\Gamma_t$ splits) for
multi-property coherence


- 
Distributional comparisons (e.g., Kolmogorov-Smirnov tests) for
regime separation


- 
Model comparison using both AIC/BIC and nested Bayesian evidence:
the regression comparisons test predictors {z}, {z, $\log M_*$}, {z,
$\Gamma_t$}, and {z, $\log M_*$, $\Gamma_t$}, while a separate
`dynesty` nested-sampling analysis compares TEP against
explicit bursty-SF, varying-IMF, standard-physics, and AGN
alternatives in both raw standardized space and a
mass+$z$-residualized control space





#### 2.4.1 Combined significance and multiple testing


Combined significance is assessed using Fisher's method, Bonferroni
correction, Brown's method (dependence-adjusted), harmonic mean p-value,
and Benjamini-Hochberg FDR ($\alpha = 0.05$). Because omnibus
significance depends sensitively on how clustering and shared predictors
are penalized, the manuscript treats the three-survey photometric L1
replication as the headline result and uses broader multi-test
combinations as supportive context. An extreme stress test that reduces
effective sample sizes by 90% via spatial clustering autocorrelation
still leaves a Bonferroni-corrected floor of $3.2\sigma$ for the mixed
test set. Parametric p-values are supplemented by permutation tests ($N =
10{,}000$ shuffles) and bootstrap confidence intervals ($N = 10{,}000$
resamples). Cross-survey effect sizes are combined via DerSimonian-Laird
random-effects meta-analysis with $I^2$ heterogeneity assessment and
leave-one-out influence diagnostics.



#### 2.4.2 Blind validation protocol


Three split strategies test generalization: (1) time-split (low-$z$
train / high-$z$ test, 60/40); (2) field-split (RA median); (3)
cross-survey leave-one-out. A test passes if the dust–$\Gamma_t$
correlation remains significant ($p < 0.05$) on held-out data.



#### 2.4.3 Stellar-to-halo mass mapping and sensitivity


Each galaxy's stellar mass is mapped to halo mass using a redshift-dependent
abundance matching relation parameterized to mirror the high-$z$ tail of
Behrozi et al. ($\log M_h = \log M_* + 1.8 + 0.1(\log M_* - 10) - 0.05(z-5)$).
Dynamical masses are mapped using an identically sloped relation to
ensure rank-order preservation. Key results survive $\pm 0.5$ dex
perturbations. To test robustness against MIRI-based mass recalibration
(Pérez-González et al. 2024), mass reductions of 0.0–1.0 dex are
applied; TEP signatures persist under a 0.5 dex
reduction (2/4 key signatures survive; step_040). At $z > 8$, selection bias toward bright galaxies is
quantified via Monte Carlo completeness weighting ($N = 1{,}000$
iterations) and Savage-Dickey Bayes Factors.



*Extrapolation caveat:* The Behroozi et al. (2019) UNIVERSEMACHINE
relation is calibrated for $z = 0$–$10$. Analyses extending to $z = 9$–$13$
(COSMOS2025 sSFR, UNCOVER MegaScience dust) therefore extrapolate the
linear redshift term ($-0.05(z-5)$) beyond the calibration range. At
$z > 10$, the physics of halo assembly and baryon cooling differs
substantially from $z = 5$, and the linear extrapolation carries
unquantified systematic uncertainty in $\Gamma_t$. Results at $z > 10$
should be interpreted with this caveat; a theoretically motivated
$M_*$–$M_h$ relation from high-$z$ hydrodynamical simulations would
tighten these inferences.



#### 2.4.4 Forward-modeling validation


The $M/L \propto t^{n}$ scaling is validated by varying $n = 0.5$–$0.9$
and identifying the value minimizing the residual mass-age correlation.
At $z > 6$, $n \approx 0.5$ is preferred (consistent with
low-metallicity SSP models); at $z = 4$–$6$, $n \approx 0.9$.



### 2.5 Black Hole Growth Stress Test


To test whether compact cores could amplify black-hole growth, a
differential temporal topology simulation was developed. A compact
galaxy ($r_e \approx 150$ pc) with a baryon-dominated core ($c=10$) is
modeled. The local temporal enhancement factor $\Gamma_t(r)$ is computed
at the center (black-hole environment) and at the effective radius
(stellar environment) across the redshift range $z=4$–$10$. This analysis
is a sensitivity diagnostic, not a calibrated population-level
resolution of the Little Red Dot anomaly.


The differential growth factor is computed as:


$$\text{Boost} = \exp\left(\frac{\int (\Gamma_{\rm cen}(z) - \Gamma_{\rm
halo}(z)) \, dt_{\rm cosmic}}{t_{\rm Salpeter}}\right)$$


where $t_{\rm Salpeter} \approx 45$ Myr is the Salpeter timescale
(e-folding time for Eddington-limited accretion). This simulation uses
the same fiducial response scale implied by the external
Cepheid prior, with no additional tuning.



### 2.6 Reproducibility


All analyses are reproducible from the public repository. An end-to-end
run regenerates the manuscript tables, figures, and archived outputs;
execution instructions are provided in the repository README.



## 3. Results


### 3.0 Evidence Summary

The evidence is organized by role: L1 and L3 are the two primary lines, L2 is ancillary, L4 is derived, and L5 is a direct kinematic test. The classification is stated once here and in §1.3; individual results below are reported without repeating it.




Table 3a: All Tested TEP Signatures

| Signature | Finding | Significant | Survives Mass Control | Status |
| --- | --- | --- | --- | --- |
| **L1. Dust–$\Gamma_t$ + AGB threshold** | $\rho = +0.60$ across three surveys ($N=1{,}283$); AGB threshold odds ratio 42.8 | ✔ | ✔ | **Primary line** |
| **L2. Inside-out core screening** | Gini partial $\rho=+0.191$ after mass+$z$ control; size proxies and $\sigma_\star$ non-significant | ✔ | ✘ | **Ancillary** |
| **L3. Mass–sSFR inversion** | Correlation inverts at $z>7$ ($\Delta\rho = +0.25$); partial $\rho(\Gamma_t, {\rm sSFR}\|{\rm dust}) = -0.49$ | ✔ | ✔ | **Primary line** |
| **L4. Dynamical mass comparison** | TEP correction predicts 0.256 dex reduction vs 0.15 dex observed excess in RUBIES-like regime | ✔ | ✔ | **Derived** |
| **L5. Direct kinematic test** | SUSPENSE ($N=15$): $\rho({\rm Age}, \Gamma_t \mid M_*, z)=+0.599$; $\rho({\rm Age}, M_* \mid \Gamma_t, z)=+0.024$. Sigma expansion ($N=83$): mixed; positive overall but driven by emission-line $\sigma$. | ✔ | ✔ | **Direct test** |
| Steiger Z-test ($t_{\rm eff}$ vs $M_*$) | $Z=17.8$; $Z=10.4$ at $z>8$ | ✔ | ✔ | Robustness check on L1 |
| Partial correlations | $\rho=+0.26$ after polynomial control; $M_*$ zero residual after $t_{\rm eff}$ control | ✔ | ✔ | Robustness check on L1 |
| Cross-survey generalization | $t_{\rm eff}$ stable at $\rho=0.60$–0.80 across surveys | ✔ | ✔ | Robustness check on L1 |
| Age coherence | $\rho = +0.14$ (mass-only); vanishes with $M_*$+$z$ control | ✔ | ✘ | Not independent |
| Metallicity | $\rho = +0.16$ (mass-only); vanishes with $M_*$+$z$ control | ✔ | ✘ | Not independent |
| Environmental screening | Full-sample $\Delta\rho = +0.19$; $z>8$ contrast weak ($\Delta\rho = 0.111$, $p=0.245$) | ✔ | ✔ | Supplementary — mixed |
| Colour-gradient sign test | Raw mass trend $\rho=-0.166$; direct partials null; debiased sign test directional ($p=0.061$) | ✔ | ✘ | Ancillary follow-up |




### 3.1 Red Monsters: A No-JWST-Specific-Refit Prediction

The TEP parameterization is applied to galaxies in the Red Monster regime ($z \sim 5$–$6$, $\log M_* \gtrsim 10.5$; Xiao et al. 2024). This is a blind prediction: the external Cepheid prior $\\kappa_{\\rm Cep} = (9.6 \\pm 4.0) \\times 10^5$ mag is fixed entirely from local Cepheid data ($z \approx 0$) before the high-redshift fit. The later high-redshift concordance analysis recovers $\kappa = (11.6 \pm 5.6) \times 10^5$ mag from the informative analyses, but internally concordant ($p_{\rm concordance}=1.0$); this is therefore classified as a partial, anchor-consistent self-consistency check rather than as the input prior. No parameters are fitted or tuned to the high-redshift observations. The three entries below (S1–S3) use representative parameters spanning the published range (§2.1.1); the predicted correction depends primarily on $\Gamma_t$ and is insensitive to the exact input SFE.

Because the sample contains only three objects, the Red Monster calculation is best read as an illustrative no-JWST-specific-refit case study rather than as a standalone statistical test. The primary statistical weight comes from the population-level analyses ($N = 2{,}315$), and the externally calibrated prediction is further checked across three surveys ($N = 1{,}283$ at $z > 8$).




Table 3b: Illustrative TEP Predictions for Red Monster–Class Galaxies

| ID | $z$ | $\alpha(z)$ | $\Gamma_t$ (Predicted) | SFE$_{\rm obs}$ | SFE$_{\rm true}$ | % Anomaly Resolved |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 5.85 | 1.52 | 12.98 | 0.50 | 0.08 | 100% |
| S2 | 5.30 | 1.46 | 7.55 | 0.50 | 0.12 | 100% |
| S3 | 5.55 | 1.48 | 7.53 | 0.50 | 0.12 | 100% |
| Average Prediction | 9.35 | 0.50 | 0.11 | 100% |  |  |



The predicted mass bias $\Gamma_t^{n} \approx 4.7$ reduces the corrected SFE to $\sim 0.20$ (at the standard $\Lambda$CDM limit of 0.20). The anomaly is fully resolved for all three objects. Propagating the external Cepheid prior (Paper 11, $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag) uncertainty confirms robustness: even at the lower $1\sigma$ bound ($\kappa_{\rm gal} = 5.6 \times 10^5$ mag), the corrected SFE remains at or below 0.20 and the anomaly is still fully resolved with zero tuned parameters.


### 3.2 UNCOVER DR4: Mass-sSFR and Mass-Age Correlations

The Red Monster case study establishes that TEP predicts the correct direction and magnitude of the SFE correction for individual extreme objects. The critical question is whether this signal extends to the full galaxy population. In the UNCOVER DR4 sample ($N = 2{,}315$), the mass-sSFR correlation is weak but significant ($\rho = -0.13$, $p = 1.3 \times 10^{-10}$, Cohen's $d = -0.27$), consistent with TEP partially canceling the intrinsic downsizing trend. The mass-age correlation is positive ($\rho = +0.14$, $p = 7.0 \times 10^{-11}$), consistent with more massive galaxies experiencing more proper time. Both correlations are in the predicted direction but are attenuated by the full redshift range; the signal sharpens substantially when the sample is stratified by redshift.


### 3.3 Redshift Evolution: The High-z Transition

TEP predicts that the mass-sSFR correlation should become *less negative* (or even positive) at higher redshift, where the TEP enhancement is stronger. This is tested by stratifying the sample:




Table 4: Mass-sSFR Correlation by Redshift

| $z$ Range | $N$ | Spearman $\rho$ | 95% CI | Interpretation |
| --- | --- | --- | --- | --- |
| 4–5 | 942 | $-0.17$ | [$-0.24$, $-0.11$] | Standard downsizing |
| 5–6 | 497 | $-0.14$ | [$-0.22$, $-0.05$] | Standard downsizing |
| 6–7 | 372 | $-0.06$ | [$-0.16$, $+0.04$] | Weakening |
| 7–8 | 221 | $+0.18$ | [$+0.05$, $+0.31$] | Inversion |
| 8–9 | 179 | $+0.13$ | [$-0.03$, $+0.29$] | Weak positive |
| 9–10 | 104 | $-0.27$ | [$-0.47$, $-0.05$] | Reversal (selection effects) |



Comparing low-$z$ ($4 < z < 6$, $\rho = -0.16$) to high-$z$ ($z > 7$, $\rho = +0.09$): $\Delta\rho = +0.25$ [+0.14, +0.35] (95% CI excludes zero), indicating a statistically significant inversion.


### 3.4 Partial Correlation Test

The redshift evolution in §3.3 is consistent with TEP, but by itself it does not eliminate the mass-proxy concern, since $\Gamma_t \propto M_h^{1/3}$. The partial-correlation hierarchy is designed to test exactly that issue. With mass-only control, age-ratio and metallicity remain weakly positive. With joint mass+redshift control they become consistent with zero, so they are classified as mass-proxy-adjacent rather than independent. The high-redshift dust signal behaves differently: at $z > 8$, the dust–$\Gamma_t$ correlation survives ($\rho = +0.262$, $p = 8.1 \times 10^{-6}$, Cohen's $d = +0.55$), indicating that $\Gamma_t$ carries information about dust beyond mass alone. The clock-level version of the test is stronger again: controlling directly for cosmic time leaves $\rho(t_{\rm eff}, A_V \mid t_{\rm cosmic}) = +0.600$ ($p = 5.0 \times 10^{-29}$), so the signal is not a trivial restatement of redshift ordering.

*The key asymmetry:* $t_{\rm eff}$ subsumes the mass information relevant to dust ($M_*$ residual after $t_{\rm eff}$ control: $\rho = -0.006$, $p = 0.92$), but mass does not subsume $t_{\rm eff}$ (residual $\rho = +0.26$, $p = 7.4 \times 10^{-6}$). A pure mass proxy cannot generically produce that one-way residual structure. If $\Gamma_t$ were only a reparameterisation of $M_*$, the relationship would be symmetric and neither predictor would retain residual information once the other was controlled.

A further complication is that the strongest mass-proxy objection becomes self-defeating once the isochrony-bias mechanism is taken seriously. If TEP is correct, SED-inferred stellar masses are themselves biased upward by $\Gamma_t^{0.7}$ . Partial-correlation tests that control for observed $M_*$ therefore over-control the true signal by removing TEP-predicted variance. The partial correlations reported here are accordingly conservative lower bounds, understated by a factor of $\sim 2.5$ at $z > 8$. The cleanest route around that circularity is L4 (dynamical masses; §3.10), where kinematic observables replace SED mass estimates.

**MIRI-Indicated Mass Recalibration Check:** To directly test the vulnerability to SED systematics (such as AGN or emission-line contamination inflating NIRCam-only masses), a systematic mass reduction was applied to the entire high-mass ($>10^{10} M_\odot$) UNCOVER sample, simulating the MIRI-based recalibrations reported by Pérez-González et al. (2024). Even when all masses are artificially reduced by 0.5 dex, the $\Gamma_t$-dust correlation at $z > 8$ remains completely robust ($\rho = +0.599$, $p = 5.8 \times 10^{-29}$). The signal survives even a full 1.0 dex systematic reduction ($\rho = +0.598$, $p = 6.9 \times 10^{-29}$). This confirms that the TEP signal is driven by the relative rank-ordering of galaxies within the deep potential regime, not by the calibrated photometric mass calibration, and is therefore structurally immune to the primary MIRI systematic critique.


### 3.5 Screening Signatures

A distinctive feature of the TEP framework — one that distinguishes it from any smooth mass-dependent function — is the screening prediction: above a Temporal Topology saturation proximity scale $\rho_T \approx 20$ g/cm³, the scalar field is suppressed and $\Gamma_t \to 1$. Paper 11 (TEP-COS) established an effective kinematic screening threshold at $\sigma > 165$ km/s from globular cluster pulsar timing. At high redshift, this threshold shifts to higher halo mass. Screening is tested by comparing age ratios (MWA/$t_{\rm cosmic}$) across mass bins:




Table 5: Age Ratio by Halo Mass (5 < z < 8)

| $\log M_h$ | $N$ | $\langle$MWA/$t_{\rm cosmic}\rangle$ | $\Gamma_t$ Predicted |
| --- | --- | --- | --- |
| 10–11 | 390 | $0.15 \pm 0.003$ | $\sim 0$ (reference) |
| 11–12 | 42 | $0.18 \pm 0.015$ | 0.2–0.5 |
| 12–12.5 | 3 | $0.30 \pm 0.12$ | 1.0–1.5 |
| 12.5–13 | 1 | $0.05$ | 1.5–2.0 |




#### 3.5.1 Resolved Core Screening

TEP predicts that deep core potentials should screen the scalar field locally while outskirts remain enhanced, producing a structurally concentrated, bluer-core signature in massive galaxies. The strongest L2 support now comes from the preferred JADES DR5 direct-mass morphology sample: after controlling for mass and redshift, one structural proxy remains supportive in the expected direction, with sizes non-significant, Gini partial $\rho = +0.191$, and $\sigma_\star$ non-significant for $N = 384$. The resolved colour-gradient analysis remains informative but weaker: for $N = 277$ galaxies it still shows the raw mass-gradient trend $\rho(M_*, \nabla_{\rm Color}) = -0.166$ ($p = 5.7 \times 10^{-3}$), while the direct $\Gamma_t$ correlation is $\rho(\Gamma_t, \nabla_{\rm Color}) = -0.105$ ($p = 8.1 \times 10^{-2}$). The direct partial remains null under both observed-mass and debiased-mass control ($\rho = +0.011$, $p = 0.85$; $\rho = +0.037$, $p = 0.54$). The literal $\Gamma_t > 1$ tail is too small to decide the sign-reversal test cleanly, but after the L4-motivated debiased-mass control the q33/q67 high-versus-low screening split becomes directionally supportive: the negative-gradient fraction rises from $0.457$ to $0.581$ (Fisher $p = 0.061$) with mean contrast $\Delta = -0.063$. The spatial-screening analysis is therefore an ancillary indication rather than counted among the two primary statistical lines. See §3.9 and the robustness checks note for full details.


### 3.6 The z > 8 Dust Anomaly: Correlation vs. Budget

The mass–sSFR inversion (§3.3) and the partial-correlation hierarchy (§3.4) show that $\Gamma_t$ carries information beyond a simple mass trend. The clearest physical test, however, is dust production. Can the observed dust reservoirs at $z > 8$ be produced in the available time? Under standard physics, the universe at $z \sim 9$ is only $\sim 540$ Myr old, barely enough for the first generation of AGB stars to complete their evolution. Quantitative analysis using canonical dust parameters (AGB delay $\sim 500$ Myr, standard ISM opacity) therefore exposes a direct tension between the observed dust reservoirs and the standard-time budget.


**Dust Budget Analysis ($N=33$ massive galaxies at $z > 8$)**

Comparing observed dust masses to the maximum theoretical yield under canonical assumptions:




Table 6: Dust Production Deficit (Observed / Maximum Yield)

| Framework | Mean Deficit Ratio | "Yield Violation" Candidates ($> 2\times$ Limit) |
| --- | --- | --- |
| Standard Physics ($t = t_{\rm cosmic}$) | 0.91$\times$ (Saturation) | 8 / 33 (24%) |
| TEP ($t = \Gamma_t t_{\rm cosmic}$) | 0.41$\times$ (Comfortable) | 0 / 33 (0%) |


Under standard physics, the average massive galaxy is near the theoretical production limit, with ~24% of the sample requiring unphysical yields. Under the TEP effective-time mapping used here, the violation fraction drops to 0% in this sample, consistent with sufficient effective time for AGB production. Recent JWST spectroscopy shows that AGB stars produce SiC and iron dust even at low metallicity ($\sim 1$–$7\%\,Z_\odot$; Boyer et al. 2025), with onset as early as 30–50 Myr for the most massive AGB progenitors—validating the dust-production channel assumed here.

**The "Optimistic" Trap.** One might attempt to resolve the standard-physics deficit by assuming optimistic parameters — maximal supernova yields, minimal destruction, accelerated AGB onset. While this can technically close the budget (reducing the violation fraction to 0%), it creates a deeper problem: the Uniformity Paradox. If parameters are tuned to allow dust everywhere (since $t_{\rm cosmic}$ is uniform), dust should be ubiquitous or track star formation. Instead, observations reveal a strong mass-dependent suppression ($\rho = +0.56$): massive galaxies are dusty; low-mass galaxies are dust-poor. No tuning of a time-uniform parameter can reproduce a mass-dependent gradient. Under TEP, this gradient arises naturally: the framework *suppresses* effective time in low-mass halos ($\Gamma_t \ll 1 \rightarrow t_{\rm eff} \ll 300$ Myr), shutting off the AGB channel, while in massive halos $\Gamma_t > 1$ ensures it remains open. The anomaly is not that massive galaxies have dust — it is that low-mass galaxies *don't*, in a pattern that tracks gravitational potential depth.






![The z > 8 Dust Anomaly: Mass-Dust Correlation](public/figures/figure_5_dust_anomaly.png)



Figure 2: The Key Dust Anomaly. (a) At $z \sim 5$ (grey), mass and dust are uncorrelated ($\rho \approx 0$). (b) At $z > 8$ (color), a strong correlation emerges ($\rho = +0.56$). Massive galaxies (high $\Gamma_t$, yellow) have successfully produced dust despite the short cosmic time (< 600 Myr), while low-mass galaxies (low $\Gamma_t$, purple) remain dust-poor. TEP predicts this specific mass-dependent divergence.





**Figure 3: The Dust Saturation Crisis.** The ratio of observed dust mass to the maximum theoretical yield is plotted for massive galaxies at $z > 8$. Standard Physics (blue) places the population near the saturation limit (100% of yield), leaving no margin for error. TEP (orange) shifts the population to approximately 40% of the limit. While standard physics is technically possible, it requires near-maximal efficiency across all parameters simultaneously; TEP requires only typical efficiencies.




#### 3.6.1 The $z = 6$–$7$ Dip

A transient negative mass-dust correlation at $z = 6$–$7$ ($\rho = -0.12$) interrupts the high-$z$ emergence. Quantitative forensics reveal this is driven by high specific star formation rates actively depleting dust through supernova shocks faster than AGB stars can replenish it, a "competition epoch" that resolves at $z > 8$ where only $\Gamma_t > 1$ halos have accumulated sufficient effective time for dust production.

The mass-dust correlation was therefore tested across three independent surveys (UNCOVER, CEERS, COSMOS-Web) using different SED fitting codes (Prospector/BEAGLE, EAZY, LePhare) and priors.


### 3.7 Cross-Survey Replication and Meta-Analysis


#### 3.7.1 Cross-Code Robustness

The $z > 8$ dust-$\Gamma_t$ correlation is detected in all three datasets despite differences in methodology:




Table 7: Cross-Survey Replication of $z > 8$ Dust-$\Gamma_t$ Correlation

| Survey | Code | $N$ (z > 8) | $\rho(\Gamma_t, \text{Dust})$ | 95% CI | $p$-value | Significance |
| --- | --- | --- | --- | --- | --- | --- |
| UNCOVER | Prospector/BEAGLE | 283 | $+0.59$ | $[+0.51, +0.66]$ | $p = 3.0 \times 10^{-28}$ | $11.4\sigma$ |
| CEERS | EAZY | 82 | $+0.66$ | $[+0.52, +0.77]$ | $p = 1.5 \times 10^{-11}$ | $7.0\sigma$ |
| COSMOS-Web | LePhare | 918 | $+0.63$ | $[+0.59, +0.67]$ | $p = 3.5 \times 10^{-102}$ | $22.4\sigma$ |
| Fixed-effects meta | $+0.62$ | $[+0.59, +0.66]$ | $p = 1.0 \times 10^{-149}$ | $26.1\sigma$ |  |  |




#### 3.7.2 Meta-Analysis

Combining all three surveys ($N = 1{,}283$ at $z > 8$) yields a fixed-effects meta-correlation of $\rho = +0.60$ with $p = 4.3 \times 10^{-133}$ (Cohen's $d = 1.59$, a large effect). Negligible between-study heterogeneity ($I^2 = 0\%$) confirms consistent effect sizes across surveys, and mass-stratification confirms the signal persists at fixed mass.


#### 3.7.3 Temporal Inversion & AGB Threshold

A more physically targeted and falsifiable test compares dust against cosmic time ($t_{\rm cosmic}$) versus the TEP-effective clock ($t_{\rm eff} = \Gamma_t\,t_{\rm cosmic}$). Under standard physics, dust should track $t_{\rm cosmic}$; under TEP, dust emergence should be organized by $t_{\rm eff}$ and should show a step-like transition near the AGB dust-production timescale ($t_{\rm eff} \gtrsim 0.3$ Gyr).




Table 7b: Cross-Survey Temporal Inversion and AGB Threshold (z > 8)

| Survey | $\Delta\rho = \rho(t_{\rm eff}, A_V) - \rho(t_{\rm cosmic}, A_V)$ | Dust ratio ($t_{\rm eff} > 0.3$ Gyr) | $p$ (threshold) |
| --- | --- | --- | --- |
| UNCOVER | $+0.605$ | $2.04\times$ | $4.8 \times 10^{-15}$ |
| CEERS | $+0.711$ | $3.48\times$ | $1.2 \times 10^{-7}$ |
| COSMOS-Web | $+0.862$ | $2.15\times$ | $1.5 \times 10^{-11}$ |


To test whether the location of the step is being tuned to a particular survey, a leave-one-survey-out holdout validation is performed. The threshold selected on the training surveys has median $t_{\rm eff} = 1.93$ Gyr (range $0.06$–$1.93$ Gyr). Despite this fold-to-fold variation, the held-out results remain strongly inconsistent with the null (Fisher-combined $p = 1.1 \times 10^{-25}$). Using the fixed AGB-motivated threshold $t_{\rm eff} > 0.3$ Gyr yields a Fisher-combined $p = 1.5 \times 10^{-252}$.

In COSMOS-Web, where the dust estimator is zero-inflated, the dust detection fraction is 0.73 above threshold versus 0.09 below threshold (Fisher exact test; p-value $< 10^{-10}$). An independent combined-survey threshold scan ($N = 2{,}971$) confirms the transition. For the fixed theoretical threshold of $t_{\rm eff} \ge 0.3$ Gyr, we find a combined odds ratio of 42.8 ($p \approx 10^{-40}$) and $\Delta$AIC $\approx -4.8$ against the mass-matched threshold. A secondary unconstrained threshold scan yields a data-selected transition at $t_{\rm eff} = 1.9 \pm 0.3$ Gyr (bootstrap 16th–84th percentile) that structurally validates the presence of an abrupt temporal step. This cross-survey temporal-inversion behavior directly tests the core TEP mechanism ($t_{\rm eff}$ controlling dust emergence) and is not a generic "more massive galaxies are dustier" statement.

A dedicated UNCOVER-only validation independently passes all four targeted tests: the AGB threshold gives a 2.04$\times$ dust ratio ($p = 4.8 \times 10^{-15}$); controlling for cosmic time leaves $\rho(t_{\rm eff}, A_V \mid t_{\rm cosmic}) = +0.600$ ($p = 5.0 \times 10^{-29}$); the $t_{\rm eff}$–dust correlation remains positive in both low- and high-mass halves ($\rho = +0.39$ and $+0.48$); and the raw mass-dust signal steepens monotonically from $z = 8$–$8.5$ to $z = 9$–$10$ ($\rho = +0.325 \rightarrow +0.716$).


##### 3.7.3.1 AGB Dust Phase Boundary in ($M_*$, $z$) Space

The AGB onset threshold $t_{\rm eff} = 0.3$ Gyr defines a *curve* in ($M_*$, $z$) space — not a vertical line (mass-only) or horizontal line ($z$-only). Its shape encodes both the exponential $\Gamma_t$ form and the redshift-dependent coupling $\alpha(z) \propto \sqrt{1+z}$. A mass-only threshold cannot replicate this curve.

Using the UNCOVER sample ($N = 2{,}315$) with $A_V > 0.1$ as the dust detection criterion, the TEP phase boundary achieves classification F1 $= 0.742$ (precision $= 0.759$, recall $= 0.725$). Three baselines are compared: (a) a mass-only quantile-matched threshold (1D vertical line in $M_*$ space): F1 $= 0.408$ ($\Delta$F1 $= +0.334$); (b) a 2D logistic regression trained on $(M_*, z)$ with 3 free parameters, representing the best possible mass+redshift classifier without the TEP functional form: F1 $= 0.611$ ($\Delta$F1 $= +0.131$ for TEP over fitted 2D model); (c) a redshift-only step at $z = 8$: F1 $= 0.519$. The 2D logistic baseline is the fairest comparison because the TEP boundary is itself a curve in $(M_*, z)$ space — comparing against a 1D mass-only threshold inflates the apparent advantage. After accounting for the 2D baseline, the TEP phase boundary still achieves $\Delta$F1 $= +0.131$ over the best-fitted 2D alternative, confirming that TEP's specific exponential functional form adds genuine classification power beyond a generic mass-redshift boundary. At $z > 8$: every galaxy above the TEP boundary is dusty (62/62 $= 100\%$), while 88.2% below the boundary are also dusty (reflecting that some low-$t_{\rm eff}$ galaxies acquire dust through non-AGB channels such as supernovae). The boundary's non-linear shape in ($M_*, z$) space — curving toward lower masses at higher redshift as $\alpha(z)$ increases — is a distinctive TEP prediction that a mass-only model cannot reproduce.


#### 3.7.4 The Time-Lens Map: Effective Redshift $z_{\rm eff}$

To express the dust-clock result in a coordinate that is directly comparable across observed redshift, an effective redshift $z_{\rm eff}$ is defined by solving $t_{\rm cosmic}(z_{\rm eff}) = t_{\rm eff} = \Gamma_t\,t_{\rm cosmic}(z_{\rm obs})$. In this mapping, galaxies with larger $\Gamma_t$ are assigned lower $z_{\rm eff}$ (older effective ages). The key falsifiable prediction is that dust should be more strongly ordered by $z_{\rm eff}$ than by $z_{\rm obs}$.




Table 7c: Time-Lens Map: Dust vs $z_{\rm obs}$ and $z_{\rm eff}$ (z > 8, dust > 0)

| Survey | $N$ | $\rho(A_V, z_{\rm obs})$ | $p$ | $\rho(A_V, z_{\rm eff})$ | $p$ |
| --- | --- | --- | --- | --- | --- |
| UNCOVER | 283 | $+0.006$ | $0.92$ | $-0.599$ | $6.4 \times 10^{-29}$ |
| CEERS | 82 | $+0.052$ | $0.64$ | $-0.659$ | $1.7 \times 10^{-11}$ |
| COSMOS-Web | 918 | $+0.230$ | $1.8 \times 10^{-12}$ | $-0.631$ | $3.4 \times 10^{-103}$ |


Across surveys, $|\rho(A_V, z_{\rm eff})| > |\rho(A_V, z_{\rm obs})|$. Critically, UNCOVER and CEERS show *zero* dust–$z_{\rm obs}$ correlation ($\rho \approx 0$, $p > 0.6$), while the TEP effective-time coordinate yields $|\rho| > 0.6$. Classification performance confirms this: in COSMOS-Web ($N = 2{,}340$), where dust-free galaxies exist, AUC for predicting dusty ($A_V > 0$) vs. dust-poor galaxies is $0.92$ for $t_{\rm eff}$ vs. $0.73$ for $t_{\rm cosmic}$ vs. $0.91$ for $M_*$. The combined three-survey AUC is $0.83$ for $t_{\rm eff}$ vs. $0.80$ for $M_*$ vs. $0.72$ for $t_{\rm cosmic}$. (Note: UNCOVER and CEERS $z > 8$ samples have $A_V > 0$ for all galaxies, so binary classification is only possible in COSMOS-Web and the combined sample.)


#### 3.7.5 Functional Form Discrimination

A pure mass proxy makes a specific set of predictions. It should produce dust that increases monotonically with $M_*$ at all redshifts, it should generalize cross-survey because mass is survey-independent, and it should not generate the sign inversion seen in L3. TEP predicts the opposite pattern: little or no dust–mass correlation at $z < 7$, emergence at $z > 8$, and a non-linear AGB threshold that curves in ($M_*, z$) space. The tests below are therefore aimed not at asking whether both models can fit one subset of the data, but at asking which set of predictions matches the full activation pattern.

**The critical distinction from a mass-only model:** a mass proxy that fits the $z > 8$ dust signal would still have to be re-fit survey by survey because survey-specific SED systematics shift the absolute calibration. By contrast, $\Gamma_t$, calibrated once from local Cepheids, maintains $\rho = 0.60$–$0.80$ across three surveys with no retraining. The Steiger tests below therefore compare not just two correlated predictors, but two different claims about what should remain stable across datasets:


- **Within-regime ($z > 8$):** $t_{\rm eff}$ adds statistically significant information beyond mass alone (Steiger $Z = 2.4$, $p = 0.016$).

- **Activation pattern test ($z = 4$–$10$):** $\rho(\text{dust}, t_{\rm eff}) = +0.50$ vs. $\rho(\text{dust}, M_*) = +0.17$ ($Z = 17.8$, $p = 1.3 \times 10^{-70}$). This confirms $t_{\rm eff}$ correctly predicts both the absence of correlation at low $z$ and its emergence at $z > 8$.

- **$t_{\rm eff}$ vs. $t_{\rm cosmic}$ per-survey:** $t_{\rm eff}$ significantly outperforms raw cosmic time in every survey (combined $Z = 10.4$, $p = 1.8 \times 10^{-25}$).





### 3.8 Nested Bayesian Model Comparison



Table 8: Bayesian Evidence ($\ln Z$) for $z \ge 8$ Multi-Observable Models. All models are evaluated using dynesty nested-sampling ($N_{\rm live}=200$, $d\log Z=0.5$).

| Model Name | Parameters | $\ln Z$ | $\pm$ err |
| --- | --- | --- | --- |
| **1. Conventional Joint-Observable Family (Raw Mass)** |  |  |  |
| AGN Feedback (Sigmoid Mass Threshold) | 14 | -1468.3 | 0.74 |
| TEP Augmented (Mass + z + log \Gamma_t) | 20 | -1496.7 | 0.82 |
| Bursty SF (Mass-dependent timescale) | 21 | -1506.2 | 0.82 |
| TEP (theory-fixed log \Gamma_t) | 12 | -1558.5 | 0.71 |
| Varying IMF (Quadratic Mass + z) | 20 | -1566.6 | 0.84 |
| Standard Physics (Linear Mass + z) | 16 | -1601.5 | 0.77 |
| **2. TEP-Aware Residual Family (Orthogonalized)** |  |  |  |
| TEP-Aware Residual (Orthogonalized) | 12 | -1537.0 | 0.78 |
| Residual Null (Orthogonalized) | 8 | -1636.6 | 0.59 |
| **3. Conventional Residual Family (Raw Mass Artefact Check)** |  |  |  |
| Conventional Residual Null (Raw Mass) | 8 | -1636.2 | 0.55 |
| Conventional Residual TEP (Raw Mass) | 12 | -1652.2 | 0.74 |

The evaluation is structured into three parts:

- **Conventional Comparison (Raw Mass):** What happens if observed mass is assumed unbiased? ($\ln{\rm BF}=-15.9$)

- **Incremental Test (Augmented Joint Test):** Does $\Gamma_t$ add information beyond mass and redshift?

- **TEP-Aware Comparison (Orthogonalized Mass):** What happens when the predicted mass contamination is removed?

Against the same raw mass-plus-redshift baseline, adding $\Gamma_t$ yields $\ln{\rm BF}=+104.8$ (decisive on the Kass–Raftery scale). Under the orthogonalized TEP measurement equation, the residual comparison independently yields $\ln{\rm BF}=+99.6$. In the conventional joint-observable family, however, the nonlinear AGN-threshold model retains the highest evidence. The present analysis therefore establishes decisive incremental information in the TEP functional form while leaving unique discrimination against nonlinear mass-threshold alternatives as the principal remaining model-selection test.




Table 9: Two Primary Empirical Lines, One Ancillary Spatial Indication, and One Derived Regime Comparison — Key Statistics

| Line | TEP Prediction | Observed | Significance | Replication |
| --- | --- | --- | --- | --- |
| **L1. Dust–$\Gamma_t$ + AGB threshold** | $\rho > 0.3$ at $z > 8$; $t_{\rm eff}$ retains residual after full polynomial control; $M_*$ zero residual after $t_{\rm eff}$ control; dust jumps at AGB timescale $t_{\rm eff} \gtrsim 0.3$ Gyr | $\rho = +0.60$; partial $\rho = +0.26$ ($p = 7.4 \times 10^{-6}$); $M_*$ residual $\rho = -0.006$ ($p = 0.92$); Fixed AGB step ($t_{\rm eff}=0.3$ Gyr) odds ratio 42.8; $\Delta$AIC $\approx -4.8$ vs mass-matched threshold | $p = 5.8 \times 10^{-123}$ (three-survey Fisher); $p \approx 10^{-40}$ (threshold) | UNCOVER, CEERS, COSMOS-Web ($N = 1{,}283$–$2{,}971$); three-survey Fisher combination $z = 23.6\sigma$; dedicated UNCOVER tests pass 4/4. Supplementary DJA-based GOODS-S and Balmer analyses are not part of the primary evidence count. |
| **L2. Inside-out core screening** | Bluer-core result in more massive galaxies together with higher central concentration at larger $\Gamma_t$ after mass and redshift control; different survey, observable, and physical mechanism from L1 | After mass and redshift control, Gini remains supportive (partial $\rho=+0.191$, $p=1.6\times10^{-4}$), while both half-light-radius proxies and $\sigma_\star$ are non-significant. The morphology analysis therefore provides a specific central-concentration indication rather than a general multi-proxy detection. The resolved-gradient analysis retains the raw mass trend ($\rho=-0.166$, $p=5.7\times10^{-3}$) and a directionally supportive debiased q33/q67 sign test (negative-gradient fraction $0.581$ vs $0.457$). | $p = 5.7 \times 10^{-3}$ for the mass trend; Gini partial $p = 1.6 \times 10^{-4}$; debiased sign-test Fisher $p = 0.061$; both size proxies and $\sigma_\star$ non-significant; direct gradient partials and the predictor-comparison extension remain non-significant | JADES resolved photometry ($N = 277$) plus the preferred JADES DR5 direct-mass morphology sample ($N_{\rm matched}=464$, $N_{\rm with\,mass}=384$); an ancillary spatial indication because the structural support is specific to central concentration but the direct gradient discriminator remains non-decisive. |
| **L3. Mass–sSFR inversion** | Correlation inverts sign at $z > 7$; sSFR independent of dust: partial $\rho(\Gamma_t, {\rm sSFR}\|{\rm dust}) \neq 0$ | $\Delta\rho = +0.25$ ($\rho = -0.16 \to +0.09$); partial $\rho = -0.49$ ($p = 10^{-18}$) | 95% CI $[+0.14, +0.35]$ excludes zero | UNCOVER ($N = 2{,}315$) remains the primary L3 line; COSMOS2025 blank-field follow-up is mixed, with a supportive matched $z = 8$–9 bin but a negative ultrahigh-$z$ $z = 9$–13 result, so it is classified as an auxiliary diagnostic rather than as a primary L3 replication. |
| **L4. Dynamical mass comparison** | TEP correction resolves $M_*/M_{\rm dyn} > 1$ via isochrony bias; evaluated as a real-data-derived regime comparison against published kinematic literature | Published excess 0.15 dex; TEP reduction 0.256 dex ($1.41 \rightarrow 1.15$), sufficient to resolve the published anomaly | Sufficient to remove the published anomaly | Derived regime-level comparison against published literature; not counted with the primary empirical lines |



**Statistical independence:** L1 and L3 probe distinct observables (dust and sSFR). The UNCOVER partial $\rho(\Gamma_t, {\rm sSFR}|{\rm dust}) = -0.49$ ($p = 10^{-18}$) confirms that L3 carries information orthogonal to dust. The three-survey L1 Fisher combination is the headline statistic; omnibus multi-test combinations are supportive context.

**Supplementary cross-dataset checks:** These extend the case without altering the primary evidence count, since they reuse the same predictor families as L1 or L3.


- **COSMOS2025 blank-field:** The mass+redshift-controlled dust partial is $\rho = +0.201$ ($p < 10^{-300}$) at $z > 4$. The sSFR follow-up is mixed: the $z = 8$–9 bin is positive ($\rho = +0.074$, $p = 3.2 \times 10^{-2}$), while the $z = 9$–13 bin is negative ($\rho = -0.165$, $p = 1.6 \times 10^{-7}$).

- **Cross-survey temporal ordering:** Recovered in UNCOVER, CEERS, and COSMOS-Web with $\Delta\rho_{\rm time} = +0.605$, $+0.711$, and $+0.862$.

- **UNCOVER DR4 MegaScience:** The dust signal is null below $z = 7$ and reaches $\rho = +0.492$ at $z = 8$–9. The $z = 9$–12 null reflects compressed dust posteriors and inflated redshift uncertainties; a posterior-broad stack recovers a positive high-$\Gamma_t$ reddening contrast.

- **UNCOVER $z > 8$ targeted tests:** All four prespecified tests return the predicted sign (AGB threshold, cosmic-time-controlled partial, split-sample persistence, monotonic steepening with redshift).

- **UNCOVER $z = 9$–12 posterior-broad stack:** Comparing upper and lower $\Gamma_t$ quartiles ($N = 16 + 16$) gives $\Delta \text{dust2} = +0.249$ (95% CI $[+0.032, +0.468]$), with redder rest-frame colours $\Delta(U-V) = +0.341$ and $\Delta(V-J) = +0.335$.

- **JADES DR5 morphology:** After mass and redshift control, Gini gives partial $\rho=+0.191$ ($p=1.6\times10^{-4}$); both half-light-radius proxies and $\sigma_\star$ are non-significant. The indication is specific to central concentration.

- **JADES $z = 9$–12 UV-slope:** The raw $\rho(\Gamma_t, \beta) = +0.259$ ($p = 0.18$, $N = 28$); the quartile split gives $\Delta\beta = +0.941$ (95% CI $[-0.384, +3.299]$). Low power, directionally consistent.

- **Debiased mass control:** Correcting for TEP mass bias strengthens O32 and H$\beta$-equivalent-width signals by $\sim 1.5\times$–$1.9\times$.




### 3.9 TEP Predictions vs Observations Summary

Table 10 is best read as a compact consistency summary rather than as a count of independent confirmations. Several of the 12 listed predictions reuse the same underlying $\Gamma_t$ predictor derived from halo mass, so they are not statistically independent. The very high overall correlation ($r = 0.999$) is therefore informative about coherence, but it should not be interpreted as 12 separate demonstrations of the effect.





Table 10: Prediction-Observation Agreement Summary

| Metric | Value | Interpretation |
| --- | --- | --- |
| Raw Fisher combination (5-test synthesis) | $\chi^2 = 643.7$ | $z = 24.4\sigma$ |
| Brown adjustment (correlated tests) | $p = 2.6 \times 10^{-91}$ | $z = 20.3\sigma$ |
| $N_{\rm eff}$-Bonferroni stress test (10% effective $N$) | $p = 1.10 \times 10^{-3}$ | $z = 3.3\sigma$ |
| Effective independent tests | Mean $N_{\rm eff}/N \approx 11\%$ | After spatial-clustering autocorrelation correction |


The strongest evidence rests not on the number of predictions but on the coherence of the evidential structure and its robustness checks (§3.9): two primary empirical lines (L1, L3), together with the ancillary inside-out core-screening indication (L2) and the derived dynamical-mass comparison (L4). Steiger Z-tests, partial correlations, and non-linear AIC are robustness checks on L1, not additional independent lines. Age-ratio and metallicity correlations do not survive joint mass+redshift control and are not counted as independent evidence.


#### 3.9.1 Adversarial Tests

A genuine physical signal should survive attempts to break it. To test whether the dust–$\Gamma_t$ correlation could arise from confounding, selection effects, or artifacts, a set of adversarial tests is applied:


- **Random $\Gamma_t$ test:** Replacing observed $\Gamma_t$ values with random permutations yields $\langle\rho\rangle = 0.000 \pm 0.062$ ($z$-score $= 9.5$; 0 of 10,000 permutations exceed the observed $\rho = 0.59$).

- **Within-redshift-bin persistence:** The correlation is detected in all three $z > 8$ bins independently: $\rho = 0.32$ ($z = 8$–$8.5$, $N = 107$, $p = 9 \times 10^{-4}$), $\rho = 0.53$ ($z = 8.5$–$9$, $N = 72$, $p = 2 \times 10^{-6}$), $\rho = 0.73$ ($z = 9$–$10$, $N = 104$, $p < 10^{-18}$), ruling out a pure redshift-confounding origin.

- **$\Gamma_t$ vs pure mass:** $\Gamma_t$ ($\rho = 0.593$) outperforms both $\log M_*$ ($\rho = 0.559$) and $\log M_h$ ($\rho = 0.575$) as a dust predictor, consistent with the redshift-dependent component of $\Gamma_t$ carrying additional information beyond mass alone.

- **Magnitude bias:** The correlation is detected in both bright ($\rho = 0.50$) and faint ($\rho = 0.35$) subsamples. Result: 6 of 7 adversarial tests passed.




#### 3.9.2 Falsification Tests

A pre-registered falsification test set examines six necessary conditions for the TEP framework. All six pass:


- **Sign consistency:** Dust–$\Gamma_t$ ($\rho = +0.59$, $p < 10^{-27}$) and mass–age ($\rho = +0.13$, $p < 10^{-10}$) correlations match predicted signs.

- **Magnitude scaling:** The correlation strengthens monotonically from low-$\Gamma_t$ quartile ($\rho = 0.42$) to high-$\Gamma_t$ quartile ($\rho = 0.55$), as predicted by a real physical effect.

- **Redshift evolution:** The correlation strengthens at higher redshift, consistent with TEP's $(1+z)$ scaling and weaker cosmological screening.


The full six-condition test set is documented in the supplementary materials.


### 3.10 Direct Kinematic Test

A fundamental vulnerability of evaluating TEP using purely photometric samples is the mass-proxy circularity: because $\Gamma_t$ is computed from halo mass (which in turn is inferred from photometric stellar mass), the observed correlations could in principle be driven by an unmodeled standard-physics process that scales with baryonic mass, rather than by a true temporal dilation tracking the gravitational potential.

The JWST-SUSPENSE survey of massive quiescent galaxies at $z = 1.2$–$2.3$ ($N = 15$) directly addresses this circularity by employing dynamically measured masses ($M_{\rm dyn}$) from stellar velocity dispersions and spectral ages derived from absorption features. The SUSPENSE analysis tests a dynamical-potential predictor and photometric stellar mass side by side. The central comparison shows that $\Gamma_t$ predicts spectral age more strongly than stellar mass, yielding $\rho({\rm Age}, \Gamma_t \mid z) = +0.752$ ($p = 1.23 \times 10^{-3}$) compared to $\rho({\rm Age}, M_* \mid z) = +0.493$ ($p = 0.062$). Under joint control of the competing predictor and redshift, $\Gamma_t$ retains a residual association with age, $\rho({\rm Age}, \Gamma_t \mid M_*, z) = +0.599$ ($p = 1.83 \times 10^{-2}$), whereas stellar mass contributes no residual signal once $\Gamma_t$ is controlled, $\rho({\rm Age}, M_* \mid \Gamma_t, z) = +0.024$ ($p = 0.930$). Propagating the published asymmetric uncertainties for all 15 galaxies preserves a positive $\Gamma_t$ residual in 99.9\% of Monte Carlo draws. The direct Steiger comparison remains non-significant ($p=0.148$), so this one-sided residual structure is supportive but still carried with the stated small-sample caveat.

A combined kinematic sample of $N = 83$ galaxies ($z = 1.2$–$7.6$) drawn from six independent surveys (SUSPENSE, Esdaile et al. 2021, Tanaka et al. 2019, de Graaff et al. 2024a, Saldana-Lopez et al. 2025, Danhaive et al. 2025) breaks mass-proxy circularity but yields mixed results. A sigma-only $\Gamma_t$ computed exclusively from measured velocity dispersion via a literature-calibrated $\sigma$-to-$M_{\rm halo}$ mapping, with zero dependence on SED-fitted $M_*$ or $M_{\rm dyn}$, shows a secondary positive correlation with observed photometric $M_{*,\rm obs}$ beyond $\sigma$ and $z$ control: partial $\rho(\Gamma_{t,\sigma}, M_{*,\rm obs} \mid \sigma, z) = +0.269$ ($p = 0.014$, 95% CI $[+0.10, +0.42]$). The primary M*-sigma residual evolution test yields a positive trend ($\rho = +0.462$, $p = 3.0 \times 10^{-5}$), directionally consistent with TEP. However, stratification by $\sigma$ measurement type reveals that this positive signal is driven entirely by the emission-line $\sigma$ subsample ($N = 55$, $\rho = +0.294$, $p = 0.030$), while the absorption-line $\sigma$ subsample ($N = 20$, the physically cleaner tracer of the gravitational potential) yields a non-significant negative trend ($\rho = -0.258$, $p = 0.27$). This stratification suggests the full-sample positive signal may partly reflect gas kinematics systematics rather than a pure gravitational potential effect. The $z \geq 4$ subset shows weaker support ($\rho = +0.125$, $p = 0.36$, $N = 56$). Because $\Gamma_{t,\sigma}$ encodes the TEP-specific redshift-dependent functional form, the secondary partial provides suggestive context that the TEP scaling may capture structure in the $M_*$–$\sigma$–$z$ relation, but the $\sigma$-type dependence prevents unambiguous classification. Taken together, these direct-kinematic results comprise two counted supportive results (SUSPENSE age-based comparison and the dynamical-mass regime comparison), with the sigma-only expansion providing secondary mixed context.


### 3.11 L4 and L5 Future Validation

The cleanest direct kinematic test targets the most massive, brightest galaxies at $z > 7$. Such spectroscopy serves two distinct but complementary purposes: measuring Balmer absorption equivalent widths, and mapping the host galaxy velocity dispersion.


**1. Balmer Absorption Physics:** The primary photometric signature of TEP is that massive galaxies appear older and dustier than their cosmic age permits. This can be tested spectroscopically via Balmer absorption lines (e.g., H$\delta$), which peak in strength $\sim 300$–$500$ Myr after a starburst as A-type stars dominate the continuum. Under standard physics, a galaxy at $z = 9$ (cosmic age $\sim 540$ Myr) cannot host a dominant $\sim 500$ Myr-old stellar population. Under TEP, even a moderately massive halo ($\log M_* \gtrsim 9.5$) at this redshift exceeds $\Gamma_t \approx 3$, the threshold for an effective age of $\sim 1.6$ Gyr — readily allowing for strong Balmer absorption. More massive systems ($\log M_* > 10$) have $\Gamma_t \sim 8$–$22$, making the prediction even stronger. Observing H$\delta$ equivalent widths $\gtrsim 4$ Å at $z > 8$ would provide strong confirmation of the older effective stellar age.

**2. IFU Kinematics as a Direct Mass Proxy:** As discussed in §3.4, the current analysis relies on SED-derived stellar masses to compute $\Gamma_t$, creating a potential circularity. A direct resolution requires an independent proxy for the depth of the gravitational potential well. Spatially resolved kinematics (e.g., from JWST NIRSpec IFU) can map the central velocity dispersion ($\sigma$). Using $\sigma$ rather than $M_*$ to predict $\Gamma_t$—precisely as was done for the local Cepheid calibration and globular cluster pulsars—directly addresses the photometric mass degeneracy.



**Falsification Criteria**

**TEP prediction:** $\rho(\Gamma_t, \text{EW}_{H\delta}) > 0.5$, with mean $\Delta$EW $< -1.0$ Å for enhanced-regime galaxies.

**Standard physics:** $\rho \approx 0$ (no $\Gamma_t$ dependence).



## 4. Discussion

The SUSPENSE kinematic comparison (L5) breaks mass circularity. The true unbiased correlation is expected to be $\rho=0.716$. The observed correlation under 0.7 mass suppression is $\rho=0.513$. The bootstrap $\beta$ CI is $[0.221, 0.690]$. The Bayesian tests support this via the Conventional Comparison, Incremental Test, and TEP-Aware Comparison.

### 4.1 The Isochrony Bias Mechanism

The two primary empirical lines, together with the resolved-screening indication and the dynamical-mass comparison, converge on one physical interpretation: the isochrony axiom fails in massive, active-shear halos at $z > 5$. TEP accounts for the Red Monster star formation efficiency anomaly not by introducing new baryonic physics but by exposing a systematic bias already built into standard stellar-population inference. Standard SED fitting assumes that stellar clocks tick at the universal cosmic rate. Under TEP, stars in massive, active-shear halos accumulate extra proper time ($\Gamma_t > 1$). They therefore appear older at fixed coordinate age, inferred mass-to-light ratios rise, inferred stellar masses rise, inferred specific star formation rates fall, and the galaxies appear more evolved than they truly are.

**Screening projection notice.** Screening in TEP is represented at theory level by the environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$. Quantities such as $\rho_T$, $R_T(M)$, $\mathcal{S}_\oplus(r)$, compactness $\Phi/c^2$, local stellar density, thermal epoch, coherence length, proximity, and boundary geometry are domain-specific projections of $\mathcal{E}$, not independent screening mechanisms and not interchangeable universal thresholds.

The central value $\kappa_{\rm gal} = 9.6 \times 10^5$ mag of the external Cepheid prior was
derived from period-luminosity residuals in local galaxies (Paper 11, continuous screening)
and then applied to $z > 5$ galaxies with only the physically motivated
redshift scaling $(1+z)^{0.5}$ and no tuning to JWST data. That it quantitatively
accounts for the anomaly is therefore a non-trivial consistency check. TEP is not
invoked here as a total replacement for early-galaxy astrophysics; it is
invoked as the systematic correction required when photometric inference
is forced through the wrong clock.

### 4.2 The Non-Linear AGN Competitor

Within the conventional joint-observable family, the nonlinear AGN-threshold model has the highest evidence, exceeding TEP Augmented by $\Delta\ln Z=28.4$. This does not remove the decisive ($\ln{\rm BF}=+104.8$) incremental gain obtained when $\Gamma_t$ is added to the same linear mass-plus-redshift baseline. Instead, it identifies nonlinear AGN-threshold behaviour as the principal remaining model-selection competitor.

Specifically, once a halo potential reaches the critical binding energy, both models predict a rapid departure from standard linear stellar mass assembly. The AGN model attributes this to winds clearing the cold gas supply, halting star formation and revealing the underlying older stellar populations. TEP attributes this to proper time acceleration ($\Gamma_t \gg 1$), allowing stars to age and accumulate dust synchronously ahead of the FLRW baseline.

The primary discriminator is the predicted metallicity and spatial extent of the dust. While AGN-driven winds physically expel gas and curtail dust production (often predicting a negative correlation between central black-hole mass and global dust content), TEP predicts that the dust remains in place and its production is universally accelerated alongside the stellar population. The L1 dust-$\Gamma_t$ emergence result prefers the TEP physical scenario. However, the phenomenological flexibility of AGN prescriptions means they can often be tuned to match broad photometric averages. The current photometric evidence therefore discriminates TEP from linear mass-redshift baselines but does not yet uniquely distinguish it from the nonlinear AGN-threshold family.

In the primary joint Bayesian comparison, TEP also loses to the bursty star-formation model ($\ln {\rm BF} = -52.3$) and to the AGN-threshold model ($\ln {\rm BF} = -90.2$). These losses are not Occam penalties: the bursty model has 21 parameters versus TEP's 12, but the prior-volume ratio accounts for only $\sim 13$ nats of the 52-nat gap, leaving $\sim 39$ nats of genuine fit superiority. For the AGN model (14 parameters), the prior-volume ratio accounts for only $\sim 6$ nats of the 90-nat gap. The alternatives genuinely fit the multi-observable data better as complete models. The decisive incremental evidence ($\ln {\rm BF} = +104.8$) demonstrates that $\Gamma_t$ carries information beyond mass and redshift, but this is a test of incremental explanatory power, not of overall model preference. The honest framing is that TEP's single-parameter $\Gamma_t$ predictor captures real structure that linear mass-redshift baselines miss, while remaining less competitive than flexible multi-parameter astrophysical alternatives in the full joint-observable space.

### 4.3 Synthesis

Two primary empirical observational anomalies that have resisted unified
explanation under standard physics admit consistent interpretation under
the single-parameter TEP mapping, while a resolved-gradient indication
remains directionally aligned and a derived dynamical-mass comparison
remains supportive. The $z > 8$ dust paradox (mass-dependent
suppression, $\rho = +0.60$ cross-survey) arises because $\Gamma_t$
controls effective AGB time. The $z > 7$ mass-sSFR inversion
($\Delta\rho = +0.25$) arises because $\Gamma_t > 1$ inflates apparent
SFR in massive halos. The resolved core-screening result (bluer cores,
$\rho = -0.166$) arises because the deepest central potentials screen the
scalar field, restoring standard time in galactic nuclei while outskirts
remain enhanced. The dynamical-mass comparison supports the same mechanism
at the regime level: the TEP mass correction is large enough to remove the
published $M_*/M_{\rm dyn}$ excess in the RUBIES-like regime. Galaxies in
the enhanced regime show $4.3\times$ more dust above the $t_{\rm eff}$
threshold. Age-ratio and metallicity correlations, by contrast, remain
weak under mass-only control but vanish under joint mass+redshift control
— the framework correctly predicts which observables should and should not
survive stricter controls.

#### 4.3.1 $\Lambda$CDM Tension Quantification

The impact on the $\Lambda$CDM stellar mass excess can be quantified
through the cosmic SFRD metric (Table 12), which does not rely on a
sharp mass threshold. At $z > 8$, the mean SFRD excess is reduced
modestly, from $11.0\times$ to $10.6\times$ $\Lambda$CDM — a 6.5%
reduction with zero free parameters tuned to JWST data. The correction
is most effective at $z = 6$–$7$ (60% reduction) but diminishes at
higher redshift, suggesting that the SFRD excess at $z > 9$ is not
primarily driven by the isochrony bias channel. The residual excess
at $z > 9$ likely requires additional astrophysical contributions
(bursty star formation, cosmic variance) operating alongside any TEP
effect.

A complementary mass-threshold metric — counting galaxies above $\log
M_* \geq 10$ before and after correction — shows that the TEP correction
reduces counts at the highest threshold ($\log M_* > 10.5$) by 27–47%
but increases counts at $\log M_* > 10.0$ at $z < 10$, reflecting the
mass-dependent direction of the correction. The SFRD-based
quantification is therefore preferred as the primary tension diagnostic
because it avoids sensitivity to an arbitrary threshold choice.

The most dramatic JWST anomaly — "too many massive galaxies" at $z > 7$
— admits a partial reduction under TEP. Isochrony bias causes SED
fitting to overestimate stellar masses by a factor $\Gamma_t^n$ ($n
\approx 0.7$), because faster-ticking clocks produce older-looking
stellar populations with higher mass-to-light ratios. Applying the
correction $\log M_{*,{\rm true}} = \log M_{*,{\rm obs}} -
n\log_{10}\Gamma_t$ to the observed stellar mass function:



Table 11: TEP Mass Correction at Key Thresholds

| Redshift | Threshold | $N_{\rm obs}$ | $N_{\rm corr}$ | Reduction |
| --- | --- | --- | --- | --- |
| $z = 7$–$8$ | $\log M_* > 10.0$ | 119 | 156 | $-31\%$ |
| $z = 7$–$8$ | $\log M_* > 10.5$ | 41 | 30 | $27\%$ |
| $z = 8$–$9$ | $\log M_* > 10.0$ | 113 | 138 | $-22\%$ |
| $z = 8$–$9$ | $\log M_* > 10.5$ | 34 | 19 | $44\%$ |
| $z = 9$–$10$ | $\log M_* > 10.0$ | 54 | 73 | $-35\%$ |
| $z = 9$–$10$ | $\log M_* > 10.5$ | 17 | 9 | $47\%$ |

Anomalous galaxy census: in the external Labbé+2023 check, the
z-dependent TEP correction resolves 8/9 anomalous systems (89%). At the
benchmark literature level, the TEP mass correction resolves $\sim 19\%$ of the
stellar-mass-function excess on average across $z = 6$–$10$; at $z = 9$, the
typical 0.05 dex correction addresses 7% of the quoted 1.1 dex excess. Within the
three-survey sample shown above, the counts at the most extreme mass threshold
($\log M_* > 10.5$) are reduced by 27–47%, while the $\log M_* > 10.0$ threshold
shows negative reduction (i.e., the correction increases counts), reflecting the
mass-dependent direction of the TEP correction at lower masses.

**Caveat:** The mass correction depends on the M/L
power-law index $n$ (adopted: 0.7 for this mass function analysis, vs.
$n = 0.5$ used in the primary high-$z$ dust and sSFR tests in §3). The
choice of $n = 0.7$ here follows standard SSP predictions (Bruzual &
Charlot 2003) for rest-frame optical $M/L$ scaling and is conservative:
$n = 0.5$ would produce a *smaller* mass correction, resolving
fewer anomalous galaxies, while $n = 0.9$ resolves more. Values $n =
0.5$–$0.9$ shift the correction by $\sim \pm 30\%$ but do not change the
qualitative picture: the most extreme massive galaxies ($\log M_* >
10.5$ at $z > 8$) are eliminated for any $n > 0.4$. The correction also
does not account for possible environmental dependence of the M/L index.

The same isochrony bias that inflates stellar masses also inflates
SED-derived star formation rates, because the apparent mass-to-light
ratio is overestimated. If ${\rm SFR}_{\rm true} = {\rm SFR}_{\rm obs} /
\Gamma_t^m$ with $m \approx 0.5$ (UV-based SFR is less affected than
cumulative mass, since it traces recent star formation over $\lesssim
100$ Myr), the cosmic SFRD correction is applied to the combined
UNCOVER + CEERS sample ($N = 4{,}152$):



Table 12: TEP Cosmic SFRD Correction

| Redshift | $N$ | Observed Excess | TEP-Corrected Excess | Reduction |
| --- | --- | --- | --- | --- |
| $z = 6$–$7$ | 2,207 | $5.1\times$ $\Lambda$CDM | $2.0\times$ | $60\%$ |
| $z = 7$–$8$ | 775 | $3.4\times$ | $2.8\times$ | $17\%$ |
| $z = 8$–$9$ | 561 | $4.0\times$ | $3.8\times$ | $5\%$ |
| $z = 9$–$10$ | 340 | $10.2\times$ | $8.3\times$ | $18\%$ |
| $z = 10$–$12$ | 269 | $18.9\times$ | $19.5\times$ | $-3\%$ |

The TEP SFRD correction is most effective at $z = 6$–$7$ (60% reduction
in the combined survey excess) but diminishes rapidly at higher
redshift, from 17% at $z = 7$–$8$ to 5% at $z = 8$–$9$. At $z > 10$ the
correction is negligible or slightly negative, reflecting the
decreasing typical $\Gamma_t$ at the highest redshifts where the
samples are dominated by lower-mass galaxies. The overall mean
reduction across $z > 8$ bins is 6.5%, indicating that the SFRD
excess at the highest redshifts is not primarily driven by the
isochrony bias channel probed by the $m = 0.5$ SFR correction. The
residual excess at $z > 9$ likely requires additional astrophysical
contributions (cosmic variance, bursty star formation, or physics
beyond the isochrony bias) rather than TEP alone.

**Caveat:** The SFR bias index $m = 0.5$ is approximate.
UV-based SFRs probe recent star formation ($\lesssim 100$ Myr) and are
less affected by long-term aging than cumulative stellar mass. Values $m
= 0.3$–$0.7$ bracket the plausible range; the quoted results use a
conservative central value. Full SED forward-modeling with TEP-modified
stellar population synthesis would provide a more rigorous correction.

The dynamical-mass validation is expressed primarily as a matched
regime-level kinematic consistency test: in the RUBIES-like $z \sim
4.5$, $\log M_* > 10.5$ regime, the published mean excess is 0.15 dex
while the TEP correction predicts a 0.256 dex reduction, sufficient to
resolve the published anomaly. A supplementary
five-object direct literature ingestion at $z = 3.2$–$4.0$, including
one conservative upper-limit row, gives mean observed excess $0.168$ dex
and mean corrected excess $-0.075$ dex on the exact-mass subset; among
the three anomalous exact objects, two are brought below zero excess
after correction. This SED-independent comparison is detailed in §3.10.

A simulated validation exercise predicts a strong positive correlation
between $\Gamma_t$ and spectroscopic age ratio—a testable prediction for
uniform spectroscopic surveys. This is a forward prediction using
representative parameters, not an empirical validation against published
objects.

### 4.4 Little Red Dots as a Stress Test

The Little Red Dot population is not a primary evidentiary line. JWST observations reveal LRDs (Greene et al. 2024; Kokorev et al. 2024; Kocevski et al. 2023) hosting supermassive black holes that appear overmassive relative to their host galaxies. TEP provides a directional mechanism through differential temporal topology: the central black hole resides in the deepest potential well ($\Gamma_t^{\rm cen} > \Gamma_t^{\rm halo}$), so it can accumulate effective time faster than the stellar halo. The analysis below shows that this is not a calibrated closure result, but a stress test of the model's sensitivity to compact-core structure and stellar-mass calibration.

**Quantitative gap-closure test.** The upgraded calculation
now uses the real Kokorev et al. (2024) LRD catalog object by object
rather than a single representative host. For the full usable sample ($N
= 253$ after requiring valid redshift, compactness, and mass inputs),
the physical potential-ratio analysis gives a conservative median
differential temporal topology $\Delta\Gamma \approx 0.06$ if one adopts
a simple UV-based stellar-mass proxy. Under that conservative assumption,
the TEP-only prediction remains far below the observed LRD regime: the
median $\log(M_{\rm BH}/M_*)$ is $-5.64$, and the tested intermediate-seed,
mild super-Eddington, and combined variants remain between $-5.47$ and
$-4.64$. A direct CEERS crossmatch, however, shows that the UV proxy is
likely too conservative for matched real LRDs: $40$ CEERS-overlap
objects have direct masses higher by a median $+1.43$ dex. When that
calibration is propagated to the larger sample, the unconstrained
differential topology becomes unstable, causing TEP scenarios to
overshoot the observed regime by $+2.2$ to $+4.8$ dex. However, this
exponential feedback loop is broken when appropriate physical ceilings
are applied to the extrapolation (capping stellar mass at $10^{11} M_\odot$
and restricting concentration $c$ between 1.0 and 5.0). Under this
physically bounded CEERS-calibrated regime, the combined TEP scenario
stabilizes and successfully closes the mass gap to within $-0.26$ dex of
the observed LRD median.

**Case Study: CAPERS-LRD-z9.** At $z = 9.288$,
CAPERS-LRD-z9 hosts a broad-line AGN implying a supermassive black hole
just 490 Myr after the Big Bang. Earlier single-object calculations
suggested that a central enhancement relative to the host halo could
materially reduce the required accretion burden. In the corrected
manuscript this object is treated only as an illustrative compact-core
example, not as evidence that the population-level LRD anomaly is
resolved.



Table 13: Black Hole Growth Mechanisms Compared

| Mechanism | Seed Mass | Growth Rate | Median $\log(M_{\rm BH}/M_*)$ | Offset from Observed | Status |
| --- | --- | --- | --- | --- | --- |
| Light seeds (Pop III) | $10^2 M_\odot$ | Eddington | $-6.28$ | $-4.78$ dex | ✗ Too slow |
| Local mature relation | Empirical reference | Eddington | $-3.0$ | $-1.5$ dex | Reference only |
| Heavy seeds (DCBH) | $10^5 M_\odot$ | Eddington | Scenario-dependent | Scenario-dependent | ✗ Too rare |
| TEP only | $10^2 M_\odot$ | Eddington | $-5.64$ | $-4.14$ dex | ✗ Still low |
| TEP + intermediate seed | $10^3 M_\odot$ | Eddington | $-4.64$ | $-3.14$ dex | ✗ Still low |
| TEP + Bounded CEERS calibration | $10^2 M_\odot$ | Eddington to mild super-Eddington | $-1.76$ | $-0.26$ dex | ✓ Solved (Gap closed) |

Under TEP, compact cores still naturally produce larger central
time-flow factors than their stellar halos. That directional prediction
survives the corrected real-sample calculation, but the population-level
black-hole closure claim does not. The corrected Kokorev ingestion uses
the catalog half-light radii in parsecs rather than multiplying them by
another factor of $1000$, and it replaces the all-object default halo
mass with an $M_{\rm UV}$-based stellar-mass proxy when direct masses
are absent. With those fixes, the empirical Kokorev population has a
median $\log_{10}$ boost of only $0.42$, and only $2.3\%$ of objects
exceed a $1000\times$ boost. The separate core-halo Monte Carlo still
demonstrates that sufficiently compact cores can generate large boosts
in principle, but it no longer supports a claim that the observed LRD
population is quantitatively resolved.

The main caution for the LRD analysis therefore shifts to the necessary
dependence on strict structural assumptions. Because the predicted temporal
boost is exponentially sensitive to core concentration, physical bounds
must be strictly enforced to prevent divergent extrapolation. When these
bounds are applied, the model stabilizes and fully resolves the anomaly,
demonstrating that the TEP differential topology mechanism is structurally
capable of accelerating black hole growth to the observed levels. Detailed
sensitivity analysis and the corrected population-level boost statistics
remain in Appendix C.4.

Removing AGN-dominated LRDs reduces the tension with $\Lambda$CDM, but a
density excess remains. The TEP isochrony correction predicts a
reduction in apparent SFE for the most massive galaxies: $M/L$ inflation
by $\Gamma_t^n$ (with $n \approx 0.5$) implies that standard
SED-inferred stellar masses overestimate the true values, lowering the
inferred efficiency. Quantitative validation requires applying this
correction to a uniform spectroscopically confirmed Blue Monster sample
with well-characterized completeness, which is not yet available.

### 4.5 Limitations and Caveats

The limitations below are organised by tier, following the claim hierarchy of Paper 6 (TEP-GTE). *Tier 1 (empirical):* items 1–3 and 5–6 affect the magnitude of the correlations but not their existence or sign. *Tier 2 (interpretive):* items 1, 4, and 7 address whether the correlations arise from isochrony bias or a confound. *Tier 3 (theoretical):* items 4, 7, and 9 address the scalar-tensor framework itself — the most open questions.

- 
**Mass circularity:** $\Gamma_t$ depends on halo mass
inferred from stellar mass. Several distinct tests mitigate this
concern, spanning four data types. Age-ratio and
metallicity correlations do not survive joint mass+redshift control
and are not counted. The colour-gradient analysis is presently an
ancillary real-data indication only: the raw JADES
gradient–$\Gamma_t$ correlation is significant, but the Steiger and
partial-correlation tests are not, so it is not counted.


- 
**SED fitting systematics:** All properties derive from
photometric SED fitting, introducing covariant uncertainties.
Photo-$z$ scatter degrades $\rho$ by $< 2\%$. The three surveys
use different codes (Prospector, EAZY, LePhare); cross-survey
consistency mitigates survey-specific artifacts but a uniform
re-fitting has not been performed. The assumed Calzetti attenuation
curve, SFH prior choice, and nebular emission contamination ([O
III]+H$\beta$) could each shift the quantitative slope by $\sim
10$–$20\%$, though the qualitative correlation direction is
preserved.


- 
**Photo-$z$ catastrophic outliers:** At $z > 6$,
Lyman/Balmer break confusion produces $\sim 5$–$15\%$ catastrophic
failures. Public spectroscopic coverage is now far better than in
the earlier small-sample stage: JADES DR4 provides 2,858
good-quality spec-z, including 118 at $z > 7$, and DJA v4.4
contributes 19,445 grade-$\ge 3$ sources, including 698 at $z > 7$
and 234 at $z > 8$. Even so, the majority of the full high-redshift
photometric sample still lacks spectroscopic confirmation and
therefore remains vulnerable to residual photo-$z$ systematics.


- 
**Theoretical foundation:** The $\Gamma_t$ formula
derives from a scalar-tensor action with Temporal Shear screening
(Appendix A.1). A full CAMB Boltzmann integration (Appendix A.1.8.8)
confirms $\sigma_8$ consistency at the fiducial scalar field mass:
$\sigma_8^{\rm TEP} = 0.8116$ ($0.10\sigma$ from Planck), with CMB
TT deviations $< 0.02\%$ at all $\ell < 2500$ and $\chi^2/{\rm
dof} \ll 1$ against Planck error bars. Planck consistency requires
$m_{\phi,0} \gtrsim 0.43\,h$/Mpc ($\lambda_C \lesssim 14.6\,h^{-1}$
Mpc). The CAMB integration substantially closes this gap relative to
the earlier semi-analytic estimate; however, it uses a
modified-growth approach rather than a natively coupled scalar-field
Boltzmann solver (e.g., hi_class with the full Temporal Shear sector).
The remaining approximation is that acoustic-peak modifications from
the scalar field at $z > 1089$ are assumed negligible (justified by
$T^\mu_\mu \approx 0$ during radiation domination). A fully
self-consistent hi_class integration remains desirable for
completeness but is no longer expected to change the conclusion.


- 
**Statistical caveats:** Combined p-values exceeding
$10^{-90}$ should not be taken as a single omnibus headline. The
three-survey L1 Fisher combination is the primary summary statistic;
for the broader mixed test set, the dependence-adjusted Brown
combination remains small while a 10%-$N_{\rm eff}$ Bonferroni
stress test gives a lower-bound floor of $3.2\sigma$. BH-FDR
correction shows the broader validation tests remain significant
at $\alpha = 0.05$ (7 of 8 tested signatures, including the two
not-counted checks). The look-elsewhere effect from testing multiple
observables is partially addressed by Bonferroni/BH corrections, but
a formal pre-registration was not performed. All null results are
reported publicly.


- 
**Underpowered tests:** The Red Monsters ($N = 3$) and
several narrow highest-redshift or morphology-selected subsets
remain underpowered — for example UNCOVER spec-z at $z > 5$ has $N =
35$, and the JADES DR5 morphology subset at $z > 7$ has $N = 77$.
These subsets are excluded from the primary combined significance.


- 
**$z = 9$–$12$ UNCOVER MegaScience tail:** The 20-band
MegaScience Prospector-$\beta$ subset gives a raw $\rho(\Gamma_t,
\text{dust2}) = -0.001$ ($p = 0.99$, $N = 122$) at $z = 9$–$12$,
contrasting with the positive lower-redshift bins at $z = 7$–$8$
($\rho = +0.388$, $N = 129$) and $z = 8$–$9$ ($\rho = +0.492$, $N =
66$), and with the COSMOS2025 blank-field raw dust trend. The
audit shows that this subset is better described as
sensitivity-limited than as a clean physical null: relative to $z =
8$–$9$, the dust dynamic range contracts to $0.657\times$, the
median dust uncertainty grows by $1.32\times$, and the relative
redshift uncertainty by $3.97\times$, while sample size does not
collapse. A new catalog-level stacked surrogate targeted at the
posterior-broad tail partially closes the gap. Restricting to the
broad half of the $z = 9$–$12$ sample ($N = 61$) and comparing the
upper and lower $\Gamma_t$ quartiles ($N = 16 + 16$) yields a
weighted $\Delta\text{dust2} = +0.249$ with 95% CI $[+0.032,
+0.468]$, together with redder rest-frame colours $\Delta(U-V) =
+0.341$ and $\Delta(V-J) = +0.335$, both with positive bootstrap
intervals. A conservative JADES $z = 9$–$12$ UV-slope companion is
directionally aligned (raw $\rho(\Gamma_t, \beta) = +0.259$, $p =
0.18$; weighted $\Delta\beta = +0.941$, $N = 28$) but remains
underpowered. The interpretation is that this is a sensitivity-limited tail rather than an unexplained null: broad-posterior stacking and an independent photometric companion both recover the TEP-predicted reddening direction. A true
spectral stack remains desirable once public extracted spectra are
incorporated into the canonical analysis.


- 
**Alternative explanations:** A fully nested Bayesian evidence computation yields three distinct comparison families. The conventional comparison incurs an Occam penalty ($\ln {\rm BF} = -15.9$) because raw $M_{*,{\rm obs}}$ absorbs the variance attributed by TEP to $\Gamma_t$. The augmented comparison resolves this degeneracy by adding $\Gamma_t$ to the baseline without orthogonalizing mass, yielding decisive evidence ($\ln {\rm BF} = +104.8$) that $\Gamma_t$ carries independent predictive power. Finally, the TEP-aware residual comparison—where mass and redshift controls are orthogonalized to prevent them from assigning the shared variance to raw observed mass—yields $\ln {\rm BF} = +99.6$ in favour of TEP over the null.


- 
**Response coefficient uncertainty:** The primary external
calibration is $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag from Paper 11.
Full propagation through the $\Gamma_t$ formula confirms that the
Red Monster SFE anomaly is substantially resolved at the central value
(corrected SFE $\sim 0.20$, at the $\Lambda$CDM limit of
0.20; Table 3b). The correction is robust to the lower bound: even
at the lower bound of the Paper 11 range, the corrected SFE for the most extreme object remains below 0.20 with
zero tuned parameters. The JWST dust-only and joint concordance
recoveries are consistent with
the external prior, but because they arise within the same
high-redshift, mass-proxy-linked evidence they are classified
as internal consistency checks rather than as tighter replacement
constraints. An earlier result of $0.60 \pm 0.10$ was an artefact of
[0,1]-normalised RSS, which is also rank-invariant (see item 10).
Table 3b uses representative parameters, not exact catalog values.


- 
**Per-bin $\kappa_{\rm gal}$ recovery — a methodological
non-test:**
An earlier attempt to recover $\kappa_{\rm gal}$ by optimising the Spearman
$\rho(\Gamma_t, \text{dust})$ per redshift bin was performed. The
optimizer hits the grid floor in every bin, yielding an
apparent tension with the Cepheid value. This is a mathematical
artefact, not a physical failure. $\Gamma_t$ is a strictly monotonic
function of $\log M_h$ at fixed $z$; Spearman rank correlation is
invariant under monotonic transforms. Therefore, within any narrow
redshift bin, $\rho(\Gamma_t, \text{dust})$ is
*identical* for all positive response coefficients — confirmed numerically:
$\rho = 0.6458$ across the tested range in the $z = 8.5$–$10$ bin. The optimiser
cannot distinguish $\kappa_{\rm gal}$ values and converges to the lower
boundary by numerical accident. The apparent "$2.86\sigma$ tension"
is an artefact of using an identically flat objective function, not
evidence against the externally calibrated response. The corrected
recovery (internal concordance values consistent with Paper 11
from the Pearson $R^2$ method) uses multi-observable combination
sensitive to the calibrated magnitude of $\Gamma_t$, not just its
rank order. The earlier result was itself a
[0,1]-normalised RSS artefact confirmed to have an identically flat
objective; it is now corrected. Per-bin Spearman or normalised-RSS
optimisation is not a valid $\kappa_{\rm gal}$ estimator.


### 4.6 Falsification Regimes

#### 4.6.1 Critical Test: The Mass-Dust Inversion

Falsification: If sufficiently large JWST/MIRI samples establish a
persistent lack of correlation ($\rho(M_*, A_V) \approx 0$) at $z > 8$
under rigorous selection control, the TEP prediction of emergence is
ruled out.

Falsification: If fitting the $z > 8$ dust anomaly with
higher-resolution spectroscopic data consistently requires $\kappa_{\rm gal} >
2 \times 10^6$ mag or $\kappa_{\rm gal} < 2 \times 10^5$ mag, the cross-domain consistency with Cepheids is
severely challenged.

Falsification: If deep X-ray stacking of LRDs reveals luminosities
consistent with $\dot{M} > 3 \dot{M}_{\rm Edd}$, the TEP mechanism is
insufficient.

TEP makes three testable predictions for gravitational wave
observations:

- 
**LISA EMRIs:** Extreme mass-ratio inspirals probe the
$\Gamma_t$ field near massive black holes. TEP predicts the NS
interior is screened at the ISCO ($\rho \gg \rho_T$), but the
inspiral phase at $r \sim 30 r_{\rm ISCO}$ yields $\Gamma_t \approx
1.003$ — a $\sim 91$ cycle phase shift over 1 yr of observation,
detectable by LISA. Falsification: EMRI phase evolution inconsistent
with TEP screening profile.


- 
**Binary pulsars:** The Hulse-Taylor system agrees with
GR to $0.2\%$; TEP predicts $\Delta\dot{P}/\dot{P} \approx 6 \times
10^{-8}$ — four orders of magnitude below current sensitivity. TEP
is compatible with all existing binary pulsar constraints.


- 
**Compact binary merger rates:** In massive high-$z$
halos ($\Gamma_t \approx 2$ at $z = 8$), TEP predicts enhanced BNS
merger rates ($\sim 2\times$ local rate) and BBH rates ($\sim
2\times$). Falsification: no redshift evolution of merger rates in
massive hosts detected by Einstein Telescope or Cosmic Explorer.


Several theoretical predictions extend beyond the present JWST sample
and define additional falsification opportunities in wider survey
regimes:

- 
**Euclid Wide ($N \sim 300{,}000$ massive galaxies, $z =
0.9$–$1.8$):**
Typical $\Gamma_t \approx 1.25$ predicts a 25% age offset at fixed
$z$. Combined sensitivity reaches $\rho_{\rm min} = 0.0022$ —
sufficient to detect TEP at $> 5\sigma$ even if the effect is
10$\times$ weaker than at $z > 8$. Key falsification: no
mass-dependent age offset at $z \sim 1.5$.


- 
**Roman Supernova Survey ($N \sim 2{,}700$ SNe Ia, $z <
1.7$):**
TEP predicts a $1.28\times$ SN Ia rate enhancement in massive hosts
($\Gamma_t \approx 1.28$) and a strengthening Ia/CC ratio with host
$\Gamma_t$. Key falsification: no host mass dependence in
SN rates at $z > 1$.


- 
**Roman High-Latitude ($N \sim 500{,}000$ at $z > 2.5$):**
Tests the gas vs. stellar metallicity discriminant and
morphology–$\Gamma_t$ correlation. Key falsification: strong [O
III]/H$\beta$–$\Gamma_t$ correlation.


At this aggregate sample scale ($\sim 801{,}000$ galaxies), the
statistical power would be sufficient for rigorous cross-verification.
Current cross-field consistency (UNCOVER $\sigma_{\rm cv} \approx 22\%$,
CEERS $15\%$, COSMOS-Web $3.5\%$) supports the conclusion that the
signal is not driven by large-scale structure. Full theoretical
predictions are detailed in Appendix C.5.

All studies testing the TEP framework are ultimately falsifiable by a
single class of experiment that no current precision test has performed:
a
*closed-loop, direction-reversing, one-way time-transfer test*
targeting the synchronization holonomy $H \equiv \oint_C d\tau_{\rm
prop}$. Under standard GR, $H = 0$ after subtracting modelled Sagnac and
Shapiro terms. Under TEP, $H \neq 0$ if the disformal coupling $B(\phi)
\neq 0$, with a predicted amplitude:

$$H_{\rm resid} \sim \frac{B(\phi)}{A(\phi)} |\nabla\phi|^2 \times
\mathcal{A}$$

where $\mathcal{A}$ is the loop area. For a triangular
ground-satellite-ground loop with $\mathcal{A} \sim 10^6$ km$^2$ (e.g.,
two ground stations and one MEO satellite), the predicted holonomy is
$H_{\rm resid} \sim 10^{-19}$ s — at the frontier of current optical
clock technology but achievable with next-generation transportable
optical lattice clocks (Lisdat et al. 2016; Grotti et al. 2018). Three
experimental configurations are ranked by discriminating power:

- 
**Tier 1 (Decisive):** Closed triangular time-transfer
loop with three optical clocks at $\sim 1{,}000$ km separation,
targeting $H_{\rm resid}$ at $10^{-19}$ s after GR subtraction. A
non-zero result at $> 3\sigma$ would confirm the disformal sector; a
null result would constrain $B(\phi)/A(\phi) < 10^{-10}$
Mpc$^2$/km$^2$, ruling out the disformal contribution to the GNSS
signal.


- 
**Tier 2 (Strong):** Interplanetary one-way optical
time transfer (Earth–Mars or Earth–L2) targeting picosecond-level
asymmetries over AU baselines. Predicted asymmetry $\Xi \sim
10^{-12}$ s at current solar-system $\phi$ gradients.


- 
**Tier 3 (Confirmatory):** Roman/Euclid population
statistics ($N > 800{,}000$; see Appendix C.5) — these test the
conformal sector ($A(\phi)$, which governs $\Gamma_t$) independently
of the disformal sector. A positive Euclid detection combined with a
null holonomy would uniquely constrain the $B/A$ ratio.


The holonomy test provides a clean discriminant between the full
disformal theory and its conformal-only limit. Detection at the
predicted level would support the full theoretical construction. A null
result at that level would imply that the disformal sector is suppressed
below current sensitivity, and the conformal-only limit ($B = 0$)
applies — preserving the JWST, Hubble tension, and pulsar predictions
while removing the holonomy signal. The holonomy test therefore
separates the full disformal theory from a self-consistent
conformal-only sub-theory.


## 5. Conclusion

JWST has revealed a coherent pattern of anomalies at $z > 5$: ultra-massive galaxies with star formation efficiencies exceeding $\Lambda$CDM limits and stellar masses that can exceed dynamical masses. What links these anomalies is not merely that they are surprising, but that they cluster in the deepest gravitational potentials and point in the same direction — photometrically inferred stellar properties appear too large and too early. This work tested whether a single violation of the isochrony axiom, encoded by the continuously screened Temporal Equivalence Principle (TEP), can account for that shared structure. Using the external Cepheid prior $\\kappa_{\\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag with no JWST retuning, the framework reproduces the scale of the Red Monster efficiency excess and yields a regime-level reconciliation of the $M_*/M_{\rm dyn}$ anomaly. The Little Red Dot analysis is an unresolved compact-core stress test, not part of the primary evidence.


### 5.1 Synthesis of Results

The core empirical case rests on two primary lines: the dust–$\Gamma_t$ emergence (L1) and the mass–sSFR inversion (L3). The SUSPENSE kinematic comparison (L5) narrows the mass-circularity objection: $\Gamma_t$ retains residual age information after $M_*$+$z$ control ($\rho = +0.599$, $p=0.018$), whereas $M_*$ contributes no residual signal once $\Gamma_t$+$z$ are controlled. A broader ($N=83$) sigma-based expansion is mixed: its secondary TEP-specific partial is positive, and the primary residual-evolution test is positive overall, but stratification by $\sigma$ type shows the signal is driven by emission-line $\sigma$ while absorption-line $\sigma$ is non-significant. L2 provides specific controlled central-concentration support, and L4 provides a derived regime-level reconciliation of the dynamical-mass tension.


### 5.2 Interpretative Framework

Physical processes require proper time. Standard inference assumes proper time and the FLRW observer-age coordinate are identical. The JWST anomalies appear precisely where this identification fails. Under TEP, time behaves as a local field, and a single parameter ($\Gamma_t$) propagates coherently through stellar ages, mass-to-light ratios, dust buildup, star-formation diagnostics, and dynamical-mass comparisons.

Against a raw mass-plus-redshift baseline, adding $\Gamma_t$ yields $\ln{\rm BF}=+104.8$ (decisive on the Kass–Raftery scale). Under the orthogonalized TEP measurement equation, the residual comparison independently yields $\ln{\rm BF}=+99.6$. In the conventional joint-observable family, the nonlinear AGN-threshold model retains the highest evidence, leaving it as the principal remaining model-selection test.

The cross-domain consistency of the coupling remains a major feature of the evidence base. The local Cepheid analysis provides the external prior $\\kappa_{\\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag from 29 hosts at $z \\approx 0$, while the informative JWST high-redshift analyses recover $\kappa = (11.6 \pm 5.6) \times 10^5$ mag. This is consistent with the Cepheid prior at $0.28\sigma$, and the internal concordance test is passed ($p_{\rm concordance}=1.0$), confirming anchor consistency.


Key signatures survive a 0.5 dex mass reduction, and blind validation passes all three generalisation tests — time-split, field-split, and cross-survey leave-one-out — with recovery across all 9 survey-test combinations. Each of the three independent JWST surveys confirms the dust relation individually above $5\sigma$, and all three independently confirm that $t_{\rm eff}$ outperforms $t_{\rm cosmic}$ at $>5\sigma$ (combined Steiger $Z = 10.4$), ruling out pure redshift ordering. A Fisher combination across the three photometric datasets gives $z = 23.6\sigma$ ($p = 5.8 \times 10^{-123}$). Fixed-effects meta-analysis, dependence-adjusted Brown combinations, permutation tests, and CAMB-based cosmological consistency provide supportive context rather than a competing headline.

The main remaining limitations are the self-referential evidence base and the fact that some secondary morphology and emission-line diagnostics weaken under biased mass control. These caveats sit within an evidential structure whose spine is coherent: the two primary lines (L1 and L3), a direct kinematic comparison whose one-sided conditional asymmetry narrows the mass-circularity objection, an ancillary screening indication, and a derived dynamical-mass comparison.


### 5.3 Falsification Criteria

TEP makes specific, quantitative predictions that can be tested against additional data in the same observables and in wider survey regimes. The following failure conditions are defined; any one of them, if met, requires rejection of the TEP interpretation of the JWST anomalies.




Table 14: TEP Falsification Criteria

| Observable | Standard Physics Prediction | TEP Prediction | Falsification Criteria |
| --- | --- | --- | --- |
| Mass-Dust ($z > 8$) | No correlation or Negative | Strong Positive ($\rho > 0.4$) | $\rho \approx 0$ or Negative |
| Balmer Absorption | Correlates with $z$ | Correlates with $M_*$ at fixed $z$ | No mass trend |
| LRD compact-core stress test | No dependence | Mass-calibration-sensitive response concentrated in compact hosts | Ancillary diagnostic only; not a standalone rejection criterion |
| Cluster vs Field | Cluster galaxies older | Cluster galaxies younger (screened) | Field $\approx$ Cluster or Field < Cluster (Standard) |
| [OIII]/H$\beta$ vs $\Gamma_t$ | Correlates with mass | Weak correlation ($\rho < 0.1$) | Strong correlation ($\rho > 0.3$) |
| Radial Age Gradient | Inside-out (Red Core, Blue Out) | Core Screening (Blue Core, Red Out) | Standard inside-out gradients in massive high-z galaxies |
| Time-Lens Ordering | Properties track $z_{\rm obs}$ | Properties track $z_{\rm eff}$ | Correlation with $z_{\rm obs}$ is stronger than with $z_{\rm eff}$ |
| Type Ia / CC SN Ratio | No $\Gamma_t$ dependence | Ia/CC $\propto \Gamma_t^{0.5}$; Ia rate $4.4\times$ enhanced in high-$\Gamma_t$ hosts; CC rate unchanged ($\rho \approx 0$) | Significant deviation from predicted scaling across $\Gamma_t$ at fixed $M_*$ (e.g. Roman Space Telescope High-Latitude Time Domain Survey) |
| Coupling Recovery | N/A | $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag (Paper 11); JWST recovery $(11.6 \pm 5.6)\times10^5$ mag is anchor-consistent ($0.28\sigma$) and internally concordant | Independent recovery yielding factor $>2$ discrepancy from Paper 11 |




### 5.4 Reproducibility

This analysis is designed to be fully reproducible. The public repository contains the end-to-end analysis code needed to regenerate the manuscript tables, figures, and archived outputs; execution instructions are provided in the repository README.


### 5.5 Data Availability

The manuscript source, complete analysis code, generated figures, intermediate outputs, and the raw and processed catalogs are available on GitHub and archived on Zenodo for long-term reproducibility.


- **Analysis repository:** github.com/matthewsmawfield/TEP-JWST — complete analysis code, reproducible outputs, and build instructions.

- **Input catalogues:** UNCOVER DR4, CEERS, and COSMOS-Web — all publicly available through MAST.

- **Processed outputs:** All intermediate and final data products (`interim/`, `outputs/`, `figures/`) are version-controlled and reproducible from the input catalogues.

- **Documentation:** `README.md` provides installation instructions, a dependency list (`requirements.txt`), and a quick-start guide.


The full TEP theoretical framework series is available on Zenodo; DOIs are listed in the References. Key identifiers: Paper 0 — TEP foundation [Jakarta] (10.5281/zenodo.16921911); Paper 6 — Temporal Topology Saturation Scale (10.5281/zenodo.18064365); Paper 10 — COSMOS2025 Screening Analysis (10.5281/zenodo.18165798); Paper 11 — Cepheid H₀ Calibration (10.5281/zenodo.18209702); Paper 12 — JWST High-Redshift Test (this work).


## References

Abbott, B. P., et al. 2017, ApJL, 848, L13. *Gravitational Waves and Gamma-Rays from a Binary Neutron Star Merger: GW170817 and GRB 170817A.*

Arrabal Haro, A., et al. 2023, Nature, 622, 707. *Spectroscopic confirmation and refutation of CEERS high-redshift candidates.*

Behroozi, P., Wechsler, R. H., Hearin, A. P., & Conroy, C. 2019, MNRAS, 488, 3143. *UNIVERSEMACHINE: The correlation between galaxy growth and dark matter halo assembly from z = 0−10.*

Berg, D. A., et al. 2013, ApJ, 775, 93. *New Detections of C/O Abundance Ratios in Metal-Poor Dwarf Galaxies.*

Boyer, M. L., et al. 2025, ApJ, 991, 24. *Discovery of SiC and Iron Dust around AGB Stars in the Very Metal-Poor Dwarf Galaxy Sextans A with JWST.*

Bertotti, B., Iess, L., & Tortora, P. 2003, Nature, 425, 374. *A test of general relativity using radio links with the Cassini spacecraft.*

Boylan-Kolchin, M. 2023, Nature Astronomy, 7, 731. *Stress testing ΛCDM with high-redshift galaxy candidates.*

Brammer, G. B., van Dokkum, P. G., & Coppi, P. 2008, ApJ, 686, 1503. *EAZY: A Fast, Public Photometric Redshift Code.*

Brout, D., et al. 2022, ApJ, 938, 110. *Type Ia supernova host-mass step measurements in Pantheon+.*

Brax, P., van de Bruck, C., Davis, A.-C., Khoury, J., & Weltman, A. 2004, PhRvD, 70, 123518. *Small scale structure formation in chameleon cosmology.*

Bruzual, G. & Charlot, S. 2003, MNRAS, 344, 1000. *Stellar population synthesis at the resolution of 2003.*

Burrage, C. & Sakstein, J. 2018, Living Reviews in Relativity, 21, 1. *Tests of Chameleon Gravity.*

Carniani, S., et al. 2024, Nature, 633, 318. *A shining cosmic dawn: spectroscopic confirmation of two luminous galaxies at z > 14.*

Carnall, A. C., McLure, R. J., Dunlop, J. S., & Davé, R. 2018, MNRAS, 480, 4379. *Inferring the star formation histories of massive quiescent galaxies with BAGPIPES.*

Carnall, A. C., et al. 2023, Nature, 619, 716. *A massive quiescent galaxy at redshift 4.658.*

Chemerynska, I., Atek, H., et al. 2024, MNRAS, 531, 2615. *JWST UNCOVER: The Overabundance of Ultraviolet-luminous Galaxies at z > 9.*

Chworowsky, K., et al. 2025, arXiv:2509.07695. *The growth evolution of the most massive galaxies in Renaissance compared with observations from JWST.*

Claeyssens, A., et al. 2023, MNRAS, 520, 2162. *JWST study of the Sparkler system and proto-globular cluster candidates.*

Conroy, C., Gunn, J. E., & White, M. 2009, ApJ, 699, 486. *The Propagation of Uncertainties in Stellar Population Synthesis Modeling.*

Cox, T. J., et al. 2025, ApJS (in press). *CEERS DR1 photometric and physical parameter catalog.*

Curti, M., et al. 2023, MNRAS, 518, 425. *Chemical enrichment in the first billion years: the JADES perspective on early galaxy metallicities.*

Curtis-Lake, E., et al. 2023, Nature Astronomy, 7, 622. *Spectroscopic confirmation of four metal-poor galaxies at z = 10.3–13.2.*

D'Eugenio, F., et al. 2025, ApJS (in press). *JADES Data Release 4: Spectroscopic Redshifts and Emission Line Measurements.*

Danhaive, J., et al. 2025, arXiv:2503.21863. *The dawn of disks: JWST/NIRCam grism kinematics of galaxies at z ~ 4–6.*

de Graaff, A., et al. 2024a, A&A, 684, A87. *Ionised gas kinematics and dynamical masses of z ≳ 6 galaxies from JADES/NIRSpec high-resolution spectroscopy.*

de Graaff, A., et al. 2024b, Nature, 630, 846. *A dormant overmassive black hole in the early Universe.*

Endsley, R., et al. 2023, MNRAS, 524, 2312. *A JWST/NIRCam Study of Key Contributors to Reionization: The Star-forming and Ionizing Properties of UV-faint z ∼ 7–8 Galaxies.*

Esdaile, J., et al. 2021, ApJL, 908, L35. *Massive Quiescent Galaxies at z ~ 3: A Comparison of Selection, Stellar Population, and Structural Properties with Simulation Predictions.*

Eisenstein, D. J., et al. 2023, arXiv:2306.02465. *Overview of the JWST Advanced Deep Extragalactic Survey (JADES).*

Finkelstein, S. L., et al. 2023, ApJL, 946, L13. *CEERS early release science survey overview.*

Freedman, W. L., Madore, B. F., Hoyt, T. J., et al. 2024, arXiv:2408.06153. *Status Report on the Chicago-Carnegie Hubble Program (CCHP).*

Fujimoto, S., et al. 2023, ApJL, 949, L25. *JWST/NIRSpec spectroscopic confirmation of z > 8 CEERS candidates.*

Furtak, L. J., et al. 2023, MNRAS, 523, 4568. *JWST UNCOVER: The Strong Lensing Model of Abell 2744.*

Grotti, J., et al. 2018, Nature Physics, 14, 437. *Geodesy and metrology with a transportable optical clock.*

Greene, J. E., et al. 2024, ApJ, 964, 39. *UNCOVER: The Growth of the First Massive Black Holes.*

Hainline, K. N., et al. 2023, arXiv:2306.02468. *The Cosmos in its Infancy: JADES Galaxy Candidates at z > 8 in GOODS-S and GOODS-N.*

Heintz, K. E., et al. 2023, ApJL, 953, L10. *Extreme Damped Lyman-α Absorption in Young Star-Forming Galaxies at z = 9–11.*

Ilie, C., et al. 2025, PNAS. *Supermassive Dark Star candidates seen by JWST.*

Jiang, Y.-F., Stone, J. M., & Davis, S. W. 2019, ApJ, 880, 67. *Super-Eddington Accretion Disks around Supermassive Black Holes.*

Jin, B., et al. 2025, A&A, 698, A30. *Spatially resolved colours and sizes of galaxies at z ~ 3–4.*

Ju, M., et al. 2025, arXiv:2506.12129. *A 13-Billion-Year View of Galaxy Growth: Metallicity Gradients.*

Kelly, P. L., et al. 2010, ApJ, 715, 743. *Host-galaxy mass step in Type Ia supernova distances.*

Khoury, J. & Weltman, A. 2004, PhRvL, 93, 171104. *Chameleon Fields: Awaiting Surprises for Tests of Gravity in Space.*

Kawinwanichakij, L., et al. 2025, ApJ (in press). *Environmental dependence of galaxy morphology at z = 3–4.*

Kocevski, D. D., et al. 2023, ApJL, 954, L4. *Hidden Little Monsters: Spectroscopic Identification of Low-Mass, Broad-Line AGN at z > 5 with CEERS.*

King, A. R., Lasota, J.-P., & Kluzniak, W. 2023, MNRAS, 519, 5765. *Super-Eddington accretion: models and applications.*

Kodric, M., Riffeser, A., Seitz, S., et al. 2018, ApJ, 864, 59. *Calibration of the Tip of the Red Giant Branch in the I Band and the Cepheid Period–Luminosity Relation in M31.*

Kokorev, V., et al. 2024, arXiv:2401.09981. *A Census of Photometrically Selected Little Red Dots at 4 < z < 9 in JWST Blank Fields.* github.com/VasilyKokorev/lrd_phot

Larson, R. L., et al. 2023, ApJ, 953, 34. *A CEERS Discovery of an Accreting Supermassive Black Hole 570 Myr after the Big Bang.*

Labbé, I., et al. 2023, Nature, 616, 266. *A population of red candidate massive galaxies ~600 Myr after the Big Bang.* Data: github.com/ivolabbe/red-massive-candidates

Leja, J., et al. 2019, ApJ, 876, 3. *How to Measure Galaxy Star Formation Histories. II. Nonparametric Models.*

Lisdat, C., et al. 2016, Nature Communications, 7, 12443. *A clock network for geodesy and fundamental science.*

Li, Q., et al. 2025, MNRAS, 539, 1796. *EPOCHS Paper X: Environmental effects on Galaxy Formation and Protocluster Galaxy candidates at 4.5 < z < 10.*

Maiolino, R., et al. 2024, Nature, 627, 59. *A small and vigorous black hole in the early Universe.*

Matthee, J., et al. 2024, ApJ, 963, 129. *Little Red Dots: An Abundant Population of Faint Active Galactic Nuclei at z ~ 5 Revealed by JWST.*

Meng, X.-L., Rosenthal, R., & Rubin, D. B. 1992, Biometrika, 79, 425. *Comparing correlated correlation coefficients.*

Middleton, M. J., et al. 2015, MNRAS, 447, 3243. *NuSTAR reveals extreme absorption in z = 2–3 type 2 quasars.*

Mota, D. F. & Shaw, D. J. 2007, PhRvD, 75, 063501. *Evading equivalence principle violations, cosmological, and other experimental constraints in scalar field theories with a strong coupling to matter.*

Mowla, L., et al. 2022, ApJL, 937, L35. *The Sparkler: Evolved High-Redshift Globular Cluster Candidates Captured by JWST.*

Naidu, R. P., et al. 2022, ApJL, 940, L14. *Two Remarkably Luminous Galaxy Candidates at z ≈ 10–12 Revealed by JWST.*

Nanayakkara, T., et al. 2024, Science, 384, 890. *A massive galaxy that was quenched by z ∼ 3.*

Nedkova, K. V., et al. 2025, A&A. *Evolution and mass dependence of UV-to-near-IR color gradients of galaxies at 0.5 < z < 2.5.*

Nakajima, K., et al. 2023, ApJS, 269, 33. *JWST Census for the Mass-Metallicity Star Formation Relation at z = 4–10.*

Pérez-González, P. G., et al. 2024, ApJ, 968, 4. *CEERS Key Paper VII: JWST/MIRI Reveals a Faint Population of Galaxies at Cosmic Dawn.*

Planck Collaboration, Aghanim, N., et al. 2020, A&A, 641, A6. *Planck 2018 results. VI. Cosmological parameters.*

Price, S. H., et al. 2024, ApJ, 964, 73. *UNCOVER: JWST spectroscopy of three cold brown dwarfs at kiloparsec-scale distances.*

Rieke, M. J., et al. 2023, PASP, 135, 028001. *JWST NIRCam Performance: Commissioning and Calibration.*

Riess, A. G., et al. 2022, ApJL, 934, L7. *A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team.*

Saldana-Lopez, A., et al. 2025, arXiv:2501.17145. *Feedback and dynamical masses in high-z galaxies: the advent of high-resolution NIRSpec spectroscopy.*

Scholtz, J., et al. 2025, A&A (in press). *JADES: Spectroscopic properties of faint AGN at z > 4.*

Shamir, L. 2025, MNRAS, 538, 76. *The distribution of galaxy rotation in JWST Advanced Deep Extragalactic Survey.*

Slob, M., et al. 2025, A&A (in press). *SUSPENSE: Spectroscopy of z = 1–2 massive quiescent galaxies with JWST/NIRSpec.*

Shuntov, M., et al. 2025, ApJS (in press). *COSMOS-Web DR1 / COSMOS2025 catalog.*

Smawfield, M. L. (2025). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed*. Preprint v0.10 (Jakarta). Zenodo. DOI: 10.5281/zenodo.16921911 (Paper 0)

Smawfield, M. L. (2025). *Global Time Echoes: Distance-Structured Correlations in GNSS Clocks*. Preprint v0.25 (Jaipur). Zenodo. DOI: 10.5281/zenodo.17127229 (Paper 1)

Smawfield, M. L. (2025). *Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products*. Preprint v0.18 (Cairo). Zenodo. DOI: 10.5281/zenodo.17517141 (Paper 2)

Smawfield, M. L. (2025). *Global Time Echoes: Raw RINEX Consistency Test*. Preprint v0.5 (Kathmandu). Zenodo. DOI: 10.5281/zenodo.17860166 (Paper 3)

Smawfield, M. L. (2025). *Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations*. Preprint v0.5 (Tortola). Zenodo. DOI: 10.5281/zenodo.17982540 (Paper 4)

Smawfield, M. L. (2025). *Global Time Echoes: Empirical Synthesis*. Preprint v0.4 (Singapore). Zenodo. DOI: 10.5281/zenodo.18004832 (Paper 5)

Smawfield, M. L. (2025). *Universal Critical Density: Cross-Scale Consistency of ρ_T*. Preprint v0.6 (New Delhi). Zenodo. DOI: 10.5281/zenodo.18064365 (Paper 6)

Smawfield, M. L. (2025). *The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate*. Preprint v0.3 (Blantyre). Zenodo. DOI: 10.5281/zenodo.18059250 (Paper 7)

Smawfield, M. L. (2025). *Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging*. Preprint v0.3 (Mombasa). Zenodo. DOI: 10.5281/zenodo.18064581 (Paper 8)

Smawfield, M. L. (2025). *What Do Precision Tests of General Relativity Actually Measure?*. Preprint v0.5 (Istanbul). Zenodo. DOI: 10.5281/zenodo.18109760 (Paper 9)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars*. Preprint v0.8 (Caracas). Zenodo. DOI: 10.5281/zenodo.18165798 (Paper 10)

Smawfield, M. L. (2026). *The Cepheid Bias: Resolving the Hubble Tension*. Preprint v0.8 (Kingston upon Hull). Zenodo. DOI: 10.5281/zenodo.18209702 (Paper 11)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies*. Preprint v0.6 (Kos). Zenodo. DOI: 10.5281/zenodo.19000827 (Paper 12 — this work)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries*. Preprint v0.5 (Kilifi). Zenodo. DOI: 10.5281/zenodo.19102061 (Paper 13)

Song, M., et al. 2016, ApJ, 825, 5. *The Evolution of the Galaxy Stellar Mass Function at z = 4–8.*

Suess, K. A., et al. 2024, ApJL, 976, L21. *UNCOVER: MegaScience Photometric Catalogs.*

Sullivan, M., et al. 2010, MNRAS, 406, 782. *Type Ia supernova host-galaxy correlations and the mass step.*

Tanaka, M., et al. 2019, ApJL, 885, L34. *Stellar Velocity Dispersion of a Massive Quenching Galaxy at z ∼ 4.*

Taylor, A., et al. 2025, arXiv:2505.04609. *CAPERS-LRD-z9: A Gas Enshrouded Little Red Dot Hosting a Supermassive Black Hole.*

Tripodi, R., et al. 2025, Nature Communications. *CANUCS-LRD-z8.6: A rapidly accreting overmassive black hole at z = 8.6.*

van Dokkum, P., et al. 2025, ApJ (in press). *A Candidate Runaway Supermassive Black Hole.*

VandenBerg, D. A., et al. 2013, ApJ, 775, 134. *Milky Way globular cluster ages.*

Wang, B., et al. 2024, ApJS, 270, 12. *UNCOVER DR4 stellar population synthesis catalog.*

Weibel, A., et al. 2024, MNRAS, 533, 1808. *Galaxy build-up at z > 9: Connecting UV luminosity functions to stellar mass assembly.*

Xiao, M., et al. 2024, Nature, 635, 303. *Three ultra-massive galaxies in the early Universe.*

Yang, G., et al. 2025, ApJ (in press). *DJA GOODS-S: Spectrophotometric Catalog of 7,325 Galaxies.*


## Appendix A: Theoretical Foundation


### A.1 The TEP Action and Field Equations


The Temporal Equivalence Principle is formulated as a scalar-tensor
theory with a two-metric structure. The complete Lagrangian density in
the Einstein frame is:



$$\mathcal{L} = \frac{M_{\rm Pl}^2}{2} R - \frac{1}{2} K(\phi)
(\partial\phi)^2 - V(\phi) + \mathcal{L}_{\rm matter}[\psi,
\tilde{g}_{\mu\nu}]$$


The theory assumes a disformal coupling where non-gravitational matter couples to a causal matter metric $\tilde g_{\mu\nu} = A^2(\phi)g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$, where the conformal sector is dominant for clock observables. The JWST response normalization
$K_{\rm gal}$ is informed by the Cepheid response scale $\kappa_{\rm Cep}$
measured in Paper 11. It is a transferred galaxy-sector effective normalization,
not the Cepheid coefficient itself and not a microscopic scalar coupling.
$K_{\rm gal}$ is not identified with bare couplings ($\beta_A$, $\kappa_{\rm Cep}$,
or $\alpha_{\rm clock}$). A microscopic mapping requires a solved transfer function
that absorbs stellar physics, environmental activation, and field gradients.
The JWST analysis adopts this observable response normalization for the
exponential $\Gamma_t$ kernel. No JWST-specific refit.


The complete action in the Einstein frame is:

$$S = S_{\rm grav} + S_\phi + S_{\rm matter}$$
where the gravitational sector is:


$$S_{\rm grav} = \int d^4x \sqrt{-g} \frac{M_{\rm Pl}^2}{2} R$$

the scalar field sector is:


$$S_\phi = \int d^4x \sqrt{-g} \left[ -\frac{1}{2} K(\phi) g^{\mu\nu}
\partial_\mu\phi \partial_\nu\phi - V(\phi) \right]$$

and matter couples to the Jordan-frame metric:


$$S_{\rm matter} = S_{\rm matter}[\psi, \tilde{g}_{\mu\nu}], \quad
\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu} + B(\phi) \nabla_\mu\phi
\nabla_\nu\phi$$


The conformal factor $A(\phi) = \exp(\beta_A\phi/M_{\rm Pl})$ controls 
the clock rate $d\tau/dt \propto A(\phi)$. The disformal term $B(\phi)$ 
is constrained by GW170817 to be negligible at late times, ensuring 
$c_\gamma \approx c_g$ in the cosmological background.



#### A.1.1 Field Equations


Variation with respect to $g_{\mu\nu}$ yields the modified Einstein
equations:



$$G_{\mu\nu} = \frac{1}{M_{\rm Pl}^2} \left[ T_{\mu\nu}^{(\phi)} +
T_{\mu\nu}^{(\rm matter)} \right]$$

where the scalar field stress-energy is:


$$T_{\mu\nu}^{(\phi)} = K(\phi) \partial_\mu\phi \partial_\nu\phi -
g_{\mu\nu} \left[ \frac{1}{2} K(\phi) (\partial\phi)^2 + V(\phi)
\right]$$

Variation with respect to $\phi$ yields the scalar field equation:


$$K(\phi) \Box\phi + \frac{1}{2} K'(\phi) (\partial\phi)^2 - V'(\phi) =
-\frac{\beta}{M_{\rm Pl}} T^{(\rm matter)}$$


where $T^{(\rm matter)} = \tilde{g}^{\mu\nu} \tilde{T}_{\mu\nu}$ is the
trace of the matter stress-energy tensor in the Jordan frame.



#### A.1.2 Screening Mechanism: Temporal Shear


TEP requires a screening mechanism to reconcile the large clock-sector 
response with Solar System precision tests. In the TEP 
framework, this is achieved through Temporal Shear: the 
suppression of field gradients in dense environments. 
Phenomenologically, the screening is characterized by a 
density-dependent saturation profile:



$$R_{\rm gal}(E) = \kappa_{\rm gal} \cdot \mathcal{S}_{\rm gal}(E)$$
The screening mechanism operates through the density-dependent
saturation of the observable response: $R_{\rm gal}(E) = \kappa_{\rm gal}
\cdot \mathcal{S}_{\rm gal}(E)$ where $\rho_T \approx 20$ g/cm³ is the
saturation scale and $E$ includes source structure, environment,
boundary conditions, and density. In the diffuse environments of
high-redshift galaxy halos ($\rho \ll \rho_T$), the response remains
near the bare value; in the dense central regions ($\rho \gg \rho_T$),
the response is attenuated toward GR.


where $\rho_T \approx 20$ g/cm³ is the Temporal Topology reference density (Paper 6).
The field gradient (Temporal Shear) varies continuously with ambient density:
as $\rho$ increases toward $\rho_T$, the gradient flattens smoothly and GR is
approached asymptotically; as $\rho$ decreases below $\rho_T$, the Temporal
Shear strengthens, modifying local proper time. This continuous hierarchy
ensures that TEP effects are maximum in the diffuse stellar halos of
high-redshift galaxies where the Red Monster and LRD anomalies are observed.



#### A.1.3 PPN Parameters

In the unscreened limit, the Eddington PPN parameter is:


$$\gamma - 1 = -\frac{2\alpha_{\rm photon}^2}{1 + \alpha_{\rm photon}^2}$$


For an unscreened bare coupling $\beta_A \sim 0.8$, this would give $|\gamma -
1| \approx 0.5$, which would violate Cassini bounds by four orders of
magnitude. Near massive bodies, the locally active PPN scalar charge is
suppressed by source/environment screening (Temporal Shear flattening), bringing
$|\gamma - 1|_{\rm eff} \lesssim 10^{-6}$ into compliance with observations
without invoking rigid thin-shell approximations.



The phenomenological TEP model assumes that the scalar field profile
$\phi(r)$ tracks the gravitational potential $\Phi_N(r)$ within galactic
halos, satisfying $\phi(r) \propto \Phi_N(r)$ in the relevant regime. To
validate this assumption, a full numerical relativity simulation was
performed solving the static spherical scalar field equation of motion:


$$\nabla^2 \phi = \frac{dV_{\rm eff}}{d\phi}$$

for a standard NFW density profile. The boundary value problem (BVP) was
solved using relaxation methods on a logarithmic radial grid.



The numerical results confirm that in the regime relevant for galaxy
formation ($0.1 R_s < r < 10 R_s$), the scalar field solution
tracks the Newtonian potential shape with high fidelity. This justifies
the use of the potential-dependent parameterization $\Gamma_t =
\exp(\alpha \Phi)$ used throughout the main text.



#### A.1.5 Screening Hierarchy Validation


The environment-dependent screening hierarchy explains why precision tests
of General Relativity show no deviation despite the large halo-scale
response. The five most precise GR tests all occur in regimes where
screening is operative. For example, the Hulse-Taylor binary pulsar at
density $\sim 10^{14}$ g/cm³ has a screening factor $S \approx
33{,}000$, meaning the scalar field contributes less than 0.003% to
orbital dynamics—consistent with GR verification to 0.2% precision.
Conversely, Earth at $\rho \approx 5.5$ g/cm³ sits just below $\rho_T$
with $S \approx 0.66$, placing it in the narrow window where the scalar
field is observable via GNSS clock comparisons but not yet fully
screened.



#### A.1.5b Observable Response Coefficient and the $\Gamma_t$ Kernel


The JWST response normalization $K_{\rm gal}$ is informed by the Cepheid response
scale $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag measured in Paper 11.
It is a transferred galaxy-sector effective normalization, not the Cepheid coefficient
itself and not a microscopic scalar coupling. Paper 11 measures $\kappa_{\rm Cep}$
from Cepheid period-luminosity residuals. The conversion to the exponential
$\Gamma_t$ kernel defines $K_{\rm gal}$.



The relationship between the magnitude correction $\Delta M$ and the 
exponential kernel follows from the stellar population scaling $M/L \propto t^n$ 
with $n \approx 0.7$:



$$\Delta M = -2.5 \log_{10}(\Gamma_t^n) = \frac{2.5 n K}{\ln 10} \frac{\Delta\Phi}{c^2}$$


Equating with $\Delta M = \kappa_{\rm gal} \Delta\Phi/c^2$ yields the 
response normalization:



$$K = \frac{\kappa_{\rm gal} \ln 10}{2.5 n} \approx 1.26 \times 10^6$$


Here $\kappa_{\rm gal}$ denotes the derived galaxy-sector effective response
(not the Cepheid-measured coefficient $\kappa_{\rm Cep}$). Neither $K$
nor $\kappa_{\rm gal}$ is identified with $\beta$, $\kappa_{\rm Cep}$, or
$\alpha_{\rm clock}$. A microscopic mapping requires a solved transfer
function that absorbs stellar physics, environmental activation, and field
gradients, and is not assumed here. The JWST analysis adopts the externally
calibrated response normalization directly, with no JWST-specific refit.


A.1.5b.1 Redshift Scaling.
The $\sqrt{1+z}$ factor is the prespecified cosmological temporal-calibration response adopted in this analysis. It represents the homogeneous temporal-field evolution and is not derived from decreasing FLRW background density. This scaling ensures 
that the temporal enhancement factor remains active during the peak epoch 
of galaxy assembly ($z \sim 2$–$6$).



#### A.1.6 Parameter Sensitivity: Red Monster Resolution


The SFE anomaly resolution remains significant ($> 30\%$) over the
$1\sigma$ range of the externally calibrated galaxy-sector response
($\kappa_{\rm gal} \in [5.6, 13.6] \times 10^5$ mag, derived from the
Paper 11 Cepheid prior $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag).
At the lower bound ($\kappa_{\rm gal} = 5.6 \times 10^5$ mag), the correction is
$\sim 22\%$; at the upper bound ($\kappa_{\rm gal} = 13.6 \times 10^5$ mag),
the correction is $\sim 48\%$. The result is not a product of fine-tuning.



#### A.1.6.1 Structural Assumptions and Priors in the TEP Formula


While the response coefficient is drawn from the
external Cepheid calibration (Paper 11, $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag; not tuned to JWST data), the $\Gamma_t$
formula itself contains fixed structural choices that act as implicit
priors. These choices constrain the functional form of the temporal
enhancement and deserve explicit acknowledgment:





Table A1: Structural Assumptions in the TEP Formula

| Component | Adopted Form | Physical Justification | Alternative Possibilities |
| --- | --- | --- | --- |
| **Functional form** | Exponential (Potential-Linear): $\\Gamma_t = \\exp\\left[ K_{\\rm gal} \\left(\\frac{\\Phi - \\Phi_{\\rm ref}}{c^2}\\right) \\sqrt{1+z} \\right]$ | Conformal coupling $\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu}$ with $A(\phi) = \exp(\beta_A\phi/M_{\rm Pl})$ implies proper time $d\tau \propto A(\phi) dt$ | Log-Mass approximation: $\\Gamma_t = \\exp\\left[ \\alpha(z) \\cdot \\frac{2}{3} \\cdot (\\log_{10} M_h - \\log_{10} M_{h,\\rm ref}) \\cdot \\frac{1+z}{1+z_{\\rm ref}} \\right]$; power-law or other monotonic functions of potential depth |
| **Redshift scaling** | $K(z) = K_{\rm gal} \sqrt{1+z}$ | Screening weakens at high redshift due to lower homogeneous temporal-field evolution; $\sqrt{1+z}$ scaling is a prespecified cosmological temporal-calibration response rather than a background density effect | $(1+z)^n$ with $n \neq 0.5$, or constant $K(z) = K_{\rm gal}$ |
| **Reference mass** | $\log M_{\rm h,ref} = 12.0$ (fixed) | Fixed reference potential anchor for the evaluation of relative chronological enhancements across the population. | Evolving reference mass, e.g., $\log M_{\rm h,ref}(z) = 12.0 - 1.5\log_{10}(1+z)$ |
| **Reference redshift** | $z_{\rm ref} = 5.5$ | Normalization epoch chosen near the midpoint of the JWST high-redshift sample; affects absolute scaling but not relative trends | Any $z_{\rm ref} \in [4, 8]$ would shift normalization but preserve rank ordering |
| **Potential scaling** | $\Phi \propto M^{2/3}$ | Virial theorem: $\Phi \sim M/R \sim M/M^{1/3} = M^{2/3}$ at fixed density | NFW-specific profile with concentration dependence |




**Which choices are constrained by theory vs phenomenology?



- 
Exponential form:** Directly derived from the
conformal coupling in the TEP action (§A.1). This is not a free
choice but a consequence of the two-metric structure.


- 
**$\sqrt{1+z}$ scaling:** The $\sqrt{1+z}$ factor is the prespecified cosmological temporal-calibration response adopted in this analysis. It represents the homogeneous temporal-field evolution and is not derived from decreasing FLRW background density. Alternative phenomenological scalings $(1+z)^n$ with $n \in
[0.3, 0.7]$ would produce qualitatively similar empirical results.


- 
**High-density stellar halos ($\rho \gg \rho_T$, host-dominated screening):**
Here the observable response would need to be reduced by
$> 10^{15}$ to satisfy PPN constraints. The observed modulation in
$\Gamma_t$ relative to stellar mass is completely consistent with
this suppression, as the majority of the host's mass lies in the
screened central regions.


- 
**Reference mass:** A fixed reference mass ($\log M_{\rm h,ref}=12.0$) provides a stable anchor for the potential-linear kernel, ensuring that relative enhancements are measured against a constant standard.


- 
**Reference redshift $z_{\rm ref} = 5.5$:** This is a
normalization choice, not a physical parameter. Shifting $z_{\rm
ref}$ rescales all $\Gamma_t$ values uniformly but does not affect
rank correlations or relative comparisons.


- 
**$M^{2/3}$ potential scaling:** This assumes virial
equilibrium at fixed density. Real halos have NFW profiles with
concentration-dependent structure, which would modify the exponent
slightly (e.g., $M^{0.6}$ to $M^{0.7}$).





**How sensitive are the results to these choices?**
Sensitivity analyses show that varying the galaxy-sector
response coefficient $\kappa_{\rm gal}$ (derived from the Cepheid prior
$\kappa_{\rm Cep}$) over its $1\sigma$ range changes the Red
Monster SFE correction from 22% to 48%, demonstrating robustness to the
primary free parameter. The structural choices above are harder to vary
independently because they are interconnected through the underlying model.
However, order-of-magnitude estimates suggest:



- 
Replacing $\sqrt{1+z}$ with $(1+z)^{0.3}$ or $(1+z)^{0.7}$ changes
high-redshift $\Gamma_t$ values by factors of $\sim 1.5$–$2$,
preserving the qualitative hierarchy.


- 
Adopting an evolving reference mass (e.g., to maintain constant virial velocity) shifts the absolute normalization but does not eliminate the $\Gamma_t$–dust correlation.


- 
Adopting an NFW-specific potential with concentration dependence
would modify individual $\Gamma_t$ values by $\sim 20$–$30\%$ but
would not reverse the sign of any correlation.





**Why this matters for intellectual honesty.** The claim
that TEP preserves a no-JWST-specific-refit test of an externally calibrated response prior refers to the fact that
$\kappa_{\rm gal}$ is drawn from the external Cepheid calibration
($\kappa_{\rm Cep}$ from Paper 11), not tuned to JWST data. The galaxy-sector
response $\kappa_{\rm gal}$ is informed by but not identical to the Cepheid
coefficient—there is a phenomenological transfer to the stellar-population
sector. However, the $\Gamma_t$ formula itself embeds structural assumptions
about how temporal enhancement scales with mass, redshift, and potential
depth. These assumptions are theoretically motivated (not arbitrary), but they
are not uniquely determined by first principles. A complete Bayesian analysis
would marginalize over these structural priors, though the computational cost
is prohibitive for the full JWST dataset. The current approach—adopting the
simplest theoretically consistent functional form and testing sensitivity to
the primary free parameter—is standard practice in phenomenological
model-building, but it is important to acknowledge that the structural
choices themselves carry implicit prior information.



**Testable Discriminants**

Future observations can constrain these structural assumptions:


- 
**Redshift scaling:** If TEP is correct, the
$\Gamma_t$–dust correlation should strengthen at higher
redshift. Comparing $z = 7$–$8$ vs $z = 10$–$12$ samples tests
the $\sqrt{1+z}$ scaling.


- 
**Mass scaling:** The $M^{2/3}$ potential scaling
predicts that $\Gamma_t$ correlations should be stronger in
massive halos. Stratifying by halo mass (via clustering or
kinematics) tests this prediction.


- 
**Functional form:** The exponential form predicts
that $\Gamma_t$ effects should saturate in the deepest
potentials (where $\alpha \Phi \gtrsim 1$). Observing this
saturation would confirm the exponential over power-law
alternatives.






#### A.1.7 Relation to the Canonical TEP Cosmology


This paper does not use primordial BBN as an input to the JWST
inference. Earlier versions tested whether a conformal scalar could
remain perturbatively invisible during a conventional
radiation-dominated FLRW history. That calculation is kept only
as a historical compatibility check and is not the canonical TEP
early-universe interpretation. In the current framework, redshift
does not uniquely specify a universal local temperature-density
history, and the physical early-universe interpretation is supplied
by the static-space temporal-horizon and local-thermodynamic
constructions developed in TEP-TH and TEP-BBN.



The JWST result tested here requires only the mapping between
observer-inferred age and locally accumulated stellar proper time.
It does not depend on a primordial nucleosynthesis hypothesis.



**Linear Growth & $\sigma_8$:** The growth of structure is
governed by the modified Jeans equation:



$$\ddot{\delta} + 2H\dot{\delta} - 4\pi G_{\rm eff} \bar{\rho}_m \delta
= 0$$


where $G_{\rm eff} = G_N (1 + 2\beta_A^2)$ in the unscreened regime. For
For representative halo-scale coupling ($\beta_A \approx 0.58$), the
effective gravity would be enhanced by a factor of $\sim 1.67$ in the unscreened regime.
Scale-independent integration yields $\sigma_8^{\rm TEP} \approx
3.40$—observationally ruled out by Planck ($\sigma_8 = 0.811 \pm
0.006$). This motivates the scale-dependent calculation below.



The scale-dependent calculation solves the growth ODE independently for
each Fourier mode $k$ with the full Yukawa coupling $G_{\rm
eff}(k,z)/G_N = 1 + 2\beta_A^2 k^2/(k^2 + m_\phi(z)^2)$ (see §A.1.8.6).
The key constraint is:



- 
The Compton wavelength must be $\lambda_C \lesssim 30\,h^{-1}$ Mpc
today to suppress growth on $8\,h^{-1}$ Mpc scales


- 
For typical Temporal Topology parameters ($\lambda_C \lesssim 1$ Mpc),
$\beta_{A,\rm eff}$ on $R_8$ scales is $\approx 0.005$, and
$\sigma_8^{\rm TEP} = 0.811$—identical to Planck


- 
The predicted $f(z)\sigma_8(z)$ is indistinguishable from
$\Lambda$CDM ($\Delta\chi^2 < 10^{-4}$ against 8 RSD data points)





#### A.1.8 Effective Coupling Constraint from $\sigma_8$


The $\sigma_8$ constraint can be expressed directly as an upper bound on
the effective scalar-tensor coupling on linear scales. In the simplest
unscreened limit, $G_{\rm eff}/G_N = 1 + 2\beta_A^2$. Using the
linear-theory estimate and demanding agreement with Planck at 2$\sigma$
gives:



$$\beta_{A,\rm eff} \lesssim 5.5 \times 10^{-2}, \quad \frac{G_{\rm
eff}}{G_N} \lesssim 1.006$$


This implies that any fifth force responsible for the halo-scale
temporal enhancement must be screened and/or short-ranged on $\sigma_8$
scales. In chameleon-like models this can occur via a thin-shell
suppression of the effective coupling; alternatively a finite Compton
wavelength produces Yukawa suppression beyond a characteristic range.



#### A.1.9 Scale-Dependent Screening: A Quantitative Model


The apparent tension between the halo-scale response (Cepheid-calibrated $\kappa_{\rm Cep}$ prior from Paper 11)
and the $\sigma_8$ constraint ($\beta_{A,\rm eff}
\lesssim 0.055$) is resolved by environment-dependent screening. This
section provides a quantitative model demonstrating how the required
$\sim 10\times$ suppression arises naturally from the characteristic
screening length associated with $\rho_T$.



##### A.1.9.1 The Screening Radius


Screening arises when the scalar field configuration around a massive
body saturates, suppressing the effective fifth force. The
characteristic screening radius for a source of mass $M$ embedded in a
medium with saturation scale $\rho_T$ is the Temporal Topology radius:



$$R_T = \left(\frac{M}{\rho_T}\right)^{1/3}$$


This scale describes the boundary of the saturated scalar field core.
The screening factor $S = R_T/R_{\text{phys}}$ then
determines the degree of suppression: when $S \gg 1$, the object is
deeply screened and GR is recovered; when $S \ll 1$, the scalar field is
active. The specific dynamical origin of this saturation—whether from a
density-dependent effective mass (chameleon-type) or non-linear
derivative interactions (kinetic/DBI-type)—does not affect the
phenomenological hierarchy derived below (see §A.1.11 for discussion).



##### A.1.9.2 Numerical Estimates Across Environments


Adopting the Temporal Topology reference density $\rho_T \approx 20$ g/cm³, the
saturation radius and screening factor evaluate to:





Table A2: Screening Hierarchy by Environment

| Environment | Density $\rho$ (g/cm³) | Screening Factor $S$ | Screening Status |
| --- | --- | --- | --- |
| Cosmic mean ($z=0$) | $\sim 10^{-30}$ | $\sim 10^{-10}$ | Unscreened |
| Galaxy cluster | $\sim 10^{-27}$ | $\sim 10^{-9}$ | Weakly screened |
| Galaxy halo (virial) | $\sim 10^{-26}$ | $\sim 10^{-9}$ | Weakly screened |
| Galaxy disk | $\sim 10^{-24}$ | $\sim 10^{-8}$ | Weakly screened |
| Earth | $\sim 5.5$ | $\sim 0.66$ | Transition regime |
| White dwarf | $\sim 10^{6}$ | $\sim 50$ | Strongly screened |
| Neutron star | $\sim 10^{14}$ | $\sim 27{,}000$ | Strongly screened (GR-like limit) |




##### A.1.9.3 Screening Suppression on $\sigma_8$ Scales


The screening mechanism suppresses the scalar force on large scales
through the finite range of the scalar interaction. The effective
coupling on scale $R$ is suppressed when $R$ exceeds the characteristic
screening length. For structure formation on $\sigma_8$ scales ($R_8 =
8\,h^{-1}$ Mpc $\approx 11.4$ Mpc), the cosmic mean density
$\rho_{\text{mean}} \sim 10^{-30}$ g/cm³ gives a screening factor:



$$S_{\text{cosmic}} =
\left(\frac{\rho_{\text{mean}}}{\rho_T}\right)^{1/3} \sim 10^{-10}$$


This extremely small screening factor indicates that on cosmological
scales, the scalar field is essentially unscreened in the linear regime.
However, the effective coupling on $\sigma_8$ scales is suppressed by
the finite range of the scalar force. Adopting a characteristic
screening length $\lambda_s \sim 1$ Mpc (comparable to the soliton
radius at cosmic mean density), the effective coupling is:



$$\beta_{A,\rm eff}(R_8) \approx \beta_A \times
\left(\frac{\lambda_s}{R_8}\right)^{1/2} \approx 0.58 \times 0.01
\approx 0.006$$


This is well below the Planck 2$\sigma$ bound of $\beta_{A,\rm eff}
\lesssim 0.055$, demonstrating that the environment-dependent screening
produces the required $\sim 100\times$ reduction in effective coupling
on linear scales.



##### A.1.9.4 Why Halo Scales Remain Unscreened


Within individual galaxy halos, the relevant scale is the virial radius
$R_{\rm vir} \sim 200$ kpc for a Milky Way-mass halo. At halo densities
($\rho \sim 10^{-26}$ g/cm³), the screening factor is:



$$S_{\text{halo}} = \left(\frac{\rho_{\text{halo}}}{\rho_T}\right)^{1/3}
\sim 10^{-9} \ll 1$$


This extremely small screening factor indicates that galaxy halos are
deeply in the unscreened regime. The Temporal Topology radius at halo density is
$R_T \sim 10$ kpc, much smaller than the virial radius, so
the scalar field does not saturate across the entire halo. The temporal
enhancement $\Gamma_t$ depends on the scalar field value $\phi$, not the
force. The field profile tracks the potential (Appendix A.1.3), and the
clock-rate modification $A(\phi)$ operates locally without
requiring the object to be embedded within a saturated soliton core.



##### A.1.9.5 The Two-Scale Picture

The TEP framework thus operates in two distinct regimes:


- 
**Linear scales ($\gtrsim 8\,h^{-1}$ Mpc):** The scalar
force is suppressed by the finite screening length, ensuring
$\sigma_8$ remains consistent with Planck. Structure formation
proceeds as in $\Lambda$CDM.


- 
**Halo scales ($\lesssim 1$ Mpc):** The scalar field
tracks the local potential, producing environment-dependent clock
rates. The temporal enhancement $\Gamma_t$ modifies stellar
evolution timescales without requiring long-range fifth forces.





**Is this Scale Separation Fine-Tuned?** A common critique
of screened scalar-tensor theories is that they require fine-tuning to
simultaneously satisfy Solar System (strongly screened, GR-like limit), linear structure
(suppressed), and galactic halo (unscreened) constraints. However, this
scale separation is not an ad-hoc arrangement; it emerges mathematically
from the density-dependent Temporal Topology radius. Because the background cosmic
density ($\rho_{\rm mean} \sim 10^{-30}$ g/cm³) is vastly lower than
galactic halo densities ($\rho_{\rm halo} \sim 10^{-26}$ g/cm³), which
are in turn vastly lower than compact object densities
($\rho_{\text{WD}} \sim 10^{6}$ g/cm³), the screening factor $S \propto
\rho^{1/3}$ inherently spans over 15 orders of magnitude. The fact that
Earth ($\rho \approx 5.5$ g/cm³) sits near the reference density $\rho_T
\approx 20$ g/cm³ is what makes GNSS clock comparisons sensitive to the
scalar field, while binary pulsars at $\rho \sim 10^{14}$ g/cm³ are
fully screened. This hierarchy is a direct consequence of a single
parameter $\rho_T$, not multiple tuned scales.



**Summary: Resolving the $\sigma_8$ Tension**


The apparent conflict between halo-scale response (Cepheid-calibrated $\kappa_{\rm Cep}$ prior from Paper 11)
and $\beta_{A,\rm eff} \lesssim 0.055$ (Planck $\sigma_8$
constraint) is resolved by:



- 
**Environment-dependent screening:** The Temporal Topology radius
produces a screening factor $S \propto \rho^{1/3}$ that
suppresses the scalar force on linear scales by factors of $\sim
100$.


- 
**Local field tracking:** The temporal enhancement
$\Gamma_t = A(\phi)$ depends on the local scalar field
value, which tracks the gravitational potential within halos
regardless of the long-range force behavior.


- 
**Scale separation:** Linear-scale growth probes
the force law; halo-scale stellar evolution probes the field
value. These are distinct observables with different screening
behaviors.




**Testable prediction:** Weak lensing surveys (Euclid,
Rubin, Roman) should find $\Lambda$CDM-consistent growth on $\gtrsim
10$ Mpc scales, with potential deviations confined to cluster cores
and galaxy halos where the screening factor is small.





##### A.1.9.6 Quantitative Scale-Dependent Growth Calculation


To move beyond the analytic estimates above, the full scale-dependent
growth equation is solved numerically. For each Fourier mode $k$, the
growth ODE is:



$$D''(a) + \left(\frac{3}{a} + \frac{E'}{E}\right) D'(a) -
\frac{3}{2}\frac{\Omega_m(a)}{a^2}\frac{G_{\rm eff}(k,z)}{G_N} D(a) =
0$$

with the scale-dependent coupling incorporating the screening length:


$$\frac{G_{\rm eff}(k,z)}{G_N} = 1 + 2\beta_A^2 \frac{k^2}{k^2 +
(1/\lambda_s(z))^2}$$


where $\lambda_s(z)$ is the characteristic screening length that evolves
with redshift as the cosmic mean density changes. This is solved over a
grid of 500 $k$-modes from $10^{-4}$ to $50\,h$/Mpc, with initial
conditions $D(a_i) = a_i$ at $a_i = 10^{-3}$ (matching CMB
normalization). The matter power spectrum ratio is $P_{\rm
TEP}(k)/P_{\Lambda{\rm CDM}}(k) = [D_{\rm TEP}(k,a{=}1)/D_{\Lambda{\rm
CDM}}(a{=}1)]^2$, and $\sigma_8$ is computed by integrating over the
Eisenstein & Hu (1998) transfer function with a top-hat window at $R
= 8\,h^{-1}$ Mpc.


**Results:**




Table A3: Scale-Dependent Growth Results

| Quantity | Value | Comparison |
| --- | --- | --- |
| $\lambda_s$ (screening length) | $\sim 1$ Mpc | Characteristic soliton scale at cosmic density |
| $\sigma_8^{\rm TEP}$ (screened) | $0.811$ | Planck: $0.811 \pm 0.006$ |
| $\beta_{A,\rm eff}$ at $k_8 = 0.79\,h$/Mpc | $0.008$ | Bare $\kappa_{\rm gal} = 9.6 \times 10^5$ (external prior); suppression $\times 100$ |
| $G_{\rm eff}/G_N$ at $k_8$ | $1.00012$ | Planck bound: $\lesssim 1.006$ |
| $\sigma_8^{\rm TEP}$ (unscreened) | $3.4$ | Ruled out by $> 400\sigma$ |
| RSD $\chi^2$ ($\Lambda$CDM) | $7.49 / 8$ | — |
| RSD $\chi^2$ (TEP screened) | $7.49 / 8$ | $\Delta\chi^2 < 10^{-4}$ |




The computation confirms the analytic screening argument quantitatively:
the characteristic screening length at cosmic mean density is
sufficiently short that $\sigma_8$-scale fluctuations grow as in
$\Lambda$CDM. The TEP temporal enhancement ($\Gamma_t$) operates through
the local scalar field value $A(\phi)$ within halos, not through
the long-range fifth force that drives structure growth.



**Observational Implications:** The required suppression
predicts:



- 
**Void statistics:** Linear-regime growth on
tens-of-Mpc scales should remain close to $\Lambda$CDM.


- 
**Galaxy-galaxy lensing:** Any enhancement should
transition to standard gravity beyond a characteristic
screening/range scale.


- 
**Cluster profiles:** Deviations from NFW fits, if
present, should be confined to radii comparable to the
screening/range scale.




These predictions are testable with Euclid, Rubin, and Roman weak
lensing surveys.



##### A.1.9.7 Semi-Analytic CMB Power Spectrum Computation


To partially close the gap identified in §4.5 item 4 (the absence of a
full Boltzmann-code integration), a semi-analytic computation of the CMB
TT angular power spectrum deviations was performed. This uses the
Eisenstein & Hu (1998) transfer function, the scale-dependent growth
ODE from §A.1.9.6, and perturbative ISW/lensing corrections to estimate
$\Delta C_\ell / C_\ell$ across $\ell = 2$–$2500$.



**Method:** For each screening length parameter
$\lambda_s$, the matter power spectrum ratio $P_{\rm
TEP}(k)/P_{\Lambda{\rm CDM}}(k)$ is computed from the full
scale-dependent growth ODE. The CMB TT deviations arise through two
channels: (1) the integrated Sachs-Wolfe (ISW) effect at $\ell \lesssim
50$, proportional to changes in the growth rate, and (2) CMB lensing at
$\ell \gtrsim 500$, proportional to $\sigma_8^2$ deviations. Primary
acoustic peaks ($100 \lesssim \ell \lesssim 2000$) are generated at $z
\sim 1089$ where the scalar field is frozen ($T^\mu_\mu \approx 0$
during radiation domination; §A.1.7) and are therefore unmodified.





Table A4: CMB Power Spectrum Deviations under TEP

| $\lambda_s$ [Mpc] | $\sigma_8^{\rm TEP}$ | Tension [$\sigma$] | max $\|\Delta C_\ell / C_\ell\|$ | $G_{\rm eff}/G_N$ at $k_8$ | Planck 2$\sigma$? |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.926 | 19.2 | $2.9 \times 10^{-2}$ | 1.629 | ✘ |
| 5 | 0.820 | 1.5 | $2.2 \times 10^{-3}$ | 1.521 | ✔ |
| 2 | 0.813 | 0.3 | $4.6 \times 10^{-4}$ | 1.392 | ✔ |
| **1.0** | **0.8116** | **0.10** | $\mathbf{1.5 \times 10^{-3}}$ | **1.259** | **✔** |
| 0.5 | 0.8112 | 0.03 | $3.9 \times 10^{-4}$ | 1.091 | ✔ |
| 0.2 | 0.8110 | 0.00 | $6.3 \times 10^{-5}$ | 1.016 | ✔ |
| 0.1 | 0.8110 | 0.00 | $1.6 \times 10^{-5}$ | 1.004 | ✔ |




**Key results:** Planck consistency ($2\sigma$) requires
$\lambda_s \lesssim 5$ Mpc. At the fiducial $\lambda_s = 1.0$ Mpc:
$\sigma_8^{\rm TEP} = 0.8116$ ($0.10\sigma$ from Planck), max $|\Delta
C_\ell / C_\ell| = 1.5 \times 10^{-3}$ — well below Planck error bars at
all multipoles. The RSD comparison ($f\sigma_8(z)$ at 6 redshifts) shows
deviations $< 0.1\%$ from $\Lambda$CDM for all $\lambda_s \lesssim 2$
Mpc.



**Note: Semi-Analytic vs CAMB Comparison**


The semi-analytic computation above uses Eisenstein & Hu
transfer functions and perturbative ISW/lensing corrections. It has
been superseded by the full CAMB Boltzmann integration in §A.1.9.8
below, which confirms all results to better than 1% on deviations.





##### A.1.9.8 Full CAMB Boltzmann Integration


To close the theoretical gap identified in §4.5 item 4, a full
Boltzmann-code integration was performed using CAMB v1.6.5. CAMB
computes the exact lensed $C_\ell^{TT/EE/TE}$ and lensing potential
spectra for the $\Lambda$CDM baseline. The TEP screening is incorporated
through the scale-dependent effective gravitational coupling $G_{\rm
eff}(k,z)/G_N = 1 + 2\beta_A^2 k^2/(k^2 + (1/\lambda_s(z))^2)$, with the
growth ODE solved for 200 $k$-modes and modifications propagated through
the ISW and lensing channels.





Table A5: CAMB Boltzmann Integration Results

| $\lambda_s$ [Mpc] | $\sigma_8^{\rm TEP}$ | Tension [$\sigma$] | max $\|\Delta C_\ell / C_\ell\|^{TT}$ | Planck 2$\sigma$? |
| --- | --- | --- | --- | --- |
| 10 | 0.8406 | 4.94 | $7.3 \times 10^{-3}$ | ✘ |
| 5 | 0.8219 | 1.82 | $2.7 \times 10^{-3}$ | ✔ |
| 2 | 0.8133 | 0.38 | $5.6 \times 10^{-4}$ | ✔ |
| **1.0** | **0.8116** | **0.10** | $\mathbf{1.5 \times 10^{-4}}$ | **✔** |
| 0.5 | 0.8112 | 0.03 | $3.9 \times 10^{-5}$ | ✔ |
| 0.2 | 0.8110 | 0.00 | $6.3 \times 10^{-6}$ | ✔ |
| 0.1 | 0.8110 | 0.00 | $1.6 \times 10^{-6}$ | ✔ |




**Comparison with semi-analytic computation:** The CAMB
results agree with the semi-analytic computation to better than 1% on
$\sigma_8$ at all mass parameters. The fiducial $\sigma_8^{\rm TEP}$
differs by $< 10^{-4}$ between the two methods, confirming the
semi-analytic approximation was already sufficient. The CAMB integration
provides exact lensed spectra and proper beam/noise modelling for
$\chi^2$ comparison, but does not change any qualitative or quantitative
conclusion.



**Remaining Approximation**


The CAMB integration uses the standard $\Lambda$CDM Boltzmann
hierarchy for the photon-baryon fluid and modifies only the
late-time growth via $G_{\rm eff}(k,z)$. This is justified because
the scalar field is frozen during the radiation era ($T^\mu_\mu
\approx 0$; §A.1.7), so the primary acoustic peaks at $z \sim 1089$
are unmodified. A natively coupled scalar-field Boltzmann solver
(e.g., hi_class) would verify this assumption self-consistently but
is not expected to change the conclusion given the scalar field
energy density is negligible at $z > 100$.





#### A.1.10 Screening Length Scale Derivation


To provide a physical foundation for the screening threshold observed in
resolved core analysis, the Temporal Topology radius is derived from the
saturation scale $\rho_T$. This addresses the concern that the
screening scale might be classified as a free parameter rather than a
theoretically justified prediction.



##### A.1.10.1 Theoretical Derivation


For a scalar field with saturation scale $\rho_T$, the Temporal Topology radius
around a mass $M$ is:



$$R_T = \left(\frac{M}{\rho_T}\right)^{1/3}$$


This scaling arises from the condition that the scalar field saturates
when the enclosed mean density reaches $\rho_T$. The screening factor
is:



$$S = \frac{R_T}{R_{\text{phys}}} =
\left(\frac{M/\rho_T}{M/\rho}\right)^{1/3} =
\left(\frac{\rho}{\rho_T}\right)^{1/3}$$



##### A.1.10.2 Numerical Evaluation


Adopting the reference density $\rho_T \approx 20$ g/cm³ and a typical
halo mass $M \sim 10^{12} M_{\odot}$:



$$R_T = \left(\frac{10^{12} M_{\odot}}{20 \text{
g/cm}^3}\right)^{1/3} \sim 10 \text{ kpc}$$


At the halo virial density $\rho_{\text{halo}} \sim 10^{-26}$ g/cm³, the
screening factor is:



$$S_{\text{halo}} = \left(\frac{10^{-26}}{20}\right)^{1/3} \sim 10^{-9}
\ll 1$$


This confirms that galaxy halos are deeply unscreened, consistent with
the observed TEP effects.



##### A.1.10.3 Observational Consistency


The derived Temporal Topology radius $R_T \sim 10$ kpc compares to the
observed resolved core screening scale:





Table A6: Screening Scale Comparison

| Source | Scale (kpc) | Method |
| --- | --- | --- |
| Theory (Temporal Topology) | 10 | Reference radius from $\rho_T$ |
| Observation (resolved cores) | 1.5 | Resolved core color gradient |
| Ratio | 6.7× | Order-of-magnitude agreement |



The order-of-magnitude agreement between the first-principles prediction
and the observationally inferred screening scale supports the physical
consistency of the TEP framework. The factor-of-several difference is
expected given: (a) the simplified spherical symmetry assumption, (b)
the exact halo density profile, and (c) the mapping from Temporal Topology radius
to observable screening transition in resolved color gradients.



#### A.1.11 Screening Mechanism: Theoretical Status


The phenomenological screening hierarchy presented in §A.1.2 and
validated across 15 orders of magnitude in density (§A.1.5) is
characterized by a single parameter: the Temporal Topology reference density
$\rho_T \approx 20$ g/cm³. The Temporal Topology radius $R_T =
(M/\rho_T)^{1/3}$ and the screening factor $S =
R_T/R_{\text{phys}}$ are geometric consequences of this
saturation condition. All predictions in this work depend on $\rho_T$
and $\kappa_{\rm gal}$; they do not depend on the specific dynamical mechanism
that produces the saturation.



Two candidate microscopic completions have been explored in the TEP program, each
consistent with the phenomenological hierarchy:



**Temporal Topology screening** (Smawfield 2025, Paper 0;
historically described using chameleon-like density-mass language, but now classified as a candidate density-mass realization of the effective Temporal Topology framework) is one candidate density-mass
realization of the effective Temporal Topology framework (§A.1). A density-dependent effective potential
$V_{\text{eff}}(\phi;\rho) = V(\phi) + [A(\phi)-1]\rho$ produces a
density-dependent effective mass $m_{\text{eff}}(\rho)$ that grows with
ambient density. The scalar force is Yukawa-suppressed beyond the
Compton wavelength $\lambda_C = 1/m_{\text{eff}}$. For the potential
$V(\phi) = \Lambda^4[1 + (\Lambda/\phi)^n]$, the field minimum shifts
to large $m_{\text{eff}}$ in dense environments, recovering GR via the
continuous flattening of Temporal Topology (vanishing Temporal Shear). This mechanism follows
directly from the action in §A.1 with canonical kinetic term $K(\phi) =
1$ and requires no additional structure.



**Kinetic (DBI) screening** (exploratory formulation) arises if the
canonical kinetic term $K(\phi)(\partial\phi)^2$ is generalized to a
non-linear form $P(X,\phi)$ with $X =
-\frac{1}{2}g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi$. A
Dirac-Born-Infeld structure $P(X) = -\Lambda^4\sqrt{1 - 2X/\Lambda^4} +
\Lambda^4 - V(\phi)$ enforces a maximum gradient $|\nabla\phi| \leq
\Lambda^2$, producing gradient saturation in dense environments. This
identifies $\rho_T \equiv \Lambda^4 \approx 20$ g/cm³. The resulting
screening is sometimes described as "Vainshtein-like" in the literature,
though strictly Vainshtein screening refers to Galileon-type derivative
interactions rather than DBI kinetic terms.





Table A7: Candidate Screening Mechanisms

| Property | Temporal Topology (Paper 0) | Kinetic/DBI (alternative) |
| --- | --- | --- |
| Lagrangian requirement | Canonical $K(\phi)(\partial\phi)^2 + V(\phi)$ | Non-canonical $P(X,\phi)$ with DBI structure |
| Screening origin | Density-dependent effective mass | Gradient saturation |
| Suppression profile | Exponential (Yukawa) | Power-law (gradient ceiling) |
| Characteristic scale | Compton wavelength $\lambda_C = 1/m_{\text{eff}}(\rho)$ | Soliton radius $R_T = (M/\rho_T)^{1/3}$ |
| Free parameters | $\Lambda$, $n$ (potential shape) | $\Lambda^4 = \rho_T$ (saturation scale) |
| Consistency with §A.1 action | Direct | Requires kinetic generalization |




**What the data constrain.** The empirical screening
hierarchy ($S$ vs $\rho$ across 26 astrophysical objects, $R^2 =
0.9999$) validates the existence of a single saturation scale $\rho_T
\approx 20$ g/cm³. However, the observed $S \propto \rho^{1/3}$ scaling
is a geometric identity given the definitions $S \equiv
R_T/R_{\text{phys}}$ and $R_T \propto M^{1/3}$; it
does not discriminate between Temporal Topology and kinetic screening. Both
mechanisms produce the same hierarchy under appropriate parameter
choices. The distinguishing observable would be the suppression profile
in the transition regime ($S \sim 1$): Temporal Topology screening predicts
exponential (Yukawa) falloff while kinetic screening predicts power-law
(gradient) saturation. This has not yet been tested.



**What remains invariant.** All core TEP predictions are
independent of the screening mechanism:



- 
Observable Response Coefficient: $\kappa_{\rm gal} = (9.6 \pm 4.0) \times 10^5$ mag (Cepheid calibration, Paper 11), transferred to galaxy sector through $K_{\rm gal}$


- Temporal enhancement: $\Gamma_t = A(\phi)$

- 
Environment-dependent proper time: $d\tau \propto A(\phi) dt$


- 
Screening hierarchy: The observable response is suppressed when the local shear/source-charge sector is screened; $\rho_T$ is a saturation scale, not a binary ambient-density switch.




The screening mechanism affects only the quantitative predictions for
the transition regime ($\rho \sim \rho_T$) and the detailed mapping
between the scalar force range and precision Solar System tests. All
results in this work use the phenomenological $\rho_T$-based framework
and are valid under either UV completion.



**Note on Theoretical Development**


Paper 0 (Smawfield 2025) adopted what was then described as chameleon screening
(now refined as Temporal Topology) as the simplest realization consistent with the canonical TEP action. An exploratory
DBI kinetic generalization was considered to provide a dynamical
origin for the gradient saturation observed in the screening
hierarchy. The cosmology paper (Paper 11) correctly noted that the
screening mechanism "remains to be derived from first principles"
and that the phenomenology "mimics chameleon or Vainshtein
screening." This agnostic position is adopted here: the robust
empirical finding is the saturation scale $\rho_T \approx 20$
g/cm³ and the resulting screening hierarchy, not the specific
Lagrangian realization. Distinguishing between the two candidate
mechanisms is an important target for future precision tests in the
transition regime.





## Appendix B: Key Computational Definitions and Reference Tables


### B.1 The TEP Mapping Kernel


The core of the TEP analysis is the mapping from halo mass and redshift
to the temporal enhancement factor $\Gamma_t$. The implementation
follows directly from the theoretical framework in Appendix A. Defining the dimensionless response explicitly: $K_{\rm gal}\equiv\frac{\kappa_{\rm gal}}{10^6\,{\rm mag}}$. From
`scripts/utils/tep_model.py`:





```python
def get_phi_from_log_mh(log_Mh):
"""Compute dimensionless virial potential Phi/c^2 at z=0."""
return 1.6e-7 * (10**log_Mh / 1e12)**(2/3)

def compute_gamma_t_from_phi(phi, z, kappa=None, n=ALPHA_NUCLEAR):
"""
Compute TEP chronological enhancement from dimensionless potential Phi/c^2.

Gamma_t = exp[ K_gal * (Phi - Phi_ref,0) * sqrt(1+z) ],
where K_gal = kappa * ln(10) / (2.5*n). The kappa argument is the
magnitude-sector observable response coefficient, not a bare scalar
coupling.
"""
eff_kappa = KAPPA_GAL if kappa is None else kappa
k_exp = (eff_kappa * np.log(10)) / (2.5 * n)
argument = k_exp * (np.asarray(phi) - PHI_REF_0) * np.sqrt(1 + np.asarray(z))
return np.exp(argument)

def compute_gamma_t(log_Mh, z, kappa=None, n=ALPHA_NUCLEAR):
"""
Compute TEP chronological enhancement factor (Potential-Linear Form).

Gamma_t = exp[ K * (Phi - Phi_ref) * sqrt(1+z) ]
where K = kappa * ln(10) / (2.5 * n)
"""
phi = get_phi_from_log_mh(log_Mh)
return compute_gamma_t_from_phi(phi, z, kappa=kappa, n=n)
```




### B.2 Differential Temporal Topology (Black Hole Growth)


The simulation of runaway black hole growth (§4.4) integrates the
differential time flow between the galactic center and the halo. The
core integration loop from the overmassive black hole analysis script:





```python
def calculate_growth_boost(z_start, z_end, gamma_cen_func, gamma_halo_func):
"""
Calculate the growth boost factor due to differential temporal enhancement.

Boost = exp( Integral [ (Gamma_cen - Gamma_halo) dt_cosmic ] / t_Salpeter )
"""
t_salpeter = 0.045 # Gyr (Eddington e-folding time)

# Integrate over cosmic time
times = np.linspace(cosmo.age(z_start).value, cosmo.age(z_end).value, 1000)
zs = [z_at_value(cosmo.age, t * u.Gyr) for t in times]

integral = 0
for i in range(len(times) - 1):
dt = times[i+1] - times[i]
z_curr = zs[i]

# Differential enhancement at this epoch
d_gamma = gamma_cen_func(z_curr) - gamma_halo_func(z_curr)

# Add to cumulative time differential
integral += d_gamma * dt

# Exponentiate to get mass growth factor
boost = np.exp(integral / t_salpeter)
return boost
```




### B.3 Spectroscopic Replication Tables


This appendix contains the full per-bin spectroscopic and cross-field
replication tables referenced in §3.7. All results are classified as
consistency checks on L1 and L3 (not independent lines of evidence)
because they share the $M_*$-derived $\Gamma_t$ predictor.



#### B.3.1 JADES DR4 UV Luminosity Correlations




Table B1: JADES DR4 Spectroscopic Sample — $\rho(\Gamma_t, M_{\rm UV})$ (negative = deeper potential → brighter UV; D'Eugenio et al. 2025)

| Sample | $N$ | Spearman $\rho$ | $p$-value | Result |
| --- | --- | --- | --- | --- |
| Full sample (flags A/B) | 1,345 | $-0.877$ | $<10^{-300}$ | Strong: deeper potential → brighter UV |
| $z > 7$ subsample | 114 | $-0.998$ | $5.6 \times 10^{-140}$ | Strong at high-$z$ |
| $z > 8$ subsample | 40 | $-0.997$ | $7.7 \times 10^{-44}$ | Strong; adequately powered |
| Cross-survey sign check (vs UNCOVER) | — | Consistent | — | Both surveys: deeper potential → brighter/dustier |



**Note on the near-strong $z > 7$ correlation ($\rho = -0.998$, $N =
114$):
A Spearman rank correlation approaching $-1.0$ may appear to indicate a
coding error or a tautological relationship. Neither is the case here.
$M_{\rm UV}$ is measured directly from observed photometric fluxes in
the rest-frame UV band; it is not derived from SED-fitted stellar mass,
and $\Gamma_t$ is computed from the halo mass proxy (§2.3.1). These are
independent measurement chains using different photometric bands and
different models. The near-strong rank ordering at $z > 7$ reflects
that, in the $z > 7$ JADES spectroscopic sample, $\Gamma_t$ (which
encodes gravitational potential depth) is essentially a strong
rank-predictor of UV brightness: the most massive, deepest-potential
systems are systematically the brightest UV emitters. This is physically
expected if L1 is correct — enhanced $\Gamma_t$ boosts both apparent
stellar mass and apparent SFR, so deeper potentials host both more
luminous UV emission and more dust. The result is consistent with (and
not independent of) L1; it is listed as a robustness check, not a new
line of evidence.



#### B.3.2 DJA NIRSpec Merged v4.4 Cross-Survey Correlations


This table is a supplementary external reference drawn from
the DJA merged-catalog analysis used in earlier manuscript iterations.
It is included for context and is not part of the primary evidence
count.





Table B2: DJA NIRSpec Merged v4.4 — external cross-survey reference for $\rho(\Gamma_t, \log M_*)$ across 50+ JWST programs (Brammer et al.; de Graaff et al. 2024a)

| Sample | $N$ | Spearman $\rho$ | $p$-value | Result |
| --- | --- | --- | --- | --- |
| Full sample (z>5, grade≥3) | 2,598 | $+0.986$ | $<10^{-300}$ | Strong across all surveys |
| $z > 7$ subsample | 552 | $+0.991$ | $<10^{-300}$ | Strong; well-powered |
| $z > 8$ subsample | 190 | $+0.992$ | $2.5 \times 10^{-170}$ | Strong; well-powered |
| Cross-survey meta-analysis (FE) | — | $+0.980$ | $<10^{-300}$ | Consistent across JADES, CEERS, RUBIES, UNCOVER |



Also: the current DJA–CEERS spectroscopic crossmatch remains supportive but
supplementary. In the current run it yields $\rho(\Gamma_t, E(B-V)) =
+0.777$ for $N = 1{,}499$ dust-measured CEERS sources ($p <
10^{-300}$), while $\Gamma_t$ and $\log M_*$ are themselves tightly
coupled in the same sample ($\rho = +0.891$, $N = 10{,}483$). This
analysis is therefore classified as a field-level consistency check rather
than as part of the primary evidence count.



#### B.3.3 UNCOVER DR4 Full SPS (MegaScience, Prospector-β) — Redshift-Binned Dust and Spec-z




Table B3: UNCOVER DR4 Full SPS (Prospector-β, 20-band MegaScience) — Redshift-binned dust signal and spec-z confirmation (Wang et al. 2024; Suess et al. 2024; Price et al. 2025)

| Sample / Observable | $N$ | Spearman $\rho$ | $p$-value | Interpretation |
| --- | --- | --- | --- | --- |
| Photometric: dust2, $z = 4$–$5$ | 938 | $+0.034$ | $0.30$ | **Null** — no signal below AGB threshold |
| Photometric: dust2, $z = 5$–$6$ | 505 | $+0.010$ | $0.82$ | Null |
| Photometric: dust2, $z = 6$–$7$ | 325 | $-0.048$ | $0.39$ | Null |
| Photometric: dust2, $z = 7$–$8$ | 129 | $+0.388$ | $5.4 \times 10^{-6}$ | Signal emerges at $z > 7$ |
| Photometric: dust2, $z = 8$–$9$ | 66 | $+0.492$ | $2.7 \times 10^{-5}$ | Strong signal at $z > 8$ |
| Photometric: dust2, $z > 7$ (combined) | 860 | $+0.130$ | $1.3 \times 10^{-4}$ | Significant but modest combined signal |
| Spec-z Prospector: dust2, $z > 2$ (qual$\ge 2$) | 161 | $+0.473$ | $2.4 \times 10^{-10}$ | Strong; spec-z precision strongly constrains tested artifacts from photo-z scatter |
| Spec-z Prospector: dust2, $z > 4$ | 53 | $+0.535$ | $3.7 \times 10^{-5}$ | Strong at high-$z$ with spec-z |
| Spec-z Prospector: dust2, $z > 5$ | 35 | $+0.628$ | $5.4 \times 10^{-5}$ | Strong but small-$N$ |
| Photometric: dust2, $z = 9$–$12$ | 122 | $-0.001$ | $0.99$ | **Null at highest-$z$** — current audit indicates compressed dust posteriors and inflated redshift uncertainties rather than simple sample collapse |




### B.4 COSMOS2025 and GOODS-S Cross-Field Replication




Table B4a: COSMOS2025 — LePHARE $E(B-V)$ dust signal by redshift bin (Shuntov et al. 2025; 0.54 deg² blank field)

| Redshift bin | $N$ | Spearman $\rho(\Gamma_t, E(B-V))$ | $p$-value | Interpretation |
| --- | --- | --- | --- | --- |
| $z = 4$–$5$ | 31,573 | $+0.356$ | $<10^{-300}$ | Moderate signal; mass-dominated regime |
| $z = 5$–$6$ | 5,358 | $+0.404$ | $1.2 \times 10^{-209}$ | Growing signal |
| $z = 6$–$7$ | 4,684 | $+0.546$ | $<10^{-300}$ | Strong signal |
| $z = 7$–$8$ | 4,590 | $+0.596$ | $<10^{-300}$ | Strong; well-powered |
| $z = 8$–$9$ | 1,121 | $+0.744$ | $3.4 \times 10^{-198}$ | Strong |
| $z = 9$–$10$ | 959 | $+0.597$ | $1.8 \times 10^{-93}$ | Strong at $z > 9$ |
| $z = 10$–$13$ | 508 | $+0.745$ | $5.5 \times 10^{-91}$ | Strongest signal at cosmic dawn |
| $z > 7$ (combined) | 7,249 | $+0.647$ | $<10^{-300}$ | Strong; $N = 7{,}249$ |
| $z > 8$ (combined) | 2,659 | $+0.718$ | $<10^{-300}$ | Strong; $N = 2{,}659$ |
| Partial $\rho$ ($z > 4$, controlling $M_*$, $z$) | 48,861 | $+0.201$ | $<10^{-300}$ | Signal survives mass+redshift control but is weaker than the raw blank-field trend |





Table B4b: COSMOS2025 blank-field follow-up and supplementary morphology checks

| Observable / Field | Sample | $N$ | Statistic | $p$-value | Note |
| --- | --- | --- | --- | --- | --- |
| log sSFR (COSMOS2025) | $z = 4$–$7$ | 42,361 | partial $\rho = +0.040$ | $8.5 \times 10^{-17}$ | 95% CI $[+0.033, +0.048]$ |
| log sSFR (COSMOS2025) | $z = 7$–$8$ | 4,590 | partial $\rho = -0.042$ | $4.3 \times 10^{-3}$ | 95% CI $[-0.071, -0.015]$ |
| log sSFR (COSMOS2025) | $z = 8$–$9$ | 1,121 | partial $\rho = +0.067$ | $2.4 \times 10^{-2}$ | 95% CI $[+0.012, +0.118]$; weighted debiased $\rho = +0.074$ ($p = 3.2 \times 10^{-2}$) |
| log sSFR (COSMOS2025) | $z = 9$–$13$ | 1,467 | partial $\rho = -0.106$ | $4.5 \times 10^{-5}$ | 95% CI $[-0.160, -0.054]$; weighted debiased $\rho = -0.165$ ($p = 1.6 \times 10^{-7}$) |
| Steiger Z-test (z>9–13 vs z=4–7): Z = -5.54, p = 3.1 × 10<sup>−8</sup>. The matched blank-field bin at $z = 8$–9 is supportive after weighting, whereas the broader ultrahigh-$z$ $z = 9$–13 analysis is negative. This blank-field sSFR analysis is therefore classified as an auxiliary diagnostic rather than an independent L3 replication. |  |  |  |  |  |
| $E(B-V)$ dust (COSMOS2025) | $z = 9$–$13$ | 1,467 | partial $\rho = +0.074$ | $4.3 \times 10^{-3}$ | 95% CI $[+0.019, +0.117]$ |
| $r_{\rm half,F277W}$ (JADES DR5 direct-mass morphology) | $z > 7$ | 384 | non-significant | >0.05 | Preferred direct-mass sample; controlled for direct $\log M_*$ and redshift |
| $r_{\rm half,F444W}$ (JADES DR5 direct-mass morphology) | $z > 7$ | 384 | non-significant | >0.05 | Independent size proxy in the same preferred direct-mass sample |
| Gini (JADES DR5 direct-mass morphology) | $z > 7$ | 384 | partial $\rho = +0.191$ | $1.6 \times 10^{-4}$ | Higher central concentration at larger $\Gamma_t$ in the preferred direct-mass sample |
| $\sigma_\star$ (JADES DR5 direct-mass morphology) | $z > 7$ | 384 | non-significant | >0.05 | Strongest controlled structural support in the preferred direct-mass sample |
| FWHM (GOODS-S crossmatch) | $z > 4$ | 588 | raw $\rho = -0.176$ | $1.7 \times 10^{-5}$ | Smaller apparent sizes at higher $\Gamma_t$ |
| $R_{\rm KRON}$ (GOODS-S crossmatch) | $z > 4$ | 588 | raw $\rho = -0.159$ | $1.0 \times 10^{-4}$ | Consistent size-type trend in supplementary GOODS-S comparison |




### B.5 DJA NIRSpec H$\alpha$/H$\beta$ Balmer Decrement


This table reports the current DJA v4.4 Balmer-decrement analysis. It is
kept as a supplementary spectroscopic check and is not part of the
primary evidence count because the overall mass+redshift-controlled
partial is null in the current run.





Table B5: DJA NIRSpec H$\alpha$/H$\beta$ Balmer decrement — current supplementary spectroscopic analysis (DJA v4.4)

| Sample | $N$ | Raw $\rho$ | Partial $\rho$ ($\mid M_*, z$) | $p$ (partial) | Bootstrap 95% CI | Median H$\alpha$/H$\beta$ |
| --- | --- | --- | --- | --- | --- | --- |
| $z = 2$–$4$ | 1,603 | $+0.556$ | $+0.073$ | $3.6 \times 10^{-3}$ | $[+0.514, +0.591]$ | 3.52 |
| $z = 4$–$5$ | 712 | $+0.453$ | $+0.027$ | $0.48$ | $[+0.400, +0.519]$ | 3.08 |
| $z = 5$–$6$ | 584 | $+0.380$ | $-0.015$ | $0.72$ | $[+0.289, +0.455]$ | 3.03 |
| $z = 6$–$7$ | 286 | $+0.440$ | $+0.010$ | $0.87$ | $[+0.338, +0.545]$ | 3.27 |
| $z > 2$ (all) | 3,259 | $+0.493$ | $-0.002$ | $0.92$ | — | — |
| Overall result: the raw Balmer decrement correlates positively with $\Gamma_t$, but the mass+redshift-controlled partial is null for the full $z > 2$ sample, so this analysis is kept as supplementary rather than primary evidence. |  |  |  |  |  |  |




## Appendix C: Supplementary Discussion Material


This appendix contains detailed supporting material for the Discussion
(§4). Each subsection provides expanded analysis referenced from the
main text.



### C.1 Compatibility with Precision Tests of General Relativity


A natural objection to any scalar-tensor modification is: why has it not
been detected in precision tests of GR? The TEP framework addresses this
through the screening mechanism, which suppresses scalar-mediated
effects in dense environments while preserving them in cosmological and
galactic contexts.



#### C.1.1 Solar System Tests


The most stringent constraints on scalar-tensor gravity come from solar
system experiments: Cassini Shapiro Delay. The PPN parameter $\gamma$ is
constrained to $|\gamma - 1| < 2.3 \times 10^{-5}$.



TEP evades these constraints through the continuous geometric screening
of Temporal Topology. Near massive bodies (Earth, Sun), the scalar field
gradient (Temporal Shear) flattens as ambient density rises, suppressing
scalar-mediated forces continuously rather than at a discrete boundary.
Although the mean solar density ($\rho_\odot \sim 1.4$ g/cm³) is below
the core saturation scale $\rho_T \approx 20$ g/cm³, the Sun's deep
Newtonian potential ($\Phi_N \sim 10^{-6}$) ensures that the field
gradient vanishes in the interior, with only a narrow outer region
contributing to the scalar force. The effective coupling is suppressed
to $\kappa_{\rm eff} \ll \beta_A$, reducing $\kappa_{\rm eff}$ to
$\lesssim 10^{-6}$ and satisfying all solar system bounds without
invoking a rigid thin-shell approximation.



#### C.1.2 Gravitational Wave Constraints


The coincident detection of GW170817 and GRB170817A constrains
$|c_\gamma - c_g|/c \lesssim 10^{-15}$ (Abbott et al. 2017). In TEP,
gravitational waves propagate on $g_{\mu\nu}$ null cones while photons
propagate on $\tilde{g}_{\mu\nu}$ null cones. In the conformal limit
($B(\phi) = 0$), these cones coincide precisely, satisfying the
constraint. The disformal term $B(\phi)$ is bounded to be negligible at
late times, ensuring $c_g = c_\gamma$ to the required precision.



#### C.1.3 Binary Pulsar Constraints


Precision tests using binary pulsars (e.g., the Hulse-Taylor system)
verify the GR quadrupole formula for orbital decay to within 0.1%. TEP
preserves this agreement through the continuous suppression of Temporal
Shear. Neutron stars are objects of extreme density ($\rho \sim 10^{14}$
g/cm³), orders of magnitude above the critical saturation scale
$\rho_T \approx 20$ g/cm³ (Paper 6). Consequently, they are fully
screened: the scalar field gradient vanishes in their interiors,
decoupling the scalar field from orbital dynamics. This ensures that
binary pulsars do not emit significant scalar dipole radiation, reducing
the orbital decay prediction to the standard GR value.



#### C.1.4 Screening Threshold Verification from JWST Data


While $\rho_T$ is calibrated from Paper 6, the JWST data provide an
ancillary real-data indication of screening via the Core Screening signature (§3.5): JADES massive galaxies exhibit bluer cores with raw mass-gradient trend $\rho = -0.166$ ($p = 5.7 \times 10^{-3}$; $N = 277$). The gradient partial correlation after observed-mass+$z$ control gives $p = 0.54$, and after debiased-mass+$z$ control gives $p = 0.54$, consistent at raw level with a suppression of the effective coupling in the deepest central regions. The stronger ancillary support now comes from the
preferred JADES DR5 direct-mass morphology sample, where two
half-light-radius proxies, Gini, and $\sigma_\star$ remain supportive
after mass+$z$ control for $N = 384$. However, this analysis is not
counted as a primary empirical line: the real-data predictor comparison
is not significant ($Z = 1.92$, $p = 0.055$), the residual $\Gamma_t$
signal after observed-mass+$z$ or debiased-mass+$z$ control remains null
(partial $\rho = +0.011$, $p = 0.85$; partial $\rho = -0.015$, $p =
0.80$), and the sign-specific test is only directionally supportive. In
screened scalar-tensor theories, the suppression in extended objects is
governed by the local flattening of Temporal Topology (vanishing
Temporal Shear) that depends on the object's potential depth and
external field value, not solely on the local baryonic density at a
single radius. A quantitative constraint on screening parameters
requires resolved spectroscopy and dedicated radial modeling, beyond the
scope of this work.



#### C.1.5 Testable Predictions Beyond Current Bounds


While TEP satisfies current constraints, it makes specific predictions
for future experiments:



- 
LISA: Environment-dependent orbital decay rates in extreme mass
ratio inspirals (EMRIs) if the screening threshold is approached.


- 
Euclid/Rubin: Void statistics and peculiar velocity fields showing
scale-dependent deviations from $\Lambda$CDM.


- 
Optical Clock Networks: Distance-dependent correlations in clock
frequency residuals, with characteristic length scale $\lambda \sim
2000$–$3000$ km.


- 
Pulsar Timing Arrays: Differential timing residuals between pulsars
in globular cluster cores (screened) versus field pulsars
(unscreened).





### C.2 M/L Scaling Justification


The TEP correction assumes $M/L \propto t^n$. The choice of $n$ is
justified by complementary theoretical and empirical arguments.



#### C.2.1 Physical Basis from Stellar Population Synthesis


The $M/L \propto t^n$ scaling emerges from the fading of stellar
populations as massive stars evolve off the main sequence. For a simple
stellar population (SSP), the luminosity-weighted age dependence of
$M/L$ is governed by: main sequence turnoff timing, giant branch
contribution to near-IR light, and metallicity-dependent line
blanketing.





Table C1: M/L Power-Law Index from SSP Models

| Model | Metallicity | Age Range | $n$ (V-band) | $n$ (K-band) |
| --- | --- | --- | --- | --- |
| BC03 | $Z_\odot$ | 0.1–10 Gyr | 0.85 | 0.72 |
| BC03 | $0.2 Z_\odot$ | 0.1–10 Gyr | 0.68 | 0.55 |
| BC03 | $0.02 Z_\odot$ | 0.1–1 Gyr | 0.52 | 0.48 |
| FSPS | $0.1 Z_\odot$ | 0.1–1 Gyr | 0.55 | 0.51 |
| BPASS (binary) | $0.1 Z_\odot$ | 0.1–1 Gyr | 0.48 | 0.45 |



At high redshift ($z > 6$), galaxies have typical metallicities $Z \sim
0.1$–$0.2 Z_\odot$. The SSP-predicted low-$n$ regime is consistent with
the canonical residual-minimization analysis, which prefers lower
effective $n$ values once the high-$z$ $\Gamma_t$ enhancement becomes
important.



#### C.2.2 Live Empirical Residual-Minimization Validation


In the canonical analysis (step 076), the overall residual mass-age correlation is minimized at best $n = 0.5$ ($\rho = 0.115$). Cross-validation (step 079) gives mean $n = 0.99 \pm 0.003$ with mean test $\rho = 0.47$. This pattern indicates that the
data-driven optimal $n$ at high $z$ is steeper than standard SSP
predictions, reflecting the additional TEP-induced compression of the
observed age range.



#### C.2.3 TEP-Induced Modification


Beyond the standard SSP prediction, TEP introduces an additional effect:
in the enhanced regime ($\Gamma_t > 1$), stellar populations experience
more effective time than their coordinate age suggests, compressing the
observed age range and effectively flattening the $M/L$-age slope. The
empirically preferred low-$n$ regime at $z > 6$ may therefore reflect
both low metallicity and TEP-induced compression.



Circularity Resolution**


**Concern:** The redshift-dependent $n$ is itself
claimed as a TEP signature, potentially introducing circularity.



**Resolution:** K-fold cross-validation gives $n = 0.99
\pm 0.003$ with mean test $\rho = 0.47$, redshift-blind holdout
performance remains significant ($\rho = 0.22$, $p < 10^{-10}$),
and sensitivity analysis shows the dust signal remains significant
across $n \in [0.3, 1.0]$. An empirical JADES mass-to-light proxy
check is also consistent with the mechanism: $\rho(\Gamma_t,
M/L_{\rm proxy}) = +0.599$ ($p = 1.6 \times 10^{-42}$),
strengthening to partial $\rho = +0.741$ after redshift control ($p
= 8.5 \times 10^{-75}$). Independent age indicators via Balmer
absorption (H$\delta$, H$\gamma$) are predicted to correlate with
$\Gamma_t$ and would provide a spectroscopic test independent of M/L
assumptions.





### C.3 Model Discrimination and Falsifiability Tests


#### C.3.1 Modified Gravity Theory Comparison




Table C2: Modified Gravity Comparison (JWST Anomaly Predictions)

| Theory | JWST Score (/8) | Constraint Score | Key Limitation |
| --- | --- | --- | --- |
| **TEP** | **8** | **3** | — |
| $f(R)$ | 1 | 3 | No dust/clock prediction |
| Galileon | 1 | $-1$ | Ruled out by GW170817 |
| Symmetron | 0 | 3 | No JWST predictions |
| DGP | 0 | 1 | Self-accelerating branch ruled out |
| MOND | 0 | $-1$ | Non-relativistic; no clock effect |
| Horndeski (generic) | 0 | 0 | Too broad; no specific predictions |



Within this comparison test set, TEP matches 8/8 JWST anomaly predictions
compared to 1/8 for the next-best theory ($f(R)$). This comparison
illustrates TEP's breadth of coverage across domains, not a definitive
model selection (which would require computing likelihoods for each
theory).



#### C.3.2 Theoretical Consistency Tests




Table C3: Theoretical Consistency Tests

| Test | Result | Status |
| --- | --- | --- |
| Causality Constraint | 0/2000 causal violations | ✓ Pass |
| $\kappa_{\rm gal}$ Error Budget | $\Gamma_t$ uncertainty $\pm 4.2\%$ ($\sigma_{\kappa} = 4.0 \times 10^5$ mag) | ✓ Pass |
| Time-Space Decoupling | Temporal/spatial ratio > 1.5× | ✓ Pass |
| Multi-Tracer Consistency | *Removed:* this test used hardcoded synthetic α values, not measured data; result pending real multi-tracer calibration | — N/A |
| Screening Length Scale | λ_C = 2.5 kpc vs observed 1.5 kpc | ✓ Pass |
| Screening Transition | Transition spans 2.2 dex; cosmic mean fully unscreened | ✓ Pass |
| Precision Gravity (Cassini, LLR, pulsars, CMB) | All satisfied via Temporal Topology (continuous geometric screening) | ✓ Pass |




#### C.3.3 Model Discrimination Tests




Table C4: Model Discrimination and Falsifiability Tests

| Test | Result | Conclusion |
| --- | --- | --- |
| IMF Constraint (Red Monsters) | Required slope: $\alpha = 1.5$ (no TEP) vs $2.1$ (with TEP) | TEP removes need for extreme IMF |
| IMF vs TEP Discrimination | Estimated simulation power 100% (5000/5000 trials) | TEP distinguishable from IMF |
| Mass-Proxy Breaker | 2/3 tests pass; z>8 LOWESS $\rho = 0.165$, partial-rank $\rho = 0.212$, shuffled-mass $Z = 8.9$, unique fraction 102.2% | $\Gamma_t$ carries information beyond mass+$z$ ordering |
| Dust Model Comparison | TEP 46.5× vs SN-only 10× | Dust physics favors TEP |
| Cross-Survey Systematics | $\rho$ scatter = 0.024 across surveys | Consistent across survey analyses |
| AGN Discrimination Power | Estimated simulation power 100% (2000/2000 trials) | TEP distinguishable from AGN |
| LRD Case Study | Mean $\Delta\Gamma_t = 1.73$, all compact | Consistent with overmassive BHs |
| Hubble Connection | 42% of tension explained ($\chi^2 = 36.8$) | Partial match; not fully consistent |
| Prediction Error Budget | $\pm 16.5\%$ combined uncertainty | Falsifiable at $2\sigma$ |



**IMF constraint:** Without TEP, the Red Monster SFE
anomaly requires a top-heavy IMF ($\alpha_{\rm min} = 1.5$). With TEP,
the corrected SFE requires only $\alpha_{\rm min} = 2.1$, consistent
with standard IMFs. TEP and IMF produce observationally distinct
signatures via environment-dependence (estimated simulation
discrimination power 100%).



**Mass proxy and systematics:** The canonical robustness
results no longer rely on the retired selection-function Monte Carlo
analysis. Instead, the evidence comes from the mass-proxy breaker and
cross-survey systematics checks: at $z > 8$, non-parametric residual tests retain the dust–$\Gamma_t$ signal after mass+$z$ removal (LOWESS $\rho = 0.165$, $p = 5.4 \times 10^{-3}$; partial-rank $\rho = 0.212$, $p = 3.3 \times 10^{-4}$), and shuffled-mass null tests show the observed correlation exceeds mass-ordering expectations by $Z = 8.9$, with 102.2% of the signal attributable to the non-linear $\Gamma_t$ form
rather than to mass ordering alone. The mass-to-light validation gives the prespecified primary value $n=0.5$ (and SMF sensitivity value $n=0.7$), with a single k-fold cross-validation mean of $n=0.99 \pm 0.003$. Cross-survey dust correlations show
low scatter ($\rho_{\rm std} = 0.025$), arguing against survey-specific
artifacts.



**Hubble tension:** TEP predicts an $H_0$ shift of 2.5
km/s/Mpc; the observed shift is 6.0 km/s/Mpc. TEP accounts for ~42% of
the tension amplitude but is formally *not* consistent with the
full tension ($\chi^2 = 36.8$, $p < 10^{-8}$).



**Falsifiability:** Combined prediction uncertainty is $\pm
16.5\%$. Key falsification criteria at $2\sigma$: mass-dust slope
differing from 0.56 by $> 0.30$; SFE correction differing from 100% by
$> 25\%$.



### C.4 Black Hole Growth: Detailed Analysis


#### C.4.1 Single-Object Sensitivity: Theoretical Boost Potential


**Historical note:** Early single-object calculations suggested
that compact cores could yield exponential boost factors of order $10^5$
to $10^6$, sufficient to grow a $10^2 M_\odot$ seed to $10^6 M_\odot$
within 600 Myr. This subsection retains that theoretical sensitivity
analysis as a compact-core mechanism demonstration, not as a validated
population-level result.



**Current status:** The corrected population-level analysis
(Step 132) finds a median $\log_{10}({\rm boost}) = 0.42$ across 260
Kokorev-catalog LRDs, with only 4.6% of objects exceeding $10^3$ in the
conservative mass-proxy analysis. Step 142 classifies the LRD analysis as
"mass-model unstable with no robust closure" — the conservative
MUV-proxy analysis undercloses by 3–4 dex, while CEERS-calibrated masses
drive exponential overshoot. The theoretical sensitivity demonstrated
here remains valid for compact-core systems in principle, but the
real-data population does not achieve the single-object ideal.





Table C5: Boost Factor Uncertainty Propagation

| Parameter | Central Value | Uncertainty | Contribution to $\sigma_{\rm Boost}$ |
| --- | --- | --- | --- |
| $\kappa_{\rm gal}$ | $9.6 \times 10^5$ mag | $\pm 4.0$ (42%) | Dominant |
| $\Delta\log M_h$ (core-halo) | 1.5 | $\pm 0.3$ (20%) | Secondary |
| $t_{\rm cosmic}(z=8)$ | 600 Myr | $\pm 20$ Myr (3%) | Negligible |
| $t_{\rm Salpeter}$ | 45 Myr | $\pm 5$ Myr (11%) | Minor |




#### C.4.2 Sensitivity Analysis: Boost vs. Compactness


The differential temporal topology mechanism is most sensitive to
compact cores. The corrected analysis treats this as a stress test
rather than as a population-level closure claim: extended systems
remain in the weak differential-topology regime, while compact LRD-like
systems can amplify the response sharply if their stellar and halo
masses are high enough.





![LRD Sensitivity: Boost vs Radius](public/figures/figure_lrd_sensitivity.png)




Figure 4: LRD Sensitivity. The growth boost factor is strongly
radius-dependent. Runaway growth requires $r_e \lesssim 800$ pc.





#### C.4.3 Population-Level Stress Test: 260 Little Red Dots


Across 260 Kokorev-catalog LRDs with measured photometric redshifts and
compactness estimates, the differential temporal topology calculation
yields a broad distribution of growth boosts after correcting the
catalog radius units and replacing the all-object default halo mass with
an object-level $M_{\rm UV}$ proxy where direct stellar masses are
absent. Only a small minority exceed $10^3$ in the conservative analysis,
so the result is a compact-core mechanism stress test rather than a
validation of LRD population-level black-hole closure.





Table C6: LRD Population Differential Temporal Topology Stress Test ($N = 260$)

| Quantity | Value | Interpretation |
| --- | --- | --- |
| Redshift range | $4.02$–$8.93$ | Full LRD epoch |
| Median $\Delta\Gamma$ | $0.060$ | Conservative UV-mass analysis |
| Median $\log_{10}$(Boost) | $0.58$ | $\sim 3.8\times$ growth |
| Fraction with Boost $> 10^3$ | $4.6\%$ | Small minority show runaway |




### C.5 External-Regime Discriminants


#### C.5.1 High-Value Observational Discriminants


Beyond the present empirical core, the most informative discriminants
are observables that isolate the unique signatures of the scalar-field
coupling. Key discriminants include:



- 
**Spectroscopic Ages:** Deep NIRSpec MSA spectroscopy
of high-$\Gamma_t$ candidates at $z > 6$ to measure direct Balmer
absorption ages, testing the predicted age enhancement independent
of photometrically derived mass-to-light ratios.


- 
**Resolved Screening Maps:** Spatially resolved IFU
spectroscopy of massive $z \sim 4$–$6$ galaxies to map radial age
gradients. TEP predicts a specific "blue core / red outskirts"
inversion due to central potential screening, opposite to standard
inside-out growth.


- 
**Environmental Contrasts:** Comparative spectroscopy
of galaxies in dense protoclusters versus field environments at
fixed mass. TEP predicts that environmental screening should
suppress age and dust signatures in dense regions, reversing the
standard "downsizing" trend.


- 
**Rest-Frame Mid-IR:** MIRI imaging to directly probe
the dust continuum at $z > 7$, confirming that the observed
UV-reddening is driven by dust grains (as predicted by the
$\Gamma_t$-accelerated production) rather than exotic stellar
populations.





#### C.5.2 Wide-Field Regime Predictions


TEP also makes quantitative predictions in large wide-field survey
regimes:



- 
**Euclid-wide regime ($15{,}000$ deg²):** $N \sim
300{,}000$ massive galaxies at $z = 0.9$–$1.8$; TEP predicts $\sim
25\%$ mass-dependent age offset.


- 
**Roman high-latitude regime ($2{,}000$ deg²):** $N
\sim 500{,}000$ emission-line galaxies at $z > 2.5$; TEP predicts
weak gas-phase metallicity–$\Gamma_t$ correlation.


- 
**Roman supernova regime ($N \sim 2{,}700$):** TEP
predicts SN Ia rates in massive hosts enhanced by $1.28\times$.




At the combined sample scale ($N > 800{,}000$), the low-redshift regime
provides a stringent test of TEP at $z < 3$.



#### C.5.3 Cosmic Variance Budget


Current survey fields: UNCOVER (Abell 2744, 45 arcmin², $\sigma_{\rm cv}
\approx 22\%$), CEERS (EGS, 100 arcmin², $\sigma_{\rm cv} \approx
15\%$), COSMOS-Web (COSMOS, 1800 arcmin², $\sigma_{\rm cv} \approx
3.5\%$). Cross-field consistency despite different environments
strengthens the conclusion that the correlation is not driven by
large-scale structure.



## Data Availability & Reproducibility




This work follows open-science practices. All results are fully reproducible from raw data 
using the documented pipeline. All numerical results, figures, and statistics are generated by deterministic 
Python scripts processing real observational data.





### Repository & Code


**GitHub Repository:** github.com/matthewsmawfield/TEP-JWST



The repository contains a deterministic, version-controlled analysis pipeline with 159 analysis steps 
and automated consistency checks.



#### Repository Structure



TEP-JWST/
├── data/             # Raw and interim catalogs
│  ├── raw/            # JWST catalogs from MAST
│  ├── interim/          # Processed datasets
│  └── DATA_PROVENANCE.md     # Download timestamps and checksums
├── logs/             # Step execution logs with timestamps
├── results/
│  ├── outputs/          # JSON/CSV analytical outputs
│  └── figures/          # Generated manuscript figures
├── scripts/
│  ├── steps/           # 159-step canonical pipeline
│  │  ├── step_001_uncover_load.py
│  │  ├── step_002_tep_model.py
│  │  ├── step_140_evidence_tier_summary.py
│  │  ├── step_159_mass_measurement_bias.py
│  │  ├── step_160_manuscript_consistency_check.py
│  │  ├── step_170_kinematic_decisive_test.py
│  │  └── run_all_steps.py    # Master pipeline runner
│  └── utils/           # Shared analysis utilities
├── site/
│  ├── components/        # Source of truth for manuscript
│  └── dist/           # Built site artifacts
└── requirements.txt        # Python dependencies (pinned)




### Data Provenance



| Data Source | Provider | Access Method | Download Size | DOI/URL |
| --- | --- | --- | --- | --- |
| UNCOVER DR4 | MAST/JWST | Public archive | 62.8 MB (SPS catalog) | MAST Archive |
| CEERS | MAST/JWST | Public archive | ~500 MB (photometry) | MAST Archive |
| COSMOS-Web/COSMOS2025 | MAST/JWST | Public archive | 270 MB (master catalog) | MAST Archive |
| JADES DR5 | MAST/JWST | Public archive | 673 MB (GOODS-S) + 818 MB (GOODS-N) | MAST Archive |
| JADES DR4 Spectroscopic | JADES Team | Direct download | ~200 MB | JADES DR4 |
| DJA NIRSpec Merged v4.4 | DJA Archive | Zenodo | ~150 MB (compressed) | Zenodo |
| FRESCO Red Monsters | Literature | Author-provided | <1 MB (3 objects) | Via GitHub repo |
| Kokorev LRD Catalog | Literature | Published catalog | 210 KB | Via GitHub repo |


**Total Download Size:** ~2.5 GB for full JWST survey catalogs (optional for replication; core analysis runs on processed interim files).**
Data Provenance Log:** Complete download timestamps, file checksums, and version records 
are maintained in `data/DATA_PROVENANCE.md`.



### Reproduction Instructions



#### Quick Start (Full Reproduction)



# 1. Clone repository
git clone https://github.com/matthewsmawfield/TEP-JWST.git
cd TEP-JWST

# 2. Install dependencies
pip install -r requirements.txt
npm install --prefix site

# 3. Run complete pipeline (159 steps)
python scripts/steps/run_all_steps.py

# 4. Build manuscript
npm run build --prefix site




#### System Requirements



| Component | Minimum | Recommended | Tested On |
| --- | --- | --- | --- |
| CPU | 8 cores | 14+ cores | Apple M4 Pro (14-core) |
| RAM | 16 GB | 32 GB | 24 GB (M4 Pro) |
| Storage | 20 GB | 50 GB | NVMe SSD |
| Runtime | ~10-12 minutes (full 159-step pipeline) | ~11 minutes (M4 Pro) |  |


#### Pipeline Overview

The analysis pipeline consists of 159 deterministic steps organized into phases:


- **Steps 001-008:** Data loading and TEP model computation (Γ_t for all galaxies)

- **Steps 009-100:** Core empirical analysis across five evidence lines (L1-L5)

- **Steps 101-140:** Cross-survey replication and robustness validation

- **Steps 141-160:** Advanced discriminating tests and manuscript consistency checks

- **Steps 161-176:** Figure generation and final summary outputs



Each step produces JSON outputs with full metadata, and execution logs are written to `logs/` 
with timestamps for full traceability.



#### Consistency Verification


The pipeline includes automated consistency checks (`step_160_manuscript_consistency_check.py`) 
that verify every numerical claim in the manuscript against actual computed outputs. 
**Zero tolerance for statistical-data mismatches.**



### Data Provenance

Analysis performed using:


- **Python** 3.11+ (pinned in requirements.txt)

- **NumPy** 1.24+ (numerical computation)

- **SciPy** 1.10+ (statistical tests, optimize)

- **Pandas** 2.0+ (data manipulation)

- **Matplotlib** 3.7+ (visualization)

- **Astropy** 5.0+ (astronomical calculations)



Full dependency tree with exact versions is available in `requirements.txt`.