# The Cepheid Bias: Resolving the Hubble Tension
**Matthew Lukin Smawfield**  
Version: v0.9 (Kingston upon Hull)
First published: 11 January 2026 · Last updated: 21 August 2026  
DOI: 10.5281/zenodo.18209702

---

## Abstract

The Hubble Tension—the persistent $5\sigma$ discrepancy between local distance-ladder measurements ($H_0 \approx 73.0\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and early-universe CMB inference ($H_0 = 67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$)—represents a significant challenge in precision cosmology. This paper tests whether a component of the Hubble tension can be represented as an environment-dependent Cepheid clock bias, as predicted by the Temporal Equivalence Principle (TEP).

The hypothesis tested here is that Cepheid variable stars function as environment-dependent "standard clocks." Under TEP, proper time is a dynamical scalar field that couples universally to non-gravitational matter. Because stellar pulsation rates operate in local proper time, calibrating classical Cepheids in deep-potential SN Ia host galaxies against diffuse, low-mass anchor galaxies systematically misestimates host distance moduli. When interpreted through a universal Period--Luminosity relation, this clock-rate anomaly mimics diminished luminosity, leading to underestimated distances and an inflated local Hubble constant.

In standard distance-ladder linear regression, unconstrained latent host distance moduli $\mu_i$ algebraically absorb host-level environmental shifts. A standard-ladder projection test that inserts a TEP environmental column into the SH0ES design matrix therefore yields an apparent null result. This structural degeneracy is resolved by formulating the calibrator-host distance ladder as a generative clock-aware model coupled to the Hubble flow.

This generative framework is tested using the complete public Riess et al. (2022) sample of 37 distinct SN Ia host galaxies, utilizing a homogeneous kinematic potential coordinate $u_\phi=V_{\rm rot}/\sqrt{2}$ derived from pinned HyperLEDA rotation velocities and continuous environmental screening. Across all 37 hosts, the generative endpoint likelihood yields $\Gamma_X=(1.156\pm0.958)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical $250\ {\rm km/s}$ velocity variance, strengthening to $1.64\sigma$ under data-driven Pantheon+ velocity scatter ($182.1\ {\rm km/s}$) and $1.97\sigma$ at $150\ {\rm km/s}$, with 100% leave-one-host-out sign stability (37 of 37 refits positive). In the cosmologically constrained host-level expansion-rate likelihood across the 33 Hubble-flow hosts (Step 42), the combined endpoint response is recovered at $\Gamma_X=(1.165\pm0.979)\times10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$; $2.07\sigma$ at $150\ {\rm km/s}$; and $1.99\sigma$ under full $33\times33$ SH0ES covariance GLS), yielding a conventional Hubble-flow intercept of $H_{\rm app}=68.36\pm0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ to $68.62\pm1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$. When evaluated in a joint multi-block framework with independent TRGB distances and geometric anchors (Step 44), the multi-block likelihood consistently favours an environmental response under either single-parameter allocation ($\kappa_{\rm Cep} = (0.400 \pm 0.270)\times 10^6\ {\rm mag}$ or $\beta_X = (1.631 \pm 1.250)\times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$), resolving the local Hubble constant to $H_{\rm app} = 68.76\pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$. Under the restricted TEP Cepheid-channel closure ($\beta_X=0$), the full response is allocated to the Cepheid channel ($\kappa_{\rm Cep}^{\rm equiv} = (0.369 \pm 0.310)\times 10^6\ {\rm mag}$); this allocation is physically specified but remains conditional on host-specific aperture validation.

Propagating the TEP potential correction through the distance ladder yields a unified local Hubble constant of $H_0=66.65\pm1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, consistent with Planck CMB observations at $0.45\sigma$ ($0.31\sigma$ bootstrap). Full-matrix propagation through the 3,490-row SH0ES system confirms exact reference-gauge invariance. Finally, single-galaxy differential tests provide internal signatures with the sign predicted by TEP free from host-to-host peculiar velocity systematics: M31 inner versus outer Cepheids exhibit an empirical Period--Luminosity offset of $+0.356 \pm 0.136\ {\rm mag}$ ($2.6\sigma$), increasing to $+0.630 \pm 0.195\ {\rm mag}$ ($3.24\sigma$) under spatial PHAT matching, while OGLE-IV Cepheids in the LMC independently exhibit radial potential stratification at $+0.0284 \pm 0.0086\ {\rm mag}$ ($3.3\sigma$).

*Keywords:* Hubble tension -- Cepheid variables -- distance ladder -- galaxy rotation -- peculiar velocities -- temporal equivalence principle

## 1. Introduction

### 1.1 The Hubble Tension and Distance Ladder Systematics

The discrepancy between the local expansion rate of the Universe measured via the classical distance ladder and the value inferred from early-universe cosmological observations has emerged as one of the most pressing foundational problems in modern physics. The SH0ES collaboration reports a local Hubble constant of $H_0 = 73.04 \pm 1.04\ {\rm km\,s^{-1}\,Mpc^{-1}}$ using Cepheid-calibrated Type Ia supernovae (Riess et al. 2022; hereafter R22), whereas cosmic microwave background (CMB) measurements by the Planck satellite under flat $\Lambda$CDM establish $H_0 = 67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$ (Planck Collaboration VI 2020). This exceeds $5\sigma$ in statistical significance, resisting resolution through conventional astrophysical systematics, photometric crowding, or standard cosmological parameters.

The standard distance ladder relies on the assumption that stellar standard candles calibrated in nearby anchor galaxies (the Milky Way, Large Magellanic Cloud, Small Magellanic Cloud, and NGC 4258) behave identically in distant host galaxies. However, the anchor galaxies possess predominantly shallow gravitational potential wells, whereas the SN Ia host galaxies sample substantially deeper galactic potential wells. If stellar pulsation clocks respond dynamically to the local gravitational potential, an environmental gradient between calibrators and hosts will introduce a systematic bias into the inferred distance scale.

### 1.2 The Temporal Equivalence Principle

The Temporal Equivalence Principle (TEP) provides a covariant scalar-tensor framework that elevates proper time from a fixed background coordinate parameter to a dynamical physical field. The action is formulated on a two-metric spacetime:

\begin{equation}
S = \int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R - \frac{1}{2}(\nabla\phi)^2 - V(\phi)\right] + S_m[\tilde g_{\mu\nu}, \Psi_m] ,
\end{equation}

where gravity is governed by the metric $g_{\mu\nu}$, while all non-gravitational matter fields, atomic transitions, and stellar clocks couple universally to the causal matter metric $\tilde g_{\mu\nu} = A^2(\phi)g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$, following the convention established in the foundational TEP theory paper (Smawfield 2025, Paper 0). In gravitational bound states, the conformal factor satisfies $A(\phi) < 1$, slowing the rate of proper time relative to cosmic background time.

The absolute matter-clock rate relative to cosmic time is defined as

\begin{equation}
r_j \equiv \frac{(d\tilde\tau/dt)_j}{(d\tilde\tau/dt)_{\rm cosmic}} .
\end{equation}

Throughout the TEP framework, the physical rate ordering across galactic environments is strictly bounded:

\begin{equation}
0 < r_{\rm core} < r_{\rm disk} < r_{\rm cosmic} \equiv 1 .
\label{eq:rate_hierarchy}
\end{equation}

No clock runs faster than cosmic background time; clocks in active-shear galactic disks are simply less slowed than clocks in dense nuclear cores. Environmental chameleon screening $S(\rho)$ suppresses scalar fifth forces in high-density regimes while preserving dynamical time-rate variations on galactic and cosmological scales.

| Regime | Absolute Rate | Physical Interpretation |
| --- | --- | --- |
| Cosmic background | $r_{\rm cosmic} = 1$ | Unscreened cosmological reference |
| Galactic disk | $0 < r_{\rm disk} < 1$ | Moderate potential depth; active scalar shear |
| Galactic core | $0 < r_{\rm core} < r_{\rm disk}$ | Deep potential depth; strongly screened |

### 1.3 Stellar Pulsation Physics and the Observing Chain

Classical Cepheid variables are dynamical pulsation clocks whose pulsation periods $P_{\rm local}$ are governed by hydrodynamic acoustic transit times across the stellar envelope in local matter-frame proper time. When observed in the heliocentric telescope frame and corrected for the systemic redshift $z_{\rm spec}$ of the host galaxy, the inferred rest-frame period is:

\begin{equation}
P_{\rm rest}^{\rm inf} = \frac{P_{\rm obs}}{1 + z_{\rm spec}} = P_{\rm local}\cdot q_i ,
\qquad
q_i \equiv \frac{r_{\rm spec,i}}{r_{\rm Cep,i}} .
\label{eq:intro_period_ratio}
\end{equation}

Under the TEP core--disk tracer closure, systemic spectroscopy weighted toward a more deeply slowed region than the Cepheid field gives $r_{\rm spec} < r_{\rm Cep}$, hence $q_i < 1$, mathematically generating the observed period contraction. Through the Leavitt Period--Luminosity law ($M_W = \alpha + \beta \log P$), this period shift alters the inferred distance modulus by $\Delta \mu_i = \kappa_{\rm Cep} X_i$, where $X_i$ is the environmental potential coordinate.

### 1.4 Scope and Structure of this Study

This paper presents a comprehensive empirical test of the TEP framework across multiple independent observational tiers using public astronomical data:

1. Single-galaxy differential tests in M31 and the LMC, isolating the potential-clock coupling free from host-to-host peculiar velocity uncertainties.

2. Homogeneous, exact-identifier kinematic potential reconstruction for all 37 distinct R22 SN Ia host galaxies from pinned HyperLEDA rotation velocities and continuous screening.

3. Generative endpoint likelihood evaluation across flow-velocity models and peculiar velocity dispersions.

4. Mathematical resolution of the latent-modulus parameter absorption in standard distance-ladder design matrices using the TEP-native generative ladder.

5. Complete computational reproducibility: the resolution of the design matrix degeneracy and the full-ladder generative likelihoods are accompanied by an open-source, end-to-end Python pipeline, ensuring transparent verification of all statistical claims.

## 2. Data and Methods

### 2.1 Public Distance-Ladder Data and Independent Sample Reconstruction

The empirical distance ladder is reconstructed from the public R22 data release, comprising the 3,490-row generalized least-squares design matrix, data vector, full covariance matrix, and the Pantheon+SH0ES supernova catalogue (Scolnic et al. 2022; Brout et al. 2022). Solving the unmodified baseline system yields $H_0 = 73.0434 \pm 1.0072\ {\rm km\,s^{-1}\,Mpc^{-1}}$ with $\chi^2 = 3552.7063$ for 3,444 degrees of freedom, reproducing the published SH0ES solution.

The R22 calibrator sample contains 42 SNe Ia situated in 37 distinct host galaxies. In the host-level inference, the independent statistical unit is the individual host galaxy, as all Cepheids and multiple SNe within a single galaxy share the same host potential well. Each supernova is matched to its host galaxy using exact R22 catalogue identifiers and PGC numbers, strictly excluding angular nearest-neighbour heuristics. All 37 distinct host galaxies have measured positive Hubble-flow redshifts ($z_{\rm HD} > 0$). Of these, 34 satisfy the Hubble-flow union cut ($z_{\rm CMB} > 0.0035$ or $z_{\rm HD} > 0.0035$); 33 of those have matching Cepheid design-matrix entries and enter the primary generative likelihood (Step 42).

### 2.2 Homogeneous Kinematic Potential Coordinate

To ensure absolute reproducibility and avoid provenance errors arising from inhomogeneous mixtures of central stellar velocity dispersions and 21 cm H I linewidths, a global kinematic potential coordinate is constructed using a pinned snapshot of the HyperLEDA database. For each host and anchor galaxy, the homogenized maximum circular rotation velocity $V_{\rm rot}$ is extracted via exact PGC identifiers. The potential scale is defined as:

\begin{equation}
u_{\phi,i} = \frac{V_{{\rm rot},i}}{\sqrt{2}},
\qquad
\sigma_{u,i} = \frac{\sigma_{V,i}}{\sqrt{2}} .
\label{eq:uphi}
\end{equation}

This provides a rigorous, homogeneous proxy for the global gravitational potential depth $\Phi \sim u_\phi^2$, maintaining complete coverage across the entire 37-host sample without empirical aperture corrections.

### 2.3 Environmental Screening Formulation

The scalar field coupling in TEP is modulated by environmental density via chameleon-like screening. Host disk scale lengths are evaluated from RC3 isophotal diameters ($R_{25} = D_{25}/2$, $R_d = R_{25}/3.2$, $z_d = 0.1 R_d$) and stellar mass densities are computed at representative Cepheid radii ($r_{\rm Cep} = 1.8 R_d$). Group-scale density is obtained from the Tully (2015) 2MRS group catalogue Table 5 via exact PGC matching. The total continuous screening factor is:

\begin{align}
S_{{\rm local},i} &= \left[1 + \left(\frac{\rho_i}{0.5\ M_\odot\,{\rm pc}^{-3}}\right)^2\right]^{-1} , \\
S_{{\rm group},i} &= \left[1 + \left(\frac{N_{{\rm mb},i}}{10}\right)^{1.2}\right]^{-1} , \\
S_i &= S_{{\rm local},i} S_{{\rm group},i} .
\label{eq:screening}
\end{align}

The screened potential scale for each host is $U_i = S_i u_{\phi,i}^2$. For likelihood evaluation, the dimensionless centered environmental regressor is defined as:

\begin{equation}
\widetilde X_i = \frac{U_i - U_{\rm ref}}{c^2} - \left\langle \frac{U - U_{\rm ref}}{c^2} \right\rangle ,
\label{eq:centered_x}
\end{equation}

where centering guarantees that the fitted environmental slope is mathematically invariant to the choice of reference scale $U_{\rm ref}$.

### 2.4 Single-Galaxy Internal Differential Methodology

To test the TEP clock-rate coupling independently of host-to-host peculiar velocities and distance ladder calibration steps, internal Period--Luminosity variations are analysed within individual galaxies. In M31 (Andromeda), the homogeneous Pan-STARRS Cepheid catalog (Kodric et al. 2018) containing 2,686 variables is used, comparing Cepheids in the dense inner potential ($R < 5.0$ kpc, mean density $\rho \approx 0.31\ M_\odot\,{\rm pc}^{-3}$) against the diffuse outer disk ($R > 15.0$ kpc, $\rho \approx 0.006\ M_\odot\,{\rm pc}^{-3}$). Spatial restriction to the HST PHAT footprint is also performed. In the Large Magellanic Cloud, fundamental-mode Cepheids from OGLE-IV (Soszy&nacute;ski et al. 2015) are analysed, partitioned by galactocentric radius.

### 2.5 Generative Endpoint Likelihood and Flow Velocities

The host-level expansion relation is evaluated via the generative model:

\begin{equation}
cz_i = d_i\left(H_{\rm app} + \Gamma_X \widetilde X_i\right) + \epsilon_i ,
\label{eq:primary_likelihood}
\end{equation}

where $d_i = 10^{(\mu_i - 25)/5}\ {\rm Mpc}$, and the Gaussian variance is:

\begin{equation}
V_i = \sigma_v^2 + \left[\frac{\ln 10}{5} \left|d_i\left(H_{\rm app} + \Gamma_X \widetilde X_i\right)\right|\sigma_{\mu,i}\right]^2 + \sigma_{{\rm int},v}^2 .
\end{equation}

The likelihood is evaluated across a spectrum of peculiar-velocity dispersions: the standard R22 baseline ($\sigma_v = 250\ {\rm km\,s^{-1}}$), alternative velocity scales ($150\ {\rm km\,s^{-1}}$ and $500\ {\rm km\,s^{-1}}$), and the data-driven residual velocity scatter after bulk-flow removal from Pantheon+ ($\sigma_v = 182.1\ {\rm km\,s^{-1}}$).

### 2.6 The TEP-Native Generative Distance Ladder

In standard distance-ladder linear regression, host distance moduli $\mu_i$ are treated as unconstrained latent parameters. The linear design matrix does not generically erase observation-level Cepheid perturbations (as confirmed by the 99.7% recovery of raw row-level injections in Step 34). What it cannot identify without external constraints is an environmental perturbation that is mathematically equivalent to shifting the latent host modulus, because the 37 unconstrained $\mu_i$ parameters absorb the host-level shift ($\hat{\mu}_i \to \mu_i - \kappa_{\rm Cep} X_i$).

The true generative observable model couples the observed Cepheid distance moduli $\mu_i^{\rm obs} = \mu_i^{\rm true} - \kappa_{\rm Cep} X_i$ directly to cosmological expansion $cz_i = d_i^{\rm true}(H_{\rm app} + \beta_X X_i) + v_i$. The identifiable physical combination is:

\begin{equation}
\Gamma_X = \beta_X + \left(\frac{\ln 10}{5}\right) H_{\rm app} \kappa_{\rm Cep} .
\label{eq:gamma_relation}
\end{equation}

In Step 42, the generative ladder is formulated directly in expansion-rate space across the $N=33$ Hubble-flow SN Ia hosts with Cepheid calibrations ($z_{\rm HD} > 0.0035$ or $z_{\rm CMB} > 0.0035$, excluding local anchors):

\begin{equation}
H_{0,i} \equiv \frac{cz_i}{d_i^{\rm SH0ES}} = H_{\rm app} + \Gamma_X \widetilde X_i + \epsilon_i ,
\label{eq:step42_wls}
\end{equation}

where $d_i^{\rm SH0ES} = 10^{(\mu_i^{\rm SH0ES}-25)/5}$ Mpc, and each host is weighted by its inverse total variance:

\begin{equation}
w_i = \frac{1}{\sigma_{H_0,i}^2} = \frac{d_i^2}{\sigma_v^2 + \left(\frac{\ln 10}{5} cz_i \sigma_{\mu,i}\right)^2} .
\label{eq:step42_weights}
\end{equation}

Because $H_{0,i}$ is scale-invariant across distance, all 33 hosts contribute according to their fractional distance precision. To account for potential cross-host correlations in the SH0ES calibration chain, the model is also evaluated via Generalized Least Squares (GLS) using the full $33\times 33$ covariance matrix $\mathbf{C}_H = \mathbf{J} \mathbf{C}_\mu \mathbf{J}^T + \mathbf{C}_v$, where $\mathbf{C}_\mu = (\mathbf{L}^T \mathbf{C}^{-1} \mathbf{L})^{-1}_{\mu}$ is extracted directly from the inverted 3,490-row SH0ES normal matrix, $\mathbf{J} = \text{diag}(-\frac{\ln 10}{5} H_{0,i})$ is the Jacobian transformation, and $\mathbf{C}_v = \text{diag}(\sigma_v^2/d_i^2)$ is the peculiar velocity dispersion matrix. Setting $\beta_X = 0$ defines the restricted TEP Cepheid-channel closure, which assigns the full environmental response $\Gamma_X$ to the Cepheid distance scale ($\kappa_{\rm Cep} = \Gamma_X / [(\frac{\ln 10}{5}) H_{\rm app}]$).

The fitted intercept $H_{\rm app}$ represents the expansion rate at the sample mean environment $\langle X \rangle = 4.05 \times 10^{-8}$. To evaluate the expansion rate at physically defined reference frames, the relation maps back to:

\begin{equation}
H_{\rm ref} = H_{\rm app} - \Gamma_X \langle X \rangle,
\qquad
H_{\rm cosmic} = H_{\rm app} + \Gamma_X (X_{\rm cosmic} - \langle X \rangle) = H_{\rm ref} - \Gamma_X \left(\frac{U_{\rm ref}}{c^2}\right) ,
\label{eq:href_mapping}
\end{equation}

where $X_{\rm ref} \equiv 0$ corresponds to the anchor reference zero-point ($\sigma_{\rm ref} = 87.165\ {\rm km\,s^{-1}}$), and $X_{\rm cosmic} = -U_{\rm ref}/c^2 = -8.45 \times 10^{-8}$ corresponds to the unperturbed cosmic background (zero virial potential depth).

### 2.7 Joint Multi-Block Distance--Redshift Likelihood

While the standalone host-level regression (Step 42) measures the identifiable combined slope $\Gamma_X$, breaking the internal parameter degeneracy of the distance ladder without fixing $\beta_X=0$ requires external observational constraints. In Step 44, a joint multi-block likelihood is formulated by combining three observational data blocks:

\begin{equation}
\ln \mathcal{L}_{\rm joint}(\boldsymbol{\theta}) = \ln \mathcal{L}_{\rm redshift} + \ln \mathcal{L}_{\rm TRGB} + \ln \mathcal{L}_{\rm anchor} + \ln \mathcal{P}(\delta m) + \ln \mathcal{P}(\delta a) ,
\label{eq:joint_likelihood}
\end{equation}

where $\boldsymbol{\theta}$ encompasses the physical and nuisance parameters. The three blocks are structured as follows:

1. *Redshift--Distance Block ($N=31$ Hubble-flow hosts):* Evaluates cosmological expansion with TEP-corrected true distances $d_i^{\rm true} = d_i^{\rm obs} 10^{\kappa_{\rm Cep} X_i / 5}$, modeling $cz_i = d_i^{\rm true}(H_{\rm app} + \beta_X X_i) + v_i$. The 31 hosts represent the subset of the 33 Hubble-flow calibrator hosts after strictly excluding the two lowest-redshift systems (NGC 4424 at $z = 0.0026$ and NGC 4536 at $z = 0.0032$). The per-host variance is $\sigma_{v,i}^2 = \sigma_v^2 + (\frac{\ln 10}{5} cz_i \sigma_{\mu,i})^2 + \sigma_{{\rm int},v}^2$, where $\sigma_{{\rm int},v}$ is an intrinsic velocity dispersion nuisance parameter.

2. *TRGB Differential Block ($N=16$ non-anchor calibrators):* Compares Cepheid and Tip of the Red Giant Branch distance moduli, $\Delta \mu_i \equiv \mu_i^{\rm Cep} - \mu_i^{\rm TRGB} = \delta m - \kappa_{\rm Cep} X_i + \epsilon_i$, with variance $\sigma_{\Delta\mu,i}^2 = \sigma_{\mu,{\rm Cep},i}^2 + \sigma_{\mu,{\rm TRGB},i}^2$. Because TRGB stellar evolution clocks operate independently of pulsation physics, TRGB distances provide an external anchor on the host modulus. The 16 hosts represent the complete overlap of SH0ES Cepheid calibrators, HyperLEDA rotation velocities, and EDD/CCHP TRGB measurements (excluding geometric anchors to prevent double counting).

3. *Independent Geometric Anchor Block ($N=2$ primary anchors):* Constrains the absolute Cepheid zero-point against independent geometric distances in the Large Magellanic Cloud (detached eclipsing binaries) and NGC 4258 (water megamasers): $\Delta \mu_{\rm anc} \equiv \mu_{\rm Cep} - \mu_{\rm geo} = \delta a - \kappa_{\rm Cep} X_{\rm anc} + \epsilon_a$. M31 is excluded from this block because its distance scale is composite rather than purely geometric.

Four nested model configurations are evaluated against the data: the *Null Model* ($k=4$: $H_{\rm app}, \delta m, \delta a, \sigma_{{\rm int},v}$), the *Cepheid Model* ($k=5$: setting $\beta_X \equiv 0$ and fitting $\kappa_{\rm Cep}$), the *Velocity Model* ($k=5$: setting $\kappa_{\rm Cep} \equiv 0$ and fitting $\beta_X$), and the *Mixed Model* ($k=6$: simultaneously fitting both $\kappa_{\rm Cep}$ and $\beta_X$). The Cepheid and Velocity models represent two alternative single-degree-of-freedom closures of the underlying physical response rather than additive components. In this multi-block formulation, the three data blocks are treated as quasi-independent observational constraints; while excluding primary geometric anchors from the TRGB block prevents direct anchor duplication, shared Cepheid calibration terms are treated within individual block variances.

### 2.8 Unified Host-Level Reconstruction (Step 04)

Complementing the expansion-rate likelihood, a unified host-level reconstruction is performed directly on the complete 37-host sample. Each host's distance modulus is corrected according to its screened kinematic potential:

\begin{equation}
\mu_i^{\rm corr} = \mu_i^{\rm obs} + \kappa_{\rm Cep} S(\rho_i) \frac{u_{\phi,i}^2 - U_{\rm ref}}{c^2} ,
\label{eq:step04_correction}
\end{equation}

yielding reconstructed physical distances $d_i^{\rm corr} = 10^{(\mu_i^{\rm corr} - 25)/5}$ and individual expansion rates $H_{0,i} = cz_i / d_i^{\rm corr}$. The anchor reference zero-point is defined as $\sqrt{U_{\rm ref}} = \sigma_{\rm ref} = 87.165\ {\rm km\,s^{-1}}$, constructed from the weighted composite of local stellar velocity dispersions at the specific Cepheid disk locations within the primary calibration anchors: Milky Way solar neighbourhood ($\sigma_z = 30.0\ {\rm km\,s^{-1}}$, weight 0.20; Bovy et al. 2012), LMC stellar disk ($\sigma_{\rm disk} = 24.0\ {\rm km\,s^{-1}}$, weight 0.25; van der Marel et al. 2002), and NGC 4258 intermediate annulus ($\sigma_{\rm local} = 115.0\ {\rm km\,s^{-1}}$, weight 0.55; Kormendy & Ho 2013). As proven in Appendix D.3, the reference scale $\sqrt{U_{\rm ref}}$ is an exact gauge origin in the full linear system, absorbed into the Cepheid zero point $M_H^W$ without altering the physical response $\kappa_{\rm Cep}$.

The optimal response coefficient $\kappa_{\rm Cep}$ is determined by minimizing the residual variance and environmental gradient across the host sample ($\partial H_{0,i} / \partial u_{\phi,i} \to 0$). Honest uncertainties are estimated via a joint bootstrap ($N=1,000$ resamples with replacement), where $\kappa_{\rm Cep}$ is re-optimized for each realization, simultaneously propagating host-to-host sampling scatter and $\kappa_{\rm Cep}$ parameter variance to yield the unified Hubble constant $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$.

### 2.9 Observational Controls for Single-Galaxy Internal Gradients

Single-galaxy differential tests in M31 and the LMC provide internal empirical checks of the TEP potential hierarchy free from host-to-host peculiar velocity dispersions. To ensure these internal Period--Luminosity offsets are not generated by standard astrophysical systematics, the analysis incorporates three rigorous controls:

1. *Extinction-Free Photometry:* All Cepheid magnitudes are evaluated using the reddening-free Wesenheit index $W \equiv m_I - R_V (m_V - m_I)$ (with $R_V = 1.55$ for OGLE-IV LMC and $R_V = 1.54$ for M31), which algebraically cancels total and differential interstellar dust extinction along the line of sight.

2. *Spatial Resolution and Crowding Controls:* In M31, ground-based blending and crowding effects in the dense inner bulge are controlled by utilizing high-resolution Hubble Space Telescope Panchromatic Hubble Andromeda Treasury (PHAT) spatial matching, which isolates uncrowded stellar point spread functions and recovers the internal offset at $\Delta W = +0.630 \pm 0.195\ {\rm mag}$ ($3.24\sigma$).

3. *Metallicity Plausibility Limits:* To test whether radial metallicity gradients could mimic the observed $\Delta W$, the required metallicity sensitivity is evaluated. For the empirical M31 metallicity gradient ($\Delta [\text{O/H}] \sim 0.2\text{--}0.4\ {\rm dex}$), explaining the observed $\Delta W \approx +0.36\text{--}0.63\ {\rm mag}$ purely via metallicity would require an unphysical coefficient $\gamma > 1.5\ {\rm mag/dex}$, whereas the empirical metallicity dependence measured across the SH0ES calibrator sample is $\gamma = -0.20 \pm 0.08\ {\rm mag/dex}$.

## 3. Results

### 3.1 Host Sample Reconstruction and Potential Stratification

The exact-identifier reconstruction yields all 37 distinct R22 SN Ia host galaxies with measured positive Hubble-flow redshifts ($z_{\rm HD} > 0$), complete RC3 isophotal diameters, and pinned HyperLEDA circular rotation velocities $V_{\rm rot}$. Splitting the 37 hosts at the sample median potential scale $u_\phi = 114.62\ {\rm km/s}$ reveals an empirical stratification: the 19 shallow-potential hosts yield a mean Hubble parameter of $66.99 \pm 2.50\ {\rm km\,s^{-1}\,Mpc^{-1}}$, whereas the 18 deep-potential hosts yield $68.49 \pm 3.21\ {\rm km\,s^{-1}\,Mpc^{-1}}$, propagating the full host distance covariance matrix.

The raw potential-velocity association displays a positive Pearson correlation $r = 0.241$ ($p = 0.150$) and Spearman rank correlation $\rho = 0.188$ ($p = 0.266$). This positive gradient between host potential depth and apparent expansion rate matches the directional prediction of TEP stellar clock modulation.

![Host-level H0 values plotted against the square of the rotation-derived potential scale for 37 R22 supernova hosts](public/figures/step_03_figure_01_h0_vs_sigma.png?v=3)

Figure 1: Host-level expansion rate $cz_{\rm HD}/d$ as a function of the kinematic potential coordinate $u_\phi^2$ for all 37 R22 SN Ia host galaxies. Error bars include distance modulus covariance and peculiar velocity uncertainties.

### 3.2 Single-Galaxy Internal Verification: M31 and LMC

Single-galaxy tests provide empirical verification of the TEP potential coupling free from host distance uncertainties or peculiar-velocity flow modelling:

In M31 (Andromeda), 2,686 Cepheids from the Pan-STARRS survey are analysed across galactocentric radii. Comparing Cepheids in the dense inner potential ($R < 5.0$ kpc, mean density $\rho \approx 0.31\ M_\odot\,{\rm pc}^{-3}$, where chameleon screening is active) against those in the diffuse outer disk ($R > 15.0$ kpc, $\rho \approx 0.006\ M_\odot\,{\rm pc}^{-3}$) reveals an empirical Period--Luminosity zero-point offset, defined as $\Delta W \equiv W_{\rm inner} - W_{\rm outer}$:

> 

\begin{equation}
\Delta W_{\rm M31} \equiv W_{\rm inner} - W_{\rm outer} = +0.3560 \pm 0.1357\ {\rm mag} \quad (2.6\sigma\ {\rm significance}) .
\end{equation}

A positive $\Delta W$ indicates that inner Cepheids appear systematically fainter (possessing longer effective rest-frame periods) than their outer counterparts, consistent with the observing-chain hierarchy of Appendix C: outer-disk Cepheids undergo a larger pipeline period contraction relative to the deeper systemic tracer, producing a positive inner-minus-outer zero-point differential. When restricting to the high-precision spatial footprint of the HST PHAT survey, the internal offset sharpens to $\Delta W_{\rm PHAT} = +0.6304 \pm 0.1948\ {\rm mag}$ ($3.24\sigma$ significance). In the Large Magellanic Cloud (LMC), analysis of OGLE-IV fundamental-mode Cepheids partitioned by galactocentric radius ($R \le 0.92$ kpc versus $R \ge 2.93$ kpc) independently reveals an internal offset of $\Delta W_{\rm LMC} = +0.0284 \pm 0.0086\ {\rm mag}$ ($3.3\sigma$ significance), following the same inner-minus-outer convention. These independent single-galaxy detections are consistent with a larger inferred period contraction for Cepheids in the less-slowed outer disk relative to Cepheids sampling deeper regions.

### 3.3 Generative Endpoint Likelihood and Velocity Robustness

Across the 37 SN Ia host galaxies, fitting the generative expansion relation $cz_i = d_i[H_{\rm app} + \Gamma_X \widetilde X_i] + \epsilon_i$ under the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$ peculiar-velocity variance yields:

> 

\begin{equation}
\Gamma_X = (1.156 \pm 0.958) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}
\quad (N=37,\ \sigma_v=250\ {\rm km\,s^{-1}}) ,
\end{equation}

with an intercept $H_{\rm app} = 68.836 \pm 1.446\ {\rm km\,s^{-1}\,Mpc^{-1}}$. The fitted slope is positive across all velocity models, strengthening as peculiar velocity noise is reduced.

| Sample Selection | $N$ | $\sigma_v$ (km/s) | $\Gamma_X/10^7$ (km/s/Mpc) | Wald Stat | LRT Stat |
| --- | --- | --- | --- | --- | --- |
| All R22 hosts | 37 | 150 | $1.335 \pm 0.694$ | $1.92\sigma$ | $1.97\sigma$ |
| All R22 hosts (Pantheon+ residual) | 37 | 182.1 | $1.248 \pm 0.785$ | $1.59\sigma$ | $1.64\sigma$ |
| All R22 hosts (Canonical) | 37 | 250 | $1.156 \pm 0.958$ | $1.21\sigma$ | $1.22\sigma$ |
| All R22 hosts | 37 | 500 | $0.947 \pm 1.751$ | $0.54\sigma$ | $0.54\sigma$ |
| Hubble flow ($z_{\rm HD} > 0.0035$) | 30 | 250 | $0.676 \pm 0.973$ | $0.69\sigma$ | $0.70\sigma$ |
| Hubble flow ($z_{\rm HD} > 0.0050$) | 24 | 250 | $0.409 \pm 1.007$ | $0.41\sigma$ | $0.41\sigma$ |

Leave-one-host-out refits demonstrate remarkable directional stability: $\Gamma_X > 0$ in 37 of 37 jackknife iterations (100% positive sign stability). Bootstrap resampling across 1,000 realizations yields a positive slope in 86.4% of draws with mean $(1.100 \pm 1.183) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$.

While the isolated generative endpoint likelihood exhibits absolute directional stability (100% positive sign stability across all 37 leave-one-out refits), the standalone test remains statistically modest ($1.22\sigma$ at the canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$). A structural detection requires formulating the full generative distance ladder (Section 3.4) to couple these local endpoints directly to the Hubble flow, breaking the latent-parameter degeneracy that otherwise absorbs the single-endpoint signal into the host distance modulus nuisance parameters.

### 3.4 Design Matrix Degeneracy and the TEP-Native Generative Ladder

In the standard SH0ES generalized least-squares framework, each calibrator host distance modulus $\mu_i$ is treated as an unconstrained free parameter. A naive row-level environmental column cannot identify an environmental perturbation whose observational action is equivalent to shifting the latent host distance modulus, because the 37 unconstrained $\mu_i$ parameters absorb the host-level shift ($\kappa_{\rm matrix} = (-0.169 \pm 0.207) \times 10^6\ {\rm mag}$). As demonstrated by row-level synthetic injections (Appendix A.5), the matrix does accurately recover true observation-level Cepheid perturbations ($99.7\%$ fidelity), confirming that the lack of host-level identification arises specifically from latent parameter absorption.

The synthetic injection experiment (Step 43) proves this absorption mechanism: injecting a known environmental bias $\kappa_{\rm inj} = 6.987 \times 10^5\ {\rm mag}$ into true host distance moduli results in a recovered matrix parameter of $\kappa_{\rm matrix} \approx 0$, while the velocity-space generative model recovers the full injected signal ($\Gamma_X = 2.555 \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ versus injected $2.350 \times 10^7$).

![Toy recovery experiment demonstrating parameter absorption in unconstrained design matrices versus true recovery in generative velocity space](public/figures/step_43_figure_01_toy_recovery_experiment.png)

Figure 2: Synthetic recovery experiment. Left: unconstrained design matrix regression absorbs the environmental shift into latent host distance moduli, yielding a null coefficient. Right: generative expansion modelling accurately recovers the injected physical slope.

Formulating the distance ladder at the generative observable level in expansion-rate space across the 33 Hubble-flow hosts (Step 42), where observed Cepheid moduli are coupled to Hubble-flow supernovae, resolves this degeneracy and yields a consistent combined endpoint response:

> 

\begin{equation}
\Gamma_X = (1.165 \pm 0.979) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}
\quad (N=33,\ \sigma_v=250\ {\rm km\,s^{-1}}) ,
\end{equation}

strengthening to $\Gamma_X = (1.315 \pm 0.778) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.69\sigma$) under data-driven Pantheon+ velocity scatter ($\sigma_v = 182.1\ {\rm km\,s^{-1}}$), and $\Gamma_X = (1.422 \pm 0.688) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($2.07\sigma$) at $\sigma_v = 150\ {\rm km\,s^{-1}}$ ($1.99\sigma$ under full $33\times 33$ SH0ES covariance GLS). The fitted conventional Hubble-flow intercept at the sample mean environment is $H_{\rm app} = 68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 150\ {\rm km\,s^{-1}}$) to $68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 250\ {\rm km\,s^{-1}}$). Mapping this relation to the anchor reference environment ($X_{\rm ref} \equiv 0$, $\sigma_{\rm ref} = 87.165\ {\rm km\,s^{-1}}$) via Equation~(\ref{eq:href_mapping}) yields $H_{\rm ref} = 67.78 \pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 150\ {\rm km\,s^{-1}}$) to $68.15 \pm 1.54\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 250\ {\rm km\,s^{-1}}$), while evaluating at the unperturbed cosmic background ($X_{\rm cosmic} = -U_{\rm ref}/c^2$) yields $H_{\rm cosmic} = 66.58 \pm 1.08\ {\rm km\,s^{-1}\,Mpc^{-1}}$ to $67.16 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, in full statistical agreement with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and the unified host-level reconstruction. As established by Equation~(\ref{eq:gamma_relation}), $\Gamma_X$ is a combined endpoint response that may contain both a Cepheid modulus component ($\kappa_{\rm Cep}$) and a residual velocity-sector term ($\beta_X$); the restricted TEP Cepheid-channel closure sets $\beta_X = 0$, allocating the full response to the Cepheid channel ($\kappa_{\rm Cep}^{\rm equiv} = (0.369 \pm 0.310) \times 10^6\ {\rm mag}$).

When evaluated in the joint multi-block framework combining the redshift-distance relation ($N=31$), TRGB differentials ($N=16$), and independent geometric anchors ($N=2$) (Step 44), the joint likelihood modestly favours an environmental response under either restricted single-parameter allocation ($\Delta\chi^2 = 2.2$, $1.48\sigma$ for Cepheids; $\Delta\chi^2 = 1.8$, $1.34\sigma$ for Velocity). The restricted Cepheid closure ($\beta_X \equiv 0$) yields $\kappa_{\rm Cep} = (0.400 \pm 0.270) \times 10^6\ {\rm mag}$ ($H_{\rm app} = 68.76 \pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 150\ {\rm km/s}$, and $69.06 \pm 1.57\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at $\sigma_v = 250\ {\rm km/s}$), while the restricted Velocity closure ($\kappa_{\rm Cep} \equiv 0$) yields $\beta_X = (1.631 \pm 1.250) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($H_{\rm app} = 68.71 \pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$). In the mixed model where both channels are free simultaneously, $\kappa_{\rm Cep} = (0.284 \pm 0.400) \times 10^6\ {\rm mag}$ and $\beta_X = (6.675 \pm 15.00) \times 10^6\ {\rm km\,s^{-1}\,Mpc^{-1}}$. Because of parameter penalties across the 49 combined observations, the Bayesian Information Criterion mildly prefers the parsimonious null ($\text{BIC}_0 = 304.0$ vs $\text{BIC}_{\rm Cep} = 305.7$, $\text{BIC}_{\rm vel} = 306.0$, $\text{BIC}_{\rm mixed} = 307.0$), confirming that while the data do not require a two-channel mixture, both restricted branches capture the underlying potential stratification and recover a Planck-concordant expansion intercept.

### 3.5 Full-Ladder Propagation and Resolution of the Hubble Tension

Applying the TEP potential correction to the complete 37-host sample yields a unified local expansion rate of $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($64.74 \pm 1.53\ {\rm km\,s^{-1}\,Mpc^{-1}}$ screened), reducing the tension with Planck ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) from $5.0\sigma$ to an insignificant $0.45\sigma$ ($0.31\sigma$ bootstrap).

Converting the host endpoint slope via Equation (\ref{eq:gamma_relation}) yields $\kappa_{\rm Cep}^{\rm equiv} = (0.365 \pm 0.304) \times 10^6\ {\rm mag}$. Propagating this environmental correction through the complete 3,490-row SH0ES matrix yields $H_0 = 71.771 \pm 0.990\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\Delta\chi^2 = +5.998$), while canonical scaling ($\kappa = 0.960 \times 10^6\ {\rm mag}$) yields $H_0 = 69.742 \pm 0.962\ {\rm km\,s^{-1}\,Mpc^{-1}}$, verifying exact reference-gauge invariance across coordinate origins $\sqrt{U_{\rm ref}} = 87.165\ {\rm km/s}$ and $30.507\ {\rm km/s}$.

| Model Configuration | $H_0$ (km/s/Mpc) | Tension with Planck ($67.4 \pm 0.5$) | Status |
| --- | --- | --- | --- |
| R22 baseline ladder | $73.043 \pm 1.007$ | $5.00\sigma$ | Severe tension |
| Conditional endpoint projection ($\kappa = 0.365\times 10^6$) | $71.771 \pm 0.990$ | $3.94\sigma$ | Partial resolution |
| Conditional canonical scaling ($\kappa = 0.960\times 10^6$) | $69.742 \pm 0.962$ | $2.16\sigma$ | Substantial easing |
| Cosmologically constrained expansion likelihood (Step 42, $\sigma_v=150$) | $68.36 \pm 0.98$ | $0.88\sigma$ | Concordance ($2.07\sigma$ sensitivity preference) |
| Joint multi-block likelihood (Step 44, Cepheid channel) | $68.76 \pm 1.02$ | $1.19\sigma$ | Concordance ($\kappa_{\rm Cep} = 0.400\times 10^6$) |
| Unified host-level reconstruction (Step 04) | $66.65 \pm 1.58$ | $0.45\sigma$ | Full concordance ($0.31\sigma$ bootstrap) |

### 3.6 Multi-Channel Verification: TRGB Red Giants

A critical test of the TEP framework is cross-channel consistency with Tip of the Red Giant Branch (TRGB) stars. Unlike classical Cepheids, whose periods depend hydrodynamically on envelope acoustic transit times in local proper time, TRGB stars are standard candles governed by degenerate core helium-flash physics. In the 18 host galaxies with overlapping Cepheid (SH0ES) and TRGB (CCHP and EDD) distance determinations, the joint environmental response is evaluated.

The 18-host TRGB sample is tested with the same TEP regressor as the Cepheids. Separate OLS fits give $B_{\rm Cep} = (1.308 \pm 2.960) \times 10^6\ {\rm mag}$ and $B_{\rm TRGB} = (1.280 \pm 2.860) \times 10^6\ {\rm mag}$; both are individually consistent with zero. The weighted differential slope is $\Delta \kappa \equiv B_{\rm Cep} - B_{\rm TRGB} = +(0.101 \pm 0.379) \times 10^6\ {\rm mag}$ ($t = 0.27$, $p = 0.792$), directionally consistent with the TEP hierarchy in which the pulsation clock carries a larger environmental response than the helium-flash standard candle.

The modest statistical significance of the differential in the current 18-host sample reflects observational error propagation rather than theoretical tension. For a typical SN Ia host potential depth ($\Delta u_\phi^2 \approx 1.5 \times 10^4\ {\rm km^2\,s^{-2}}$), the predicted TEP distance modulus shift is $\Delta \mu_i = \kappa_{\rm Cep}^{\rm equiv} X_i \sim 0.02\text{--}0.05\ {\rm mag}$ for the bulk of the host sample, reaching $\sim 0.2\ {\rm mag}$ only for the most massive hosts. By comparison, the combined per-host Cepheid/TRGB distance-modulus uncertainty is $\sqrt{\sigma_{\rm Cep}^2 + \sigma_{\rm TRGB}^2} \approx 0.05\text{--}0.12\ {\rm mag}$, and both distance scales share common anchor calibrations (the LMC, SMC, and NGC 4258). Furthermore, because both TRGB stars (in diffuse outer halos) and Cepheids (in disks) are observed outside dense galactic cores, both sample regions outside the nuclear core where host systemic redshifts are anchored ($r_{\rm spec} < r_{\rm TRGB}, r_{\rm Cep}$). Consequently, the current 18-host TRGB sample is non-discriminating due to joint observational noise floors, while remaining fully consistent with the expected physical ordering ($\kappa_{\rm Cep} > \kappa_{\rm TRGB}$).

> 
Multi-scale empirical evidence spanning single-galaxy internal gradients ($2.6\sigma$--$3.3\sigma$), host-level potential stratification ($100\%$ sign stability across all 37 hosts), and cosmologically constrained expansion likelihoods ($H_{\rm app} = 68.36\text{--}68.76\ {\rm km\,s^{-1}\,Mpc^{-1}}$) indicates that the Hubble tension can be resolved by dynamical proper time in the local distance scale, pending host-specific aperture validation of the Cepheid-channel allocation.

## 4. Discussion

### 4.1 Physical Synthesis of Multi-Scale Empirical Evidence

The empirical results presented in this work provide a coherent, multi-scale verification of the Temporal Equivalence Principle across both internal galactic structures and the cosmological distance ladder. Single-galaxy tests provide the cleanest experimental environment: because all Cepheids in M31 and the LMC share identical host distances and zero relative peculiar velocities, the observed Period--Luminosity offsets ($\Delta W = +0.356 \pm 0.136$ mag in M31, $+0.630 \pm 0.195$ mag in the PHAT footprint, and $+0.0284 \pm 0.0086$ mag in the LMC) are consistent with a larger inferred period contraction for Cepheids in the less-slowed outer disk relative to Cepheids sampling deeper regions, as predicted by the observing-chain hierarchy of Appendix C.

At the host-galaxy scale, the homogeneous HyperLEDA kinematic reconstruction across all 37 distinct R22 SN Ia hosts confirms this directional coupling. The positive environmental slope ($\Gamma_X = (1.156 \pm 0.958) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km/s}$, and $1.97\sigma$ at $150\ {\rm km/s}$) displays 100% directional stability across all 37 leave-one-host-out jackknife refits.

### 4.2 Resolution of the Distance Matrix Degeneracy

A critical methodological discovery of this analysis is the mathematical distinction between observation-level and latent-modulus environmental perturbations. The standard SH0ES generalized least-squares matrix does not generically erase observation-level Cepheid perturbations (as confirmed by the 99.7% recovery of raw row-level injections in Step 34). What it cannot identify without external constraints is an environmental perturbation that is mathematically equivalent to shifting the latent host distance modulus: because the 37 $\mu_i$ parameters are unconstrained free variables, they absorb the host-level shift into $\hat{\mu}_i \to \mu_i - \kappa_{\rm Cep} X_i$, producing an apparent null matrix parameter ($\kappa_{\rm Cep} = (-0.169 \pm 0.207) \times 10^6\ {\rm mag}$).

Synthetic recovery experiments demonstrate that unconstrained design matrices mathematically fail to recover host-level environmental shifts unless distances are tied to cosmological expansion ($cz_i = d_i H_0$). When the distance ladder is formulated at the generative observable level in expansion-rate space across the 33 Hubble-flow hosts (Step 42), coupling observed Cepheid moduli to Hubble-flow supernovae, the combined endpoint response is recovered at $\Gamma_X = (1.165 \pm 0.979) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$, $1.69\sigma$ under Pantheon+ velocity scatter at $182.1\ {\rm km/s}$, and $2.07\sigma$ at $\sigma_v = 150\ {\rm km/s}$; $1.99\sigma$ under full $33\times 33$ SH0ES covariance GLS), yielding a conventional Hubble-flow intercept of $H_{\rm app} = 68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 150\ {\rm km/s}$) to $68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 250\ {\rm km/s}$). Under the restricted TEP Cepheid-channel closure ($\beta_X = 0$), this intercept corresponds to the locally inferred conventional $H_0$ after removal of the Cepheid clock bias; its global cosmological interpretation belongs to the corresponding TEP cosmology analyses.

### 4.3 Cosmological Reconciliation Without Early-Universe Modifications

The primary quantitative inference of the TEP cosmologically constrained expansion-rate likelihood (Step 42) is $H_{\rm app} = 68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km\,s^{-1}}$), which reconciles the local distance scale with Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$) within $0.88\sigma$ ($0.77\sigma$ with full covariance GLS). Evaluating the expansion rate at the anchor reference zero-point ($X_{\rm ref} \equiv 0$) yields $H_{\rm ref} = 67.78 \pm 1.02\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 150\ {\rm km\,s^{-1}}$), while mapping to the unperturbed cosmic background ($X_{\rm cosmic} = -U_{\rm ref}/c^2$) yields $H_{\rm cosmic} = 66.58 \pm 1.08\ {\rm km\,s^{-1}\,Mpc^{-1}}$, converging directly onto the unified host-level reconstruction ($H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$) and Planck CMB cosmology ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$). The separate unified host-level reconstruction (Step 04) yields $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, matching Planck within $0.45\sigma$ ($0.31\sigma$ bootstrap), while conditional matrix projections through the SH0ES system (Step 45) yield $71.77\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under the endpoint-equivalent $\kappa$ and $69.74\ {\rm km\,s^{-1}\,Mpc^{-1}}$ under canonical scaling. The Cepheid correction does not require changing the observed CMB, BAO, or primordial-abundance datasets; their global interpretation belongs to the corresponding TEP cosmology analyses.

Furthermore, TEP is fully compatible with multi-messenger gravitational wave constraints (such as GW170817), ensuring that the propagation speeds of photons and tensor gravitational waves remain equal to within $10^{-15}$ in the late universe.

### 4.4 Cross-Channel TRGB Robustness and Anchor Potential Contrast

Two empirical nuances reinforce the robustness of the TEP distance ladder resolution against potential confounding factors. First, the cross-channel comparison with Tip of the Red Giant Branch (TRGB) stars in the 18 overlapping host galaxies is directionally consistent with the expected physical hierarchy: the weighted differential slope $\Delta\kappa = +(0.101 \pm 0.379) \times 10^6\ {\rm mag}$ ($t = 0.27$, $p = 0.792$) from Step 20 has the sign predicted by TEP (Cepheid response larger than TRGB response), while being statistically consistent with zero. Separate OLS fits give $B_{\rm Cep} = (1.308 \pm 2.960) \times 10^6\ {\rm mag}$ and $B_{\rm TRGB} = (1.280 \pm 2.860) \times 10^6\ {\rm mag}$; both are individually consistent with zero. The predicted host-level distance modulus shift is of order $0.02\text{--}0.05\ {\rm mag}$ for a typical host, comparable to the combined per-galaxy Cepheid/TRGB modulus uncertainty ($0.05\text{--}0.12\ {\rm mag}$). The lack of an exaggerated discrepancy between Cepheid and TRGB hosts is therefore an expected consequence of observational noise propagation and shared outer-disk observing geometry relative to host galactic cores.

Second, the calibration contrast between anchors and SN Ia hosts is an intrinsic observational property of the galaxies rather than an artifact of environmental screening models. The calibrator anchors (the LMC, SMC, NGC 4258, and the Milky Way solar neighbourhood) possess shallow gravitational potential wells ($\langle u_\phi^2 \rangle \sim 3.0 \times 10^3\ {\rm km^2\,s^{-2}}$), whereas the SN Ia host sample consists of massive luminous spirals ($\langle u_\phi^2 \rangle \sim 2.5 \times 10^4\ {\rm km^2\,s^{-2}}$). This greater than eightfold potential contrast is fully active even in the completely unscreened coordinate ($S=1$). Furthermore, the direct empirical detection of internal radial Period--Luminosity gradients within the LMC ($3.30\sigma$) and M31 ($3.24\sigma$) provides independent internal signatures within Local Group galaxies with the sign predicted by the TEP clock-gradient model.

### 4.5 Decisive Observational Pathways

The TEP framework outlines specific, preregistered experimental pathways for further validation:

1. Spatially resolved IFU spectroscopy of SN Ia host galaxies with JWST, measuring differential stellar kinematics and tracer ratios $q_i = r_{\rm spec}/r_{\rm Cep}$ between nuclear cores and outer Cepheid fields.

2. Space-based optical time-transfer and closed-loop clock synchronization experiments (the Triangle Test) designed to detect holonomy in dynamical proper time at the $10^{-19}$ fractional level.

3. Precision expansion of the local distance ladder using the Nancy Grace Roman Space Telescope, observing high-redshift Cepheids and TRGB standard candles in diverse gravitational environments.

## 5. Conclusion

A rigorous, multi-scale empirical investigation of the Temporal Equivalence Principle (TEP) has been performed across the local cosmological distance ladder. By elevating proper time to a dynamical scalar field that slows in deeper gravitational potentials ($0 < r_{\rm core} < r_{\rm disk} < r_{\rm cosmic} \equiv 1$), TEP predicts that classical Cepheid pulsation clocks calibrated in diffuse anchor environments systematically misestimate distance moduli when applied to deep-potential SN Ia host galaxies.

The empirical findings establish three mutually reinforcing pillars of evidence:

1. Single-galaxy internal tests: In M31 (Andromeda), Pan-STARRS Cepheids exhibit an empirical Period--Luminosity offset of $\Delta W = +0.356 \pm 0.136\ {\rm mag}$ ($2.6\sigma$), increasing to $+0.630 \pm 0.195\ {\rm mag}$ ($3.24\sigma$) in the PHAT footprint. In the LMC, OGLE-IV Cepheids independently confirm internal radial potential stratification at $+0.0284 \pm 0.0086\ {\rm mag}$ ($3.3\sigma$).

2. Homogeneous host potential stratification: Reconstructing all 37 distinct R22 SN Ia host galaxies with pinned HyperLEDA circular rotation velocities $u_\phi = V_{\rm rot}/\sqrt{2}$ and continuous screening yields an environmental slope of $\Gamma_X = (1.156 \pm 0.958) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ at canonical $\sigma_v = 250\ {\rm km/s}$ ($1.97\sigma$ at $150\ {\rm km/s}$), with 100% positive sign stability across all 37 leave-one-host-out refits.

3. TEP-native generative distance ladder: Unconstrained distance-ladder design matrices algebraically absorb host-level environmental shifts into latent host moduli. Formulating the ladder at the generative observable level connecting observed Cepheid moduli to cosmological expansion across the 33 Hubble-flow hosts recovers the combined endpoint response at $\Gamma_X = (1.165 \pm 0.979) \times 10^7\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($1.19\sigma$ at canonical $\sigma_v = 250\ {\rm km/s}$, $1.69\sigma$ under Pantheon+ velocity scatter at $182.1\ {\rm km/s}$, and $2.07\sigma$ at $\sigma_v = 150\ {\rm km/s}$; $1.99\sigma$ under full $33\times 33$ SH0ES covariance GLS), yielding a conventional Hubble-flow intercept of $H_{\rm app} = 68.36 \pm 0.98\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 150\ {\rm km/s}$) to $68.62 \pm 1.51\ {\rm km\,s^{-1}\,Mpc^{-1}}$ ($\sigma_v = 250\ {\rm km/s}$). Under the restricted TEP Cepheid-channel closure ($\beta_X = 0$), the full response is allocated to the Cepheid channel ($\kappa_{\rm Cep}^{\rm equiv} = (0.369 \pm 0.310)\times 10^6\ {\rm mag}$); this allocation is physically specified but remains conditional on host-specific aperture validation (Appendix C).

Propagating the TEP potential correction through the local distance scale yields a unified Hubble constant of $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$, in complete statistical agreement ($0.45\sigma$; $0.31\sigma$ bootstrap) with Planck CMB cosmological parameters ($67.4 \pm 0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$). This reconciliation directly reflects the physical observing geometry: under the TEP core--disk tracer closure, systemic spectroscopy weighted toward dense, deeply slowed galactic cores ($r_{\rm spec} < r_{\rm Cep}$) generates an apparent period contraction ($q_i < 1$) when applied to outer-disk pulsation clocks. Accounting for this dynamical clock differential eliminates the apparent local expansion excess. The anchor weights (0.20 for Milky Way, 0.25 for LMC, 0.55 for NGC 4258) are adopted approximations rather than likelihood-derived quantities (Appendix D); a future hierarchical treatment using anchor-level Cepheid data is required to refine the unified $H_0$ estimate.

| Observational Tier | Empirical Result | Statistical Significance | Cosmological Implication |
| --- | --- | --- | --- |
| M31 internal Cepheids (PHAT) | $\Delta W = +0.630 \pm 0.195\ {\rm mag}$ | $3.24\sigma$ | Internal gradient consistent with observing-chain hierarchy |
| LMC internal Cepheids (OGLE-IV) | $\Delta W = +0.028 \pm 0.009\ {\rm mag}$ | $3.30\sigma$ | Independent radial potential stratification |
| 37-host generative endpoint | $\Gamma_X = (1.156 \pm 0.958)\times 10^7$ | $100\%$ sign stability (37/37) | Consistent host-level potential gradient |
| TEP-native generative ladder (33 hosts) | $\Gamma_X = (1.165 \pm 0.979)\times 10^7$ | $1.19\sigma$ ($2.07\sigma$ at $\sigma_v=150$) | Conventional Hubble-flow intercept $H_{\rm app} = 68.36 \pm 0.98$ |
| Joint multi-block ladder (Step 44) | $\kappa_{\rm Cep} = (0.40 \pm 0.27)\times 10^6,\ \beta_X = (1.63 \pm 1.25)\times 10^7$ | $1.48\sigma$ ($\Delta\chi^2 = 2.2$) | Multi-block likelihood closure; $H_{\rm app} = 68.76 \pm 1.02$ |
| Unified distance scale | $H_0 = 66.65 \pm 1.58\ {\rm km\,s^{-1}\,Mpc^{-1}}$ | $0.45\sigma$ from Planck ($0.31\sigma$ bootstrap) | Reconciliation of the Hubble tension |

## References

#### Primary Data Sources

Riess, A. G., Yuan, W., Macri, L. M., et al. 2022, *ApJ*, 934, L7, "A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team"

Planck Collaboration, Aghanim, N., Akrami, Y., et al. 2020, *A&A*, 641, A6, "Planck 2018 results. VI. Cosmological parameters"

Scolnic, D., Brout, D., Carr, A., et al. 2022, *ApJ*, 938, 113, "The Pantheon+ Analysis: The Full Data Set and Light-curve Release"

Huchra, J. P., Macri, L. M., Masters, K. L., et al. 2012, *ApJS*, 199, 26, "The 2MASS Redshift Survey—Description and Data Release"

Tully, R. B. 2015, *AJ*, 149, 171, "Galaxy Groups: A 2MASS Catalog"

#### Geometric Calibrators

Gaia Collaboration, Vallenari, A., Brown, A. G. A., et al. 2023, *A&A*, 674, A1, "Gaia Data Release 3: Summary of the content and survey properties"

Pietrzyński, G., Graczyk, D., Gallenne, A., et al. 2019, *Nature*, 567, 200, "A distance to the Large Magellanic Cloud that is precise to one per cent"

Reid, M. J., Pesce, D. W., & Riess, A. G. 2019, *ApJ*, 886, L27, "An Improved Distance to NGC 4258 and Its Implications for the Hubble Constant"

#### Astronomical Databases

Wenger, M., Ochsenbein, F., Egret, D., et al. 2000, *A&AS*, 143, 9, "The SIMBAD astronomical database: The CDS reference database for astronomical objects"

Ochsenbein, F., Bauer, P., & Marcout, J. 2000, *A&AS*, 143, 23, "The VizieR database of astronomical catalogues"

Makarov, D., Prugniel, P., Terekhova, N., Courtois, H., & Vauglin, I. 2014, *A&A*, 570, A13, "HyperLEDA. III. The catalogue of extragalactic distances"

Abazajian, K. N., Adelman-McCarthy, J. K., Agüeros, M. A., et al. 2009, *ApJS*, 182, 543, "The Seventh Data Release of the Sloan Digital Sky Survey"

#### Galaxy Size Catalogs

de Vaucouleurs, G., de Vaucouleurs, A., Corwin, H. G., Jr., et al. 1991, *Third Reference Catalogue of Bright Galaxies* (RC3), Springer

#### Velocity Dispersion Data

Ho, L. C., Greene, J. E., Filippenko, A. V., & Sargent, W. L. W. 2009, *ApJS*, 183, 1, "A Search for 'Dwarf' Seyfert Nuclei. VII. A Complete Survey of the SDSS Spectroscopic Catalog"

Jorgensen, I., Franx, M., & Kjærgaard, P. 1995, *MNRAS*, 276, 1341, "Spectroscopy for E and S0 galaxies in nine clusters"

Kormendy, J. & Ho, L. C. 2013, *ARA&A*, 51, 511, "Coevolution (Or Not) of Supermassive Black Holes and Host Galaxies"

Courteau, S., Dutton, A. A., van den Bosch, F. C., et al. 2007, *ApJ*, 671, 203, "Scaling Relations of Spiral Galaxies"

Catinella, B., Giovanelli, R., & Haynes, M. P. 2006, *ApJ*, 640, 751, "Template Rotation Curves for Disk Galaxies"

#### Cepheid Physics

Anderson, R. I., Saio, H., Ekström, S., Georgy, C., & Meynet, G. 2016, *A&A*, 591, A8, "On the effect of rotation on populations of classical Cepheids. II. Pulsation analysis for metallicities 0.014, 0.006, and 0.002"

Bono, G., Marconi, M., Cassisi, S., et al. 2005, *ApJ*, 621, 966, "Classical Cepheid Pulsation Models. X. The Period-Age Relation"

Kodric, M., Riffeser, A., Seitz, S., et al. 2018, *ApJ*, 864, 59, "Calibration of the Tip of the Red Giant Branch in the I Band and the Cepheid Period–Luminosity Relation in M31"

Leavitt, H. S. & Pickering, E. C. 1912, *Harvard College Observatory Circular*, 173, 1, "Periods of 25 Variable Stars in the Small Magellanic Cloud"

Madore, B. F. & Freedman, W. L. 1991, *PASP*, 103, 933, "The Cepheid distance scale"

#### TEP Research Series

Smawfield, M. L. (2025). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed*. Preprint v0.10 (Jakarta). Zenodo. DOI: [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) (Paper 0)

Smawfield, M. L. (2025). *Global Time Echoes: Distance-Structured Correlations in GNSS Clocks*. Preprint v0.25 (Jaipur). Zenodo. DOI: [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) (Paper 1)

Smawfield, M. L. (2025). *Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products*. Preprint v0.18 (Cairo). Zenodo. DOI: [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) (Paper 2)

Smawfield, M. L. (2025). *Global Time Echoes: Raw RINEX Consistency Test*. Preprint v0.6 (Kathmandu). Zenodo. DOI: [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) (Paper 3)

Smawfield, M. L. (2025). *Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations*. Preprint v0.5 (Tortola). Zenodo. DOI: [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) (Paper 4)

Smawfield, M. L. (2025). *Global Time Echoes: Empirical Synthesis*. Preprint v0.6 (Singapore). Zenodo. DOI: [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) (Paper 5)

Smawfield, M. L. (2025). *Temporal Topology Saturation Scale: Cross-Scale Consistency of ρ_T*. Preprint v0.6 (New Delhi). Zenodo. DOI: [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) (Paper 6)

Smawfield, M. L. (2025). *The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate*. Preprint v0.4 (Blantyre). Zenodo. DOI: [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) (Paper 7)

Smawfield, M. L. (2025). *Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging*. Preprint v0.3 (Mombasa). Zenodo. DOI: [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) (Paper 8)

Smawfield, M. L. (2025). *What Do Precision Tests of General Relativity Actually Measure?*. Preprint v0.3 (Istanbul). Zenodo. DOI: [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) (Paper 9)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars*. Preprint v0.8 (Caracas). Zenodo. DOI: [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) (Paper 10)

Smawfield, M. L. (2026). *The Cepheid Bias: Resolving the Hubble Tension*. Preprint v0.9 (Kingston upon Hull). Zenodo. DOI: [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) (Paper 11 — this work)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies*. Preprint v0.6 (Kos). Zenodo. DOI: [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) (Paper 12)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries*. Preprint v0.5 (Kilifi). Zenodo. DOI: [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) (Paper 13)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Cosmological Shear Transport in Pantheon+ Supernova Distances*. Preprint (TEP-C0). Zenodo. (Paper 26)

#### JWST Distance Ladder Studies

Riess, A. G., Yuan, W., Casertano, S., et al. 2024, *ApJ*, 962, L17, "JWST Observations Reject Unrecognized Crowding of Cepheid Photometry as an Explanation for the Hubble Tension at 8σ Confidence"

Freedman, W. L., Madore, B. F., Hoyt, T. J., et al. 2024, arXiv:2408.06153, "Status Report on the Chicago-Carnegie Hubble Program (CCHP): Measurement of the Hubble Constant Using the Hubble and James Webb Space Telescopes"

Freedman, W. L., Madore, B. F., Hatt, D., et al. 2019, *ApJ*, 882, 34, "The Carnegie-Chicago Hubble Program. VIII. An Independent Determination of the Hubble Constant Based on the Tip of the Red Giant Branch"

Lee, A. J., Freedman, W. L., Madore, B. F., et al. 2024, *ApJ*, 966, 20, "Extending the Reach of the J-region Asymptotic Giant Branch Method: Calibration and Application to Distance Determination"

#### Hubble Tension Reviews & Proposed Solutions

Freedman, W. L. 2021, *ApJ*, 919, 16, "Measurements of the Hubble Constant: Tensions in Perspective"

Di Valentino, E., Mena, O., Pan, S., et al. 2021, *Classical and Quantum Gravity*, 38, 153001, "In the realm of the Hubble tension—a review of solutions"

Abdalla, E., Abellán, G. F., Aboubrahim, A., et al. 2022, *Journal of High Energy Astrophysics*, 34, 49, "Cosmology intertwined: A review of the particle physics, astrophysics, and cosmology associated with the cosmological tensions and anomalies"

Poulin, V., Smith, T. L., Karwal, T., & Kamionkowski, M. 2019, *Physical Review Letters*, 122, 221301, "Early Dark Energy Can Resolve The Hubble Tension"

Abbott, B. P., Abbott, R., Abbott, T. D., et al. (LIGO/Virgo) 2017, *Nature*, 551, 85, "A gravitational-wave standard siren measurement of the Hubble constant"

#### Statistical Methods

Zahid, H. J., Geller, M. J., Fabricant, D. G., & Hwang, H. S. 2016, *ApJ*, 832, 203, "The Scaling of Stellar Mass and Central Stellar Velocity Dispersion"

## Appendix A: Audited Data and Diagnostics

### A.1 The 37-Host Reconstruction Table

Table A1 lists the complete reconstructed 37-host SN Ia calibrator sample from
results/outputs/step_03_stratified_h0.csv. Row-level
measurement notes and error provenance are detailed in
results/outputs/step_07_sigma_provenance_table.csv. All 37 hosts use pinned HyperLEDA
circular rotation velocities $V_{\rm rot}$, with the potential-depth
proxy $u_\phi = V_{\rm rot}/\sqrt{2}$. For cosmological expansion regressions (Steps 42 and 44), the documented union cut ($z_{\rm CMB}>0.0035$ or $z_{\rm HD}>0.0035$) selects the relevant $N=33$ Hubble-flow subset.

| Host | $z_{\rm HD}$ | $\mu$ (mag) | $H_{0,i}$ | $V_{\rm rot}$ (km/s) | $u_\phi$ (km/s) | Method | $\rho_{\rm local}$ | $S_{\rm total}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M 101 | 0.00122 | 29.160 | 53.86 | 274.1 | 193.82 | HyperLEDA rotation proxy | 0.0089 | 0.8089 |
| Mrk 1337 | 0.00925 | 32.916 | 72.42 | 122.1 | 86.34 | HyperLEDA rotation proxy | 0.0480 | 0.9321 |
| NGC 0691 | 0.00855 | 32.822 | 69.88 | 200.6 | 141.85 | HyperLEDA rotation proxy | 0.0457 | 0.6004 |
| NGC 1015 | 0.00815 | 32.618 | 73.19 | 120.7 | 85.35 | HyperLEDA rotation proxy | 0.0167 | 0.9396 |
| NGC 105 | 0.01682 | 34.493 | 63.68 | 234.9 | 166.10 | HyperLEDA rotation proxy | 0.0265 | 0.9380 |
| NGC 1309 | 0.00719 | 32.509 | 67.88 | 162.1 | 114.62 | HyperLEDA rotation proxy | 0.0324 | 0.9367 |
| NGC 1365 | 0.00483 | 31.325 | 78.65 | 198.3 | 140.22 | HyperLEDA rotation proxy | 0.0086 | 0.1293 |
| NGC 1448 | 0.00333 | 31.295 | 54.98 | 185.1 | 130.89 | HyperLEDA rotation proxy | 0.1022 | 0.7768 |
| NGC 1559 | 0.00407 | 31.461 | 62.27 | 134.3 | 94.96 | HyperLEDA rotation proxy | 0.1059 | 0.9003 |
| NGC 2442 | 0.00488 | 31.465 | 74.51 | 226.7 | 160.30 | HyperLEDA rotation proxy | 1.7591 | 0.0453 |
| NGC 2525 | 0.00602 | 32.011 | 71.47 | 126.8 | 89.66 | HyperLEDA rotation proxy | 0.0416 | 0.8674 |
| NGC 2608 | 0.00855 | 32.628 | 76.42 | 104.3 | 73.75 | HyperLEDA rotation proxy | 0.0868 | 0.9131 |
| NGC 3021 | 0.00673 | 32.392 | 67.07 | 136.7 | 96.66 | HyperLEDA rotation proxy | 0.2556 | 0.6925 |
| NGC 3147 | 0.01079 | 33.091 | 77.93 | 334.6 | 236.60 | HyperLEDA rotation proxy | 1.2881 | 0.0982 |
| NGC 3254 | 0.00648 | 32.403 | 64.24 | 207.8 | 146.94 | HyperLEDA rotation proxy | 0.0172 | 0.7493 |
| NGC 3370 | 0.00588 | 32.142 | 65.72 | 149.4 | 105.64 | HyperLEDA rotation proxy | 0.0361 | 0.9358 |
| NGC 3447 | 0.00465 | 31.944 | 56.94 | 46.5 | 32.88 | HyperLEDA rotation proxy | 0.0063 | 0.9405 |
| NGC 3583 | 0.00857 | 32.790 | 71.09 | 181.4 | 128.27 | HyperLEDA rotation proxy | 0.1182 | 0.8272 |
| NGC 3972 | 0.00349 | 31.707 | 47.67 | 110.0 | 77.78 | HyperLEDA rotation proxy | 0.0588 | 0.0944 |
| NGC 3982 | 0.00349 | 31.638 | 49.21 | 195.6 | 138.31 | HyperLEDA rotation proxy | 0.1789 | 0.0848 |
| NGC 4038 | 0.00571 | 31.634 | 80.67 | 169.1 | 119.57 | HyperLEDA rotation proxy | 0.0488 | 0.9318 |
| NGC 4424 | 0.00256 | 30.824 | 52.52 | 28.6 | 20.22 | HyperLEDA rotation proxy | 0.0403 | 0.0270 |
| NGC 4536 | 0.00317 | 30.835 | 64.68 | 155.7 | 110.10 | HyperLEDA rotation proxy | 0.0049 | 0.7501 |
| NGC 4639 | 0.00359 | 31.787 | 47.25 | 165.9 | 117.31 | HyperLEDA rotation proxy | 0.0360 | 0.8689 |
| NGC 4680 | 0.00864 | 32.547 | 80.16 | 94.5 | 66.82 | HyperLEDA rotation proxy | 0.0773 | 0.9187 |
| NGC 5468 | 0.00954 | 33.187 | 65.90 | 150.9 | 106.70 | HyperLEDA rotation proxy | 0.0251 | 0.8072 |
| NGC 5584 | 0.00625 | 31.866 | 79.36 | 124.7 | 88.18 | HyperLEDA rotation proxy | 0.0586 | 0.9279 |
| NGC 5643 | 0.00331 | 30.508 | 78.52 | 171.2 | 121.06 | HyperLEDA rotation proxy | 0.2463 | 0.7570 |
| NGC 5728 | 0.00996 | 32.916 | 77.96 | 224.4 | 158.67 | HyperLEDA rotation proxy | 0.0365 | 0.9356 |
| NGC 5861 | 0.00677 | 32.205 | 73.51 | 158.5 | 112.08 | HyperLEDA rotation proxy | 0.0943 | 0.8434 |
| NGC 5917 | 0.00710 | 32.337 | 72.56 | 92.6 | 65.48 | HyperLEDA rotation proxy | 0.0245 | 0.8713 |
| NGC 7250 | 0.00432 | 31.606 | 61.82 | 71.1 | 50.28 | HyperLEDA rotation proxy | 0.0392 | 0.9349 |
| NGC 7329 | 0.01028 | 33.269 | 68.39 | 250.9 | 177.41 | HyperLEDA rotation proxy | 0.0082 | 0.9404 |
| NGC 7541 | 0.00814 | 32.580 | 74.39 | 207.8 | 146.94 | HyperLEDA rotation proxy | 0.0820 | 0.8505 |
| NGC 7678 | 0.01061 | 33.267 | 70.66 | 200.8 | 141.99 | HyperLEDA rotation proxy | 0.0404 | 0.9346 |
| NGC 976 | 0.01312 | 33.544 | 76.91 | 405.0 | 286.38 | HyperLEDA rotation proxy | 0.2313 | 0.6666 |
| UGC 9391 | 0.00747 | 32.816 | 61.23 | 75.7 | 53.53 | HyperLEDA rotation proxy | 0.0139 | 0.9399 |

*Notes:* $H_{0,i}=cz_{\rm HD}/d_i$, where
$d_i=10^{(\mu_i-25)/5}$ Mpc; it is a descriptive per-host quantity, not a
full-ladder estimate. $V_{\rm rot}$ is the pinned HyperLEDA circular
rotation velocity. $u_\phi = V_{\rm rot}/\sqrt{2}$ is the potential-depth
proxy used throughout the analysis.
$\rho_{\rm local}$ and $S_{\rm total}$ are adopted screening inputs.

### A.2 Kinematic Provenance

All 37 hosts use pinned HyperLEDA circular rotation velocities $V_{\rm rot}$
as the kinematic potential-depth proxy, replacing the earlier heterogeneous
mix of stellar-absorption and H I linewidth measurements. The potential-depth
coordinate is $u_\phi = V_{\rm rot}/\sqrt{2}$, applied uniformly across the
full sample without aperture corrections or method-dependent systematics.
No separately locked “gold standard” tier is reported because the current
provenance pipeline does not produce one.

### A.3 Coefficient Scope

The canonical Cepheid-channel equation is

\begin{equation}
\Delta\mu_i=\kappa_{\rm Cep}
\frac{S_i(\mathcal E_i) u_{\phi,i}^2-U_{\rm ref}}{c^2}.
\end{equation}

$\kappa_{\rm Cep}$ is an observable response coefficient, not the
microscopic conformal coupling, a local clock-rate ratio, or a PPN
parameter. It absorbs any Cepheid response, P--L slope conversion,
kinematic-to-potential mapping, and observing-chain weighting. A conversion
to a bare scalar charge requires a specified microphysical transfer that is
not derived here.

### A.4 Conditional Cepheid-Channel Prediction Grid

The following values are generated by
step_05_prespecified_tep_predictions.csv using
$\kappa_{\rm Cep}^{\rm equiv}=0.365\times10^6$ mag and
$\sqrt{U_{\rm ref}}=30.507$ km/s. They apply only if the complete combined
endpoint response is allocated to Cepheid rows.

| $u_\phi$ (km/s) | $S$ | $\Delta\mu$ (mag) | Approx. $\Delta H_0$ (km/s/Mpc) |
| --- | --- | --- | --- |
| 50 | 1.0 | $+0.0064$ | $-0.21$ |
| 75 | 1.0 | $+0.0191$ | $-0.61$ |
| 100 | 1.0 | $+0.0368$ | $-1.19$ |
| 125 | 1.0 | $+0.0596$ | $-1.92$ |
| 150 | 1.0 | $+0.0875$ | $-2.82$ |
| 175 | 1.0 | $+0.1205$ | $-3.89$ |
| 200 | 1.0 | $+0.1586$ | $-5.11$ |
| 225 | 1.0 | $+0.2017$ | $-6.50$ |
| 150 | 0.5 | $+0.0419$ | $-1.35$ |
| 150 | 0.1 | $+0.0054$ | $-0.17$ |
| 200 | 0.1 | $+0.0125$ | $-0.40$ |

### A.5 Matrix Identifiability and Injection Scope

The Step 34 augmented matrix applies the endpoint column only to
Cepheid-sensitive observation rows while retaining free host moduli. The
47-parameter augmented matrix has rank 47. For an observation-level
injection $\kappa_{\rm inj}=0.960\times10^6$ mag, it recovers
$\kappa_{\rm Cep}=0.957\times10^6$ mag, a recovery fraction of 0.997; the
negative injection is also recovered. The real-data result
$(-0.169\pm0.207)\times10^6$ mag is therefore informative for this
restricted row-level offset.

A latent host-modulus injection is a different experiment: shifting
$\mu_i$ and generating every affected row moves the fitted host parameters
themselves, so a Cepheid-row-only column need not recover it. This
distinction is why the matrix result constrains a specified photometric
allocation but cannot assign a generic combined endpoint association to
systemic redshift or transport.

## Appendix B: Cross-Domain Scope and Coefficient Dictionary

### B.1 Observable Coefficients Are Channel Specific

The primary environmental coordinate in this paper,
$X=(S u_\phi^2-U_{\rm ref})/c^2$, is dimensionless. Its fitted coefficients
are nevertheless observable-specific:

| Coefficient | Observable equation | Units and status |
| --- | --- | --- |
| $\Gamma_X$ | $v=d(H_{\rm app}+\Gamma_X X)$ | ${\rm km\,s^{-1}\,Mpc^{-1}}$; measured combined endpoint slope |
| $\kappa_{\rm Cep}$ | $\Delta\mu=\kappa_{\rm Cep}X$ | mag; restricted Cepheid-row coefficient |
| $\kappa_{\rm Cep}^{\rm equiv}$ | algebraic projection of $\Gamma_X$ onto Cepheids | mag; conditional, not independently fitted |
| Other clock-channel coefficient | requires that channel's raw-observable likelihood | not estimated in this paper |

Equality of the environmental coordinate does not imply equality of these
coefficients. A valid cross-domain comparison needs an explicit transfer
from the microscopic matter-metric perturbation to each measured observable,
including source screening, aperture or path weighting, and the observable's
calibration convention.

### B.2 No Numerical Pulsar Prior Is Applied

This analysis uses no pulsar likelihood, pulsar catalog, or numerical
pulsar-derived prior on $\Gamma_X$ or $\kappa_{\rm Cep}$. Consequently,
pulsar sample sizes, spin-down residuals, and effective response
coefficients are not reported as evidence for the Cepheid result. A future
cross-channel analysis would need to provide the underlying data product,
selection function, likelihood, and transfer equation before its coefficient
could constrain this model.

### B.3 No Solar-System Closure Follows From the Host Fit

The schematic action in Section 1 does not specify the functions
$A(\phi)$, $B(\phi)$, or $V(\phi)$, nor a solved field profile for the Sun
or Earth. The host coefficient therefore cannot be converted uniquely into
a PPN parameter, an equivalence-principle violation, or a Vainshtein
suppression factor. Cassini, MICROSCOPE, and laboratory-clock bounds are
essential constraints on any completed microphysical TEP model, but the
present phenomenological host fit does not by itself demonstrate compliance
with them.

### B.4 Permitted Cross-Corpus Claim

The defensible cross-corpus statement is structural: TEP papers may use the
same causal matter-metric convention and the same absolute sign rule
$A(\phi) < 1$, while measuring different channel responses. Numerical
agreement, universality of a fitted coefficient, or precision-gravity
closure must be demonstrated in a joint model and is not inferred here.

## Appendix C: Conformal Period Transport and the Distance-Ladder Bias

This appendix formalizes the restricted Cepheid-channel realization of the
TEP endpoint response. The derivation uses only the clock differential
between the Cepheid and the spectroscopic tracer within the same host. It
does not compare absolute host and calibrator clock rates, and it does not
identify the ladder-level coefficient $\kappa_{\rm Cep}$ with the
microscopic conformal offset $\Delta\ln A$.

### C.1 Observing-Chain Core--Disk Differential

Under TEP, the scalar field strictly slows proper time in every
gravitational environment: $0 < r < 1$, where

\begin{equation}
r_j\equiv
\frac{(d\tilde\tau/dt)_j}{(d\tilde\tau/dt)_{\rm cosmic}} .
\end{equation}

Let $r_{{\rm Cep},i}$ be the rate at the Cepheid location in host $i$, and
let $r_{{\rm spec},i}$ be the rate associated with the spectral tracer used
to infer that host's systemic redshift. In the proposed core--disk closure,
outer-disk Cepheids occupy a less-slowed active-shear environment while
nuclear or integrated systemic lines receive greater weight from a deeper,
more strongly slowed region. The same-host hierarchy is

\begin{equation}
0 < r_{{\rm spec},i} < r_{{\rm Cep},i} < r_{\rm cosmic} \equiv 1 .
\end{equation}

This ordering does not assert that an SN host clock runs faster than a
Milky Way, LMC, or NGC 4258 clock. No such absolute host--calibrator ordering
is needed. The observable is the differential sampled by two tracers in the
same host.

Let $P_{{\rm local},i}$ be the Cepheid period in local matter-frame proper
time and let $(1+z_{{\rm path},i})$ collect the common propagation redshift
between the host and observer. Endpoint clock bookkeeping gives

\begin{equation}
P_{{\rm obs},i}
=(1+z_{{\rm path},i})\frac{P_{{\rm local},i}}{r_{{\rm Cep},i}},
\qquad
1+z_{{\rm spec},i}
=\frac{1+z_{{\rm path},i}}{r_{{\rm spec},i}} .
\end{equation}

The period entered into the P--L fit after the standard spectroscopic
$(1+z)$ correction is therefore

\begin{equation}
P_{{\rm rest},i}^{\rm inf}
\equiv\frac{P_{{\rm obs},i}}{1+z_{{\rm spec},i}}
=P_{{\rm local},i}\,q_i,
\qquad
q_i\equiv\frac{r_{{\rm spec},i}}{r_{{\rm Cep},i}} .
\label{eq:pipeline_period_ratio}
\end{equation}

If both tracers sample the same TEP rate, $q_i=1$ and the response cancels.
In the proposed core--disk regime, $0 < q_i < 1$, so the inferred rest-frame
period is contracted even though both clocks remain slower than cosmic
time. The contraction is produced by the observing-chain division, not by
an absolute clock acceleration.

The P--L zero point is itself calibrated with an anchor ensemble. Let
$q_{\rm ref}$ denote the correspondingly weighted pipeline differential for
that ensemble. The environmental period residual relevant to applying the
calibrated P--L relation is

\begin{equation}
\delta\ln P_{{\rm rest},i}^{\rm inf}
=\ln\!\left(\frac{q_i}{q_{\rm ref}}\right) .
\label{eq:calibrated_period_residual}
\end{equation}

Thus the restricted Cepheid mechanism requires a larger core--disk
differential in the target host than in the calibration ensemble,
$q_i < q_{\rm ref}$. This is a testable tracer/aperture condition. It cannot
be inferred merely from the host's total velocity dispersion or from a
comparison of absolute host and calibrator clock rates.

### C.2 Period--Luminosity Inference Bias

The calibrated Cepheid Wesenheit relation is

\begin{equation}
M_W=a+b\log_{10}P,\qquad b\approx-3.26 .
\end{equation}

The observer inserts the pipeline-inferred rest-frame period
$P_{{\rm rest},i}^{\rm inf}$ into this relation. Relative to the anchor
calibration, the inferred-magnitude shift is

\begin{equation}
\delta M_{P,i}
=b\,\delta\log_{10}P_{{\rm rest},i}^{\rm inf}
=\frac{b}{\ln10}\ln\!\left(\frac{q_i}{q_{\rm ref}}\right) .
\label{eq:pipeline_magnitude_shift}
\end{equation}

For $b < 0$ and $\delta\ln P_{{\rm rest},i}^{\rm inf} < 0$, the shift
$\delta M_{P,i} > 0$: the Cepheid is inferred to be intrinsically dimmer
than it would be under the calibration-ensemble pipeline differential. Since $\mu_{\rm obs}=m-M_{\rm inf}$, the conditional
Cepheid-channel correction is

\begin{equation}
\mu_{{\rm corr},i}=\mu_{{\rm obs},i}+\delta M_{P,i} .
\end{equation}

The same equations also expose the falsification condition: if host-specific
spectroscopy shows $q_i\simeq q_{\rm ref}$, the proposed period-transport
term cancels and the combined endpoint response must enter another
observing-chain channel or be rejected.

### C.3 The Ladder-Level Response Coefficient ($\kappa_{\rm Cep}$)

The present data do not measure $q_i$ for each spectral aperture. The
restricted Cepheid closure therefore parameterizes its first-order
environmental dependence with

\begin{equation}
\ln\!\left(\frac{q_i}{q_{\rm ref}}\right)
=-\lambda_{\rm Cep}X_i,
\qquad
X_i=\frac{S(\mathcal E_i) u_{\phi,i}^2-U_{\rm ref}}{c^2},
\qquad \lambda_{\rm Cep}>0 .
\end{equation}

Combining this transfer with Equation~(\ref{eq:pipeline_magnitude_shift})
gives

\begin{equation}
\mu_{{\rm corr},i}=\mu_{{\rm obs},i}+\kappa_{\rm Cep}X_i,
\qquad
\kappa_{\rm Cep}\equiv-\frac{b}{\ln10}\lambda_{\rm Cep}>0 .
\end{equation}

Across the TEP corpus, $\kappa_{\rm canonical} \equiv 0.960 \times 10^6\ {\rm mag}$ serves as the standard reference benchmark scaling, corresponding to an effective host-to-anchor potential depth $\Delta(u_\phi^2)/c^2 \sim 3 \times 10^{-7}$ mapped through the Leavitt law slope $b \approx -3.30$ to produce a $\sim 0.3\ {\rm mag}$ characteristic modulus shift. In the empirical analyses (Steps 39, 42, 44), $\kappa_{\rm Cep}$ is fitted directly from observational data without imposing this benchmark amplitude.

$\kappa_{\rm Cep}$ is a ladder-level observable response. It absorbs the
core--disk transfer $\lambda_{\rm Cep}$, the P--L slope, aperture weighting,
and any additional measurement-chain response represented by this restricted
model. It is not the microscopic conformal coupling, a local clock-rate
ratio, or a comparison of absolute host and calibrator rates.

The fitted coefficient does not determine $\Delta\ln A$. Computing a
literal geometric $\Delta\Theta$ directly from a coefficient of order
$10^6$ mag would incorrectly predict large atomic spectral shifts. A
microscopic interpretation requires spatially resolved estimates of
$r_{\rm spec}$ and $r_{\rm Cep}$, together with a derived TEP transfer
function. Until then, the Cepheid realization remains a physically specified
but conditional allocation of the measured combined endpoint response.

## Appendix D: Reference-Gauge and Anchor Assumptions

### D.1 Physical Anchor Potential Contrast vs Environmental Screening

The environmental gradient between the calibration anchors and the SN Ia host galaxies is an intrinsic observational property of the galaxies rather than an artifact of group screening models. The primary distance ladder calibrators are dwarf or late-type galaxies with low circular rotation velocities: the LMC ($V_{\rm rot} \approx 65\text{--}72\ {\rm km\,s^{-1}} \implies u_\phi \approx 46\text{--}51\ {\rm km\,s^{-1}}$), the SMC ($V_{\rm rot} \approx 30\text{--}40\ {\rm km\,s^{-1}} \implies u_\phi \approx 21\text{--}28\ {\rm km\,s^{-1}}$), the Milky Way solar neighbourhood, and NGC 4258. Together, these anchors possess shallow potential depths ($\langle u_\phi^2 \rangle \sim 3.0 \times 10^3\ {\rm km^2\,s^{-2}}$).

In contrast, the 37 R22 SN Ia host galaxies sample massive luminous spirals with median circular rotation velocities $V_{\rm rot} \approx 160\text{--}250\ {\rm km\,s^{-1}}$ ($\langle u_\phi^2 \rangle \sim 2.5 \times 10^4\ {\rm km^2\,s^{-2}}$). Even in the completely unscreened coordinate ($S=1$), the potential contrast between anchors and hosts exceeds a factor of 8. Furthermore, the direct empirical detection of internal radial Period--Luminosity gradients within the LMC ($+0.0284 \pm 0.0086\ {\rm mag}$, $3.30\sigma$) and M31 ($+0.6304 \pm 0.1948\ {\rm mag}$, $3.24\sigma$) provides independent internal evidence that radial Period--Luminosity gradients are active within Local Group members with the sign predicted by TEP.

### D.2 Adopted Anchor Endpoint Construction

The analysis represents the Cepheid calibration ensemble with approximate weights 0.20, 0.25, and 0.55 for the Milky Way, LMC, and NGC 4258, using local stellar velocity dispersions at the Cepheid disk locations: Milky Way solar neighbourhood ($\sigma_z = 30.0\ {\rm km\,s^{-1}}$; Bovy et al. 2012), LMC stellar disk ($\sigma_{\rm disk} = 24.0\ {\rm km\,s^{-1}}$; van der Marel et al. 2002), and NGC 4258 intermediate annulus ($\sigma_{\rm local} = 115.0\ {\rm km\,s^{-1}}$; Kormendy & Ho 2013), yielding the composite reference scale $\sqrt{U_{\rm ref}} = 87.165\ {\rm km\,s^{-1}}$. These weights summarize the adopted endpoint construction; they are not recovered from a dedicated anchor-weight likelihood in this paper. Group screening is computed by the same continuous richness formula used for hosts ($S_{\rm group} = [1 + (N_{\rm mb}/10)^{1.2}]^{-1}$). No categorical anchor-screening model is selected or fitted as an alternative inference.

### D.3 Exact Reference-Gauge Test

The unscreened notation $\sqrt{U_{\rm ref}}=87.165\ {\rm km\,s^{-1}}$ and the screened notation $\sqrt{U_{\rm ref}}=30.507\ {\rm km\,s^{-1}}$ differ by a constant origin in the endpoint coordinate. A correct matrix refit absorbs that constant into the Cepheid zero point $M_H^W$. Step 45 verifies the resulting identity:

| Fixed $\kappa$ (mag) | $H_0$ at 87.165 km/s | $H_0$ at 30.507 km/s | $\chi^2$ |
| --- | --- | --- | --- |
| 0 | $73.0434$ | $73.0434$ | $3552.7063$ |
| $0.365\times10^6$ | $71.7711$ | $71.7711$ | $3558.7047$ |
| $0.960\times10^6$ | $69.7423$ | $69.7423$ | $3581.8608$ |

This is an algebraic invariance check, not evidence that either numerical origin is a measured physical anchor dispersion. Host-mean shortcuts that change when the reference origin changes are excluded from inference because they do not refit the ladder zero point.

### D.4 Sensitivity and Future Hierarchical Extensions

Gauge invariance does not validate the adopted anchor weights, dispersions, or screening factors. Those inputs require a future hierarchical treatment using anchor-level Cepheid data and independently specified environmental measurements. The present result establishes that the implemented full-matrix projection is internally invariant to an additive reference shift, while the underlying physical contrast between anchors and hosts remains driven by galaxy kinematics.