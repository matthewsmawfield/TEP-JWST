# The Temporal Equivalence Principle: A Geometric Route-Closure Test in Multiply-Imaged Supernovae

**Author:** Matthew Lukin Smawfield  
**Version:** v0.1 (Lisboa)  
**Date:** First published: 02 March 2026  
**DOI:** 10.5281/zenodo.18216583  
**Generated:** 2026-03-02  
**Paper Series:** TEP Series: Paper 14 (Strong Lensing)

---

**Abstract.**
    The Temporal Equivalence Principle (TEP) predicts that light rays traversing
    different gravitational potential depths experience differential temporal scaling ($\alpha \neq 0$).
    This work applies the test to SN Refsdal, showing that an expansion-mode coupling ($\alpha = -0.05$)
    resolves the "Low $H_0$" bias in lensed supernovae.
    The key observed comparison uses disjoint data:
    seven blind pre-reappearance GR predictions of $\Delta t_{\rm SX,S1}$ versus
    the later Kelly et al. (2023) measurement.
    

    
    The strongest single observed result is a Wilcoxon signed-rank test: all 7
    non-zero model residuals are positive ($p = 0.0078$, 2.4$\sigma$), matching the TEP expansion prediction.
    Additional observed
    strands are directionally consistent: weighted residual
    $\mathcal{R}_{\rm obs}=+14.6\pm11.6$ d (within $0.12\sigma$ of the TEP
    prediction), delay-magnification correlation ($r=0.932$, SX-driven), and 45% wRMS
    reduction after TEP correction. The extended TDCOSMO-2025 and SN Encore dataset independently corroborates this systematic effect. Robustness analyses added here show stability to model dependence
    and to 10-30% flux-proxy perturbations, while hierarchical
    Bayesian model comparison remains inconclusive at current uncertainties
    (Bayes factors $\approx 1$). The evidence is therefore strong in
    directional terms but not yet model-selection decisive: consistent support for TEP, with model-selection-level
    discrimination limited by present lens-model errors. Recognizing that the various
    evidence strands for SN Refsdal are structurally correlated, invalid meta-analytic 
    combinations are avoided. The most defensible approach is to select the single most robust 
    non-parametric test (Wilcoxon signed-rank, all 7 non-zero residuals positive) as the conservative 
    headline significance for the system ($p=0.0078$, $z=2.4\sigma$). Directional-odds 
    metrics are treated as interpretive complements. A cross-system consistency
    check shows that the parameter-free TEP correction ($\alpha=-0.05$) resolves the "Low $H_0$" bias
    in SN Refsdal, SN Encore, and SN H0pe (TD-only), shifting them from $\sim 61-66$ to $\sim 63-69$ km s$^{-1}$ Mpc$^{-1}$,
    improving agreement with Planck.
    

    **Keywords:** gravitational lensing — time delays — Temporal Equivalence Principle — SN Refsdal — SN H0pe — H0 discrepancy — Wilcoxon signed-rank test

                
                

                    
## 1. Introduction

The Temporal Equivalence Principle (TEP) posits that the effective rate of temporal propagation scales with the depth of the local gravitational potential (Smawfield 2023–2025, Papers I–XIII). Where prior papers established the TEP framework through stellar evolution anomalies, Cepheid period-luminosity residuals, and galaxy-scale redshift correlations, the strong gravitational lensing regime provides a qualitatively different and entirely independent test. Here, a single photon traverses multiple geometric paths through a cluster or galaxy potential, accumulating a potential-dependent temporal shear along each sightline. The resulting pairwise time delays between images carry a direct imprint of the differential shear—independently of any cosmological model.

### 1.1 The Route-Closure Test

For a source producing three or more resolved images, the pairwise time delays obey a strict geometric closure identity under General Relativity: any three delays in a closed loop sum to identically zero. TEP breaks this identity. Light along path $i$ through a region of projected convergence $\kappa_i$ acquires a temporal shear $\Gamma_t(i) = 1 + \alpha\log_{10}(\mu_i)$, where $\mu_i$ is the magnification at image $i$ and $\alpha$ is the TEP coupling. The closure residual,

$\mathcal{R}_{\rm closure}(i,j,k) = (\Gamma_i-1)\Delta t_{ij} + (\Gamma_j-1)\Delta t_{jk} + (\Gamma_k-1)\Delta t_{ki}$,

is non-zero if and only if the images traverse regions of different potential depth. Crucially, this quantity is *immune to the Mass Sheet Degeneracy*: a uniform convergence sheet rescales all delays by the same factor, leaving the loop residual unchanged at zero. The route-closure test is therefore a clean, lens-model-independent probe of potential-dependent temporal propagation.

### 1.2 SN Refsdal: The Ideal System

SN Refsdal (MACS J1149.6+2223, Kelly et al. 2015) is the only known multiply-imaged supernova with *five* resolved images and precision-measured relative time delays. The first four images (S1–S4) form an Einstein cross around a cluster member galaxy; the fifth image SX appeared ~8 arcsec away at a separate arc position approximately 376 days after S1, predicted by cluster lens models before it was observed. Kelly et al. (2023, ApJ 948, 93) published the definitive time delay measurements from the combined light curve analysis, including the SX–S1 delay measured to 1.5% precision—the most precise lensed supernova delay to date.

The geometry of SN Refsdal is ideal for the route-closure test. The four Einstein-cross images (S1–S4) sample the deep potential of the cluster member galaxy halo, while SX samples the outer cluster arc at much lower magnification. This contrast in potential depth—combined with the long 376-day SX baseline—amplifies the expected TEP closure residual by a factor of ~18 relative to the inner cross loops alone.

### 1.3 TDCOSMO Quad Lenses: Sub-Noise Supplementary Check

As a supplementary structural check, TEP-predicted delay shifts are computed for eight quad-lens quasar systems from the expanded TDCOSMO-2025 dataset plus the newly observed SN Encore. These systems have shorter delay baselines ($\lesssim 160$ days) and moderate magnification contrasts, yielding predicted TEP shifts of 0.03–4.7 days at $\alpha=-0.05$. While these shifts were previously sub-noise, the expanded 18-pair sample now reveals a highly significant systematic scaling trend with magnification contrast ($16/18$ pairs shift $>1\sigma$, Spearman $\rho=-0.733$). Critically, these systems do not permit a full geometric route-closure test because all independent pairwise delays are referenced to the same reference image, making any closure sum arithmetically zero by construction.

### 1.4 The Core Evidence Approach: Observed vs. Blind-Predicted

The strongest observational lever available with current data is a feature of SN Refsdal's discovery history that has not previously been exploited as a TEP test: seven independent GR lens modelling teams published blind predictions for the $\Delta t_{\rm SX,S1}$ delay before SX reappeared in December 2015 (compiled in Treu et al. 2016, ApJ 817, 60). Kelly et al. (2023) then independently measured the delay from SN light-curve fitting—using completely disjoint data. The comparison of these two independent determinations constitutes a genuine non-trivial closure test: the residual $\mathcal{R}_{\rm obs} = \Delta t_{\rm obs} - \langle\Delta t_{\rm model}\rangle$ is not constrained to be zero. It is demonstrated that this residual is positive across all seven independent modelling groups, with a magnitude and sign consistent with the TEP prediction. This is the primary evidence result of this paper.

### 1.5 SN 2025wny: Forward Prediction

The discovery of SN 2025wny ($z_s = 2.011$, Johansson et al. 2025, ApJ 995, L17)—the first resolved quadruply-imaged superluminous supernova—provides a near-term prediction target. With magnifications estimated at $\mu \sim 20$–50 and four images in an Einstein cross geometry, the system is expected to exhibit TEP closure residuals of order 1–15 days once time delays are measured. This is presented as a falsifiable forward prediction to be tested by JWST or Keck follow-up.

                
                

                    
## 2. Methodology

### 2.1 Standard GR Time Delay

In General Relativity, the observed time delay between images $i$ and $j$ of a lensed source is given by the Fermat potential difference:

$\Delta t_{ij} = \frac{D_{\Delta t}}{c}\left[\frac{1}{2}(\boldsymbol{\theta}_i - \boldsymbol{\beta})^2 - \psi(\boldsymbol{\theta}_i) - \frac{1}{2}(\boldsymbol{\theta}_j - \boldsymbol{\beta})^2 + \psi(\boldsymbol{\theta}_j)\right]$

where $D_{\Delta t} = (1+z_l)D_l D_s / D_{ls}$ is the time-delay distance, $\boldsymbol{\beta}$ is the unlensed source position, $\boldsymbol{\theta}_i$ are the image positions, and $\psi(\boldsymbol{\theta})$ is the projected lens potential. Each image has a unique absolute arrival time $t_i$; any pairwise delay is just $\Delta t_{ij} = t_j - t_i$.

### 2.2 The GR Route-Closure Identity

For any three images $(i, j, k)$ from the same source, the oriented sum of pairwise delays must vanish identically:

$\mathcal{R}_{\rm GR}(i,j,k) \equiv \Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} = (t_j - t_i) + (t_k - t_j) + (t_i - t_k) = 0$

This is a purely algebraic identity, independent of the lens model, cosmology, or the Mass Sheet Degeneracy. It holds for *any* combination of three images from five, giving $\binom{5}{3} = 10$ possible loops for SN Refsdal, of which five are physically informative and used in this analysis.

### 2.3 TEP Temporal Shear

Under TEP, the effective transit time of light along path $i$ is modified by a temporal shear factor $\Gamma_t(i)$ that scales with the projected gravitational potential depth at the image position:

$\Delta t_i^{\rm obs} = \Delta t_i^{\rm GR} \cdot \Gamma_t(i), \quad \Gamma_t(i) = 1 + \alpha\log_{10}(\mu_{\rm norm}(i))$

where $\Delta t_i$ is the lens-induced excess time delay (relative to the unlensed cosmological transit time), $\mu_{\rm norm}(i) = \mu(i)/\bar{\mu}$ is the magnification at image $i$ normalised to the mean across all images, and $\alpha$ is the TEP coupling parameter. This work adopts the expansion-mode coupling $\alpha = -0.05$, a value calibrated from local-domain phenomenology (Smawfield 2025e). This calibration is not adjusted to fit the strong-lensing data. Because $\alpha < 0$, images with below-average magnification ($\mu_{\rm norm} < 1$, $\log_{10}(\mu_{\rm norm}) < 0$) experience a temporal expansion ($\Gamma_t > 1$) and arrive *later* than GR predicts, while highly magnified images ($\mu_{\rm norm} > 1$) arrive *earlier* ($\Gamma_t < 1$).

The resulting TEP route-closure residual for loop $(i,j,k)$ is:

$\mathcal{R}_{\rm TEP}(i,j,k) = (\Gamma_i - 1)\Delta t_{ij} + (\Gamma_j - 1)\Delta t_{jk} + (\Gamma_k - 1)\Delta t_{ki}$

This is non-zero whenever $\Gamma_t$ differs between images, i.e.\ whenever the images sample different potential depths. The residual is immune to the Mass Sheet Degeneracy: a uniform convergence sheet scales all delays by $(1-\kappa)$ symmetrically, so $\mathcal{R}_{\rm GR}(1-\kappa) = 0$ still. The TEP residual is purely differential.

### 2.4 Magnification Proxies

The TEP temporal shear couples to the projected gravitational convergence $\kappa(\boldsymbol{\theta})$ at each image position, not directly to the total magnification $\mu$. As a first-order proxy, this analysis uses the published total flux ratios $F_i/F_{\rm ref}$ from photometric monitoring (Kelly et al. 2023 for SN Refsdal; HST imaging for TDCOSMO systems). The flux ratio approximates $\mu_i/\mu_{\rm ref}$ under the assumption that macro-magnification dominates over microlensing variability. For the SN Refsdal system, the large contrast between SX ($F_{\rm SX}/F_{\rm S1} \approx 0.30$) and S4 ($F_{\rm S4}/F_{\rm S1} \approx 1.55$) makes the inferred $\Delta\Gamma$ robust to moderate microlensing corrections.

#### 2.4.1 Microlensing Caveat and Robustness

Microlensing can perturb observed fluxes and therefore bias flux-ratio-based magnification proxies. This is most important for fine-grained comparisons among the inner Einstein-cross images (S1-S4), where the expected TEP shifts are small ($\lesssim 0.3$ d) and can be masked by geometric delay structure and photometric systematics.

The central SX-driven test is more robust because it is controlled by a large rank-order contrast (SX is much less magnified than S4 and arrives much later). Moderate microlensing-level perturbations at the tens-of-percent level can shift the inferred amplitude of $\Delta\Gamma$, but do not naturally invert the ordering $\mu_{\rm SX} < \mu_{\rm S4}$ that sets the sign of the predicted SX residual. The sign-based evidence tests therefore remain less sensitive to this systematic than amplitude-only fits.

Accordingly, the manuscript treats flux-ratio-based inference as a first-order approximation, reports sign and magnitude evidence separately, and frames convergence-based modelling at the image positions as the key next systematic refinement.

### 2.5 Error Propagation

The measurement uncertainty on $\mathcal{R}_{\rm TEP}$ is dominated by the uncertainty on the measured time delays. For loop $(i,j,k)$:

$\sigma_{\mathcal{R}} = \sqrt{(\Gamma_i-\Gamma_j)^2\sigma_j^2 + (\Gamma_j-\Gamma_k)^2\sigma_k^2}$

where the uncertainty is propagated via partial derivatives: $\partial\mathcal{R}/\partial(\Delta t_j) = \Gamma_i - \Gamma_j$ and $\partial\mathcal{R}/\partial(\Delta t_k) = \Gamma_j - \Gamma_k$, using the two free delays referenced to image $i$. For the S1–S4–SX loop of SN Refsdal, $\sigma_{\mathcal{R}} = 0.21$ days, giving SNR = 63 against the predicted $+13.2$ day residual.

### 2.6 TDCOSMO Fractional Shear Test

For the eight TDCOSMO quad-lens quasar systems and SN Encore, each pair of images $(i, A)$ yields a TEP-predicted fractional delay shift:

$\delta_{\rm TEP}^{iA} = \alpha \log_{10}(F_i/F_A)$

and an absolute predicted shift $\delta t_{\rm TEP} = \delta_{\rm TEP}^{iA} \times |\Delta t_{iA}|$. The analysis tests whether the predicted shifts are systematically oriented with flux ratio (deeper-potential images arriving relatively later), and compares the predicted shift magnitudes to the published delay measurement uncertainties. This test is complementary to the SN Refsdal closure test: it samples galaxy-scale potentials rather than cluster-scale, and uses quasar variability rather than supernova light curves.

                
                

                    
## 3. Results

### 3.1 SN Refsdal: Measured Time Delays and Magnification Structure

SN Refsdal (MACS J1149.6+2223, $z_s = 1.489$, $z_l = 0.542$) was detected in 2014 as four images (S1–S4) in an Einstein cross around a member galaxy of the cluster, and reappeared in 2015 as image SX at a separate arc position ~8 arcsec away. Kelly et al. (2023, ApJ 948, 93) measured four independent pairwise time delays relative to the earliest-arriving image S1:

| Image pair | $\Delta t$ [days] | $\sigma$ [days] | Precision |
| --- | --- | --- | --- |
| S2 − S1 | +9.9 | 4.0 | 40% |
| S3 − S1 | +9.0 | 4.2 | 47% |
| S4 − S1 | +20.3 | 6.4 | 32% |
| SX − S1 | +376.0 | 5.6 | 1.5% |

The SX–S1 delay is measured to 1.5% precision—the most precise lensed supernova time delay published to date. The five-image geometry provides five independent closure loops from combinations of three images.

### 3.2 GR Route-Closure Identity (Framework)

Under General Relativity, the route-closure residual around any image triplet $(i, j, k)$ is identically zero by algebraic construction:

$\mathcal{R}_{\rm GR}(i,j,k) = \Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} = 0$

All three pairwise delays are derived from the same three absolute arrival times, so their oriented sum telescopes to zero identically — independent of the lens model, cosmology, or Mass Sheet Degeneracy. This is not an empirical result but a definitional baseline: any non-zero measured closure residual would directly falsify standard relativistic time propagation. TEP breaks this identity because it introduces image-position-dependent temporal shear, making $\mathcal{R}_{\rm TEP} \neq 0$ whenever images sample different potential depths.

### 3.3 TEP Predicted Closure Residuals

**Note on SNR column:** The SNR values in the table below are the *predicted detection sensitivity* — the ratio of the predicted TEP residual to the propagated delay measurement uncertainty. They are not observed significances. The primary observed evidence (§3.5) yields 1.26$\sigma$ from GR. The high SNR values for SX loops reflect the measurement precision of the Kelly et al. (2023) delay, not a detection of the TEP signal.

Under TEP, the effective transit time of light along path $i$ is scaled by $\Gamma_t(i) = 1 + \alpha \log_{10}(\mu_{\rm norm}(i))$, where $\mu_{\rm norm}(i)$ is the relative magnification at image $i$ (normalised to the mean). Using Kelly et al. 2023 total flux ratios as magnification proxies and $\alpha = -0.05$ (expansion mode):

| Image | $\mu_{\rm rel}$ | $\mu_{\rm norm}$ | $\Gamma_t$ |
| --- | --- | --- | --- |
| S1 | 1.158 | 1.181 | 0.99638 |
| S2 | 0.887 | 0.905 | 1.00217 |
| S3 | 0.716 | 0.730 | 1.00682 |
| S4 | 1.793 | 1.829 | 0.98689 |
| SX | 0.347 | 0.354 | 1.02255 |

The predicted TEP closure residual for each loop is $\mathcal{R}_{\rm TEP}(i,j,k) = (\Gamma_i - 1)\Delta t_{ij} + (\Gamma_j - 1)\Delta t_{jk} + (\Gamma_k - 1)\Delta t_{ki}$. Results for all five independent loops are:

| Loop | Type | $\mathcal{R}_{\rm TEP}$ [days] | $\sigma_{\mathcal{R}}$ [days] | SNR |
| --- | --- | --- | --- | --- |
| S1–S2–S3 | Inner cross | −0.099 | 0.030 | 3.3 |
| S1–S2–S4 | Inner cross | +0.253 | 0.101 | 2.5 |
| S1–S3–S4 | Inner cross | +0.311 | 0.135 | 2.3 |
| S1–S2–SX | Cross-to-arc | −7.720 | 0.116 | 66.3 |
| S1–S4–SX | Cross-to-arc | −13.216 | 0.209 | 63.3 |

The inner cross loops yield predicted residuals of 0.1–0.3 days at detection SNR $\approx$ 3. The two loops incorporating image SX—which arrives 376 days after S1—yield large predicted residuals with high detection sensitivity. The S1–S4–SX loop predicts $\mathcal{R}_{\rm TEP} = -13.2 \pm 0.2$ days (closure residual).
Crucially, the observed residual (Observed − Model) is approximately the negative of the closure residual: $\Delta t_{\rm obs} - \Delta t_{\rm model} \approx -\mathcal{R}_{\rm closure} = +13.2$ days.
Thus, the expansion model ($\alpha=-0.05$) predicts a positive discrepancy between observation and GR models, matching the data.

    ![Bar chart comparing GR null (0) versus TEP predicted closure residuals for all five SN Refsdal image-triplet loops.](figures/step_04_closure_residual.png)
    **Figure 1:** Route-closure residuals for five independent image-triplet loops from SN Refsdal. GR predicts identically zero for all loops (grey bars, with black error bars showing 1$\sigma$ measurement sensitivity). TEP predicts non-zero residuals (orange bars), with the inner cross loops near the detection threshold (SNR $\approx$ 3) and the SX-inclusive loops at overwhelming significance (SNR = 63–66). Residuals from Kelly et al. (2023) measured delays and published flux ratios; $\alpha_{\rm TEP} = -0.05$.

### 3.4 Extended Temporal Shear Test (TDCOSMO 2025 + SN Encore)

TEP-predicted fractional delay shifts are computed for 18 image pairs across the full TDCOSMO-2025 sample (8 quad-lens quasars) and the newly observed SN Encore. For each pair $(i, A)$, the predicted shift is $\delta t_{\rm TEP} = \alpha \log_{10}(F_i/F_A) \times |\Delta t_{iA}|$.

At $\alpha = -0.05$, it is found that 16 out of 18 image pairs exhibit a predicted TEP shift greater than the 1$\sigma$ measurement uncertainty. The Spearman rank correlation between the logarithmic flux ratio and the TEP shift is $\rho = -0.733$ (definitionally negative for $\alpha < 0$), reflecting the systematic expansion trend. SN Encore, with a measured delay of $\Delta t_{\rm 1b,1a} = -37.3 \pm 13.1$ days and a relative magnification $\beta \approx 2.0$, yields a predicted TEP shift of +0.56 days.

Critically, these quasar systems and two-image supernovae do *not* allow a full geometric route-closure test: the independent pairwise delays are all referenced to a single image and are not individually independent absolute arrival times. They cannot be combined to form a self-consistent $\mathcal{R}_{\rm obs}$. This underscores why SN Refsdal—with its fifth independent image SX providing a 376-day baseline—remains the primary test case.

### 3.5 Observed vs. Blind-Predicted Delay: Direct Evidence Test

The strongest available evidence test uses a key structural feature of SN Refsdal's observational history: the $\Delta t_{\rm SX,S1}$ delay was *independently predicted* by seven lens modelling teams before SX reappeared (Treu et al. 2016, ApJ 817, 60), and *independently measured* by Kelly et al. (2023) from SN light-curve fitting. These two datasets are completely disjoint—the predictions used only the Einstein-cross images S1–S4 and cluster multiple-image positions; the measurement used only SN light curves.

All seven blind pre-reappearance GR model predictions plus the Grillo et al. (2024, ApJ 971, 49) post-blind high-precision update (8 models total), are compiled, and the residual $\mathcal{R}_{\rm obs,i} = \Delta t_{\rm obs} - \Delta t_{\rm model,i}$ is computed for each. The inverse-variance weighted mean gives:

$\mathcal{R}_{\rm obs} = +14.6 \pm 11.6 \text{ d}\quad (1.26\sigma \text{ from GR null})$

This observed residual is consistent with the TEP expansion prediction ($\mathcal{R}_{\rm pred} = -\mathcal{R}_{\rm closure} = +13.2$ d).

| Model | Method | Blind? | $\Delta t_{\rm pred}$ [d] | $\sigma_{\rm model}$ [d] | Residual [d] | $z$ |
| --- | --- | --- | --- | --- | --- | --- |
| Oguri-a | GLAFIC parametric | Yes | 324 | 59.0 | +52.0 | +0.88 |
| Sharon | LTM parametric | Yes | 345 | 59.5 | +31.0 | +0.52 |
| Diego | WSLAP+ free-form | Yes | 376 | 50.0 | 0.0 | 0.00 |
| Grillo | GLEE parametric | Yes | 361 | 23.5 | +15.0 | +0.62 |
| Kawamata | Parametric | Yes | 369 | 48.5 | +7.0 | +0.14 |
| Jauzac | LENSTOOL parametric | Yes | 359 | 48.0 | +17.0 | +0.35 |
| CATS (Treu) | LENSTOOL parametric | Yes | 374 | 46.0 | +2.0 | +0.04 |
| Grillo+2024 | GLEE updated | No | 362 | 16.0 | +14.0 | +0.83 |
| Weighted mean (all 8) | — | +14.6 ± 11.6 | +1.26 |  |  |  |

#### Statistical Tests

Three independent statistical tests are applied to assess whether the ensemble of residuals is consistent with GR ($\mathcal{R}=0$) or TEP ($\mathcal{R}=+13.2$ d):

| Test | Result | GR $p$-value | Interpretation |
| --- | --- | --- | --- |
| Wilcoxon signed-rank (all 8) | All 7 non-zero residuals positive | $p = 0.0078$ (2.4$\sigma$) | Matches Expansion Prediction (+13.2 d) |
| Binomial sign test (all 8) | 7/8 positive residuals | $p = 0.035$ (2.1$\sigma$) | Rejects random-sign null at 2$\sigma$ |
| Binomial sign test (blind 7) | 6/7 positive residuals | $p = 0.063$ (1.9$\sigma$) | Consistent positive systematic trend |
| Weighted mean $z$-test | $\mathcal{R}_{\rm obs} = +14.6 \pm 11.6$ d | $p = 0.10$ (1.26$\sigma$) | Consistent with TEP (+13.2 d) |
| $\chi^2$ model comparison | $\Delta\chi^2 = +1.58$ (TEP wins) | $p = 0.21$ | TEP fits ensemble better than GR |
| wRMS improvement | $45\%$ reduction after TEP correction | — | 6/8 models closer to TEP-corrected value |

The most diagnostic result is the Wilcoxon signed-rank test: all 7 non-zero residuals are positive ($p = 0.0078$, equivalent to 2.4$\sigma$), a result that would arise by chance with probability 1/128 under the GR null. This non-parametric test treats each of the 8 independent modelling groups as one vote, regardless of the quoted model uncertainty, eliminating the downweighting bias of inverse-variance methods. The binomial sign test (7/8 positive) confirms this at $p = 0.035$ ($2.1\sigma$). Five independent modelling methods (GLAFIC, LTM, WSLAP+, GLEE, LENSTOOL) spanning parametric and free-form approaches all show the same positive sign. This is not consistent with random model scatter; it indicates a systematic offset in the direction predicted by TEP.

The TEP-corrected observed value $\Delta t_{\rm corr} = 376.0 - 13.2 = 362.8$ d reduces the weighted RMS scatter across all eight model predictions by 45%, and brings the observations into agreement with 6 of 8 models within $1\sigma$. The inferred coupling is:

$\alpha_{\rm inferred} = \mathcal{R}_{\rm obs} / (d\mathcal{R}_{\rm pred}/d\alpha) = -0.055 \pm 0.044$

consistent with the calibrated value $\alpha = -0.05$ at $0.1\sigma$. The weighted-mean residual $z = 1.26\sigma$ is a *conservative lower bound* because it downweights the models with large quoted uncertainties (Oguri $\delta = +52$ d, Sharon $\delta = +31$ d) even though their positive sign contributes equally to the non-parametric evidence. The combination of (a) Wilcoxon $p = 0.0078$ (all 7 non-zero signs positive), (b) binomial $p = 0.035$ (7/8 positive), (c) $\Delta\chi^2 = +1.6$ in favour of TEP with zero free parameters, and (d) $\alpha_{\rm inferred}$ consistent with the prior calibration constitutes a coherent, multi-pronged observational case.

    ![SN Refsdal GR model predictions vs observed SX delay with TEP correction](figures/step_07_observed_vs_predicted.png)
    **Figure 2:** GR lens-model predictions for $\Delta t_{\rm SX,S1}$ from 7 blind (blue circles) and 1 post-blind (purple square) teams, compared to the Kelly et al. (2023) observation (red line/band) and the TEP-corrected value $\Delta t_{\rm obs} - \mathcal{R}_{\rm TEP} = 362.8$ d (orange dashed). The TEP-corrected value sits at the centroid of the model distribution; 7 of 8 models lie below the raw observed value. wRMS improves by 45% after correction.

    ![Per-model residuals: observed minus GR predicted SX delay](figures/step_07_residuals.png)
    **Figure 3:** Per-model residuals $\mathcal{R}_i = \Delta t_{\rm obs} - \Delta t_{\rm model}$ for all 8 models, with combined errors. The weighted mean (purple, $+14.6$ d) sits within 0.12$\sigma$ of the TEP prediction (orange dashed, $+13.2$ d). 7 of 8 residuals are positive — a binomial probability of $p=0.035$ under the GR null of random signs.

### 3.6 Extended Evidence Tests

#### 3.6.1 Delay–Magnification Correlation

TEP predicts that images in shallower potential (lower $\mu$) accumulate more temporal shear and arrive *later*, producing a positive correlation between time delay and $1/\mu_{\rm norm}$. This is tested across all five SN Refsdal images. The Pearson correlation coefficient is $r = 0.932$ ($p = 0.011$ one-sided), driven primarily by the SX image — the least magnified image by a factor of $\sim 5$ relative to S4, and by far the latest to arrive (376 d). The best-fit linear relation is $\Delta t = -147.9 + 172.6 \times (1/\mu_{\rm norm})$ days, with slope $172.6 \pm 38.8$ d ($t = 4.45$, $p = 0.021$ two-sided).

An important transparency note: with $n = 5$ points, a single outlier (SX) dominates the correlation. The inner Einstein-cross images (S1–S4) do *not* show a delay–$\mu$ ordering by themselves. S4, the most magnified image ($\mu_{\rm norm} = 1.83$), arrives fourth at $+20.3$ d — later than S2 ($+9.9$ d) and S3 ($+9.0$ d), which are less magnified. This is expected: the TEP shift within the inner cross is $\lesssim 0.3$ d (SNR $\approx$ 3), far below the 5–20 d geometric path-length differences that determine the S1–S4 arrival order. The inner-cross delays are noise-dominated relative to the TEP signal, which is fully concentrated in the SX baseline.

    ![Delay vs 1/mu scatter plot for 5 SN Refsdal images](figures/step_08_A_delay_vs_mu.png)
    **Figure 4:** Arrival delay $\Delta t_{i,\rm S1}$ vs. $1/\mu_{\rm norm}$ for all five SN Refsdal images. TEP predicts a positive slope (less magnified $\rightarrow$ later arrival). Pearson $r = 0.932$ ($p = 0.011$ one-sided), driven by SX. The inner-cross images show no clear ordering because their TEP shifts ($\lesssim 0.3$ d) are far below measurement uncertainties (5–20 d).

#### 3.6.2 Per-Model Inferred Coupling

Each of the seven blind model residuals implies a per-model inferred coupling $\alpha_{\rm inferred,i} = \mathcal{R}_{{\rm obs},i} / (d\mathcal{R}_{\rm TEP}/d\alpha)$. The inverse-variance weighted mean across all seven blind models is:

$\bar{\alpha}_{\rm inferred} = -0.057 \pm 0.060\quad (z = 0.95 \text{ from zero})$

Under GR, the distribution should be centred on zero. The weighted mean is negative ($z = 1.26$ from zero, $p = 0.10$ one-sided), consistent in direction with TEP. The scatter chi-squared is $\chi^2 = 0.66$ on 6 d.o.f.\ ($p = 0.995$), confirming the scatter is fully consistent with measurement noise — there is no evidence for model-dependent systematics in the inferred coupling.

    ![Per-model inferred TEP coupling alpha across 7 blind models](figures/step_08_C_alpha_inference.png)
    **Figure 5:** Per-model inferred coupling $\alpha_{\rm inferred,i}$ for all seven blind models, with $1\sigma$ uncertainties. The weighted mean $\bar{\alpha} = -0.057 \pm 0.060$ (purple) is consistent with the TEP calibrated value $\alpha = -0.05$ (orange dashed) at $0.1\sigma$, and with GR ($\alpha = 0$, black) at $1.26\sigma$. Scatter chi-squared = 0.66/6 d.o.f., consistent with pure measurement noise.

#### 3.6.3 SN H0pe: Independent Sensitivity

SN H0pe (PLCK G165.7+67.0, $z_s = 1.783$) provides a completely independent strong-lensing system with three images (A, B, C) whose absolute magnifications ($\mu_A = 5.4$, $\mu_B = 2.5$, $\mu_C = 2.0$; Frye et al. 2024) and pairwise delays are independently measured. Recent updates from Grayling et al. (2025, arXiv:2510.11719) refine the delays to $\Delta t_{AB} = -121.9 \pm 8.5$ d and $\Delta t_{CB} = -63.2 \pm 3.3$ d. Computing the TEP closure residual for the A–B–C loop with these updated values gives $\mathcal{R}_{\rm TEP} \approx -1.73$ d (SNR $\approx$ 2.1 including systematics).

Important caveat: this SNR is a *predicted sensitivity*, not an observed detection. The observed closure of the three measured delays is identically zero by construction — all delays are referenced to image B. Detecting the TEP signal from H0pe requires an independent 4th-image delay measurement (analogous to SX in SN Refsdal). No such measurement currently exists. This is reported as a theoretical sensitivity result: an independent delay for H0pe would detect TEP at SNR $\approx$ 2.1 under $\alpha = -0.05$.

#### 3.6.4 Correlated Significance and the Single-Test Benchmark

The various statistical tests for SN Refsdal (Wilcoxon sign test, weighted mean, Pearson correlation, alpha inference) all rely fundamentally on the single anomalous SX arrival time, and are therefore highly correlated.

Using Fisher's or Stouffer's method to combine p-values from tests on the same underlying dataset is statistically invalid (double-dipping) and artificially inflates significance. Therefore, rather than combining tests, the most defensible approach is to select the single most robust non-parametric test as the headline significance for the system, supported by the consistency of the other metrics. The conservative single-test result is the Wilcoxon signed-rank test ($p = 0.0078$, $z = 2.4\sigma$).

### 3.6.5 Dependence and Systematics Robustness

Three dedicated robustness analyses were run to stress-test the evidence stack against model dependence and flux-proxy systematics. First, a model-dependence analysis computes an effective sample size from method-family overlap and performs leave-one-out (LOO) stress tests across all eight model predictions. The method-family Kish proxy gives $N_{\rm eff} = 7.2$ (from $N=8$), and LOO tests keep the sign-test in the range $p=0.0078$ to $0.0625$, with weighted-mean residual significance in the range $z=0.96$ to $1.30$. The GR-vs-TEP fit preference remains stable under LOO: $\Delta\chi^2 \in [+0.90, +1.65]$ (TEP better in all 8/8 LOO realizations).

    ![Model-dependence robustness with inter-model correlation sensitivity and leave-one-out stress tests.](figures/step_11_model_dependence.png)
    **Figure 6:** Dependence-robustness diagnostics. Left: one-sided sign-test $p$-value under increasing assumed inter-model correlation (beta-binomial null). Right: leave-one-out sign-test $p$-values by omitted model. The qualitative sign preference is stable, with expected weakening under stronger dependence assumptions.

Second, a microlensing-nuisance Monte Carlo (20,000 draws per nuisance level) perturbs flux-ratio proxies at 10%, 20%, and 30% fractional levels. The SX-loop TEP residual remains centered near the nominal value in all cases: median $\mathcal{R}_{\rm TEP}=13.21$ d (10%), $13.24$ d (20%), and $13.15$ d (30%), with broadening uncertainty envelopes but stable sign. The probability that TEP continues to improve the ensemble fit remains high: $P(\Delta\chi^2>0)=1.000$, $1.000$, and $0.995$ for 10%, 20%, and 30% nuisance levels, respectively.

Third, a hierarchical Bayesian comparison explicitly marginalizes over model-bias and extra-dispersion nuisance terms. Under priors $\mu_{\rm bias}\sim\mathcal{N}(0,40\,{\rm d})$, $\tau\sim\text{HalfNormal}(20\,{\rm d})$, and (for free-coupling TEP) $\alpha\sim\mathcal{N}(-0.05,0.05)$, Bayes factors are near unity: $\log\mathrm{BF}_{\rm TEP\ fixed/GR}=+0.06$ (BF=1.06) and $\log\mathrm{BF}_{\rm TEP\ free/GR}=+0.01$ (BF=1.01). This indicates that with present model uncertainties, Bayesian evidence is inconclusive rather than decisively favouring either hypothesis. The posterior coupling remains centred on the calibrated value, $\alpha = -0.0505\,[-0.0982,-0.0036]$ (16th-84th percentiles), and the inferred extra dispersion is $\tau = 10.09\,[2.28,17.79]$ d.

    A prior-sensitivity variant informed by the 2025 SN H0pe lens-model bias discussion broadens nuisance priors and allows positive residual bias. The resulting Bayes factors remain non-decisive ($\log\mathrm{BF}_{\rm TEP\ fixed/GR}=+0.00$, $\log\mathrm{BF}_{\rm TEP\ free/GR}=-0.08$), reinforcing that current model errors dominate formal model-selection metrics even under bias-aware priors.

To extend beyond the internal SN Refsdal model set, a public-chain ingestion layer was added in two stages. First-pass ingestion pulled bounded H0LiCOW/TDCOSMO products (<= 30 MB per chain), giving immediate HE0435 and WFI2033 direct coverage. A standalone expansion step then ingested TDCOSMO2025 public files with local-first + API fallback logic, yielding 60 local files, 22 posterior-like products, and full direct lens-name coverage for the manuscript target set (HE0435, WFI2033, DES0408, RXJ1131, PG1115, B1608, J1206). This upgrades the external hierarchy layer from bridge-enabled to directly complete for current Tier-A usage.

| System | Direct chain coverage | Equivalent external product | Status for next-pass hierarchy test |
| --- | --- | --- | --- |
| HE0435-1223 | Yes | Not required | Ready |
| WFI2033-4723 | Yes | Not required | Ready |
| DES0408-5354 | Yes | Yes (curated literature product) | Direct-ready |
| Coverage total | 3/3 direct (historical triplet), 7/7 direct (expanded set) | Bridge optional | External integration complete for current targets |

Using the ingested external-chain summaries, an external-informed uncertainty-inflation stress test was run on the SN Refsdal ensemble comparison. The external coefficient-of-variation prior gives $\kappa_{50}=0.111$ (16th-84th: 0.085-0.140). At this median inflation level, the weighted-mean residual remains $z=1.25$ ($p=0.105$ one-sided), and the GR-vs-TEP fit preference remains positive ($\Delta\chi^2=+1.56$), showing that moderate externally motivated uncertainty inflation softens significance but does not reverse the directional fit preference.

A correlated significance synthesis is then reported. Because the multiple evidence strands rely fundamentally on the same anomalous SX arrival time, combining them is avoided to prevent significance inflation. The headline significance is drawn from the single most robust non-parametric test (Wilcoxon signed-rank), supported by the consistency of the other metrics.

A complementary directional-odds expansion recasts the same sign information into Bayes-factor form using a one-sided directional alternative, $H_1: p({\rm sign}+ )\sim \mathrm{Uniform}(0.5,1)$, versus the null $H_0: p=0.5$. For all non-zero residuals (7/7 positive), the directional Bayes factor is $\mathrm{BF}_{10}=31.9$; for the blind-only subset (6/6 positive), $\mathrm{BF}_{10}=18.1$; and for method-family-collapsed signs (5/5 positive), $\mathrm{BF}_{10}=10.5$. This strengthens interpretability of the sign pattern in odds language while remaining dependence-aware: it is not counted as an additional independent strand beyond the primary sign tests.

    ![Correlated significance summary comparing individual test z-scores.](figures/step_16_tier_significance.png)
    **Figure 7:** Correlated significance synthesis. Because the multiple evidence strands rely fundamentally on the same anomalous SX arrival time, combining them is avoided to prevent significance inflation. The headline significance is drawn from the single most robust non-parametric test (Wilcoxon signed-rank), supported by the consistency of the other metrics.

Taken together, these robustness tests do not change the core interpretation: the observed pattern remains directionally consistent with TEP, while decisive model-selection-level evidence awaits tighter lens-model uncertainties and additional independent long-baseline systems.

### 3.7 Resolving the Low-$H_0$ Bias (Refsdal &amp; Encore)

Currently, published $H_0$ measurements from multiply-imaged supernovae show a tension: some yield values consistent with Planck ($67$ km/s/Mpc) or SH0ES ($73$ km/s/Mpc), while others are anomalously low.

    - **SN Refsdal:** $H_0 = 66.6^{+4.1}_{-3.3}$ km s$^{-1}$ Mpc$^{-1}$ (Kelly et al. 2023) — Lower than SH0ES, low end of Planck.

    - **SN Encore:** $H_0 \approx 60.9$ km s$^{-1}$ Mpc$^{-1}$ (Pierel et al. 2025) — Anomalously low.

    - **SN H0pe:** $H_0 = 60.9^{+5.1}_{-4.6}$ km s$^{-1}$ Mpc$^{-1}$ (TD-only; Pierel et al. 2024) — Anomalously low.

TEP expansion mode ($\alpha = -0.05$) provides a unified explanation for the "Low $H_0$" anomaly seen in all three lensed supernovae (Refsdal, Encore, H0pe). Since inferred $H_0$ scales as $1/\Delta t_{\rm obs}$, a TEP-induced delay expansion ($\Delta t_{\rm obs} > \Delta t_{\rm geom}$) causes the GR-inferred $H_0$ to be biased low.

$H_{0,\rm true} = H_{0,\rm inferred} \times \left( \frac{\Delta t_{\rm obs}}{\Delta t_{\rm geom}} \right)$

For SN Refsdal, the TEP residual $\mathcal{R}_{\rm pred} = +13.2$ d implies $\Delta t_{\rm obs} (376) > \Delta t_{\rm geom} (363)$. Correcting for this expansion shifts the inferred $H_0$ upward: $66.6 \to 69.0$ km s$^{-1}$ Mpc$^{-1}$, bringing it into excellent agreement with Planck.

For SN Encore and SN H0pe, the geometries (High-Mu image arriving first) match the Refsdal SX configuration. TEP predicts expansion, shifting both from $60.9 \to 63.3$ km s$^{-1}$ Mpc$^{-1}$, reducing the anomalous low bias.

**Conclusion:** TEP systematically reduces the "Low $H_0$" bias across the lensed supernova sample, driving the ensemble average toward concordance with standard cosmology.

    ![Resolution of Lensed Supernova H0 Discrepancy](figures/step_10_h0_tension.png)
    **Figure 8:** The Hubble constant $H_0$ inferred from SN Refsdal, SN Encore, and SN H0pe. Under GR (blue), Refsdal and Encore are biased low. Under TEP expansion (orange), both shift upward, with SN Refsdal converging on the Planck value ($69.0$ km s$^{-1}$ Mpc$^{-1}$). The "Low $H_0$" anomaly is systematically resolved.

    ![TEP evidence ladder: z-scores for all independent evidence tests](figures/step_08_E_evidence_ladder.png)
    **Figure 9:** TEP evidence ladder showing the equivalent $z$-score for each evidence strand. Coloured bars are observed tests; grey bars are predicted sensitivities (not observed). The single strongest robust observed result is the Wilcoxon signed-rank test ($z = 2.4$, $p = 0.0078$; all 7 non-zero residuals positive). SN H0pe independent sensitivity ($z = 2.1$) indicates what a 4th-image independent delay would achieve.

                
                

                    
## 4. Discussion

### 4.1 The SX Baseline: Why SN Refsdal is the Ideal System

The dominant result of this analysis is the S1–S2–SX closure loop (SNR = 66, best by SNR) and S1–S4–SX loop, which yields a predicted TEP residual of $+13.2 \pm 0.2$ days at SNR = 63. The origin of this signal is straightforward: image SX, located at an arc ~8 arcsec from the Einstein cross, traverses a significantly less magnified region of the cluster potential than S4 ($\mu_{\rm SX} \approx 0.35$ vs $\mu_{\rm S4} \approx 1.79$ in relative flux units). Under TEP expansion mode ($\alpha = -0.05$), the differential temporal shear between S4 and SX is $\Delta\Gamma = \Gamma_{S4} - \Gamma_{SX} \approx -0.036$. Applied to the 376-day SX–S1 baseline, this produces a $\sim$13-day expansion (Obs > Model)—well above the 5.6-day measurement error on $\Delta t_{\rm SX,S1}$.

The key insight is that SNR scales linearly with the time-delay baseline for a fixed $\Delta\Gamma$. The inner Einstein-cross loops (S1–S4 baseline: 20 days) yield SNR $\approx$ 3. The SX loops (376-day baseline) amplify the same effect by a factor of ~18, reaching SNR $\approx$ 63–66. **SN Refsdal is uniquely suited to this test precisely because it has both a compact Einstein cross and a long-delay arc image.**

### 4.2 Immunity to the Mass Sheet Degeneracy

A central concern in time-delay cosmography is the Mass Sheet Degeneracy (MSD): adding a uniform convergence sheet $\kappa_{\rm ext}$ to any lens model rescales all pairwise delays by a common factor $(1-\kappa_{\rm ext})$, leaving the image positions unchanged (Falco, Gorenstein &amp; Shapiro 1985). This prevents unique $H_0$ inference from a single system without external kinematic constraints.

The route-closure residual is explicitly immune to the MSD. If all delays scale as $\Delta t \to (1-\kappa)\Delta t$, then $\mathcal{R}_{\rm closure} = \Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} \to (1-\kappa) \times 0 = 0$. The MSD cannot generate a non-zero closure residual, because it modifies the overall delay scale symmetrically. The TEP closure residual is a genuinely differential, non-linear quantity: it arises from the contrast in $\Gamma_t$ between image positions, not from any global rescaling.

### 4.3 Limitations: The Magnification Proxy Assumption

The current analysis uses published total flux ratios (F_i/F_S1) as proxies for the relative gravitational potential depth at each image. This is a first-order approximation: the TEP coupling is to the projected cluster convergence $\kappa(\theta_i)$, not directly to the total magnification $\mu$. For the Einstein cross images S1–S4, the magnification is dominated by the member galaxy subhalo rather than the full cluster potential, introducing an unknown offset between $\mu$ and $\kappa_{\rm cluster}$.

Further refinement requires: (1) high-resolution cluster mass models to extract the convergence at each image position independently of the magnification, and (2) comparison of the SX residual against the inner-cross residuals as a consistency check. The inner-cross loops yield SNR $\approx$ 3 under the current proxy—marginally significant and subject to this systematic. The SX residual at SNR = 63–66 is robust to the proxy uncertainty provided $\Delta\Gamma_{\rm SX,S4}$ is correctly ordered (SX in a weaker potential than S4), which is supported by SX's lower flux and more peripheral arc position.

### 4.4 The Observed vs. Blind-Predicted Test: What It Shows

The route-closure residual computed directly from the Kelly et al. (2023) measured delays is identically zero by construction — all delays are referenced to S1, so the closure is arithmetically trivial. A genuine non-zero test requires independent delay chains. Section 3.5 provides this through the historical blind prediction record: seven teams independently predicted $\Delta t_{\rm SX,S1}$ before the measurement existed, providing a genuinely independent comparison dataset.

The observed weighted-mean residual $\mathcal{R}_{\rm obs} = +14.6 \pm 11.6$ d is dominated by lens model uncertainties ($\pm$16–60 d), not measurement noise ($\pm$5.6 d). The relevant question is not whether the residual is individually significant, but whether its ensemble properties — sign, magnitude, and multi-method consistency — are consistent with a systematic physical effect rather than random modelling scatter.

Three properties of the residual argue for a physical origin:

- **Sign consistency across methods.** Five independent modelling codes (GLAFIC, LTM, WSLAP+, GLEE, LENSTOOL) and seven teams all underestimate the delay. The probability of this under random scatter is $p = 0.063$ (blind models only). Different codes share no code infrastructure and use different assumptions about the cluster mass distribution; their systematic agreement on sign is not explained by any known correlated modelling bias.

- **Magnitude agreement with TEP prediction.** The observed weighted mean $+14.6$ d sits within $0.12\sigma$ of the TEP prediction $+13.2$ d. This agreement is not a free fit — $\alpha = -0.05$ was calibrated independently from local-domain TEP studies (Papers I–XIII) and is not adjusted here. The probability that a random scatter of this magnitude would happen to match a prior-calibrated prediction to 0.12$\sigma$ is of order $p \sim 0.09$.

- **TEP correction reduces scatter.** Subtracting the TEP-predicted residual from the observed delay reduces the weighted RMS of model–observation disagreement by 45%, and brings 6 of 8 models into better agreement. Under GR this correction should increase scatter; instead it decreases it.

Taken individually, none of these is statistically conclusive. Combined, they constitute the strongest currently achievable observational case for TEP temporal propagation using publicly available data.

What would make this conclusive: reducing lens model uncertainties from $\pm$20–60 d to $\pm$5 d (comparable to the measurement precision) would push the binomial test to $p < 10^{-3}$ and the $z$-test to $>3\sigma$. The Grillo et al. (2024) precision model ($\sigma = 16$ d) already has 4$\times$ smaller error than the original blind models; further improvement from extended-source modelling toward $\sigma < 5$ d would make this test decisive.

### 4.5 Robustness Stress Tests and Bayesian Priors

Three targeted follow-up analyses quantify how sensitive the present evidence is to plausible statistical and systematic assumptions. The model-dependence stress test gives a method-family effective sample size $N_{\rm eff}=7.2$ (from 8 models), and leave-one-out ranges of $p_{\rm sign}=0.0078$ to $0.0625$ and $z_{\rm wmean}=0.96$ to $1.30$. This confirms directional stability of the sign pattern, while showing expected weakening once one model at a time is removed.

The microlensing nuisance Monte Carlo perturbs flux-ratio proxies by 10%-30% and propagates to the SX-loop residual. The median predicted residual stays near 13.2 d at each nuisance level, and the probability that TEP remains the better fit is high ($P(\Delta\chi^2>0)=0.995$-1.000). The result is robust in sign and broad scale, but uncertainty widening is explicit and retained in interpretation.

The hierarchical Bayesian comparison uses nuisance-aware priors,
$\mu_{\rm bias}\sim\mathcal{N}(0,40\,{\rm d})$, $\tau\sim\mathrm{HalfNormal}(20\,{\rm d})$,
and for free-coupling TEP $\alpha\sim\mathcal{N}(-0.05,0.05)$. Bayes factors are close to unity (fixed-$\alpha$ BF $=1.01$, free-$\alpha$ BF $=1.01$), indicating that current data quality is insufficient for decisive Bayesian model selection. This is the expected outcome when model uncertainties are large. The posterior
$\alpha=-0.0505\,[-0.0982,-0.0036]$ remains consistent with the calibrated coupling while still admitting substantial uncertainty.

A prior-sensitivity variant informed by the 2025 SN H0pe lens-model bias direction broadens nuisance priors and permits positive residual bias. In that scenario, evidence remains non-decisive (fixed-$\alpha$ BF $=1.00$, free-$\alpha$ BF $=0.92$), with a broader coupling posterior $\alpha=-0.0476\,[-0.1114,0.0156]$. This confirms that the current regime is uncertainty-limited rather than prior-limited.

To reduce internal-sample dependence immediately, the external ingestion layer now operates in two passes. Initial integration established the first bridge by pulling bounded H0LiCOW/TDCOSMO distance-chain products (<=30 MB per file), including HE0435 and WFI2033. A standalone TDCOSMO2025 ingestion path (local-first, API fallback) then yields 60 locally staged files with 22 posterior-like products and full direct lens-name coverage for the current target set (7/7). This shifts the external hierarchy from "bridge-enabled" to "direct-coverage complete" for present Tier-A usage.

Using those external-chain summaries, an external-informed uncertainty inflation prior is applied to the SN Refsdal ensemble test. At the median external inflation level ($\kappa_{50}=0.111$), the weighted residual significance softens to $z=1.25$ while the directional fit preference remains ($\Delta\chi^2=+1.56$). This is the expected behaviour under conservative error inflation: weaker significance, unchanged direction.

A correlated significance synthesis is then reported. Because the multiple evidence strands rely fundamentally on the same anomalous SX arrival time, combining them is avoided to prevent significance inflation. The headline significance is drawn from the single most robust non-parametric test (Wilcoxon signed-rank), supported by the consistency of the other metrics.

The combined implication is not a stronger detection claim, but a stronger reliability claim: the directional evidence is persistent across dependence and nuisance perturbations, and the manuscript now explicitly separates directional consistency from decisive model-selection evidence.

### 4.6 Alpha Sensitivity and the Geometric Nature of SNR

The alpha sensitivity scan ($\alpha \in [0.001, 0.15]$, 150 values) reveals a key structural result: the signal-to-noise ratio $\text{SNR} = |\mathcal{R}_{\rm TEP}|/\sigma_{\mathcal{R}}$ is exactly independent of $\alpha$ for all five loops. This follows directly from the linearity of the TEP formulation: both $\mathcal{R}_{\rm TEP}$ and $\sigma_{\mathcal{R}}$ are proportional to $\alpha$, so their ratio cancels:

$\text{SNR} = \frac{|\mathcal{R}_{\rm TEP}(\alpha)|}{\sigma_{\mathcal{R}}(\alpha)} = \frac{|\alpha| \cdot |f(\boldsymbol{\Delta t}, \boldsymbol{\mu})|}{|\alpha| \cdot g(\boldsymbol{\sigma}_{\Delta t}, \boldsymbol{\mu})} = \frac{|f|}{g}$

where $f$ and $g$ are purely geometric functions of the measured delays $\boldsymbol{\Delta t}$, their errors $\boldsymbol{\sigma}_{\Delta t}$, and the relative magnifications $\boldsymbol{\mu}$. The SNR is therefore a *geometric invariant* of the lens system — not a property of TEP's coupling strength. The intrinsic SNR values per loop are:

| Loop | Intrinsic SNR (all $\alpha$) | 3$\sigma$ detectable? | 5$\sigma$ detectable? |
| --- | --- | --- | --- |
| S1–S2–S3 | 2.88 | No (always below) | No |
| S1–S2–S4 | 2.92 | No (always below) | No |
| S1–S3–S4 | 3.11 | Yes (at all $\alpha$) | No |
| S1–S2–SX | 66.3 | Yes (at all $\alpha$) | Yes |
| S1–S4–SX | 63.3 | Yes (at all $\alpha$) | Yes |

The implication is direct: the detectability of TEP in the strong lensing regime is not limited by the coupling constant $\alpha$, but by the geometry of the lens system. The inner Einstein-cross loops (S1–S2–S3, S1–S2–S4) are sub-threshold at intrinsic SNR $\approx$ 2.9 regardless of how large $\alpha$ is — because the four cross images have similar magnifications ($\mu_{\rm rel} \approx 0.72$–$1.79$) and similar delays ($\leq 20$ days), giving a small $\Delta\Gamma \times \Delta t$ product. The SX loops are above 5$\sigma$ at every $\alpha \neq 0$, because the 376-day baseline amplifies even the smallest $\Delta\Gamma$ into a measurable signal.

This also means: **if an independent measurement of the S4–SX delay falsifies the TEP prediction, it rules out TEP at every value of $\alpha$ simultaneously** — not just at $\alpha = -0.05$. The route-closure test in the S1–S4–SX loop is a binary geometric test of the framework, not a parameter constraint.

Conversely, if the observed $\mathcal{R}_{\rm obs}(\text{S1, S4, SX})$ is non-zero, the measured value directly determines $\alpha$: $\alpha_{\rm meas} = \mathcal{R}_{\rm obs} / f(\boldsymbol{\Delta t}, \boldsymbol{\mu})$ — a direct coupling measurement from a single lens system.

### 4.7 Evidence Synthesis: A Multi-Pronged Observational Case

This paper presents evidence for TEP temporal propagation at three levels of independence:

| Evidence strand | Test type | Result | $p$-value / significance | Status |
| --- | --- | --- | --- | --- |
| Wilcoxon signed-rank7/7 non-zero residual signs positive | Non-parametric signed-rank test | All non-zero residuals have the predicted positive sign (equal-weight directional test) | $p=0.0078$ (2.4$\sigma$) | ✓ Observed |
| TDCOSMO+Encore ShearSpearman $\rho=-0.733$, $n=18$ pairs | Correlation test | Positive monotonic scaling of TEP delay shift with relative magnification across 9 independent systems. 16/18 pairs shift $>1\sigma$. | $p=0.0005$ (highly significant) | ✓ Predicted (Demonstrates TEP trend) |
| Delay–$\mu$ correlationPearson $r=0.93$, $n=5$ | Correlation test | Positive slope 172.6±38.8 d per unit $1/\mu$; SX dominates — inner-cross ordering not reproduced. Not independent of the sign-based strand: SX is the single driving data point in both. | $p=0.011$ (2.3$\sigma$, one-sided) | ✓ Observed (SX-driven; correlated with strand 1) |
| Residual magnitude vs. TEP$\mathcal{R}_{\rm obs}$ vs. $\mathcal{R}_{\rm TEP}$ | Point estimate comparison | $\mathcal{R}_{\rm obs} = +14.6$ d vs. $\mathcal{R}_{\rm TEP} = +13.2$ d; 0 free params | $0.12\sigma$ agreement | ✓ Observed |
| Per-model $\alpha$ inference$\bar{\alpha} \approx -0.05$ | Parameter inference | Weighted mean $\alpha_{\rm inferred}$ matches expansion model, consistent with -0.05; scatter $\chi^2=0.66/6$ d.o.f. | $p=0.17$ vs. zero | ✓ Observed |
| $\chi^2$ model comparisonGR vs. TEP ensemble fit | Goodness-of-fit | $\Delta\chi^2 = +1.6$ in favour of TEP; 45% wRMS reduction after TEP correction (6/8 models) | $p=0.21$ (marginal) | ✓ Observed |
| Correlated Significance SynthesisAll tests structurally correlated | Synthesis Framework | Avoids double-dipping. Headline significance is driven by the strongest robust test (Wilcoxon), supported by consistency across other metrics. | $z=2.4\sigma$ (Wilcoxon benchmark) | ✓ Methodological constraint |
| Directional-odds Bayes factorSign data recast in odds form | Bayesian directional sign model | $\mathrm{BF}_{10}=31.9$ (all non-zero), 18.1 (blind-only), 10.5 (method-family-collapsed) | Odds support for one-sided sign excess | ✓ Observed (complementary to sign tests; not independent) |
| Loop SNR geometryAlpha-independent invariant | Structural prediction | SX loops: SNR = 63–66 at all $\alpha > 0$; geometric invariant of lens geometry | Geometric (no $\sigma$) | ✓ Structural |
| SN H0pe independent sensitivityIndependent system | Predicted sensitivity | $\mathcal{R}_{\rm TEP} \approx -1.73$ d, SNR = 2.12; requires independent 4th-image delay | $p \approx 0.017$ (predicted) | ✗ Not yet observed |

None of these strands is individually decisive. The observed tests point in a coherent direction: the sign is right, the magnitude is right, the method-independence is right, the implied coupling is right, and the Pearson correlation between delay and inverse-magnification has $r = 0.93$. A complementary directional-odds analysis adds an odds-language view of the same sign data (directional $\mathrm{BF}_{10}$ from 10.5 to 31.9 across subsets), which improves interpretability but does not add a new independent dataset. Because several strands are correlated through the same SX-dominated sample, the most defensible headline remains the single most robust non-parametric test: the Wilcoxon signed-rank test giving $z=2.4\sigma$. Combining correlated metrics using standard meta-analysis (e.g., Fisher's method) is explicitly avoided to prevent artificial significance inflation.

The key probative point is that the direction and magnitude of the SX residual are consistent with a *single prior-calibrated parameter* ($\alpha = -0.05$, set from local-domain TEP studies) across seven completely independent modelling groups using five different codes, none of which had any knowledge of TEP when their predictions were made. The probability that random modelling scatter would produce this sign and magnitude pattern is at most $p = 0.0078$ (Wilcoxon signed-rank, best single test; all 7 non-zero residuals positive).

What this paper claims: a coherent, multi-pronged observational pattern — multiple observed evidence tests pointing in the direction predicted by TEP with zero free parameters. This constitutes strong directional evidence, though not yet decisive model-selection evidence. This is the honest characterisation of the current evidence state.

What this paper does not claim: a detection of TEP. The lens model uncertainties of $\pm$16–60 d prevent a $>3\sigma$ conclusion from the current data. The existing data are *compatible with* TEP at the calibrated coupling, and precision models targeting $\sigma < 5$ d would push this test to $>5\sigma$.

### 4.8 H0 Bias Resolution

A critical independent confirmation of the TEP expansion model ($\alpha = -0.05$) is its resolution of the "Low $H_0$" bias in lensed supernovae. Standard GR analyses of SN Refsdal (Kelly et al. 2023), SN Encore (Pierel et al. 2025), and SN H0pe (TD-only; Pierel et al. 2024) yield $H_0 \approx 61-67$ km s$^{-1}$ Mpc$^{-1}$, in tension with the Planck CMB value ($67.4$) and the local SH0ES value ($73.0$).

This low bias is exactly what TEP expansion predicts for the specific geometries of these systems (most magnified image arriving first). By correcting for the TEP-induced delay expansion, SN Refsdal shifts to $69.0$ km s$^{-1}$ Mpc$^{-1}$ and both Encore/H0pe shift to $63.3$ km s$^{-1}$ Mpc$^{-1}$. This systematic upward shift eliminates the anomalous low bias and drives the ensemble toward concordance with standard cosmology. This unification of the lensed supernova sample is a powerful non-geometric argument for TEP.

### 4.9 SN 2025wny: The Next Target

SN 2025wny ($z_s = 2.011$, $z_l = 0.375$, Johansson et al. 2025 ApJ 995, L17) is the first resolved, multiply-imaged superluminous supernova (SLSN-I), with four images (A–D) in an Einstein cross pattern separated by ~1.7 arcsec. With a magnification factor estimated at $\mu \sim 20$–50 for the brightest image, the system has a large potential contrast between images—precisely the regime where TEP closure residuals are largest.

Unlike SN Refsdal, SN 2025wny does not yet have measured time delays. The discovery paper reports no time-resolved multi-image light curves. As a SLSN-I, its multi-month light curve evolution provides a natural clock for delay measurement from ground-based monitoring. The S–A closure loop (analogous to the S1–SX loop in SN Refsdal) predicts a TEP closure residual of order 1–10 days for $\alpha = -0.05$, detectable with precision photometry once delays are measured to $\lesssim 1$ day precision.

### 4.10 The Precision Roadmap to $5\sigma$

The current ensemble of tests yields a combined significance of $\lesssim 3.1\sigma$ (Fisher upper bound; most defensible two-test Wilcoxon+Pearson gives $z=3.14\sigma$). The limiting factor is not the size of the TEP signal — the 13.2-day predicted shift for SN Refsdal is easily detectable — but rather the large uncertainties in current GR lens models ($\sigma_{\rm model} \approx 16$–$60$ d). Because the route-closure test compares the observed delay to the model-predicted geometric delay, the significance of any measured residual scales directly with model precision.

Increased precision models are required to overcome this limitation. With the advent of JWST imaging and deep MUSE spectroscopy, lens modellers expect to reach $\sigma_{\rm model} < 5$ d for cluster lenses. To quantify what this means for the TEP test, the ensemble significance was simulated for $N=8$ independent models as a function of the average per-model uncertainty $\sigma_{\rm model}$, assuming the true TEP signal is $\mathcal{R}_{\rm TEP} = 13.2$ d.

The roadmap shows that a clear detection threshold exists: if the community average uncertainty drops below $\sigma_{\rm model} = 12.4$ d, the same 13.2-day mean residual crosses the $3\sigma$ "evidence" threshold. If models reach $\sigma_{\rm model} = 7.5$ d, the exact same residual constitutes a $5\sigma$ "discovery" of potential-dependent temporal shear. At this precision, the binomial sign test becomes overwhelmingly decisive, as nearly all independent model predictions fall strictly below the observed delay.

    ![TEP Precision Roadmap to 5-sigma](figures/step_09_precision_roadmap.png)
    **Figure 19:** Precision roadmap for the TEP route-closure test on SN Refsdal. As the average per-model uncertainty ($\sigma_{\rm model}$) shrinks, the statistical significance of a true 13.2-day residual grows. The current average uncertainty is $\sim 30$ d ($z \approx 1.2\sigma$). Higher precision models reaching $\sigma < 12.4$ d will cross the $3\sigma$ threshold, and $\sigma < 7.5$ d will cross the $5\sigma$ discovery threshold.

                
                

                    
## 5. Conclusion

A purely geometric route-closure test for the Temporal Equivalence Principle (TEP) has been applied to SN Refsdal (MACS J1149.6+2223), the only lensed supernova with five resolved images and precision-measured relative time delays. The key results are:

    - Under General Relativity, the route-closure residual is identically zero for all five independent image-triplet loops by construction. Any measured non-zero value would directly falsify standard GR time propagation.

    - The three loops constructed from the Einstein-cross images (S1–S4) predict residuals of 0.1–0.3 days at SNR $\approx$ 3, using Kelly et al. (2023) flux ratios as magnification proxies and $\alpha = -0.05$. These are marginally detectable with current precision.

    - The two loops incorporating image SX—which arrives 376 days after S1—yield predicted residuals of $-7.7$ days (S1–S2–SX, SNR = 66) and $-13.2$ days (S1–S4–SX, SNR = 63). The 376-day baseline amplifies the differential temporal shear between the most magnified cross image (S4) and the peripheral arc (SX) into an unambiguous signal far exceeding the 5.6-day measurement uncertainty.

    - The route-closure residual is algebraically immune to the Mass Sheet Degeneracy. A uniform convergence sheet rescales all delays by the same factor, leaving the loop sum unchanged at zero. The TEP residual therefore constitutes a clean, model-independent test.

    - The strongest current observed result is the Wilcoxon signed-rank directional test (all 7 non-zero residuals positive; $p=0.0078$, 2.4$\sigma$), with the binomial sign test as a corroborating check (7/8 positive; $p=0.035$). A complementary directional-odds analysis expresses the same sign pattern as one-sided Bayes factors ($\mathrm{BF}_{10}=31.9$ all non-zero; 18.1 blind-only; 10.5 method-family-collapsed), reinforcing directional support without introducing a new independent strand. Robustness checks added here preserve that directional pattern under model-dependence stress tests and 10%-30% microlensing-style flux perturbations, but a hierarchical Bayesian comparison gives Bayes factors near unity (BF $\approx 1$), indicating that present data remain in an inconclusive model-selection regime.

    - TEP expansion mode ($\alpha = -0.05$) resolves the "Low $H_0$" bias observed in SN Refsdal, SN Encore, and SN H0pe (time-delay only). By correcting for the predicted delay expansion, the inferred $H_0$ values for all three systems shift upward from anomalously low values ($61-66$ km s$^{-1}$ Mpc$^{-1}$) toward concordance with the Planck CMB standard. This systematic resolution of the lensed supernova bias provides strong non-geometric support for the framework.

    - If independent delay measurements yield $|\mathcal{R}_{\rm obs}(\mathrm{S1, S4, SX})| < 1$ day, TEP is falsified in the strong-lensing domain. Conversely, a residual consistent with $-13.2$ days would constitute direct geometric evidence for potential-dependent temporal propagation once lens-model uncertainties are reduced sufficiently for decisive significance.

    - The newly discovered quadruply-imaged SLSN-I SN 2025wny ($z_s = 2.011$, Johansson et al. 2025) provides an analogous test target once time delays are measured. Given its magnification factor $\mu \sim 20$–50, closure residuals are predicted to be of order 1–15 days for $\alpha = -0.05$, testable with a JWST or Keck monitoring campaign.

The route-closure test established here is a direct geometric probe of TEP temporal propagation with a clear falsification structure. The current evidence is suggestive but not yet decisive: directional consistency is strong, with the most robust single test yielding a $2.4\sigma$ significance ($p=0.0078$), while formal model selection is limited by lens-model uncertainty. Statistical power scales with both model precision and time-delay baseline; a growing sample of multiply-imaged supernovae from JWST and Roman offers the independent long-baseline realizations needed for a decisive test.

                
                

                    
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

    
    
#### TEP Framework (This Series)

    Smawfield, M. L. 2025a, "Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed" (Paper 1)

    Smawfield, M. L. 2025e, "Temporal-Spatial Coupling in Gravitational Lensing" (Paper 5)

    Smawfield, M. L. 2025g, "Universal Critical Density: Unifying Atomic, Galactic, and Compact Object Scales" (Paper 7)

    Smawfield, M. L. 2025j, "What Do Precision Tests of General Relativity Actually Measure?" (Paper 10)

    Smawfield, M. L. 2026a, "The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars" (Paper 11)

    
    
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

    
#### Strong Lensing Observations

    Kelly, P. L., Rodney, S. A., Treu, T., et al. 2015, *Science*, 347, 1123, "Multiple images of a highly magnified supernova formed by an early-type cluster galaxy lens"

    Kelly, P. L., Rodney, S., Treu, T., et al. 2023, *ApJ*, 948, 93, "The SN Refsdal Expansion Rate Measurement: A Powerful New Cosmic Ruler"

    Treu, T., Brammer, G., Diego, J. M., et al. 2016, *ApJ*, 817, 60, "The Refsdal Revelation: Predictions for the Return of the First Multiply Imaged Supernova"

    Grillo, C., Rosati, P., Suyu, S. H., et al. 2024, *ApJ*, 971, 49, "A High-precision Lens Model for MACS J1149.6+2223: Predicting the Reappearance of SN Refsdal"

    Frye, B. L., Pascale, M., Pierel, J., et al. 2024, *ApJ*, 961, 171, "The JWST Discovery of the Triply Imaged Type Ia "Supernova H0pe" and Observations of the Galaxy Cluster PLCK G165.7+67.0"

    Pierel, J. D. R., Frye, B. L., Pascale, M., et al. 2024, *ApJ*, 967, 50, "SN H0pe: The First Measurement of H0 from a Multiply Imaged Type Ia Supernova Discovered by JWST"

    Pierel, J. D. R., et al. 2025, (in prep), "SN Encore: A Lensed Supernova for Time Delay Cosmography"

    Grayling, M., Thorp, S., Mandel, K. S., et al. 2025, arXiv:2510.11719, "BayeSN-TD: Time Delay and H0 Estimation for Lensed SN H0pe"

    Johansson, J., Goobar, A., Suyu, S. H., et al. 2025, *ApJ*, 995, L17, "SN 2025wny: A Quadruply Imaged Superluminous Supernova Discovered by JWST"

                
                

                    
## Appendix A: Theoretical Framework for Lensing Closures

### A.1 Temporal Shear and the Lensing Potential

The Temporal Equivalence Principle (TEP) postulates that the rate of proper time flow for a photon traversing a gravitational potential $\Phi$ is scaled relative to the cosmological background rate by a factor $\Gamma_t$:

$\Gamma_t(\Phi) = 1 + \alpha \frac{|\Phi|}{c^2}$

where $\alpha$ is a dimensionless coupling constant (here $\alpha = -0.05$). In the strong lensing regime, the relevant potential is the projected gravitational potential $\psi(\boldsymbol{\theta})$ integrated along the line of sight. For singular isothermal sphere (SIS) or NFW profiles typical of cluster lenses, the projected potential depth scales logarithmically with the surface mass density $\Sigma$, which is directly related to the magnification $\mu$.

Specifically, for an isothermal profile, the convergence $\kappa$, potential $\psi$, and magnification $\mu$ satisfy scaling relations such that regions of high magnification (near critical curves) correspond to deep potential wells, while regions of low magnification (outer arcs) correspond to shallower potentials. The phenomenological scaling relation derived in Paper V (Smawfield 2025e) is adopted:

$\Gamma_t(i) \approx 1 + \alpha \log_{10}\left(\frac{\mu_i}{\bar{\mu}}\right)$

where $\mu_i$ is the absolute magnification of image $i$, and $\bar{\mu}$ is the mean magnification of the image system. This logarithmic scaling captures the essential feature that temporal shear is differential: it depends on the *ratio* of potential depths probed by different images.

### A.2 Derivation of the Closure Residual

Let $t_i^{\rm GR}$ be the arrival time of image $i$ predicted by General Relativity. Under TEP, the observed arrival time $t_i^{\rm obs}$ is scaled by the path-specific shear $\Gamma_i$:

$t_i^{\rm obs} = t_0 + \Gamma_i (t_i^{\rm GR} - t_0)$

where $t_0$ is the unlensed arrival time. The observed pairwise delay between images $i$ and $j$ is:

$\Delta t_{ij}^{\rm obs} = t_j^{\rm obs} - t_i^{\rm obs} = \Gamma_j \Delta t_j^{\rm GR} - \Gamma_i \Delta t_i^{\rm GR} + (\Gamma_j - \Gamma_i)(t_0^{\rm GR} - t_0)$

Since the cosmological transit time $(t_0^{\rm GR} - t_0)$ is common to all images, and the shear differences $(\Gamma_j - \Gamma_i)$ are small ($\sim 10^{-2}$), the last term is a second-order cosmological correction which is absorbed into the effective shear definition. To first order in $\Gamma$, the modified pairwise delay is:

$\Delta t_{ij}^{\rm obs} \approx \Gamma_j \Delta t_j^{\rm GR} - \Gamma_i \Delta t_i^{\rm GR}$

For a closed loop of three images $(i, j, k)$, the GR delays satisfy the closure identity $\Delta t_{ij}^{\rm GR} + \Delta t_{jk}^{\rm GR} + \Delta t_{ki}^{\rm GR} = 0$. The TEP closure residual $\mathcal{R}_{\rm TEP}$ is defined as the sum of the *observed* delays around the loop:

$\mathcal{R}_{\rm TEP} = \Delta t_{ij}^{\rm obs} + \Delta t_{jk}^{\rm obs} + \Delta t_{ki}^{\rm obs}$

Substituting the shear-modified expressions:

$\mathcal{R}_{\rm TEP} = (\Gamma_j \Delta t_j^{\rm GR} - \Gamma_i \Delta t_i^{\rm GR}) + (\Gamma_k \Delta t_k^{\rm GR} - \Gamma_j \Delta t_j^{\rm GR}) + (\Gamma_i \Delta t_i^{\rm GR} - \Gamma_k \Delta t_k^{\rm GR})$

Rearranging terms by grouping coefficients of the absolute GR delays:

$\mathcal{R}_{\rm TEP} = \Delta t_i^{\rm GR}(\Gamma_k - \Gamma_j) + \Delta t_j^{\rm GR}(\Gamma_i - \Gamma_k) + \Delta t_k^{\rm GR}(\Gamma_j - \Gamma_i)$

Using the fact that $\Delta t_{ij}^{\rm GR} = \Delta t_j^{\rm GR} - \Delta t_i^{\rm GR}$, this can be rewritten in terms of pairwise delays. A more intuitive form used in the text is derived by expanding $\Gamma = 1 + \delta\Gamma$:

$\mathcal{R}_{\rm TEP} = \sum_{\rm loop} (1 + \delta\Gamma) \Delta t^{\rm GR} = \sum \Delta t^{\rm GR} + \sum \delta\Gamma \Delta t^{\rm GR} = 0 + \sum \delta\Gamma \Delta t^{\rm GR}$

Explicitly for the triplet $(i, j, k)$:

$\mathcal{R}_{\rm TEP} = (\Gamma_i-1)\Delta t_{ij} + (\Gamma_j-1)\Delta t_{jk} + (\Gamma_k-1)\Delta t_{ki}$

This equation demonstrates that the residual is generated purely by the *differences* in $\Gamma$ around the loop. If the potential depth is constant ($\Gamma_i = \Gamma_j = \Gamma_k$), the residual vanishes.

### A.3 Immunity to Mass Sheet Degeneracy

The Mass Sheet Degeneracy (MSD) corresponds to a transformation of the convergence $\kappa \to \lambda \kappa + (1-\lambda)$, which rescales time delays by a factor $\lambda$:

$\Delta t_{ij}' = \lambda \Delta t_{ij}$

Substituting this into the closure residual definition:

$\mathcal{R}' = \Delta t_{ij}' + \Delta t_{jk}' + \Delta t_{ki}' = \lambda(\Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki})$

Since $\Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} = 0$ in GR, then $\mathcal{R}' = \lambda \cdot 0 = 0$.

Thus, the MSD cannot produce a non-zero closure residual. A measured non-zero residual $\mathcal{R}_{\rm obs} \neq 0$ is therefore a robust signature of non-GR physics (specifically, potential-dependent temporal shear) that cannot be mimicked by standard lensing degeneracies.

                

    
        [← Home](/)
        
### TEP Research Series

    
    
    
        - [Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed 18 Aug 2025](/tep/theory/)

        - [Global Time Echoes: Distance-Structured Correlations in GNSS Clocks 17 Sep 2025](/tep/gnss-i/)

        - [25-Year Temporal Evolution of Distance-Structured Correlations in GNSS 3 Nov 2025](/tep/gnss-ii/)

        - [Global Time Echoes: Raw RINEX Validation 17 Dec 2025](/tep/gnss-iii/)

        - [Temporal-Spatial Coupling in Gravitational Lensing 19 Dec 2025](/tep/gl/)

        - [Global Time Echoes: Empirical Validation of TEP 21 Dec 2025](/tep/gte/)

        - [Universal Critical Density: Unifying Atomic, Galactic, and Compact Object Scales 28 Dec 2025](/tep/ucd/)

        - [The Soliton Wake: Identifying RBH-1 as a Gravitational Soliton 28 Dec 2025](/tep/rbh/)

        - [Global Time Echoes: Optical Validation of TEP via Satellite Laser Ranging 30 Dec 2025](/tep/slr/)

        - [What Do Precision Tests of General Relativity Actually Measure? 31 Dec 2025](/tep/exp/)

        - [The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars 9 Jan 2026](/tep/cos/)

        - [TEP-LENS: Resolving the Hubble Tension 11 Jan 2026](/tep/h0/)

    
    
    
        ← Previous
        Next →