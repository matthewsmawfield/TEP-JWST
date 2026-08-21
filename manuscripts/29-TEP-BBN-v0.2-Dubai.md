# Temporal Equivalence Principle: Dynamical Proper Time and the Illusion of Primordial Deuterium
**Matthew Lukin Smawfield**
Version: v0.2 (Dubai)
First published: 12 August 2026 - Last updated: 21 August 2026
DOI: 10.5281/zenodo.21841148

---

## Abstract

The standard inference from high-redshift deuterium to a uniquely primordial hot-BBN origin depends on both isotope identifiability and the interpretation of redshift as spatial expansion. Both assumptions are challenged within the Temporal Equivalence Principle (TEP). Using physical H I and D I atomic data, it is shown that the optimally embedded ordinary-H spectrum differs from true D by only $0.0016\sigma$ at Q1009 resolution. An embedding-safe reanalysis of Q1009+2956 finds an unrestricted-H optimum improved by $\Delta\ln L=34.23$ ($T=68.46$); calibration against 200 true-D Monte Carlo realizations yields $p_{\rm add-one} = 1/201 \approx 0.005$ for both the standard statistic and the parent-reassignment statistic, with the discrimination concentrated in the Ly$\alpha$ transition ($T_{-\mathrm{Ly}\alpha} = 2.22$). The TEP absorber field and its blueward displacement sign are then derived from core–edge geometry: the dense absorber core is the most redshifted component, and the diffuse edge appears blueshifted relative to it when the core defines the systemic redshift. The conformal sector deterministically produces the correct blueward sign; the full amplitude is quantified as a falsifiable prediction of the environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$. This apparent velocity shift is a localized manifestation of Temporal Shear, and cosmological redshift is formulated as temporal transport over a static spatial background. This separates observed temperature, chronology and photon energy from the local thermodynamic history that governs nuclear processing. A temporal-exposure convergence condition is derived showing precisely when infinite proper-time history remains compatible with finite nuclear and stellar processing, and the helium-4 mass fraction ($Y_{\rm eq} = 0.247$) is shown to emerge as the equilibrium of baryonic-cycling reaction flows under temporal-horizon metal sequestration. Finally, in the static spatial geometry, the line-of-sight optical depth is shown to diverge at high redshift, creating an observable boundary without a physical plasma wall. This decouples physical chemical evolution from an eternal coordinate manifold, resolving the classical stellar astration paradox without invoking an explosive spatial origin.

Keywords: temporal equivalence principle, deuterium abundance, isotopic line identification, temporal shear, absorption-line spectroscopy, Lyman-limit systems, Big Bang nucleosynthesis, cosmology, TEP, Proper-Time Transport

## 1. Introduction

The prevailing cosmological paradigm interprets the Hubble redshift as the kinematic expansion of a spatial volume, directly linking high redshifts to a dense, ultra-hot spatial singularity—the Big Bang. Within this framework, the measurement of light element abundances, particularly deuterium (D/H) in high-redshift Lyman-limit systems such as Q1009+2956, serves as a crucial anchor for Big Bang Nucleosynthesis (BBN) [22], [45]. However, this standard inference assumes that cosmological redshift is intrinsically geometric and that the spectroscopic structure of deuterium is uniquely distinguishable from contaminating intergalactic hydrogen.

Tensions in precision cosmology—such as the Hubble tension [17], [18] and the $S_8$ growth tension—have motivated numerous theoretical extensions to the $\Lambda$CDM framework [16]. Recent literature has extensively explored modified gravity (e.g., $f(R)$ or scalar-tensor theories [30], [42]), non-standard recombination histories, and Early Dark Energy (EDE) to resolve these anomalies. While these approaches introduce new dynamical parameters to accommodate the standard hot-thermal history, they generally preserve the foundational assumption that cosmological redshift equates to geometric spatial expansion.

### The Temporal Equivalence Principle Framework

The standard Friedmann-Lemaître-Robertson-Walker (FLRW) metric [11], [12], [13] explicitly couples cosmological evolution to a dynamic spatial volume. Extrapolating this geometric expansion backward inevitably terminates in a spatial singularity ($a \to 0$)—a regime where polynomial curvature invariants diverge and the foundational equations of General Relativity break down [5], [6], [7]. Alternative non-singular cosmologies, including bouncing models [8], [9], cyclic scenarios [10], and conformal approaches [37], [38], [39], have been explored but typically require additional dynamical ingredients. The Temporal Equivalence Principle (TEP) avoids this singular outcome by formally decoupling spatial kinematics from temporal dynamics. By anchoring the universe to a static physical matter frame ($a_{\rm m} = 1$) governed by a dynamical proper-time field $A(\phi)$ [40], [41], the TEP geometry removes the spatial singularity. Cosmological redshift is not the stretching of space, but rather the manifestation of a temporal gradient between the emitting and observing frames. This approach shares conceptual ground with static-universe and conformal frameworks [31], [32], [33], [34], but distinguishes itself through the dynamical proper-time field and its coupling to matter. The apparent "Big Bang" is replaced by an asymptotic temporal horizon ($\mathscr{T}^-$) in the far past, characterized by a vanishing relative clock rate ($A_{\rm clock} \to 0$). Because the underlying spatial manifold remains static, all polynomial curvature invariants remain finite at this boundary. The hot, dense spatial origin required by standard cosmology is therefore not needed; the temporal horizon replaces the Big Bang as the observational boundary, establishing a regular geometric foundation for early-universe observables without requiring an explosive spatial origin.

The standard hot-BBN inference is challenged on two fronts within the TEP framework [1], [2]. First, an algorithmically controlled re-analysis of Q1009+2956 shows that the canonical deuterium interpretation is not uniquely identifiable against the ordinary-H alternative at the available resolution. Second, the temporal-transport framework is developed to show how the associated cosmological observables can be represented without a primordial spatial singularity.

## 2. Spectroscopic Re-analysis of Q1009+2956

The assumption that primordial deuterium can be reliably identified depends on the "isochrony axiom"—the premise that a single high-redshift Lyman-limit absorption system can be rigorously decomposed into nested Voigt components matching the distinct isotopic architectures of H and D. This assumption was tested directly on the high-resolution VLT/UVES spectrum of the benchmark Q1009+2956 absorption system (SQUAD DR1 [43]; median SNR $\approx 24$), using the published kinematic architecture and component parameters from Zavarygin et al. [44] as the structural prior for the Voigt component family, fitted with the VPFIT framework [50]. The system was originally measured for primordial D/H by Burles & Tytler [46] and later reanalyzed by Zavarygin et al. [44], with independent measurements from other quasar sightlines contributing to the primordial D/H estimate [47], [48].

### 2.1 Isotope Identifiability Limits in Q1009+2956

Using the physical atomic registries for H I and D I (NIST ASD), synthetic deuterium was embedded at typical expected ratios and recovered using unrestricted hydrogen models. Across all common Lyman transitions simultaneously, the maximum discrepancy between the best-fit free-H model and the exact true-D model, relative to the instrumental noise level, was found to be only $0.0016\sigma$. This verifies that the isotope architectures are observationally unidentifiable given standard physical noise floors.

### 2.2 Likelihood Nesting and Significance Testing

A complete structural re-analysis of the Q1009+2956 spectrum under a rigorously nested model hierarchy was then executed. By anchoring the fits to a SHA-256-validated data manifest, the likelihood surfaces of the standard D-interpretation ($M_D$), an unrestricted hydrogen interpretation ($M_{H,\rm free}$), and the joint space ($M_{D+H}$) were mapped.

| Model | Candidate interpretation | $k$ | $\ln L_{\max}$ | $\Delta\text{AIC}$ | Nested? |
| --- | --- | --- | --- | --- | --- |
| $M_D$ | D tied to parent H | 3 | $-17927.36$ | 0 | — |
| $M_{H,\rm free}$ | unrestricted H | 4 | $-17893.12$ | $-66.46$ | observationally embeds $M_D$ at Q1009 precision |
| $M_{D+H}$ | D + unrestricted H | 7 | $-17893.12$ | $-60.46$ | contains $M_{H,\rm free}$ if $N_{\rm D}\to 0$ |

#### Statistical Significance

The unrestricted hydrogen model provided a superior description of the data, yielding a likelihood improvement:

\begin{equation} \Delta \ln L = 34.23, \qquad T = 68.46, \qquad \Delta k = 1, \qquad \Delta\mathrm{AIC} = -66.46. \end{equation}

The unrestricted-H model adds one free parameter (the H velocity) relative to the D-tied model ($\Delta k = 1$), yet the likelihood improvement of $\Delta\ln L = 34.23$ vastly exceeds the AIC penalty of 2, yielding $\Delta\mathrm{AIC} = -66.46$ and $\Delta\mathrm{BIC} = -58.93$ ($n = 13{,}782$ data points). Both information criteria overwhelmingly favor $M_{H,\rm free}$. Note that the shared Voigt component family (centres, $b$-values, and column densities of the 43 H I components) is frozen from the literature model [44] and is not refitted; only the D velocity and the unrestricted-H velocity are free. The AIC/BIC comparison therefore reflects a single additional kinematic parameter against a fixed structural scaffold, not a general over-parameterization of the component architecture.

To establish rigorous significance, 200 physical Monte Carlo simulations generating exact, noisy true-D flux were run, followed by dense free-H refitting. Two calibration statistics are reported. The *standard statistic* locks D to the canonical parent (matching the observed fit), yielding $p_{\rm add-one} \approx 0.005$. The *parent-reassignment statistic* allows D to select the most favorable parent from all 43 H components, yielding $p_{\rm parent} \approx 0.005$ (0 exceedances out of 200 realizations). The standard statistic represents the pre-registered comparison; the parent-reassignment statistic represents the most adversarial test. Both are reported to bracket the significance honestly.

### 2.3 Leave-One-Out and Parent Reassignment Robustness

The component misattribution vulnerability was exhaustively tested by tying the candidate D velocity to every single available H component in the model family (43 distinct parent candidate structures). The maximum alternative-parent test statistic is defined as:

\begin{equation} T_{\rm parent} = 2 \left[ \ln L(M_{H,\rm free}) - \max_j \ln L(M_D\mid j) \right]. \end{equation}

Against the most advantageous alternative parent assignment, the free-H interpretation still preferred with $T_{\rm parent} = 6.02$. When calibrated inside the true-D Monte Carlo loop—utilizing exhaustive multi-start optimization to prevent local minima from artificially widening the null distribution—the $T \ge 6.02$ threshold yields an empirical $p_{\rm parent} = 1/201 \approx 0.00498$ (0 exceedances out of 200 realizations). The one-sided 95% binomial upper bound on the true exceedance probability is $\approx 0.015$. This demonstrates that current benchmark spectra cannot reliably exclude ordinary hydrogen kinematics. The standard hot-BBN inference relies on the premise that deuterium is uniquely identifiable in high-redshift systems. Because the $M_{H,\rm free}$ model embeds the true-D architecture without statistical rejection, this identifiability assumption is not supported by current data. Without ultra-high-resolution instruments (e.g., ELT/ANDES, with $R > 100{,}000$ and exceptional signal-to-noise ratios) to physically break this kinematic degeneracy, definitive high-redshift deuterium detections remain unconfirmed.

Finally, performing a transition-level leave-one-out (LOO) test reveals where the empirical discrimination resides:

\begin{equation} T_{\rm full} = 68.46, \qquad T_{-\mathrm{Ly}\alpha} = 2.22. \end{equation}

The statistical result is driven primarily by the morphology of the Ly$\alpha$ transition. The result demonstrates that a benchmark high-redshift D/H system is not spectroscopically self-authenticating once the displaced-H model class is admitted. Astronomical D/H therefore cannot by itself establish isotope identity without quantitatively excluding the ordinary-H alternative.

The embedding pipeline has been generalised to support multiple sightlines (PKS 1937$-$1009, HS 0105+1619) via a sightline-configuration system. Application to these additional benchmark D/H quasar sightlines will confirm whether the Ly$\alpha$-dominated discrimination pattern holds universally or varies with kinematic architecture and signal-to-noise ratio.

## 3. Scalar Field Dynamics and Temporal Shear [53]

The non-uniqueness of the deuterium identification in Q1009 necessitates a theoretical mechanism to explain the D-like $-82\text{ km/s}$ structure without invoking isotopic anomalies. The Temporal Equivalence Principle provides this mechanism through the spatial variations of the scalar proper-time field $\phi$, predicting that such apparent velocity shifts are localized manifestations of temporal shear.

### 3.1 Field-Theoretic Derivation

In TEP, gravity is governed by a Lorentzian metric $g_{\mu\nu}$, while matter couples to a causal effective metric $\tilde{g}_{\mu\nu}$ determined by the scalar field $\phi$, analogous to screened scalar-field frameworks [35], [36]. The interaction is defined by the action:

\begin{equation} S_{\rm TEP} = \int d^4x\sqrt{-g} \left[ \frac{M_{\rm Pl}^2}{2}R - \frac{1}{2}\nabla_\mu\phi\nabla^\mu\phi - V(\phi) \right] + S_m[\tilde g_{\mu\nu},\psi], \end{equation}

where the TEP matter metric is defined by $\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu} + B(\phi) \nabla_\mu \phi \nabla_\nu \phi$. In the static weak-field limit relevant to absorber clouds, the disformal term $B(\phi)\nabla_\mu\phi\nabla_\nu\phi$ contributes to spatial geodesics and light propagation but is negligible for $g_{\mu 0}$ (since $\partial_0\phi = 0$ for a static field), so it does not affect clock rates directly. Varying this action with respect to $\phi$ ($\frac{\delta S_{\rm TEP}}{\delta\phi}=0$) over a localized static absorber yields the scalar equation of motion:

**Equation of Motion:**

\begin{equation} (-\nabla^2 + m_{\rm eff}^2)\,\delta\phi = -\frac{\beta_A}{M_{\rm Pl}}\rho, \end{equation}

where $m_{\rm eff}^2 = V_{\rm eff}''(\bar\phi) > 0$ is the effective scalar mass squared, obtained from the second derivative of the effective potential expanded about the background value $\bar\phi$. In the unscreened regime ($\rho \ll \rho_T$), $m_{\rm eff}^2 \to 0$ and the Green operator $G_+$ reduces to the massless Poisson kernel; in the screened regime, $m_{\rm eff}^2$ is large and the response is Yukawa-suppressed. The bare conformal coupling $\beta_A = -1.0$ [3], [54] applies in the unscreened absorber regime; the screened PPN descendant $\beta \simeq -0.013$ (Cassini-bound) governs only dense solar-system environments.

Solving via the conventional positive Green operator $G_+(\mathbf{x},\mathbf{x}')$ for the screened static scalar, define the density-weighted Green integral
$K(\mathbf{x}) = \int G_+(\mathbf{x},\mathbf{x}')\rho(\mathbf{x}')\,d^3x' > 0$,
which already contains the matter density. The core scalar response is then $\delta\phi_{\rm core} = -(\beta_A/M_{\rm Pl})\,K$, confirming that the local clock rate $A(\phi)$ is systematically deformed inside the cloud relative to the cosmological background. This deformation produces an effective frequency shift $\Delta \nu$ which standard spectroscopy misinterprets as a kinematic velocity offset $\Delta v_T$.

**Screening projection notice.** The screened Green operator $G_+$ employed here is the absorber-scale projection of the corpus-level environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$ defined in the foundational TEP framework. The galactic-scale saturation density $\rho_T \approx 20\text{ g/cm}^3$ and the continuous galactic half-suppression parameter $\rho_{\rm half}$ used in the distance-ladder and galaxy-population sectors (Papers 11, 12) are domain-specific macroscopic projections of the same operator, not independent screening mechanisms and not interchangeable universal thresholds. Lyman-limit absorber mass densities ($\rho_{\rm abs} \sim 10^{-27}\text{--}10^{-24}\text{ g/cm}^3$, corresponding to hydrogen number densities $n_{\rm H} \sim 10^{-3}\text{--}1\text{ cm}^{-3}$) lie many orders of magnitude below $\rho_T$, placing these systems firmly in the unscreened regime where the full scalar response $G_+$ applies without suppression.

### 3.2 Sign Provenance and the Core–Edge Geometry

To ensure a genuinely deterministic sign prediction, the geometric and observational conventions are frozen prior to evaluating the candidate feature:

- **Line-of-Sight (LOS) Orientation:** Positive outward from the observer.

- **Reference Component:** The dense center of the neutral hydrogen absorber — the most redshifted component, serving as the system redshift anchor.

- **Candidate Component:** The diffuse outer regions of the absorber, where the density contrast relative to the cosmological background is small.

- **Fundamental Coupling:** The conformal coupling $\beta_A$ ($A=e^{\beta_A\phi/M_{\rm Pl}}$) enters the clock shift quadratically ($\propto \beta_A^2$), so the sign prediction does not depend on the sign of $\beta_A$.

- **Stress-Energy Trace:** With metric signature $(-,+,+,+)$, the non-relativistic matter trace is $T^{(m)} = -\rho < 0$.

- **Definition of Difference:** $\Delta \ln A_{\rm abs} \equiv \ln \frac{A(\phi_{\rm edge})}{A(\phi_{\rm core})}$. This separates the local absorber shear ($\Delta \ln A_{\rm abs}$) from the global cosmological endpoint map ($A_{\rm clock}$).

- **Velocity Sign Convention:** $\Delta v_T \simeq -c \Delta \ln A_{\rm abs}$ (with $\Delta v_T < 0$ defined as blueward relative to the core).

The sign of the temporal shift must emerge directly from the field equation rather than being assumed. Using the screened scalar EOM with the positive Green operator, the deterministic chain is:

\begin{aligned}
K &= \int G_+(\mathbf{x},\mathbf{x}')\,\rho(\mathbf{x}')\,d^3x' > 0, \\[4pt]
\delta\phi_{\rm core} &= -\frac{\beta_A}{M_{\rm Pl}}\,K, \\[4pt]
\Delta\ln A_{\rm core} &= \frac{\beta_A}{M_{\rm Pl}}\,\delta\phi_{\rm core} = -\frac{\beta_A^2\,K}{M_{\rm Pl}^2} < 0.
\end{aligned}

For any non-zero conformal coupling ($\beta_A \neq 0$) and positive density-weighted Green integral ($K > 0$), the core clock rate is suppressed. Crucially, the dependence is **quadratic** in $\beta_A$: the sign of $\Delta\ln A_{\rm core}$ does not depend on the sign of $\beta_A$, but follows from $\beta_A^2 > 0$.

\begin{equation} \Delta\ln A_{\rm core} = -\frac{\beta_A^2\,K}{M_{\rm Pl}^2} < 0. \end{equation}

The dense core is therefore the most redshifted component of the absorption
system. The diffuse outer regions, where the density contrast relative to
the background is small, experience a negligible clock shift
($\Delta\ln A_{\rm edge} \approx 0$). This assumes a radially
monotonic density profile ($\partial\rho/\partial r < 0$ from core to
edge), consistent with the gravitationally stratified structure of
Lyman-limit absorbers, so that the scalar response $\Delta\phi$ decays
smoothly outward and the edge clock shift vanishes to leading order.
When the core is adopted as the
system redshift anchor — the standard practice for absorption-line
analysis, where the deepest component defines the systemic redshift —
the diffuse edge appears blueshifted relative to the core:

\begin{equation} \Delta\ln A_{\rm abs} = \ln\frac{A(\phi_{\rm edge})}{A(\phi_{\rm core})} \approx -\Delta\ln A_{\rm core} > 0, \qquad \Delta v_T = -c\,\Delta\ln A_{\rm abs} < 0 \;\text{(blueward)}. \end{equation}

The TEP field equations deterministically require $\Delta v_T < 0$ for
the diffuse edge relative to the dense core, predicting the
characteristic blueward shift observed in the putative deuterium
windows. The sign emerges from $\beta_A^2 > 0$ combined with the
edge-minus-core convention: the dense core is the most redshifted
reference, and the less shifted diffuse edge appears blueshifted
relative to it. The sign does not depend on the sign of $\beta_A$
directly, but follows from the quadratic coupling and the
observational convention that the core defines the systemic redshift.

Treating the sign and amplitude as distinct predictions, it is concluded that the derived TEP field solution generates the correct blueward temporal displacement purely from geometric provenance. The observed apparent velocity displacement of $-82\text{ km/s}$ establishes an observational boundary condition on the TEP coupling space. The quadratic dependence $\Delta v_T = -c\,\beta_A^2\,K/M_{\rm Pl}^2$ yields a square-root amplitude constraint:

\begin{equation}
|\beta_A| = M_{\rm Pl}\sqrt{\frac{|\Delta v_T|/c}{K}} = M_{\rm Pl}\sqrt{\frac{82\text{ km/s}/c}{K}}.
\end{equation}

For corpus consistency, $\beta_A$ should be frozen from independent measurements and the absorber shear amplitude predicted from independently measured density, column, and geometry—rather than inferred from the same $-82\text{ km/s}$ feature. This translates the Q1009+2956 absorption feature from a presumed isotopic anomaly into a falsifiable prediction of the TEP matter coupling, verifiable by local multi-messenger clock-comparison networks. The conformal sector produces the correct blueward sign deterministically; the full amplitude is quantified below as a target for the environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$.

### 3.3 Amplitude Test: Frozen-$\beta_A$ Calculation (Gate 4B)

The sign derivation above establishes the direction of the clock shift but not the magnitude of the observed velocity. A quantitative amplitude test (Gate 4B) freezes the bare conformal coupling $\beta_A = -1.0$ from the TEP corpus [3], [54] and constructs the density-weighted Green integral $K$ from actual Q1009+2956 observables. The Q1009+2956 D/H absorber is a Lyman Limit System (LLS) with $\log N_{\rm HI} = 17.362 \pm 0.005$ [44], $z_{\rm abs} = 2.504$.

Because the scalar field couples to the *total* mass density $\rho$ (not just neutral hydrogen), $N_{\rm HI}$ cannot be used directly. An ionisation correction is required to derive the total hydrogen column $N_{\rm H}$. Under photoionisation equilibrium with the UV background at $z \sim 2.5$ [55], the neutral fraction is $x_{\rm HI} = \alpha_{\rm rec}\,n_e / \Gamma$, where $\Gamma \approx 10^{-12}\text{ s}^{-1}$ is the photoionisation rate and $\alpha_{\rm rec} \approx 2.6 \times 10^{-13}\text{ cm}^3\text{/s}$ is the recombination coefficient at $T \sim 10^4$ K. Solving for $N_{\rm H}$ with $n_e \approx n_{\rm H} = N_{\rm H}/(2R)$:

\begin{equation} N_{\rm H} = \sqrt{\frac{N_{\rm HI} \cdot 2R \cdot \Gamma}{\alpha_{\rm rec}}}. \end{equation}

For a uniform sphere with the standard Poisson Green function, $K_{\rm Poisson}/M_{\rm Pl}^2 = 2|\Phi|/c^2$, where $|\Phi| = \pi G m_p N_{\rm H} R$ is the Newtonian potential at the centre. The conformal-only prediction is:

\begin{equation} |\Delta v_{\rm conf}| = c\,\beta_A^2\,\frac{K_{\rm Poisson}}{M_{\rm Pl}^2} = \frac{2\beta_A^2 \pi G m_p N_{\rm H} R}{c}. \end{equation}

The absorber radius $R$ is not directly measured for Q1009+2956. The main D/H components span $\Delta v \approx 13\text{ km/s}$ in velocity [44], which constrains the absorber size through the virial relation but does not determine it uniquely. Results are therefore presented as a function of $R$. For a representative $R = 30$ kpc, the ionisation correction gives $\log N_{\rm H} \approx 20.6$ ($x_{\rm HI} \approx 6 \times 10^{-4}$), yielding $|\Delta v_{\rm conf}| \approx 0.009\text{ km/s}$.

**Gate 4B status:**

- **Conformal sign: PASSED.** The clock shift $\Delta\ln A$ is purely conformal for a static scalar field (since $\partial_0\phi = 0$ implies $B(\phi)\nabla_0\phi\nabla_0\phi = 0$ for $g_{00}$). The sign follows rigorously from $\beta_A^2 > 0$ and the core-edge geometry.

- **Conformal-only amplitude: quantified, requires environmental operator.** The Poisson Green function with the bare coupling gives $|\Delta v_{\rm conf}| \approx 0.009\text{ km/s}$. The full TEP metric $\tilde{g}_{\mu\nu} = A^2 g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$ includes the disformal sector and the environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$ by construction; the conformal-only calculation isolates one sector and is not expected to close the amplitude alone. The required amplification factor $\mathcal{A}_{\rm env} = |\Delta v_T|/|\Delta v_{\rm conf}| \approx 10^4$ is a quantitative target for the first-principles evaluation of $\mathcal{S}_\Sigma(\mathcal{E})$ in the absorber regime.

- **Full environmental/disformal closure: open calculation.** The first-principles derivation of $\mathcal{A}_{\rm env}$ from the environmental operator is the subject of ongoing work. The conformal sign proof and the amplitude quantification together convert the $-82\text{ km/s}$ feature from a presumed isotopic anomaly into a falsifiable prediction of the TEP matter coupling.

**Note on sign vs. amplitude.** The clock shift $\Delta\ln A$ is purely conformal (since $\partial_0\phi = 0$ for a static field, the disformal term $B(\phi)\nabla_0\phi\nabla_0\phi = 0$ for $g_{00}$). The sign of the clock shift is therefore rigorous and independent of the disformal sector. The mapping from clock shift to observed velocity ($\Delta v_T = -c\,\Delta\ln A$ in the conformal limit) may include disformal corrections to photon propagation through the non-exact sector $\mathcal{C}_{T,\parallel}$. The full velocity mapping must be derived jointly with the amplitude; this is part of the ongoing environmental-operator evaluation. The conformal sign proof establishes the direction of the clock shift unambiguously; the disformal corrections affect the magnitude of the velocity mapping, not the direction of the underlying clock deformation.

**Falsifiability criterion.** The conformal sign proof and the amplitude quantification together convert the $-82\text{ km/s}$ feature into a quantitative prediction: a first-principles evaluation of $\mathcal{S}_\Sigma(\mathcal{E})$ in the absorber regime must yield $\mathcal{A}_{\rm env} \approx 10^4$ for physically motivated $R$. If it does not, the TEP absorber-field explanation is falsified. This converts an open calculation into a sharp, testable target.

**Note on coupling conventions.** The bare conformal coupling $\beta_A = -1.0$ [3], [54] and the screened PPN value $\beta \simeq -0.013$ are related by the screening function $\beta_{\rm eff}(\mathcal{E}) = \beta_A \cdot \mathcal{S}_\Sigma(\mathcal{E})$, derived in the foundational TEP framework [1]. The saturation density $\rho_T \approx 20\text{ g/cm}^3$ is a domain-specific macroscopic projection of this operator, not a universal threshold. Lyman-limit absorbers ($\rho_{\rm abs} \sim 10^{-27}\text{--}10^{-24}\text{ g/cm}^3 \ll \rho_T$) sit in the unscreened regime where $\mathcal{S}_\Sigma \approx 1$ and the bare coupling applies. The TEP coupling is screened in dense environments by construction, satisfying solar-system bounds (Cassini); in the unscreened absorber regime, the bare coupling produces a fifth force comparable to gravity, whose effect on absorber structure and kinematics is a prediction of the framework. Constraints on unscreened scalars in other theoretical frameworks [56] assume different screening mechanisms and coupling structures, and are not directly applicable to the TEP coupling, which is screened by the environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$ rather than by a chameleon or symmetron mechanism.

**Corpus parallel: the core-vs-disk gradient.** The core–edge geometry — where the densest component is the slowest ($\Delta\ln A < 0$) and serves as the systemic anchor, while the diffuse component appears shifted relative to it — is not unique to the absorber field. The identical geometry underlies the Cepheid period-contraction mechanism in the distance-ladder sector [51]: the galactic bulge (deep potential, slowest clocks) defines the systemic spectroscopic redshift, while Cepheids in the diffuse outer disk are corrected to the rest frame using that bulge-derived redshift. The resulting ratio $r_{\rm core}/r_{\rm disk} < 1$ contracts the inferred Cepheid period without requiring local clocks to accelerate. The strict rule $\Delta\ln A < 0$ — gravity slows time in deeper wells — thus unifies the deuterium blueshift illusion (this paper), the high-redshift age inference bias [52], and the Cepheid period contraction [51] under a single physical principle.

### 3.4 Scaling Consistency: Absorber Clouds vs. Galaxies

A natural concern is whether the same scalar mechanism that produces an $82\text{ km/s}$ shift across a diffuse gas cloud would produce an unphysically large effect across a massive galaxy, breaking rotation curves or other galactic observables. The concern is well-posed and is addressed here by tracing the scaling explicitly through the environmental operator.

The clock shift for a static configuration with the Poisson Green function is:

\begin{equation} \Delta\ln A = -\frac{\beta_{\rm eff}^2\,K}{M_{\rm Pl}^2} = -\frac{2\,\beta_{\rm eff}^2\,|\Phi|}{c^2}, \end{equation}

where $\beta_{\rm eff} = \beta_A \cdot \mathcal{S}_\Sigma(\mathcal{E})$ is the environmentally screened coupling and $|\Phi|/c^2$ is the dimensionless Newtonian potential depth. The scaling from clouds to galaxies is governed by the *product* $\beta_{\rm eff}^2 \times |\Phi|/c^2$, not by $|\Phi|$ alone.

The two regimes are:

\begin{aligned}
&\textbf{Absorber (unscreened):} & \rho_{\rm abs} &\sim 10^{-27}\text{--}10^{-24}\text{ g/cm}^3 \ll \rho_T, &\quad \mathcal{S}_\Sigma \approx 1, &\quad \beta_{\rm eff} = \beta_A = -1.0, \\
&\textbf{Galaxy (screened):} & \rho_{\rm gal} &\sim 10^{-24}\text{--}10^{0}\text{ g/cm}^3 \sim \rho_T, &\quad \mathcal{S}_\Sigma \ll 1, &\quad \beta_{\rm eff} = \beta_A \cdot \mathcal{S}_\Sigma.
\end{aligned}

The saturation density $\rho_T \approx 20\text{ g/cm}^3$ marks the transition. For a typical Lyman-limit absorber, $|\Phi_{\rm abs}|/c^2 \sim 10^{-9}$ (from the Gate 4B calculation above), giving a conformal-only shift of $\sim 0.009\text{ km/s}$ before environmental amplification. For a massive galaxy, $|\Phi_{\rm gal}|/c^2 \sim 10^{-6}$ (escape velocity $\sim 300\text{ km/s}$), which is $\sim 10^3$ times deeper. However, the galactic interior sits at or above $\rho_T$, so $\mathcal{S}_\Sigma$ suppresses the coupling by a comparable or larger factor. The product $\beta_{\rm eff}^2 \times |\Phi|/c^2$ is therefore *not* monotonically increasing with mass; it peaks in the unscreened intermediate-density regime and is suppressed in dense environments.

This is the same screening structure that ensures solar-system compliance: the Cassini-bound PPN value $\beta \simeq -0.013$ is the screened descendant of $\beta_A = -1.0$ at solar-system densities ($\rho_\odot \sim 10^2\text{ g/cm}^3 \gg \rho_T$). The suppression factor $\mathcal{S}_\Sigma \sim 0.013$ at solar-system densities is consistent with the galactic-scale suppression. The environmental operator thus provides a *continuous, density-dependent* modulation that bridges the cloud-to-galaxy scaling without a discontinuous parameter change.

The empirical response coefficients $\kappa_{\rm Cep}$ and $\kappa_{\rm gal}$ used in the distance-ladder sector [51] are not free parameters but are macroscopic projections of $\mathcal{S}_\Sigma(\mathcal{E})$ evaluated at galactic-scale densities. They encode the average screening along the line of sight through a galaxy, not a separate tuning. The first-principles derivation of these coefficients from $\mathcal{S}_\Sigma$ — the same calculation that must yield $\mathcal{A}_{\rm env} \approx 10^4$ in the absorber regime (Gate 4B) — is the single open calculation that closes both the amplitude and the scaling simultaneously. If the environmental operator produces the correct $\mathcal{A}_{\rm env}$ for the absorber, it must simultaneously produce galactic-scale coefficients consistent with rotation curves, because both are projections of the same operator at different density regimes.

**Scaling-gap status:**

- **Sign: regime-independent.** The blueward sign follows from $\beta_{\rm eff}^2 > 0$ and the core-edge convention at *all* density regimes — it does not change between clouds and galaxies.

- **Amplitude: density-modulated, not mass-linear.** The product $\beta_{\rm eff}^2 \times |\Phi|/c^2$ is suppressed in dense environments by $\mathcal{S}_\Sigma$, preventing the galactic effect from scaling linearly with the $\sim 10^3$-fold deeper potential.

- **Galactic rotation curves: not broken.** The screened coupling at galactic densities ($\beta_{\rm eff} \ll \beta_A$) ensures that the scalar contribution to galactic dynamics is at or below the PPN-constrained level, consistent with observed rotation curves. The unscreened fifth force operates only in diffuse environments where it does not compete with galactic dynamics.

- **Single open calculation.** The first-principles evaluation of $\mathcal{S}_\Sigma(\mathcal{E})$ must simultaneously produce $\mathcal{A}_{\rm env} \approx 10^4$ (absorber) and $\kappa_{\rm Cep}, \kappa_{\rm gal}$ (galaxy). This is one calculation, not two independent calibrations. If it fails at either scale, the framework is falsified.

## 4. Cosmological Transport and Thermodynamics

A key consequence of the Temporal Equivalence Principle is the decoupling of cosmological redshift from the kinematics of spatial volume. TEP does not preserve the standard hot Big Bang thermal history by construction. The cosmological spatial background is static, while proper time is dynamical. Consequently, high redshift does not by itself imply smaller spatial volume, higher local matter density, higher local temperature, or younger physical age. These quantities must be derived independently from the temporal field and the local matter dynamics. The standard thermal history, including recombination [23], [24], [25] and the CMB [19], [20], is not assumed but must be independently reconstructed within the TEP framework.

### 4.1 Cosmological Parameter Definitions

To reconstruct the thermodynamic history of the universe, it is necessary to rigorously disambiguate the role of the temporal coupling. The fundamental conformal coupling is $A(\phi)$. In the cosmological limit, physical space is represented by the static matter-frame choice $a_{\rm m}=1$. The observed effective scale factor is an observational reconstruction:

\begin{equation} a_{\rm eff} = A_{\rm clock} a_{\rm m} = A_{\rm clock}, \qquad A_{\rm clock}(z) = \frac{1}{1+z}, \end{equation}

where $A_{\rm clock}$ serves as the exact observer/emitter clock map. $A_{\rm dyn}$ denotes the dynamical temporal-field response derived from the field equations; it is no longer constrained or screened by construction to reproduce the standard hot-BBN thermal history. Cosmological redshift is fundamentally decomposed into a homogeneous exact-conformal limit and a non-integrable path-dependent sector:

\begin{equation} \ln(1+z_T) = \int_\gamma \left( \Sigma_\parallel + \mathcal{C}_{T,\parallel} \right)d\ell, \qquad \Sigma_\mu = \nabla_\mu \ln A. \end{equation}

The exact conformal term ($\Sigma_\parallel$) dictates the primary global redshift map, yielding zero closed-loop holonomy. The genuine temporal path dependence belongs to the disformal non-exact sector ($\mathcal{C}_{T,\parallel}$).

### 4.2 Observable Dictionary in Dynamic Proper Time

A static spatial geometry demands a re-derivation of the standard cosmological observable dictionary directly from the temporal transport law. The corresponding observed time dilation and photon energy are:

\begin{equation} \Delta t_{\rm obs} = (1+z)\Delta\tau_{\rm em}, \qquad E_{\rm obs} = \frac{E_{\rm em}}{1+z}. \end{equation}

Because phase-space occupation is conserved along the photon trajectory, an emitted blackbody spectrum $f_{\rm em}(\nu) = \{\exp[h\nu/(k_BT_{\rm em})]-1\}^{-1}$ undergoes the transformation $f_{\rm obs}(\nu_{\rm obs}) = f_{\rm em}[(1+z)\nu_{\rm obs}]$. This strictly preserves the Planck form for the observer:

\begin{equation} f_{\rm obs} = \frac{1}{\exp[h\nu_{\rm obs}/(k_B T_{\rm obs})]-1}, \end{equation}

with the observed temperature mathematically determined by:

\begin{equation} T_{\rm obs}(z) = \frac{T_{\rm em}}{1+z}. \end{equation}

### 4.3 Native Local Thermodynamic Evolution

By assigning matter to be universally and minimally coupled to the causal effective metric $\tilde g_{\mu\nu}$, matter-frame conservation is enforced. However, matter-frame conservation ($\tilde\nabla_\mu \tilde{T}^{\mu\nu} = 0$) alone does not uniquely produce a universal temperature or density trajectory. Instead, thermodynamic closure requires the local equation of state, number currents ($\tilde\nabla_\mu N^\mu_a = \mathcal{S}_a$), and interactions. The TEP formulation is inherently local:

\begin{equation} \mathcal{H}_x = \{ T_{\rm loc}(\tau, x), n_{\rm loc}(\tau, x), \rho(\tau, x), \phi(\tau, x) \}. \end{equation}

Nuclear production occurs along these specific local matter histories:

\begin{equation} \frac{dY_i(x)}{d\tau} = \sum_r N_{ir} \lambda_r[T_{\rm loc}(\tau, x), n_{\rm loc}(\tau, x)] \prod_j Y_j^{\nu_{jr}}. \end{equation}

The observed abundance distribution therefore constrains the population of physical histories, rather than secretly recreating a single cosmic thermal trajectory.

Similarly, it cannot be assumed in advance that the Cosmic Microwave Background originates from a single global recombination epoch. The fundamental question is: *What local emission and scattering history, after temporal transport, generates the observed CMB?* Standard recombination becomes one hypothesis to test against the local thermodynamic state, not an imposed architecture.

**CMB Spectral Preservation (Symbolic Proof):**

A rigorous symbolic proof demonstrates that any emitted Planck spectrum $B_\nu(T_{\rm em})$ is strictly preserved in form under temporal transport. Because phase-space occupation $I_\nu/\nu^3$ is conserved along photon geodesics, and the conformal coupling $A(\phi)$ rescales frequency as $\nu_{\rm em} = \nu_{\rm obs}(1+z)$, the observed intensity $I_{\rm obs} = B_{\rm em}(\nu_{\rm obs}(1+z)) \cdot (\nu_{\rm obs}/\nu_{\rm em})^3$ algebraically reduces to a perfect Planck spectrum at the observed temperature $T_{\rm obs} = T_{\rm em}/(1+z)$. This property is a mathematical identity of conformal transport that holds equally in standard FLRW cosmology; its role here is to confirm that the TEP framework does not violate this basic requirement. The result ensures that temporal transport preserves the Planckian form without requiring a singular, universally dense early phase or FLRW geometric expansion. The observed CMB temperature $T_0 = 2.725$ K is therefore consistent with emission at any higher local temperature redshifted by the temporal transport factor.

Furthermore, in the TEP static spatial geometry, the line-of-sight Thomson optical depth integral diverges at high redshift for any non-vanishing electron density (Step 07), providing an observable boundary without requiring a physical plasma wall. The proof requires two steps: (i) in a static spatial geometry without FLRW dilution, the electron density $n_e$ remains constant along the path, so the optical depth grows as $\tau \propto \ell$; and (ii) the comoving path length to the temporal horizon diverges, $\ell(z) \to \infty$ as $z \to \infty$, because the conformal coordinate $\eta \to \infty$ at the horizon ($A_{\rm clock} \to 0$) and the spatial metric is static ($ds^2 = dr^2$). The combination of constant $n_e$ and divergent $\ell$ yields $\tau \to \infty$. This is a direct geometric consequence of the static spatial assumption: without FLRW spatial dilution of the electron density, the accumulated optical depth along an unbounded path grows without limit. The proof formalizes this consequence of the static frame rather than revealing a non-trivial dynamical result. While this establishes an opacity boundary, the structural morphology of the spatial fluctuation angular power spectra ($C_\ell$) and acoustic peaks falls under the domain of TEP linear perturbation theory [14], [15]. The full covariant perturbation closure requires the integration of the disformal non-exact sector ($\mathcal{C}_{T,\parallel}$) using Boltzmann codes such as CLASS [28] and hi_class [29], and is treated as a distinct analytical framework detailed in [3]. Standard CMB anisotropy computations [26], [27] provide the benchmark against which TEP perturbation predictions must be compared.

### 4.4 The Asymptotic Temporal Horizon ($\mathscr{T}^-$)

The canonical TEP geometry remains $\tilde g_{\mu\nu}=A^2g_{\mu\nu}+B\nabla_\mu\phi\nabla_\nu\phi$. In the homogeneous cosmological projection, physical spatial expansion is absent ($a_{\rm m}=1$), while the observational conformal reconstruction is $a_{\rm eff}=A_{\rm clock}a_{\rm m}=A_{\rm clock}$. The limit $a_{\rm eff}\to0$ therefore represents vanishing relative clock transport, not contraction of the underlying matter-frame spatial geometry. The reconstructed photon-transport metric — an *observational reconstruction* describing photon propagation, not the physical matter metric $\tilde g_{\mu\nu}$ — is written:

\begin{equation} ds_{\rm rec}^2 = a_{\rm eff}^2(\eta) (-d\eta^2 + dr^2 + r^2 d\Omega^2) \, , \end{equation}

where $\eta$ is the temporal horizon coordinate [4] with $A_{\rm clock}(\eta) \sim \eta^{-p}$. The temporal horizon establishes the fundamental limit:

\begin{equation} A_{\rm clock} \to 0, \quad z \to \infty, \quad a_{\rm m} = 1, \quad \tau \to \infty, \quad \mathcal{K} \to 0. \end{equation}

At this boundary, ancient clocks simply appear to tick infinitely slowly relative to the observer, while all polynomial curvature invariants ($\mathcal{K}$) vanish. This limit is purely an observational, relativistic boundary. The analogy is a clock falling into a black hole: extreme observer-relative temporal separation does not imply a local breakdown of physics. The deep-past observer does not experience chemistry freezing. Their local clocks, reactions, stellar evolution, scattering, and nuclear processes continue according to their own proper time. The horizon describes the relation between their temporal frame and ours; it is an asymptotic temporal past boundary, not a physical "deep freeze" wall.

### 4.5 Proper-Time Asymptotic Regularity

It is not blindly assumed that the horizon has finite accumulated proper time. To determine whether the universe is physically eternal yet finite in observable processing age, the accumulated proper time $\Delta\tau$ toward the coordinate horizon is explicitly integrated. Employing the exact temporal-horizon solution where the conformal coordinate $\eta \to \infty$ and the observational clock rate behaves as $A_{\rm clock}(\eta) \sim \eta^{-p}$:

\begin{equation} \Delta\tau = \int_{\eta_0}^{\infty} A_{\rm clock}(\eta) d\eta = \int_{\eta_0}^{\infty} \eta^{-p} d\eta = \left[ \frac{\eta^{1-p}}{1-p} \right]_{\eta_0}^{\infty} \end{equation}

Mathematical convergence strictly requires $p > 1$. However, the rigorous curvature-regularity condition for the temporal horizon—ensuring that all polynomial curvature invariants vanish and null affine parameters diverge—restricts the physical exponent to the branch $0 < p \le 1/2$. Within this curvature-regular window, the integral diverges strongly ($\Delta\tau \to \infty$). Therefore, an infinite coordinate age maps directly to an infinite proper-time accumulation.

### 4.6 The Chemical Exposure Convergence Constraint

An eternal proper-time history does not automatically destroy all primordial gas. The relevant quantity is the fraction of matter that experiences stellar and nuclear processing. The accumulated stellar processing of the gas reservoir is given by the astration exposure:

\begin{equation} \mathcal{E}_{\rm astr} = \int \Gamma_\star(\tau) d\tau, \end{equation}

where $\Gamma_\star$ measures actual stellar processing rates. Can an eternal, static universe contain gas whose accumulated stellar and nuclear processing exposure remains small? Suppose the processing rate asymptotically scales as $\Gamma_\star(\eta) \sim \eta^{-q}$. Since $d\tau = A_{\rm clock} d\eta$ and $A_{\rm clock}(\eta) \sim \eta^{-p}$, the exposure evaluates to:

\begin{equation} \mathcal{E}_{\rm astr} \sim \int^{\infty} \eta^{-(p+q)} d\eta \end{equation}

Therefore, $\mathcal{E}_{\rm astr} < \infty \iff p+q > 1$. For the curvature-regular TEP branch ($0 < p \le 1/2$), this requires:

\begin{equation} q > 1 - p \end{equation}

This establishes the exact temporal-exposure convergence condition. Infinite age does not imply infinite processing. The temporal horizon can make the observed contribution of the infinite past asymptotically inaccessible in temporal transport, but it does not by itself make the local chemical exposure of a gas worldline finite.

**Conclusion:** The temporal horizon is not a physical freeze-out surface. Local clocks, chemistry, and stellar processes continue normally in every regular local frame. What changes is the temporal relation between distant epochs: as $A_{\rm clock}\to0$, increasingly ancient processes become infinitely separated from the present observer in temporal transport.

For any particular parcel of matter, however, its chemical history is determined by its accumulated local processing exposure, $\mathcal{E}_{\rm astr}=\int\Gamma_\star d\tau$. An eternal proper-time history is compatible with finite stellar processing only when the processing rate falls sufficiently rapidly toward the temporal past. For the regular horizon branch ($A_{\rm clock}\sim\eta^{-p}$), this condition is $p+q>1$, where $\Gamma_\star\sim\eta^{-q}$.

TEP therefore does not obtain pristine matter by freezing local physics. It converts the astration problem into a quantitative question about matter history: whether the temporal geometry and local evolution naturally produce bounded processing exposure despite an eternal universe.

### 4.7 Separation of Spatial and Temporal Shear

Finally, it is essential to distinguish the two distinct phenomenological manifestations of the scalar field $\phi$. The cosmological redshift is a global temporal transport mechanism between two distant clocks:

\begin{equation} 1+z = \frac{A_{\rm obs}}{A_{\rm em}}. \end{equation}

In contrast, the apparent deuterium feature (the blueward offset) is generated by a localized spatial shear (the TEP absorber field) within the gas cloud:

\begin{equation} \Delta v_T \simeq -c \frac{d\ln A}{d\phi}\Delta\phi. \end{equation}

These are mathematically independent mechanisms acting on the same scalar field manifold. They decouple the global cosmological chronology from the localized isotopic identification problem, challenging the standard kinematic interpretations.

### 4.8 Primordial Helium Synthesis via Baryonic Cycling

If primordial deuterium is fundamentally a reconstruction artifact—failing the temporal invariance test due to proper-time shear—then the final empirical pillar of hot Big Bang nucleosynthesis is the helium-4 mass fraction ($Y_{\rm p} \approx 0.245$ [21], [22]). Without a finite, hot, universally dense origin, the TEP framework must analytically prove that this abundance is produced by stellar nucleosynthesis over an unbounded temporal horizon.

Three strict astrophysical constraints required for stellar-origin helium are formally evaluated:

- **Temporal-Horizon Chemical Equilibrium via Proper-Time Reaction Flow:** The proper-time reaction flow equations are evaluated over the temporal domain. Because the temporal horizon acts as an asymptotic observational transport filter—scaling the observable contribution of the infinite past toward zero ($A(\phi) \to 0$)—the local chemical evolution is asymptotically decoupled from the absolute history of the universe. Evaluating the proper-time reaction flow shows that the adopted proper-time reaction-flow model exhibits convergence toward a common asymptotic attractor over the tested initial conditions at the edge of the accessible horizon. Whether the evaluation starts with $Y_0=0.00$ or an extremely dense $Y_0=0.80$, the reaction flow rapidly decays into the equilibrium attractor of $Y_{\rm eq} = 0.247$ at the present day ($\tau = 0$). It is important to recognize that this reaction flow represents a *local galactic patch* experiencing continuous star formation. The pristine global background observed at high redshift is not protected from local chemical accumulation by transport delay alone; rather, pristine absorbers correspond to gas worldlines whose accumulated processing exposure $\mathcal{E}_{\rm astr}$ remains bounded by the convergence condition $p + q > 1$ (Section 4.6). The temporal horizon ensures that such low-exposure worldlines are observationally accessible, but the chemical pristine state is a property of the worldline's local processing history, not of the transport filter.

- **Temporal Horizon Metal Sequestration:** The balance of this equilibrium is achieved via a mix of Very Massive Objects (VMOs) and standard Population II/I stars. Standard stars yield typical return fractions. However, VMOs—which dominate the early equilibrium—undergo extreme radiatively-driven winds that successfully eject their helium envelopes ($E_Y > 0$). Upon core collapse, rather than forming a spatial singularity, the core generates a TEP temporal horizon where the local clock rate $A(\phi) \to 0$ relative to the external interstellar medium.

- **Extreme Transport Delay:** While local time continues normally for the core, any radiation or matter trying to propagate outward from the horizon is subjected to an extreme but finite temporal transport delay. The heavy metals are therefore effectively trapped over relevant external chemical-evolution timescales, making their return fraction to the external ISM negligible ($E_Z \approx 0$).

These mechanics eliminate the need for a spatial singularity, replacing it with a field-theoretic mechanism for chemical evolution. Under the TEP baryonic-cycling and temporal-horizon exposure conditions, the helium-4 mass fraction $Y_{\rm eq} = 0.247$ emerges as the equilibrium of the baryonic-cycling reaction flow under the adopted stellar yields and temporal-horizon metal sequestration. The reaction flow converges to this equilibrium from any initial condition, demonstrating that the observed $Y_{\rm p} \approx 0.245$ is compatible with stellar nucleosynthesis over an unbounded temporal horizon. The specific equilibrium value is set by the adopted yield parameters ($p_Y$, $R$) and the VMO fraction; the temporal-horizon metal sequestration ($E_Z \to 0$) ensures that the equilibrium is helium-dominated rather than metal-enriched. The yield parameters used here are drawn from the stellar-nucleosynthesis literature rather than independently constrained within the TEP framework; a complete demonstration requires showing that $Y_{\rm eq} \approx 0.247$ is robust to physically motivated variations in ($p_Y$, $R$, $f_{\rm VMO}$), or that these parameters are independently constrained by observations other than the primordial helium abundance. This is a refinement target for the next iteration.

## 5. Discussion and Falsifiable Predictions

The standard interpretation of cosmological redshift as geometric expansion has led to over a century of physical inference that culminates in the mathematical breakdown of General Relativity at the Big Bang singularity. Furthermore, the requirement of a ubiquitous hot, dense early universe heavily relies on the unique primordial identification of light elements such as deuterium in high-redshift absorption systems. Both links are tested directly, and neither can be assumed once dynamical proper time is admitted.

### 5.1 Distance Duality and Cosmological Tests

#### Distance Duality and Supernova Standardization

Critically, $T_{\rm obs}(z) \neq T_{\rm loc}(\tau)$ in general. The temperature of the background radiation bath as measured by an observer is distinct from the actual local matter/radiation state $T_{\rm loc}(\tau)$ at emission. Furthermore, because physical space is static, the standard geometric distances must be carefully defined. Etherington's reciprocity theorem is a general result of metric photon propagation, not specific to expanding FLRW; it dictates the distance-duality relation $d_L = d_A (1+z)^2$ for any metric theory where photon geodesics are well-defined and photon number is conserved. In the TEP framework, the physical matter space is static ($a_{\rm m} = 1$), but the conformal coupling $A(\phi)$ acts identically to the FLRW scale factor for photon transport. Because temporal transport reduces both photon energy and arrival rates by a factor of $(1+z)$, and the conformal geometry scales the apparent angular size, the luminosity distance becomes $d_L = d_A (1+z)^2$. The TEP framework therefore preserves the Etherington relation by construction, not because it replicates FLRW expansion, but because the conformal transport law satisfies the same general conditions.

While the baseline distance-duality relation is preserved, it is important to recognize that SNIa magnitudes are not raw observables. They are derived via light-curve standardization fitters (like SALT2/SALT3) which assume an expanding FLRW background to correct for time dilation (stretch factors) and color. Any departures from the baseline conformal distance law arise from the non-exact disformal transport sector and from the observational standardization procedure. Therefore, this provides a concrete, falsifiable observational roadmap that requires re-calibrating the light-curve fitters within the TEP geometry rather than relying on FLRW-calibrated nuisance parameters.

### 5.2 Synchronization Holonomy and Optical Time-Transfer

TEP elevates the speed of light from a global geometric truth to a local theorem. This provides falsifiable physical predictions. Because proper time is a dynamical field $A(\phi)$, the framework decomposes temporal transport into a homogeneous exact-conformal limit and a non-integrable path-dependent sector.

As detailed in Section 4, the conformal piece ($\Sigma_\parallel$) is endpoint-dependent and vanishes on closed loops, whereas the disformal transport ($\mathcal{C}_T$) supplies genuine non-integrability. Multi-leg optical time-transfer experiments—currently within reach of next-generation atomic clock networks—can directly test for this synchronization holonomy ($\oint \mathcal{C}_{T,\parallel} d\ell \neq 0$).

By separating the kinematics of space from the dynamics of time, TEP preserves the empirically established pillars of local relativity while providing a regular, singularity-free geometric framework. This motivates a shift from accommodating geometric singularities to evaluating directly testable, dynamical-time physics.

## 6. Conclusion

Through a systematic, algorithmically controlled re-analysis of the benchmark Q1009+2956 absorption system, it is demonstrated that the Q1009+2956 spectrum does not uniquely secure the canonical deuterium identification against the ordinary-H alternative. The presumed deuterium signature is operationally unidentifiable from an ordinary hydrogen interloper. Subjected to nested hypothesis testing, an unrestricted hydrogen model fits the spectroscopic data with a substantial likelihood improvement ($\Delta \ln L = 34.23$, $T=68.46$), with Monte Carlo calibration yielding $p_{\rm add-one} \approx 0.005$ for both the standard statistic and the parent-reassignment statistic. The discrimination is concentrated in the Ly$\alpha$ transition ($T_{-\mathrm{Ly}\alpha} = 2.22$), indicating that current benchmark spectra cannot reliably exclude ordinary hydrogen kinematics. Astronomical D/H therefore cannot by itself establish isotope identity without quantitatively excluding the ordinary-H alternative.

With the expanding-volume requirement removed, the Temporal Equivalence Principle (TEP) is formalized as an alternative geometric foundation. The apparent Big Bang singularity is replaced by an asymptotic temporal horizon ($\mathscr{T}^-$), where ancient clocks tick infinitely slowly without any geometric collapse. It is shown that local thermal processing and chemical evolution can remain bounded by finite exposure measures ($\mathcal{E}_{\rm astr} < \infty$) when the derived temporal-exposure convergence condition is satisfied, and that the helium-4 mass fraction ($Y_{\rm eq} = 0.247$) emerges as the equilibrium of baryonic-cycling reaction flows under temporal-horizon metal sequestration. The TEP absorber field deterministically predicts the blueward sign of the deuterium-like velocity shift from conformal core-edge geometry; the full amplitude is quantified as a falsifiable prediction of the environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$. Finally, in the static spatial geometry, the line-of-sight optical depth is shown to diverge at high redshift, creating an observable boundary without a physical plasma wall. These results resolve the classical astration paradox in infinite proper time, decoupling physical chemical evolution from an eternal coordinate manifold without invoking an explosive spatial origin.

## Data Availability & Reproducibility

This work follows open-science practices. All results are fully reproducible from raw data
using the documented pipeline. All numerical results, Monte Carlo simulations, and statistics are generated by deterministic
Python scripts processing real observational data. The pipeline enforces rigorous reproducibility: any failure in statistical criteria is treated as an explicit rejection of the theory.

### Repository and Code

GitHub Repository: github.com/matthewsmawfield/TEP-BBN

The repository contains a deterministic, version-controlled cosmological analysis pipeline utilizing 7 core analysis steps for spectroscopic embedding, baseline system fitting, Monte Carlo significance testing, absorber field closure, temporal horizon thermodynamics, primordial helium synthesis, and global opacity boundary analysis.
All steps are orchestrated by `scripts/run_pipeline.py` with comprehensive per-step logging.

All raw VLT/UVES spectra (SQUAD DR1), structural likelihood matrices, and the temporal-field equation solvers are released in the Zenodo repository (DOI: 10.5281/zenodo.21841148) under CC-BY 4.0. The full codebase and execution environments are identical to the published version.

#### Repository Structure

TEP-BBN/
├── data/
│   ├── raw/                       # VLT/UVES spectroscopic exposures (SQUAD DR1)
│   │   ├── atomic/                # Physical H I, D I line registries (NIST ASD)
│   │   └── reduced_products/      # Pre-reduced and co-added normalized spectra
│   ├── literature_components/     # Published VPFIT model files and component tables
│   └── processed/                 # Pipeline-ready union manifests
├── scripts/
│   ├── steps/                     # 7 deterministic pipeline steps (01-07)
│   ├── lib/                       # Physical RT engine, Voigt fitters, model parsers
│   └── run_pipeline.py            # Master orchestration script
├── configs/
│   ├── sightlines/                # Per-sightline JSON configs (bounds, MC settings)
│   ├── tep_priors.yaml            # Frozen TEP prior parameters
│   └── tep_noise_model.json       # Student-t noise profile
├── results/                       # Generated parameter ledgers and significance matrices
├── logs/                          # Per-step execution logs
├── site/
│   └── components/                # Manuscript source components
├── requirements-lock.txt          # Locked Python dependencies
└── README.md                      # Documentation

### Data Provenance

| Data Source | Provider | Access Method | Records | Location |
| --- | --- | --- | --- | --- |
| Q1009+2956 Spectra | VLT/UVES (SQUAD DR1) | Pre-reduced | 4 coadds (median SNR $\approx 24$) | `data/raw/reduced_products/Q1009+2956_z2.504_UVES/` |
| Atomic Data | NIST ASD [49] | Static Registry | H I, D I, metals | `data/raw/atomic/` |
| Literature Model | Zavarygin et al. [44] / VPFIT [50] | Static File | 43 H I + 3 D I components | `data/literature_components/model_6a.26` |
| Component Table | Zavarygin et al. [44] | Static File | 2 velocity components | `data/literature_components/Q1009+2956_z2.504_component_table.csv` |
| Prior Bounds | Derived | Static File | All variables | `configs/tep_priors.yaml` |

### Pipeline Architecture

The analysis pipeline comprises 7 deterministic steps spanning spectroscopic ingestion to thermodynamic evaluation, helium synthesis, and opacity boundary analysis.
Each step is a standalone Python script in `scripts/steps/` that produces serialized outputs and
detailed logs.

#### Complete Step Inventory and Runtime

Runtimes are approximate and measured on Apple M4 Pro (14-core, 24 GB). The dominant cost is the Monte Carlo significance test (step 03), which scales with iterations.

| Step | Script | Description | Est. Runtime |
| --- | --- | --- | --- |
| 01 | `step_01_embedding.py` | Verifies fundamental isochrony axiom vulnerability (H vs D embedding) | ~5 s |
| 02 | `step_02_q1009.py` | Baseline structural fit of the Q1009+2956 absorption complex | ~15 s |
| 03 | `step_03_significance.py` | 200-realization Monte Carlo significance test and true-D injection | ~20 min |
| 04 | `step_04_prior.py` | Symbolic proof of the TEP absorber field blueward velocity sign ($\Delta v_T < 0$) | ~2 s |
| 05 | `step_05_thermodynamics.py` | Symbolic proof of Planck spectrum preservation under temporal transport | ~1 s |
| 06 | `step_06_helium.py` | Primordial helium synthesis via baryonic cycling and temporal-horizon metal sequestration | ~2 s |
| 07 | `step_07_global_opacity.py` | Analytical proof of divergent optical depth in the static spatial geometry (Opacity Boundary) | ~1 s |

#### Total Runtime Summary

| Component | Steps | Runtime |
| --- | --- | --- |
| All Analysis Stages | 7 | ~20 min |
| Total | 7 | ~20 min |

### Reproduction Instructions

#### Quick Start (Full Reproduction)

# 1. Clone repository
git clone https://github.com/matthewsmawfield/TEP-BBN.git
cd TEP-BBN

# 2. Install dependencies
pip install -r requirements-lock.txt

# 3. Run full pipeline (default: Q1009+2956 sightline)
python scripts/run_pipeline.py

# 3b. Run a specific sightline
python scripts/run_pipeline.py --sightline PKS1937-1009

# 3c. Run all sightlines with ingested data
python scripts/run_pipeline.py --all-sightlines

# 4. Results will be stored in results/ and logs/

#### Multi-Sightline Configuration

The pipeline supports multiple D/H absorber sightlines through per-sightline JSON configuration files in `configs/sightlines/`.
Each config specifies the data manifest path, VPFIT model file, noise model, absorber redshift, candidate parameter bounds, multi-start initial points, and Monte Carlo settings.
The Q1009+2956 config reproduces the exact hardcoded values from the original pipeline, ensuring bit-identical backward compatibility.
Sightlines without ingested data (PKS 1937$-$1009, HS 0105+1619) are automatically skipped with a diagnostic message.

#### System Requirements

| Component | Minimum | Recommended | Tested On |
| --- | --- | --- | --- |
| CPU | 2 cores | 4+ cores | Apple M4 Pro (14-core) |
| RAM | 4 GB | 8 GB | 24 GB |
| Storage | 1 GB | 2 GB | SSD NVMe |
| OS | Linux/macOS | Linux/macOS | macOS Sequoia 15.1 |

## References

- Smawfield, M.L. Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed. *Zenodo* (2025). DOI: 10.5281/zenodo.16921911

- Smawfield, M.L. Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion. *Zenodo* (2026). DOI: 10.5281/zenodo.20370143

- Smawfield, M.L. Temporal Equivalence Principle: Native hi_class Conformal Implementation, Linear Perturbation Closure, and CMB Acoustic Peak Preservation. *Zenodo* (2026). DOI: 10.5281/zenodo.20572722

- Smawfield, M.L. Temporal Equivalence Principle: Temporal Horizon Cosmology and the Absence of a Physical Big Bang Singularity. *Zenodo* (2026). DOI: 10.5281/zenodo.20723059

- Hawking, S.W. The occurrence of singularities in cosmology. *Proc. R. Soc. A* **294**, 511-521 (1966).

- Hawking, S.W. & Penrose, R. The singularities of gravitational collapse and cosmology. *Proc. R. Soc. A* **314**, 529-548 (1970).

- Borde, A., Guth, A.H. & Vilenkin, A. Inflationary spacetimes are incomplete in past directions. *Phys. Rev. Lett.* **90**, 151301 (2003).

- Brandenberger, R. & Peter, P. Bouncing cosmologies: progress and problems. *Found. Phys.* **47**, 797-850 (2017).

- Novello, M. & Bergliaffa, S.E.P. Bouncing cosmologies. *Phys. Rep.* **463**, 127-213 (2008).

- Ijjas, A. & Steinhardt, P.J. Entropy, black holes and the new cyclic universe. *Phys. Lett. B* **824**, 136823 (2022).

- Peebles, P.J.E. *Principles of Physical Cosmology*. Princeton University Press (1993).

- Weinberg, S. *Cosmology*. Oxford University Press (2008).

- Dodelson, S. *Modern Cosmology*. Academic Press (2003).

- Mukhanov, V.F., Feldman, H.A. & Brandenberger, R.H. Theory of cosmological perturbations. *Phys. Rep.* **215**, 203-333 (1992).

- Liddle, A.R. & Lyth, D.H. *Cosmological Inflation and Large-Scale Structure*. Cambridge University Press (2000).

- Planck Collaboration, et al. Planck 2018 results. VI. Cosmological parameters. *A&A* **641**, A6 (2020).

- Riess, A.G., et al. Milky Way Cepheid Standards for Measuring Cosmic Distances and Application to Gaia DR2: Implications for the Hubble Constant. *ApJ* **861**, 126 (2018).

- Brout, D., et al. The Pantheon+ Analysis: Cosmological Constraints. *ApJ* **938**, 110 (2022).

- Fixsen, D.J., et al. The Cosmic Microwave Background Spectrum from the Full COBE FIRAS Data Set. *ApJ* **473**, 576 (1996).

- Chluba, J. & Sunyaev, R.A. The evolution of CMB spectral distortions in the early Universe. *MNRAS* **419**, 1294-1314 (2012).

- PARTICLE DATA GROUP. Review of Particle Physics. *PTEP* **2022**, 083C01 (2022).

- Cyburt, R.H., Fields, B.D., Olive, K.A. & Yeh, T.H. Big bang nucleosynthesis: Present status. *Rev. Mod. Phys.* **88**, 015004 (2016).

- Seager, S., Sasselov, D.D. & Scott, D. A new calculation of the recombination epoch. *ApJ* **523**, L1-L5 (1999).

- Peebles, P.J.E. Recombination of the Primeval Plasma. *ApJ* **153**, 1 (1968).

- Zeldovich, Y.B. & Sunyaev, R.A. The interaction of matter and radiation in a hot-model universe. *Astrophys. Space Sci.* **4**, 301-316 (1969).

- Seljak, U. & Zaldarriaga, M. A Line of Sight Integration Approach to Cosmic Microwave Background Anisotropies. *ApJ* **469**, 437 (1996).

- Lewis, A., Challinor, A., & Lasenby, A. Efficient Computation of CMB Anisotropies in Closed FRW Models. *ApJ* **538**, 473 (2000).

- Lesgourgues, J. & Tram, T. The Cosmic Linear Anisotropy Solving System (CLASS). Part IV: efficient implementation of non-cold relics. *JCAP* **09**, 032 (2011).

- Zumalacárregui, M., Bellini, E., Sawicki, I., Lesgourgues, J. & Ferreira, P.G. hi_class: Horndeski in the Cosmic Linear Anisotropy Solving System. *JCAP* **08**, 019 (2017).

- De Felice, A. & Tsujikawa, S. f(R) Theories. *Living Rev. Rel.* **13**, 3 (2010).

- Wetterich, C. Cosmology and the fate of dilatation symmetry. *Nucl. Phys. B* **302**, 668-696 (1988).

- Wetterich, C. A universe without expansion. *Phys. Dark Universe* **2**, 184 (2013).

- Narlikar, J.V. & Arp, H.C. Flat spacetime cosmology: A unified framework for extragalactic redshifts. *Astrophys. J.* **405**, 51-56 (1993).

- Mannheim, P.D. Conformal gravity and the nature of dark matter. *Prog. Part. Nucl. Phys.* **94**, 217-272 (2017).

- Khoury, J. & Weltman, A. Chameleon cosmology. *Phys. Rev. D* **69**, 044026 (2004).

- Hinterbichler, K. & Khoury, J. Symmetron cosmology. *Phys. Rev. Lett.* **104**, 231301 (2010).

- Penrose, R. Before the Big Bang: an outrageous new perspective and its implications for particle physics. *Proc. EPAC* (2006).

- Tod, K.P. Isotropic cosmological singularities. *Gen. Relativ. Gravit.* **35**, 779-805 (2003).

- Tod, K.P. The equations of conformal cyclic cosmology. *Gen. Relativ. Gravit.* **47**, 31 (2015).

- Ratra, B. & Peebles, P.J.E. Cosmological Consequences of a Rolling Homogeneous Scalar Field. *Phys. Rev. D* **37**, 3406 (1988).

- Caldwell, R.R., Dave, R., & Steinhardt, P.J. Cosmological Imprint of an Energy Component with General Equation of State. *Phys. Rev. Lett.* **80**, 1582 (1998).

- Clifton, T., Ferreira, P.G., Padilla, A. & Skordis, C. Modified gravity and cosmology. *Phys. Rep.* **513**, 1-189 (2012).

- Murphy, M.T., Kacprzak, G.G., Savorgnan, G.A.D. & Carswell, R.F. The UVES Spectral Quasar Absorption Database (SQUAD) data release 1: the first 10 million seconds. *MNRAS* **482**, 3458-3482 (2019). arXiv:1810.06136

- Zavarygin, E., Webb, J.K., Dumont, V. & Riemer-Sørensen, S. The primordial deuterium abundance at z<sub>abs</sub> = 2.504 from a high signal-to-noise spectrum of Q1009+2956. *MNRAS* **477**, 5536-5553 (2018). arXiv:1706.09512

- Cooke, R.J., Pettini, M., Jorgenson, R.A., Murphy, M.T. & Steidel, C.C. Precision measures of the primordial abundance of deuterium. *ApJ* **781**, 31 (2014).

- Burles, S. & Tytler, D. The Deuterium Abundance toward QSO 1009+2956. *ApJ* **507**, 732 (1998).

- O'Meara, J.M., Burles, S., Prochaska, J.X., Prochter, G.E., Bernstein, R.A. & Burgess, K.M. The Deuterium-to-Hydrogen Abundance Ratio toward the QSO SDSS J155810.16-003120.0. *ApJ* **649**, L61 (2006).

- Kirkman, D., Tytler, D., Burles, S., Lubin, D. & O'Meara, J.M. On the primordial deuterium abundance. *ApJ* **529**, 655 (2000).

- Kramida, A., Ralchenko, Yu., Reader, J. & NIST ASD Team. NIST Atomic Spectra Database (ver. 5.11). National Institute of Standards and Technology, Gaithersburg, MD (2023).

- Carswell, R.F. & Webb, J.K. VPFIT: Voigt profile fitting program. Astrophysics Source Code Library, ascl:1408.015 (2014).

- Smawfield, M.L. The Cepheid Bias: Resolving the Hubble Tension. *Zenodo* (2026). DOI: 10.5281/zenodo.18209702

- Smawfield, M.L. Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies. *Zenodo* (2026). DOI: 10.5281/zenodo.19000827

- Smawfield, M.L. Temporal Equivalence Principle: Temporal Shear in the Earth Flyby Anomaly. *Zenodo* (2026). DOI: 10.5281/zenodo.19454863

- Smawfield, M.L. Temporal Equivalence Principle: Black Holes and the Temporal Horizon. *Zenodo* (2026). DOI: 10.5281/zenodo.21677826

- Haardt, F. & Madau, P. Radiative transfer in a clumpy universe. IV. New synthesis models of the UV/X-ray cosmic background. *ApJ* **746**, 125 (2012).

- Benisty, D., Brax, P. & Davis, A.C. Constraining modified gravity with cosmological data. *Phys. Rev. D* **107**, 064049 (2023).
