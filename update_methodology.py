
html_content = r"""<div class="methodology">
    <h2>2. Data and Methods</h2>

    <h3>2.1 Data</h3>

    <h4>2.1.1 Red Monsters (FRESCO)</h4>
    <p>The motivating case study is the set of three spectroscopically confirmed ultra-massive galaxies reported by Xiao et al. (2024). The published stellar masses and inferred efficiencies are treated as the observational inferences to be reinterpreted under TEP, rather than attempting an independent re-fit of the original photometry.</p>

    <h4>2.1.2 UNCOVER DR4</h4>
    <p>For population-level tests, the UNCOVER DR4 stellar population synthesis catalog is used (Wang et al. 2024). The pipeline applies quality cuts and constructs a high-redshift sample with $4 < z < 10$ and $\log(M_*/M_\odot) > 8$, yielding $N = 2{,}315$ galaxies. For multi-property analyses (age ratio, metallicity, dust), a subset with complete measurements is used (e.g., $N = 1{,}108$ for the partial-correlation and split tests).</p>

    <h4>2.1.3 Independent replications and spectroscopic validation</h4>
    <p>To evaluate independent replication of the $z > 8$ dust result, catalogs for CEERS are used (Cox et al. 2025; Finkelstein et al. 2023) and COSMOS-Web (Shuntov et al. 2025). For spectroscopic validation of key correlations, the combined UNCOVER+JADES spectroscopic sample is used at $z > 4$ (N = 147). The UNCOVER DR4 zspec subset with quality flag $\ge 2$ (N = 122) forms the core of this compilation.</p>

    <h4>2.1.4 MIRI-based mass calibration context</h4>
    <p>Recent JWST/MIRI analyses (Pérez-González et al. 2024) show that NIRCam-only SED fits can overestimate stellar masses at $z > 5$ because of age-attenuation degeneracy and emission-line contamination. When MIRI photometry is included, the number density of the most massive systems decreases and some candidates are reclassified as dusty or line-dominated sources. The photometry is not reprocessed in this work, but published masses are treated as conservative upper bounds and MIRI-based studies serve as an external check on the interpretation of the extreme-mass tail.</p>

    <div class="data-table">
        <table>
            <caption>Table 1: Observational Datasets</caption>
            <thead>
                <tr><th>Dataset</th><th>Role</th><th>Sample Size</th><th>Redshift Range</th><th>Mass Cut ($\log M_*$)</th><th>Key Reference</th></tr>
            </thead>
            <tbody>
                <tr><td>Red Monsters</td><td>Case Study</td><td>3</td><td>$5.3 < z < 5.9$</td><td>$> 10.5$</td><td>Xiao et al. (2024)</td></tr>
                <tr><td>UNCOVER DR4</td><td>Primary Statistical Sample</td><td>2,315</td><td>$4 < z < 10$</td><td>$> 8.0$</td><td>Wang et al. (2024)</td></tr>
                <tr><td>CEERS DR1</td><td>Independent Replication</td><td>82</td><td>$z > 8$</td><td>$> 8.0$</td><td>Cox et al. (2025)</td></tr>
                <tr><td>COSMOS-Web</td><td>Large-Volume Check</td><td>918</td><td>$z > 8$</td><td>$> 8.0$</td><td>Shuntov et al. (2025)</td></tr>
                <tr><td>JADES/UNCOVER</td><td>Spectroscopic Validation</td><td>147</td><td>$z > 4$</td><td>$> 7.5$</td><td>Eisenstein+23; Price+24</td></tr>
            </tbody>
        </table>
    </div>

    <p>Related MIRI-supported analyses of Little Red Dots (LRDs) at $z > 4$ find that inferred stellar masses can shift by up to orders of magnitude depending on the assumed AGN contribution. This motivates a conservative stance in the interpretation of compact red sources and provides a systematic-control context for any extreme-mass claims in the literature.</p>

    <h3>2.2 Key Terminology</h3>
    <p>The following terms are used consistently throughout this work:</p>

    <div class="data-table">
        <table>
            <caption>Table 1b: Glossary of Key Terms</caption>
            <thead>
                <tr><th>Term</th><th>Symbol</th><th>Definition</th></tr>
            </thead>
            <tbody>
                <tr><td><strong>Temporal Enhancement Factor</strong></td><td>$\Gamma_t$</td><td>The ratio of effective proper time to cosmic time experienced by stellar populations in a given gravitational potential. $\Gamma_t > 1$ indicates enhanced time flow (deeper potentials); $\Gamma_t < 1$ indicates suppressed time flow (shallower potentials).</td></tr>
                <tr><td><strong>Temporal Shear</strong></td><td>—</td><td>The spatial gradient of $\Gamma_t$ across a galaxy or environment. Differential temporal shear refers to the difference in $\Gamma_t$ between two regions (e.g., galactic center vs. halo).</td></tr>
                <tr><td><strong>Isochrony Bias</strong></td><td>—</td><td>The systematic error in inferred stellar properties (mass, age, SFR) arising from the assumption that stellar clocks tick at the cosmic rate everywhere. Under TEP, this assumption is violated in deep potential wells.</td></tr>
                <tr><td><strong>Screening</strong></td><td>—</td><td>The suppression of TEP effects in regions where the local density exceeds a critical threshold ($\rho_c \approx 20$ g/cm³), restoring standard GR physics. Two types are distinguished:</td></tr>
                <tr><td style="padding-left:2em;"><em>Core Screening</em></td><td>—</td><td>Screening within a single galaxy, where the deep central potential suppresses TEP ($\Gamma_t \to 1$) while the outskirts remain enhanced. Produces bluer cores and redder outskirts.</td></tr>
                <tr><td style="padding-left:2em;"><em>Environmental Screening</em></td><td>—</td><td>Screening by the ambient group or cluster potential, causing galaxies in dense environments to appear younger than isolated field galaxies of the same mass.</td></tr>
                <tr><td><strong>Effective Time</strong></td><td>$t_{\rm eff}$</td><td>The proper time experienced by stellar populations: $t_{\rm eff} = t_{\rm cosmic} \times \Gamma_t$.</td></tr>
            </tbody>
        </table>
    </div>

    <h3>2.3 Derived quantities</h3>

    <h4>2.3.1 Halo mass proxy</h4>
    <p>For each galaxy, the pipeline uses an abundance-matching relation to map stellar mass to halo mass. This mapping is used solely to construct $\Delta\log(M_h)$ for the TEP parameterization.</p>

    <h4>2.3.2 The TEP Metric Coupling</h4>
    <p>The temporal enhancement factor $\Gamma_t$ is not an ad-hoc fitting function but is derived from first principles within the scalar-tensor framework developed in Paper 1 (Smawfield 2024, <em>The Temporal Equivalence Principle: A Two-Metric Foundation</em>). The complete derivation, including stability analysis and cosmological constraints, is presented there; this section summarizes the key steps from action to observable.</p>

    <div class="callout callout-info">
        <h4>Derivation Summary</h4>
        <p>The $\Gamma_t$ formula follows a rigorous chain:</p>
        <ul>
            <li><strong>Action</strong>: Two-metric scalar-tensor theory with conformal coupling $A(\phi)$</li>
            <li><strong>Field Equations</strong>: Variation yields Klein-Gordon equation for $\phi$ sourced by matter</li>
            <li><strong>Screening Solution</strong>: In virialized halos, $\phi$ tracks potential depth $\Phi \propto M_h^{2/3}$</li>
            <li><strong>Proper Time</strong>: Clock rates scale as $d\tau/dt = A(\phi)^{1/2} = \exp(\beta\phi/M_{\rm Pl})$</li>
            <li><strong>Observable</strong>: $\Gamma_t = \exp[\alpha(z) \times \frac{2}{3} \times \Delta\log M_h \times (1+z)/(1+z_{\rm ref})]$</li>
        </ul>
        <p>Each step is detailed below. For the full treatment including ghost-freedom proofs and cosmological perturbation analysis, see Paper 1.</p>
    </div>

    <h5>2.3.2.1 The Two-Metric Action</h5>
    <p>The TEP framework posits a single spacetime manifold endowed with two metrics: a gravitational metric $g_{\mu\nu}$ (to which gravity couples) and a causal matter metric $\tilde{g}_{\mu\nu}$ (to which all non-gravitational fields couple). The metrics are related by a disformal map:</p>
    <div class="equation">
        $$\tilde{g}_{\mu\nu} = A(\phi) g_{\mu\nu} + B(\phi) \nabla_\mu\phi \nabla_\nu\phi$$
    </div>
    <p>where $\phi$ is the time field, $A(\phi) = \exp(2\beta\phi/M_{\rm Pl})$ is a universal conformal factor, and $B(\phi)$ encodes small disformal corrections bounded by multi-messenger constraints ($|c_\gamma - c_g|/c \lesssim 10^{-15}$ from GW170817). The full action in the Einstein frame is:</p>
    <div class="equation">
        $$S = \int d^4x \sqrt{-g} \left[ \frac{M_{\rm Pl}^2}{2} R - \frac{1}{2} K(\phi) g^{\mu\nu} \partial_\mu\phi \partial_\nu\phi - V(\phi) \right] + S_{\rm matter}[\psi, \tilde{g}_{\mu\nu}, \phi]$$
    </div>
    <p>where $R$ is the Ricci scalar, $K(\phi) > 0$ is the kinetic function, and $V(\phi)$ is the scalar potential. Matter fields $\psi$ couple universally to $\tilde{g}_{\mu\nu}$, ensuring the Weak Equivalence Principle holds in the matter frame.</p>

    <h5>2.3.2.2 The Scalar Field Equation</h5>
    <p>Varying the action with respect to $\phi$ yields the Klein-Gordon equation in curved spacetime:</p>
    <div class="equation">
        $$K(\phi) \Box\phi + \frac{1}{2} K'(\phi) (\partial\phi)^2 - V'(\phi) = -\frac{\beta}{M_{\rm Pl}} T^{\mu}_{\;\mu} A(\phi)$$
    </div>
    <p>where $T^{\mu}_{\;\mu}$ is the trace of the matter stress-energy tensor. For non-relativistic matter, $T^{\mu}_{\;\mu} \approx -\rho c^2$, giving a source term proportional to the local matter density. In the quasi-static limit relevant for virialized halos, this reduces to:</p>
    <div class="equation">
        $$\nabla^2\phi \approx \frac{\beta \rho A(\phi)}{K(\phi) M_{\rm Pl}} + \frac{V'(\phi)}{K(\phi)}$$
    </div>
    <p>The competition between the matter coupling (first term) and the bare potential (second term) determines the screening behavior. In dense regions, the matter term dominates, driving $\phi$ toward values where $A(\phi) \to 1$. In diffuse regions, the potential term dominates, allowing $\phi$ to relax to values where $A(\phi) > 1$.</p>

    <h5>2.3.2.3 Proper Time and the Enhancement Factor</h5>
    <p>A clock's proper time increment is $d\tau = \sqrt{-\tilde{g}_{\mu\nu} dx^\mu dx^\nu}/c$. For slow observers with small $\partial_0\phi$ compared to spatial gradients, $\tilde{g}_{00} \approx A(\phi) g_{00}$, yielding:</p>
    <div class="equation">
        $$\frac{d\tau}{dt} \approx A(\phi)^{1/2} = \exp\left(\frac{\beta\phi}{M_{\rm Pl}}\right)$$
    </div>
    <p>This rescaling does not alter local Lorentz invariance: in any freely falling lab for $\tilde{g}$, Minkowski physics holds and $c$ is invariant. What varies is the mapping between distant proper-time standards. The temporal enhancement factor $\Gamma_t$ is defined as the ratio of effective proper time to coordinate time relative to a reference environment:</p>
    <div class="equation">
        $$\Gamma_t \equiv \frac{d\tau/dt}{(d\tau/dt)_{\rm ref}} = \exp\left[\frac{\beta(\phi - \phi_{\rm ref})}{M_{\rm Pl}}\right]$$
    </div>

    <h5>2.3.2.4 Mapping to Observables</h5>
    <p>In screened scalar-tensor theories (chameleon, symmetron, or Vainshtein mechanisms), the scalar field $\phi$ tracks the local gravitational potential. For a virialized halo of mass $M_h$, the potential depth scales as $\Phi \sim GM_h/R_{\rm vir} \sim M_h^{2/3}$ at fixed overdensity. Parameterizing $\beta(\phi - \phi_{\rm ref})/M_{\rm Pl} = \alpha(z) \times \Phi/\Phi_{\rm ref}$ and using the virial scaling gives:</p>
    <div class="equation">
        $$\Gamma_t = \exp\left[\alpha(z) \times \frac{2}{3} \times \Delta\log(M_h) \times \frac{1+z}{1+z_{\rm ref}}\right]$$
    </div>
    <p>where $\Delta\log(M_h) = \log(M_h/M_{\rm ref})$ is the logarithmic halo mass excess relative to the reference mass ($\log M_{h, \rm ref} = 12.0$). The pre-factor $\frac{2}{3}$ arises from the virial scaling $\Phi \propto M^{2/3}$. The redshift factor $(1+z)/(1+z_{\rm ref})$ encodes the weakening of screening at high redshift due to lower cosmic density.</p>

    <h5>2.3.2.5 Connection to Brans-Dicke and Chameleon Theories</h5>
    <p>The TEP framework is a specific realization within the broader class of scalar-tensor theories. In Brans-Dicke theory, the coupling $\omega_{\rm BD}$ relates to the TEP parameter via $\alpha_0^2 \approx 1/(2\omega_{\rm BD} + 3)$. The measured $\alpha_0 = 0.58 \pm 0.16$ corresponds to $\omega_{\rm BD} \approx 0.5$, which would be ruled out by solar system tests without screening. The chameleon mechanism resolves this: the effective mass $m_\phi(\rho)$ grows with ambient density, suppressing the scalar force in dense environments (Earth, solar system) while allowing cosmological dynamics.</p>
    <p>The screening condition is:</p>
    <div class="equation">
        $$\rho > \rho_c \approx 20 \text{ g/cm}^3 \quad \Rightarrow \quad \Gamma_t \to 1 \text{ (screened)}$$
    </div>
    <p>This critical density, derived independently from pulsar timing in globular clusters (Paper 7, TEP-UCD), ensures that TEP effects are suppressed in the solar system while remaining active in galactic halos and the high-redshift universe.</p>

    <h5>2.3.2.6 The TEP Screening Mechanism</h5>
    <p>A natural objection is that standard GR predicts gravitational time dilation ($\Gamma_t < 1$). The TEP framework, as developed in Paper 1, requires a scalar-tensor theory where the conformal factor $A(\phi) > 1$ in regions of low ambient density. The specific Lagrangian achieving this while satisfying all Solar System constraints is detailed in Paper 1 (§4.3); the key elements are summarized here.</p>
    <p>Consider the action:</p>
    <div class="equation">
        $$S = \int d^4x \sqrt{-g} \left[ \frac{M_{\rm Pl}^2}{2} R - \frac{1}{2} (\partial\phi)^2 - V(\phi) \right] + S_{\rm matter}[\psi, A^2(\phi) g_{\mu\nu}]$$
    </div>
    <p>with the specific potential and coupling:</p>
    <div class="equation">
        $$V(\phi) = \Lambda^4 \left(1 + \frac{\Lambda}{\phi}\right), \quad A(\phi) = \exp\left(\frac{\beta \phi}{M_{\rm Pl}}\right)$$
    </div>
    <p>where $\Lambda \sim 10^{-3}$ eV is the dark energy scale and $\beta \sim 0.3$ is the coupling strength. The effective potential in the presence of matter density $\rho$ is:</p>
    <div class="equation">
        $$V_{\rm eff}(\phi) = V(\phi) + \rho A(\phi) = \Lambda^4 \left(1 + \frac{\Lambda}{\phi}\right) + \rho \exp\left(\frac{\beta \phi}{M_{\rm Pl}}\right)$$
    </div>
    <p>The key insight is the sign of the coupling. In standard chameleon theories, $A(\phi) = \exp(-\beta\phi/M_{\rm Pl})$ with $\beta > 0$, so the field is driven to smaller values (and $A < 1$). In the TEP framework (Paper 1), the sign is $A(\phi) = \exp(+\beta\phi/M_{\rm Pl})$. The field is still driven to a minimum of $V_{\rm eff}$, but now:</p>
    <ul>
        <li>In diffuse environments ($\rho \ll \rho_c$): The potential term dominates, allowing $\phi$ to relax to large values, giving $A(\phi) > 1$ (temporal enhancement active). This is the regime of galactic halos and high-$z$ galaxies.</li>
        <li>In dense environments ($\rho \gtrsim \rho_c \approx 20$ g/cm³): The matter coupling dominates, driving $\phi \to 0$ and $A(\phi) \to 1$ (screening). This is the Solar System regime where GR is recovered.</li>
    </ul>
    <p>Note: The critical density $\rho_c \approx 20$ g/cm³ (Paper 7, TEP-UCD) represents the core saturation scale of the scalar sector. At galactic scales, an effective screening transition occurs at much lower densities ($\rho_{\rm trans} \approx 0.5 M_\odot/\text{pc}^3$; Paper 12), which governs the phenomenology of Cepheid hosts and galaxy bulges.</p>
    <p>The thin-shell mechanism operates identically to standard chameleons. For a spherical body of radius $R$ and density $\rho_{\rm in}$ embedded in a background of density $\rho_{\rm out}$, the effective coupling is suppressed by:</p>
    <div class="equation">
        $$\alpha_{\rm eff} = \alpha_0 \times \frac{\Delta R}{R} \approx \alpha_0 \times \frac{\phi_{\rm out} - \phi_{\rm in}}{6\beta M_{\rm Pl} \Phi_N}$$
    </div>
    <p>where $\Phi_N = GM/Rc^2$ is the Newtonian potential. For the Sun ($\Phi_N \sim 10^{-6}$, $\rho \sim 1$ g/cm³), the thin-shell factor is $\Delta R/R \lesssim 10^{-6}$, reducing $\alpha_{\rm eff}$ to $\lesssim 10^{-6}$ and satisfying Cassini bounds ($|\gamma - 1| < 2.3 \times 10^{-5}$).</p>
    <p>The physical picture, consistent with the screening hierarchy established in Papers 7 and 11, is:</p>
    <ul>
        <li>Solar System ($\rho \sim 1$–$10$ g/cm³): Dense matter screens the scalar field via the thin-shell mechanism. Clocks tick at the GR rate ($A \approx 1$). All precision tests (Cassini, LLR, perihelion precession) are satisfied.</li>
        <li>Galactic Halos ($\rho \sim 10^{-24}$ g/cm³): Diffuse matter is far below the screening threshold. The field relaxes to its unscreened value, and clocks in deep potential wells (massive galaxy centers) experience $A > 1$, i.e., temporal enhancement. This is the regime probed by JWST observations.</li>
        <li>High Redshift: Lower cosmic mean density at early times means weaker ambient screening globally, amplifying the unscreened effect as $\alpha(z) = \alpha_0\sqrt{1+z}$ (Paper 12).</li>
    </ul>
    <p>This mechanism demonstrates that temporal enhancement ($\Gamma_t > 1$) is not in conflict with GR or Solar System tests. The apparent paradox arises from conflating the kinematic gravitational redshift (a photon effect, governed by $g_{00}$) with the dynamical clock rate (a matter effect, governed by $A(\phi)$). In the TEP framework, photons still redshift climbing out of potential wells, but atomic clocks in diffuse environments tick faster due to the scalar coupling.</p>
    <p>Caveat: The full theoretical treatment, including stability analysis, ghost-freedom proofs, and cosmological perturbation constraints, is presented in Paper 1. Cosmological viability (structure formation, BBN) remains an area for future work. The purpose of this summary is to show that the $\Gamma_t$ formula used in this observational analysis is derived from first principles, not fitted ad hoc.</p>

    <div class="data-table">
        <table>
            <caption>Table 2: TEP Model Parameters (Fixed)</caption>
            <thead>
                <tr><th>Parameter</th><th>Value</th><th>Source</th><th>Description</th></tr>
            </thead>
            <tbody>
                <tr><td>$\alpha_0$</td><td>$0.58 \pm 0.16$</td><td>TEP-H0 (Paper 12)</td><td>Coupling strength from Cepheids</td></tr>
                <tr><td>$z_{\rm ref}$</td><td>5.5</td><td>TEP-H0</td><td>Reference redshift for calibration</td></tr>
                <tr><td>$\log M_{h, \rm ref}$</td><td>12.0</td><td>TEP-COS</td><td>Reference halo mass ($\Gamma_t=1$)</td></tr>
                <tr><td>$\rho_c$</td><td>20 g/cm³</td><td>TEP-UCDCritical density for screening</td></tr>
            </tbody>
        </table>
    </div>

    <figure>
        <img src="public/figures/figure_1_tep_model.png" alt="The TEP Metric Coupling Gamma_t as a function of redshift and halo mass" style="width:100%; max-width:800px;">
        <figcaption>Figure 1: The TEP Metric Coupling $\Gamma_t(M_h, z)$. The enhancement factor increases with halo mass (potential depth) and redshift (weakening screening). The reference mass ($\log M_h = 12$) defines $\Gamma_t = 1$ (cosmic time flow). Massive halos at high redshift experience significant temporal enhancement ($\Gamma_t > 1$), while low-mass halos are suppressed ($\Gamma_t < 1$).</figcaption>
    </figure>

    <p>Parameter Calibration. Crucially, $\alpha_0 = 0.58$ and $z_{\rm ref} = 5.5$ are fixed based on independent calibration from Paper 12 (TEP-H0). In that work, $\alpha_0$ was derived from the period-luminosity residuals of Cepheids in massive hosts (e.g., M31, NGC 4258) relative to low-mass hosts. This leaves zero free parameters to be tuned to the JWST data.</p>

    <h4>2.3.3 Effective time and isochrony bias correction</h4>
    <p>An effective time is defined as $t_{\rm eff} = t_{\rm cosmic}\,\Gamma_t$, where $t_{\rm cosmic}$ is computed from a fiducial cosmology (Planck18). Under the isochrony-bias model used here, the mass-to-light ratio is assumed to scale as $M/L \propto t^{0.7}$. This power-law scaling is a robust feature of standard stellar population synthesis models (e.g., FSPS, BC03) for constant star formation histories at ages $t < 1$ Gyr (Conroy 2013). This implies a mass bias factor $\Gamma_t^{0.7}$, such that</p>
    <div class="equation">
        $$M_{*,\rm true} = M_{*,\rm obs}/\Gamma_t^{0.7}, \quad \mathrm{SFE}_{\rm true} = \mathrm{SFE}_{\rm obs}/\Gamma_t^{0.7}.$$
    </div>

    <h3>2.4 Statistical procedures</h3>
    <p>Associations are quantified using Spearman rank correlations and bootstrap confidence intervals. To address confounding by redshift and stellar mass, partial-correlation analyses implemented via residualization are employed. In addition to correlation-based tests, the following are reported:</p>
    <ul>
        <li>Stratified comparisons (e.g., high vs low $\Gamma_t$ splits) for multi-property coherence</li>
        <li>Distributional comparisons (e.g., Kolmogorov-Smirnov tests) for regime separation</li>
        <li>Model comparison using AIC/BIC for regression models that compare predictors {z}, {z, $\log M_*$}, {z, $\Gamma_t$}, and {z, $\log M_*$, $\Gamma_t$}</li>
    </ul>

    <h4>2.4.1 Selection bias quantification</h4>
    <p>At $z > 8$, only bright, star-forming galaxies are detected, biasing toward higher SFR, younger ages, and lower dust. To quantify these biases:</p>
    <ul>
        <li>Detection completeness is estimated as a function of mass and redshift using a sigmoid model based on typical JWST depth ($m_{\rm lim} \approx 28.5$ mag in F444W)</li>
        <li>Monte Carlo resampling (N = 1,000 iterations) with inverse-completeness weighting provides bias-corrected correlation estimates and adjusted p-values</li>
        <li>Bayes Factors are computed using the Savage-Dickey approximation to provide evidence ratios independent of frequentist thresholds</li>
    </ul>
    <p>For small-N bins (e.g., the high-mass $z > 7$ sample with $N = 10$), power analysis indicates a minimum detectable effect size of $|\rho| \approx 0.63$ at 80% power. This limitation is explicitly acknowledged in the interpretation of screening signatures (§3.6).</p>

    <h4>2.4.2 Mass sensitivity analysis</h4>
    <p>Recent MIRI studies (Pérez-González et al. 2024) indicate that NIRCam-only stellar masses may be overestimated by 0.5–1 dex at $z > 5$. To test robustness against this systematic, the pipeline applies mass reductions of 0.0, 0.3, 0.5, 0.7, and 1.0 dex, recomputes halo masses and $\Gamma_t$, and re-runs all key correlation tests. A signature is considered robust if it remains significant ($p < 0.05$) after a 0.5 dex mass reduction.</p>

    <h4>2.4.3 Forward-modeling validation</h4>
    <p>To address the concern that the $M/L \propto t^{0.7}$ assumption is adopted rather than tested, a forward-modeling approach is employed:</p>
    <ul>
        <li>The M/L power-law index is varied ($n = 0.5$–$0.9$) and the value that minimizes the residual mass-age correlation after TEP correction is identified</li>
        <li>Alternative hypotheses (bursty SFH, metallicity variations) are tested by computing partial correlations controlling for dust and metallicity</li>
        <li>If the mass-age correlation persists after controlling for standard astrophysical confounders, the TEP interpretation is supported</li>
    </ul>

    <h3>2.5 Black Hole Growth Simulation</h3>
    <p>To test the "Little Red Dot" resolution, a differential temporal shear simulation was developed (pipeline step 41). A compact galaxy ($r_e \approx 150$ pc) with a baryon-dominated core ($c=10$) is modeled. The local temporal enhancement factor $\Gamma_t(r)$ is computed at the center (Black Hole environment) and at the effective radius (Stellar environment) across the redshift range $z=4$–$10$.</p>
    <p>The differential growth factor is computed as:</p>
    <div class="equation">
        $$\text{Boost} = \exp\left(\frac{\int (\Gamma_{\rm cen}(z) - \Gamma_{\rm halo}(z)) \, dt_{\rm cosmic}}{t_{\rm Salpeter}}\right)$$
    </div>
    <p>where $t_{\rm Salpeter} \approx 45$ Myr is the Salpeter timescale (e-folding time for Eddington-limited accretion). This simulation uses the same $\alpha_0=0.58$ parameter calibrated from Cepheids, with no additional tuning.</p>
    <p>The simulation of runaway black hole growth (Section 4.12) integrates the differential time flow between the galactic center and the halo. The core integration loop from <code>scripts/steps/step_41_overmassive_bh.py</code>:</p>

    <h3>2.6 Reproducibility</h3>
    <p>A complete run is executed with <code>python scripts/steps/run_all_steps.py</code>, which produces step-indexed outputs in <code>results/outputs</code> and step-indexed logs in <code>logs</code>.</p>
</div>"""

with open('/Users/matthewsmawfield/www/TEP-JWST/site/components/3_methodology.html', 'w') as f:
    f.write(html_content)
