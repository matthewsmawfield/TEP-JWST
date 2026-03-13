# Author Note: Screening Mechanism — Theoretical Status

**Date:** 13 March 2026  
**Re:** Screening mechanism across the TEP paper series  
**Conclusion:** Paper 1's chameleon screening is theoretically consistent. No revision required.

---

## Summary

A cross-paper audit revealed that the TEP series uses different screening mechanisms without adequate acknowledgement: Paper 1 adopts chameleon screening, Papers 7 and 9 introduce a DBI kinetic generalization producing "Vainshtein-like" screening, and Paper 11 hedges correctly by calling the mechanism phenomenological.

On closer analysis, Paper 1 was not wrong. The chameleon mechanism follows directly from Paper 1's canonical action. The later papers quietly changed the Lagrangian by introducing a DBI kinetic term (Paper 7, Box 6.5) — a different theory, not a correction of an error. The empirical data ($S \propto \rho^{1/3}$) do not discriminate between the two mechanisms because the scaling is a geometric tautology given how $S$ and $R_{\text{sol}}$ are defined.

The JWST paper (Paper 13) now takes a mechanism-agnostic position: all predictions depend on the phenomenological saturation density $\rho_c \approx 20$ g/cm³, not on the specific dynamical mechanism. Both chameleon and DBI/kinetic screening are presented as viable UV completions. This is the scientifically honest position and is consistent with Paper 11's existing language.

---

## The Lagrangian Discrepancy

Paper 1 writes the TEP action with a canonical kinetic term:

$$S = \int d^4x \sqrt{-g} \left[ \frac{M_{\rm Pl}^2}{2} R - \frac{1}{2} K(\phi)(\partial\phi)^2 - V(\phi) \right] + S_{\rm matter}[\psi, \tilde{g}_{\mu\nu}]$$

This action, with $K(\phi) = 1$ and a chameleon potential $V(\phi) = \Lambda^4[1 + (\Lambda/\phi)^n]$, naturally produces chameleon screening via $V_{\text{eff}}(\phi;\rho) = V(\phi) + [A(\phi)-1]\rho$. There is no mechanism for Vainshtein or kinetic screening in this action. Paper 1's §4.3 is internally consistent.

Paper 7 (Box 6.5) introduces a different action with a DBI non-canonical kinetic term:

$$P(X,\phi) = -\Lambda^4\sqrt{1 - 2X/\Lambda^4} + \Lambda^4 - V(\phi)$$

This enforces gradient saturation at $|\nabla\phi| \leq \Lambda^2$, identifying $\rho_c \equiv \Lambda^4 \approx 20$ g/cm³. This is not a correction of Paper 1 — it is a generalization of the kinetic sector. In the low-gradient limit ($2X \ll \Lambda^4$), the DBI term reduces to the canonical term, so Paper 1's results hold as a limiting case.

---

## Why the $\rho^{1/3}$ Scaling Does Not Discriminate

Paper 7 itself acknowledges this (§4.2, interpretation note): "the regression $S \propto \rho^{0.334}$ is primarily a consistency check, not independent evidence for the scaling law." The screening factor $S = R_{\text{sol}}/R_{\text{phys}}$ with $R_{\text{sol}} = (M/\rho_c)^{1/3}$ algebraically gives $S \propto \rho^{1/3}$ for any spherical object. This is geometry, not a signature of a specific mechanism.

A chameleon potential tuned to the same $\rho_c$ would produce the same soliton radius, the same screening hierarchy, and the same $R^2 = 0.9999$ fit. The distinguishing observable would be the suppression profile in the transition regime ($S \sim 1$): Yukawa falloff (chameleon) vs. power-law saturation (DBI/kinetic). This has not been tested.

---

## What the Data Actually Constrain

The robust empirical finding across Papers 7–13 is:

- A single saturation density $\rho_c \approx 20$ g/cm³ produces a consistent screening hierarchy across 15 orders of magnitude in density
- This $\rho_c$ coincides with the onset of electron degeneracy pressure
- Earth sits in the transition regime ($S \approx 0.66$), making GNSS clocks sensitive to the scalar field
- Compact objects (white dwarfs, neutron stars, pulsars) are strongly screened, recovering GR

These findings are mechanism-independent. They depend on $\rho_c$ and $\alpha_0$, not on whether the underlying dynamics are chameleon or DBI.

---

## Current Paper 13 Position

The JWST paper (Paper 13, Appendix A.1.11) now presents:

- Screening as phenomenological, characterized by $\rho_c$
- Chameleon screening (Paper 1) as one viable UV completion, consistent with the canonical action
- Kinetic/DBI screening (Papers 7, 9) as a second viable UV completion, requiring a kinetic generalization
- Table A6 comparing the two mechanisms side by side
- Explicit statement that the $\rho^{1/3}$ scaling does not discriminate between them
- Identification of the transition-regime suppression profile as the key future test
- All predictions in the main text use only the phenomenological $\rho_c$ framework

This is consistent with Paper 11's position that the mechanism "mimics chameleon or Vainshtein screening" and "remains to be derived from first principles."

---

## Implications for Paper 1

Paper 1's chameleon screening in §4.3 is theoretically consistent with its own action. No correction is needed. If a future preprint revision is desired for other reasons, the author may wish to add a brief note to §4.3 acknowledging that the kinetic sector may be generalized (as explored in Papers 7, 9) and that the specific screening mechanism is an open theoretical question, but this is not urgent and the existing text is not incorrect.

The core TEP predictions — $\Gamma_t$, $\alpha_0$, synchronization holonomy, all metrology tests — are independent of the screening mechanism and remain unchanged across all papers.

